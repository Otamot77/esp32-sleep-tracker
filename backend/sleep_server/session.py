from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time

from sleep_dsp.models import EpochFeatures
from sleep_dsp.pipeline import RealtimeDspPipeline
from sleep_product.analysis import analyze_night
from sleep_product.history import HistoryStore
from sleep_physiology.models import SpO2Calibration
from sleep_physiology.pipeline import SlowPhysiologyPipeline

from .clock import ClockSynchronizer
from .config import settings
from .influx_writer import InfluxWriter
from .identity import make_session_id
from .recording import PacketRecorder


MIN_STAGE_SESSION_EPOCHS = 20


@dataclass
class SessionState:
    session_id: str
    boot_id: int
    started_at_ns: int

    clock: ClockSynchronizer = field(
        default_factory=ClockSynchronizer
    )
    dsp: RealtimeDspPipeline = field(
        default_factory=RealtimeDspPipeline
    )

    slow_physiology: SlowPhysiologyPipeline | None = None

    epochs: list[tuple[EpochFeatures, int]] = field(
        default_factory=list
    )

    previous_sequence: int | None = None

    packet_count: int = 0
    epoch_count: int = 0
    sequence_gap_events: int = 0
    duplicate_packets: int = 0

    connected_clients: int = 0
    reconnects: int = 0
    finalized: bool = False

    disconnect_generation: int = 0
    finalize_task: asyncio.Task | None = None

    recorder: PacketRecorder | None = None


class SessionManager:
    def __init__(
        self,
        influx: InfluxWriter,
        history: HistoryStore,
    ) -> None:
        self.influx = influx
        self.history = history

        self._sessions: dict[str, SessionState] = {}

    def acquire(
        self,
        boot_id: int,
        hello_device_time_us: int,
        server_time_ns: int,
    ) -> tuple[SessionState, bool]:
        session_id = make_session_id(
            settings.device_id,
            boot_id,
        )

        state = self._sessions.get(session_id)
        resumed = state is not None and not state.finalized

        if state is None or state.finalized:
            state = SessionState(
                session_id=session_id,
                boot_id=boot_id,
                started_at_ns=(
                    server_time_ns
                    - int(hello_device_time_us) * 1000
                ),
            )

            state.recorder = PacketRecorder(
                settings.recording_directory,
                session_id,
            )

            calibration = None

            if (
                settings.spo2_cal_a is not None
                and settings.spo2_cal_b is not None
                and settings.spo2_cal_c is not None
            ):
                calibration = SpO2Calibration(
                    a=settings.spo2_cal_a,
                    b=settings.spo2_cal_b,
                    c=settings.spo2_cal_c,
                )

            state.slow_physiology = SlowPhysiologyPipeline(
                calibration=calibration
            )

            self._sessions[session_id] = state
        else:
            state.reconnects += 1

        state.connected_clients += 1
        state.disconnect_generation += 1

        if (
            state.finalize_task is not None
            and not state.finalize_task.done()
        ):
            state.finalize_task.cancel()

        accepted = state.clock.add_anchor(
            hello_device_time_us,
            server_time_ns,
        )

        if not accepted:
            print(
                f"[clock] HELLO anchor rejected "
                f"session={session_id}"
            )

        return state, resumed

    def release(self, state: SessionState) -> None:
        if state.finalized:
            return

        state.connected_clients = max(
            0,
            state.connected_clients - 1,
        )

        if state.connected_clients > 0:
            return

        state.disconnect_generation += 1
        generation = state.disconnect_generation

        state.finalize_task = asyncio.create_task(
            self._finalize_after_grace(
                state,
                generation,
            )
        )

    async def _finalize_after_grace(
        self,
        state: SessionState,
        generation: int,
    ) -> None:
        try:
            await asyncio.sleep(
                settings.reconnect_grace_seconds
            )
        except asyncio.CancelledError:
            return

        if (
            state.connected_clients != 0
            or state.disconnect_generation != generation
            or state.finalized
        ):
            return

        await self.finalize(state)

    async def finalize(
        self,
        state: SessionState,
    ) -> None:
        if state.finalized:
            return

        state.finalized = True
        ended_at_ns = time.time_ns()

        clock_estimate = state.clock.estimate

        if len(state.epochs) >= MIN_STAGE_SESSION_EPOCHS:
            features_only = [
                features
                for features, _timestamp_ns in state.epochs
            ]

            baseline = self.history.baseline()

            analysis = await asyncio.to_thread(
                analyze_night,
                state.session_id,
                features_only,
                baseline,
            )

            for prediction, (
                _features,
                timestamp_ns,
            ) in zip(
                analysis.predictions,
                state.epochs,
            ):
                self.influx.write_stage_prediction(
                    prediction,
                    state.session_id,
                    timestamp_ns,
                )

            self.influx.write_night_summary(
                analysis,
                state.started_at_ns,
                ended_at_ns,
                sequence_gaps=state.sequence_gap_events,
                duplicate_packets=state.duplicate_packets,
                clock_drift_ppm=clock_estimate.drift_ppm,
                rejected_clock_anchors=(
                    clock_estimate.rejected_anchors
                ),
            )

            await asyncio.to_thread(
                self.history.append,
                analysis,
                state.started_at_ns,
                ended_at_ns,
            )

            self.influx.flush()

            print(
                "[night] "
                f"session={state.session_id} "
                f"TST={analysis.summary.total_sleep_minutes:.1f}min "
                f"eff={analysis.summary.sleep_efficiency_percent:.1f}% "
                f"recovery={analysis.recovery.score:.0f}"
                f"{'*' if analysis.recovery.provisional else ''} "
                f"gaps={state.sequence_gap_events} "
                f"duplicates={state.duplicate_packets} "
                f"drift={clock_estimate.drift_ppm:+.1f}ppm"
            )
        elif state.epochs:
            print(
                "[night] session too short for final staging: "
                f"{len(state.epochs)} epochs"
            )

        if state.recorder is not None:
            state.recorder.close()

        print(
            "[session] FINALIZED "
            f"id={state.session_id} "
            f"packets={state.packet_count} "
            f"epochs={state.epoch_count} "
            f"reconnects={state.reconnects}"
        )

        self._sessions.pop(
            state.session_id,
            None,
        )

    async def finalize_all(self) -> None:
        states = list(self._sessions.values())

        for state in states:
            if not state.finalized:
                await self.finalize(state)
