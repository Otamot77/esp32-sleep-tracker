from __future__ import annotations

import asyncio
import time

from sleep_server.protocol import (
    HelloPacket,
    ImuPacket,
    PacketParser,
    PpgPacket,
    ProtocolError,
    SyncPacket,
    TelemetryPacket,
)

from .config import settings
from .influx_writer import InfluxWriter
from .session import SessionManager, SessionState
from .sequence import SequenceStatus, classify_sequence


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    sessions: SessionManager,
) -> None:
    peer = writer.get_extra_info("peername")

    parser = PacketParser()
    state: SessionState | None = None

    # Preserve bytes received before a fragmented HELLO becomes complete.
    pending_record_bytes = bytearray()

    print(f"[tcp] connected peer={peer}")

    try:
        while True:
            chunk = await reader.read(4096)

            if not chunk:
                break

            received_ns = time.time_ns()

            if state is None:
                pending_record_bytes.extend(chunk)
            elif state.recorder is not None:
                state.recorder.append(chunk)

            try:
                packets = parser.feed(chunk)
            except ProtocolError as exc:
                print(f"[protocol] peer={peer}: {exc}")
                break

            for packet in packets:
                if isinstance(packet, HelloPacket):
                    if state is None:
                        state, resumed = sessions.acquire(
                            boot_id=packet.boot_id,
                            hello_device_time_us=(
                                packet.device_time_us
                            ),
                            server_time_ns=received_ns,
                        )

                        if state.recorder is not None:
                            state.recorder.append(
                                bytes(pending_record_bytes)
                            )

                        pending_record_bytes.clear()

                        print(
                            "[session] "
                            f"{'RESUME' if resumed else 'START'} "
                            f"id={state.session_id} "
                            f"peer={peer}"
                        )
                    else:
                        if packet.boot_id != state.boot_id:
                            print(
                                "[protocol] boot ID changed "
                                "inside one TCP connection"
                            )
                            return

                        state.clock.add_anchor(
                            packet.device_time_us,
                            received_ns,
                        )

                    continue

                if state is None:
                    print(
                        "[protocol] sensor/control data "
                        "received before HELLO"
                    )
                    return

                if isinstance(packet, SyncPacket):
                    accepted = state.clock.add_anchor(
                        packet.device_time_us,
                        received_ns,
                    )

                    estimate = state.clock.estimate

                    print(
                        "[clock] "
                        f"session={state.session_id} "
                        f"anchor={'ok' if accepted else 'rejected'} "
                        f"drift={estimate.drift_ppm:+.1f}ppm "
                        f"n={estimate.anchors}"
                    )
                    continue

                if isinstance(packet, TelemetryPacket):
                    measurement_ns = state.clock.to_wall_ns(
                        packet.device_time_us
                    )

                    estimate = state.clock.estimate

                    sessions.influx.write_device_health(
                        telemetry=packet,
                        session_id=state.session_id,
                        measurement_ns=measurement_ns,
                        sequence_gaps=(
                            state.sequence_gap_events
                        ),
                        duplicate_packets=(
                            state.duplicate_packets
                        ),
                        clock_drift_ppm=(
                            estimate.drift_ppm
                        ),
                        rejected_clock_anchors=(
                            estimate.rejected_anchors
                        ),
                    )

                    print(
                        "[health] "
                        f"session={state.session_id} "
                        f"queue={packet.queue_depth} "
                        f"high={packet.queue_high_watermark} "
                        f"dropped={packet.dropped_packets} "
                        f"reconnects={packet.reconnect_count} "
                        f"send_failures={packet.send_failures}"
                    )
                    continue

                if not isinstance(
                    packet,
                    (PpgPacket, ImuPacket),
                ):
                    continue

                if state.previous_sequence is not None:
                    check = classify_sequence(
                        state.previous_sequence,
                        packet.sequence,
                    )

                    if (
                        check.status
                        is SequenceStatus.DUPLICATE_OR_OLD
                    ):
                        state.duplicate_packets += 1

                        print(
                            "[duplicate] "
                            f"session={state.session_id} "
                            f"previous={state.previous_sequence} "
                            f"current={packet.sequence}"
                        )

                        # A retransmitted sample must never enter DSP twice.
                        continue

                    if check.status is SequenceStatus.GAP:
                        state.sequence_gap_events += (
                            check.missing
                        )

                        print(
                            "[gap] "
                            f"session={state.session_id} "
                            f"previous={state.previous_sequence} "
                            f"current={packet.sequence} "
                            f"missing≈{check.missing}"
                        )

                state.previous_sequence = packet.sequence
                state.packet_count += 1

                measurement_ns = state.clock.to_wall_ns(
                    packet.device_time_us
                )

                sessions.influx.write_packet(
                    packet,
                    state.session_id,
                    measurement_ns,
                )

                completed_epochs = state.dsp.add(packet)

                completed_slow = (
                    state.slow_physiology.add(packet)
                    if state.slow_physiology is not None
                    else []
                )

                for estimate in completed_slow:
                    estimate_ns = state.clock.to_wall_ns(
                        estimate.end_device_time_us
                    )

                    sessions.influx.write_slow_physiology(
                        estimate,
                        state.session_id,
                        estimate_ns,
                    )

                    spo2_text = (
                        f"{estimate.spo2_percent:.1f}%"
                        if estimate.spo2_calibrated
                        and estimate.spo2_usable
                        else "uncalibrated"
                    )

                    print(
                        "[slow] "
                        f"session={state.session_id} "
                        f"SpO2={spo2_text} "
                        f"R={estimate.ratio_of_ratios:.3f} "
                        f"RR={estimate.respiratory_rate_bpm:.1f} "
                        f"RRq={estimate.respiratory_confidence:.2f}"
                    )

                for features in completed_epochs:
                    state.epoch_count += 1

                    epoch_ns = state.clock.to_wall_ns(
                        features.end_device_time_us
                    )

                    sessions.influx.write_epoch_features(
                        features,
                        state.session_id,
                        epoch_ns,
                    )

                    state.epochs.append(
                        (features, epoch_ns)
                    )

                    print(
                        "[epoch] "
                        f"session={state.session_id} "
                        f"#{state.epoch_count} "
                        f"HR={features.mean_hr_bpm:.1f} "
                        f"RMSSD={features.rmssd_ms:.1f}ms "
                        f"motion={features.motion_fraction:.2f} "
                        f"quality={features.ppg_quality:.2f} "
                        f"usable={features.usable_for_sleep_staging}"
                    )

                if state.packet_count % 5000 == 0:
                    estimate = state.clock.estimate

                    print(
                        "[tcp] "
                        f"session={state.session_id} "
                        f"packets={state.packet_count} "
                        f"gaps={state.sequence_gap_events} "
                        f"drift={estimate.drift_ppm:+.1f}ppm"
                    )

    finally:
        if state is not None:
            sessions.release(state)

            print(
                "[tcp] disconnected "
                f"session={state.session_id} "
                f"peer={peer}; waiting "
                f"{settings.reconnect_grace_seconds:.0f}s "
                "before finalizing"
            )
        else:
            print(
                f"[tcp] disconnected peer={peer} "
                "before HELLO"
            )

        writer.close()

        try:
            await writer.wait_closed()
        except ConnectionError:
            pass


async def run() -> None:
    influx = InfluxWriter()

    from sleep_product.history import HistoryStore

    history = HistoryStore(settings.history_path)
    sessions = SessionManager(influx, history)

    server = await asyncio.start_server(
        lambda reader, writer: handle_client(
            reader,
            writer,
            sessions,
        ),
        settings.tcp_host,
        settings.tcp_port,
    )

    addresses = ", ".join(
        str(sock.getsockname())
        for sock in server.sockets or []
    )

    print(f"[server] listening on {addresses}")

    try:
        async with server:
            await server.serve_forever()
    finally:
        await sessions.finalize_all()
        influx.flush()
        influx.close()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[server] stopped")
