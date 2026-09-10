from __future__ import annotations

from collections import deque

import numpy as np

from sleep_dsp.constants import (
    GRAVITY_M_S2,
    IMU_SAMPLE_RATE_HZ,
    PPG_SAMPLE_RATE_HZ,
)
from sleep_server.protocol import (
    ImuPacket,
    PpgPacket,
)

from .models import (
    SlowPhysiologyEstimate,
    SpO2Calibration,
)
from .oximetry import estimate_oximetry
from .respiration import estimate_respiratory_rate


class SlowPhysiologyPipeline:
    """
    Rolling physiological estimates that need more context than one 30-second
    sleep-stage epoch.

    Output cadence: every 30 seconds
    Window:         last 120 seconds
    """

    def __init__(
        self,
        calibration: SpO2Calibration | None = None,
        window_seconds: float = 120.0,
        output_seconds: float = 30.0,
    ) -> None:
        self.window_us = int(
            window_seconds * 1_000_000
        )
        self.output_us = int(
            output_seconds * 1_000_000
        )

        self.calibration = calibration

        self.ppg: deque[PpgPacket] = deque()
        self.imu: deque[ImuPacket] = deque()

        self.latest_ppg_us: int | None = None
        self.latest_imu_us: int | None = None
        self.latest_device_us: int | None = None
        self.first_ppg_us: int | None = None
        self.first_imu_us: int | None = None
        self.next_output_us: int | None = None

    def add(
        self,
        packet: PpgPacket | ImuPacket,
    ) -> list[SlowPhysiologyEstimate]:
        if isinstance(packet, PpgPacket):
            self.ppg.append(packet)
            self.latest_ppg_us = packet.device_time_us

            if self.first_ppg_us is None:
                self.first_ppg_us = packet.device_time_us
        else:
            self.imu.append(packet)
            self.latest_imu_us = packet.device_time_us

            if self.first_imu_us is None:
                self.first_imu_us = packet.device_time_us

        if self.latest_device_us is None:
            self.latest_device_us = packet.device_time_us
        else:
            self.latest_device_us = max(
                self.latest_device_us,
                packet.device_time_us,
            )

        if self.next_output_us is None:
            self.next_output_us = (
                (
                    packet.device_time_us
                    // self.output_us
                )
                + 1
            ) * self.output_us

        self._trim(packet.device_time_us)

        return self._emit_ready()

    def _trim(self, latest_us: int) -> None:
        cutoff = latest_us - self.window_us - self.output_us

        while (
            self.ppg
            and self.ppg[0].device_time_us < cutoff
        ):
            self.ppg.popleft()

        while (
            self.imu
            and self.imu[0].device_time_us < cutoff
        ):
            self.imu.popleft()

    def _emit_ready(
        self,
    ) -> list[SlowPhysiologyEstimate]:
        results: list[SlowPhysiologyEstimate] = []

        if (
            self.next_output_us is None
            or self.latest_device_us is None
        ):
            return results

        while self.latest_device_us >= self.next_output_us:
            end_us = self.next_output_us
            start_us = end_us - self.window_us

            # Do not emit until a full 120-second window exists.
            if start_us >= 0 and self._has_full_history():
                results.append(
                    self._estimate(
                        start_us,
                        end_us,
                    )
                )

            self.next_output_us += self.output_us

        return results

    def _has_full_history(self) -> bool:
        if (
            self.first_ppg_us is None
            or self.first_imu_us is None
            or self.latest_ppg_us is None
            or self.latest_imu_us is None
        ):
            return False

        ppg_period_us = int(1_000_000 / PPG_SAMPLE_RATE_HZ)
        imu_period_us = int(1_000_000 / IMU_SAMPLE_RATE_HZ)

        return (
            self.latest_ppg_us - self.first_ppg_us
            >= self.window_us - ppg_period_us
            and self.latest_imu_us - self.first_imu_us
            >= self.window_us - imu_period_us
        )

    def _estimate(
        self,
        start_us: int,
        end_us: int,
    ) -> SlowPhysiologyEstimate:
        ppg = [
            p
            for p in self.ppg
            if start_us <= p.device_time_us < end_us
        ]

        imu = [
            p
            for p in self.imu
            if start_us <= p.device_time_us < end_us
        ]

        duration_s = (end_us - start_us) / 1_000_000.0
        ppg_complete = len(ppg) >= (
            duration_s * PPG_SAMPLE_RATE_HZ * 0.90
        )
        imu_complete = len(imu) >= (
            duration_s * IMU_SAMPLE_RATE_HZ * 0.90
        )
        window_complete = ppg_complete and imu_complete

        red = np.asarray(
            [p.red for p in ppg],
            dtype=float,
        )

        ir = np.asarray(
            [p.ir for p in ppg],
            dtype=float,
        )

        if imu:
            ax = np.asarray(
                [p.ax for p in imu],
                dtype=float,
            )
            ay = np.asarray(
                [p.ay for p in imu],
                dtype=float,
            )
            az = np.asarray(
                [p.az for p in imu],
                dtype=float,
            )
            gx = np.asarray(
                [p.gx for p in imu],
                dtype=float,
            )
            gy = np.asarray(
                [p.gy for p in imu],
                dtype=float,
            )
            gz = np.asarray(
                [p.gz for p in imu],
                dtype=float,
            )

            accel_g = (
                np.sqrt(ax * ax + ay * ay + az * az)
                / GRAVITY_M_S2
            )

            gyro_rad_s = np.sqrt(
                gx * gx + gy * gy + gz * gz
            )

            moving = (
                np.maximum(
                    accel_g - 1.0,
                    0.0,
                ) > 0.05
            ) | (
                gyro_rad_s > 0.1745329252
            )

            motion_fraction = float(
                np.mean(moving)
            )
        else:
            motion_fraction = 1.0

        ox = estimate_oximetry(
            red=red,
            ir=ir,
            sample_rate_hz=PPG_SAMPLE_RATE_HZ,
            motion_fraction=motion_fraction,
            calibration=self.calibration,
        )

        resp = estimate_respiratory_rate(
            ir=ir,
            sample_rate_hz=PPG_SAMPLE_RATE_HZ,
            motion_fraction=motion_fraction,
        )

        return SlowPhysiologyEstimate(
            start_device_time_us=start_us,
            end_device_time_us=end_us,
            ratio_of_ratios=ox.ratio_of_ratios,
            spo2_percent=ox.spo2_percent,
            spo2_quality=ox.quality,
            spo2_calibrated=ox.calibrated,
            spo2_usable=ox.usable and window_complete,
            respiratory_rate_bpm=(
                resp.breaths_per_minute
            ),
            respiratory_confidence=(
                resp.confidence
            ),
            respiration_usable=(
                resp.usable and window_complete
            ),
            motion_fraction=motion_fraction,
        )
