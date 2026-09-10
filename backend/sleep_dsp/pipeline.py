from __future__ import annotations

from collections import deque

import numpy as np

from sleep_server.protocol import ImuPacket, PpgPacket

from .constants import EPOCH_SECONDS
from .epochs import extract_epoch_features
from .models import EpochFeatures


class RealtimeDspPipeline:
    def __init__(self, epoch_seconds: float = EPOCH_SECONDS) -> None:
        self.epoch_us = int(epoch_seconds * 1_000_000)

        self.ppg: deque[PpgPacket] = deque()
        self.imu: deque[ImuPacket] = deque()

        self.epoch_start_us: int | None = None
        self.latest_device_us: int | None = None

    def reset(self) -> None:
        self.ppg.clear()
        self.imu.clear()
        self.epoch_start_us = None
        self.latest_device_us = None

    def add(
        self,
        packet: PpgPacket | ImuPacket,
    ) -> list[EpochFeatures]:
        if self.epoch_start_us is None:
            self.epoch_start_us = (
                packet.device_time_us // self.epoch_us
            ) * self.epoch_us

        if isinstance(packet, PpgPacket):
            self.ppg.append(packet)
        else:
            self.imu.append(packet)

        if self.latest_device_us is None:
            self.latest_device_us = packet.device_time_us
        else:
            self.latest_device_us = max(
                self.latest_device_us,
                packet.device_time_us,
            )

        return self._emit_ready_epochs()

    def _emit_ready_epochs(self) -> list[EpochFeatures]:
        results: list[EpochFeatures] = []

        if (
            self.epoch_start_us is None
            or self.latest_device_us is None
        ):
            return results

        while True:
            end_us = self.epoch_start_us + self.epoch_us

            if self.latest_device_us < end_us:
                break

            results.append(
                self._process_epoch(self.epoch_start_us, end_us)
            )

            self.epoch_start_us = end_us
            self._drop_before(self.epoch_start_us)

        return results

    def _process_epoch(
        self,
        start_us: int,
        end_us: int,
    ) -> EpochFeatures:
        ppg = [
            p for p in self.ppg
            if start_us <= p.device_time_us < end_us
        ]

        imu = [
            p for p in self.imu
            if start_us <= p.device_time_us < end_us
        ]

        return extract_epoch_features(
            start_device_time_us=start_us,
            end_device_time_us=end_us,
            ppg_t_us=np.array(
                [p.device_time_us for p in ppg],
                dtype=np.int64,
            ),
            red=np.array([p.red for p in ppg], dtype=float),
            ir=np.array([p.ir for p in ppg], dtype=float),
            imu_t_us=np.array(
                [p.device_time_us for p in imu],
                dtype=np.int64,
            ),
            ax=np.array([p.ax for p in imu], dtype=float),
            ay=np.array([p.ay for p in imu], dtype=float),
            az=np.array([p.az for p in imu], dtype=float),
            gx=np.array([p.gx for p in imu], dtype=float),
            gy=np.array([p.gy for p in imu], dtype=float),
            gz=np.array([p.gz for p in imu], dtype=float),
        )

    def _drop_before(self, cutoff_us: int) -> None:
        while self.ppg and self.ppg[0].device_time_us < cutoff_us:
            self.ppg.popleft()

        while self.imu and self.imu[0].device_time_us < cutoff_us:
            self.imu.popleft()
