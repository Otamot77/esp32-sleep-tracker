from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


NOMINAL_NS_PER_US = 1000.0
MAX_DRIFT_PPM = 500.0
MAX_ANCHORS = 32
MAX_ANCHOR_RESIDUAL_NS = 250_000_000


@dataclass(frozen=True)
class ClockEstimate:
    slope_ns_per_us: float
    drift_ppm: float
    anchors: int
    rejected_anchors: int


class ClockSynchronizer:
    """
    Maps ESP32 monotonic microseconds to server wall-clock nanoseconds.

    Each HELLO/SYNC gives an anchor:
        (device_time_us, server_receive_time_ns)

    A centered least-squares fit estimates clock slope/drift while clamping
    implausible oscillator error. Old buffered sensor samples can then be
    timestamped by when they were measured, not when backlog finally arrived.
    """

    def __init__(self) -> None:
        self._anchors: deque[tuple[int, int]] = deque(
            maxlen=MAX_ANCHORS
        )
        self._slope = NOMINAL_NS_PER_US
        self._reference_device_us = 0
        self._reference_server_ns = 0
        self._rejected = 0

    def add_anchor(
        self,
        device_time_us: int,
        server_time_ns: int,
    ) -> bool:
        if self._anchors:
            predicted = self.to_wall_ns(device_time_us)
            residual = abs(server_time_ns - predicted)

            # Once there are several good anchors, reject a sync that was
            # obviously delayed by a server stall/network backlog.
            if (
                len(self._anchors) >= 3
                and residual > MAX_ANCHOR_RESIDUAL_NS
            ):
                self._rejected += 1
                return False

        self._anchors.append(
            (int(device_time_us), int(server_time_ns))
        )

        self._refit()
        return True

    def _refit(self) -> None:
        if not self._anchors:
            return

        if len(self._anchors) == 1:
            device_us, server_ns = self._anchors[0]
            self._slope = NOMINAL_NS_PER_US
            self._reference_device_us = device_us
            self._reference_server_ns = server_ns
            return

        device = np.asarray(
            [a[0] for a in self._anchors],
            dtype=np.float64,
        )
        server = np.asarray(
            [a[1] for a in self._anchors],
            dtype=np.float64,
        )

        device_mean = float(np.mean(device))
        server_mean = float(np.mean(server))

        centered_device = device - device_mean
        centered_server = server - server_mean

        denominator = float(
            np.dot(centered_device, centered_device)
        )

        if denominator <= 0.0:
            slope = NOMINAL_NS_PER_US
        else:
            slope = float(
                np.dot(
                    centered_device,
                    centered_server,
                )
                / denominator
            )

        ppm_fraction = MAX_DRIFT_PPM / 1_000_000.0

        slope = float(
            np.clip(
                slope,
                NOMINAL_NS_PER_US * (1.0 - ppm_fraction),
                NOMINAL_NS_PER_US * (1.0 + ppm_fraction),
            )
        )

        self._slope = slope
        self._reference_device_us = int(round(device_mean))
        self._reference_server_ns = int(round(server_mean))

    def to_wall_ns(
        self,
        device_time_us: int,
    ) -> int:
        if not self._anchors:
            raise RuntimeError(
                "ClockSynchronizer has no clock anchor"
            )

        delta_us = (
            int(device_time_us) -
            self._reference_device_us
        )

        return int(round(
            self._reference_server_ns +
            delta_us * self._slope
        ))

    @property
    def estimate(self) -> ClockEstimate:
        drift_ppm = (
            (self._slope / NOMINAL_NS_PER_US) - 1.0
        ) * 1_000_000.0

        return ClockEstimate(
            slope_ns_per_us=self._slope,
            drift_ppm=float(drift_ppm),
            anchors=len(self._anchors),
            rejected_anchors=self._rejected,
        )
