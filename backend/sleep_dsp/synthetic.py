from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import GRAVITY_M_S2


@dataclass(frozen=True)
class SyntheticEpoch:
    ppg_t_us: np.ndarray
    red: np.ndarray
    ir: np.ndarray

    imu_t_us: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    gz: np.ndarray


def make_epoch(
    duration_s: float = 30.0,
    ppg_rate_hz: int = 100,
    imu_rate_hz: int = 50,
    heart_rate_bpm: float = 60.0,
    rr_modulation_ms: float = 45.0,
    moving: bool = False,
    seed: int = 1,
) -> SyntheticEpoch:
    rng = np.random.default_rng(seed)

    beat_times = []
    t = 0.5
    beat_index = 0

    while t < duration_s:
        rr_ms = (
            60_000.0 / heart_rate_bpm
            + rr_modulation_ms * np.sin(
                2.0 * np.pi * beat_index / 5.0
            )
        )
        beat_times.append(t)
        t += rr_ms / 1000.0
        beat_index += 1

    ppg_t = np.arange(0.0, duration_s, 1.0 / ppg_rate_hz)
    pulse = np.zeros_like(ppg_t)

    for beat_t in beat_times:
        pulse += np.exp(-((ppg_t - beat_t) / 0.055) ** 2)
        pulse += 0.20 * np.exp(
            -((ppg_t - beat_t - 0.24) / 0.08) ** 2
        )

    baseline = 1000.0 * np.sin(2.0 * np.pi * 0.05 * ppg_t)

    ir = 120_000.0 + baseline + 18_000.0 * pulse
    red = 100_000.0 + 0.7 * baseline + 11_000.0 * pulse

    ir += rng.normal(0.0, 250.0, ppg_t.size)
    red += rng.normal(0.0, 220.0, ppg_t.size)

    imu_t = np.arange(0.0, duration_s, 1.0 / imu_rate_hz)

    if moving:
        ax = rng.normal(0.0, 1.4, imu_t.size)
        ay = rng.normal(0.0, 1.4, imu_t.size)
        az = GRAVITY_M_S2 + rng.normal(0.0, 1.2, imu_t.size)

        gx = rng.normal(0.0, 0.45, imu_t.size)
        gy = rng.normal(0.0, 0.45, imu_t.size)
        gz = rng.normal(0.0, 0.45, imu_t.size)

        artifact = 5000.0 * rng.normal(0.0, 1.0, ppg_t.size)
        ir += artifact
        red += 0.8 * artifact
    else:
        ax = rng.normal(0.0, 0.025, imu_t.size)
        ay = rng.normal(0.0, 0.025, imu_t.size)
        az = GRAVITY_M_S2 + rng.normal(0.0, 0.035, imu_t.size)

        gx = rng.normal(0.0, 0.003, imu_t.size)
        gy = rng.normal(0.0, 0.003, imu_t.size)
        gz = rng.normal(0.0, 0.003, imu_t.size)

    return SyntheticEpoch(
        ppg_t_us=(ppg_t * 1_000_000).astype(np.int64),
        red=red,
        ir=ir,
        imu_t_us=(imu_t * 1_000_000).astype(np.int64),
        ax=ax,
        ay=ay,
        az=az,
        gx=gx,
        gy=gy,
        gz=gz,
    )
