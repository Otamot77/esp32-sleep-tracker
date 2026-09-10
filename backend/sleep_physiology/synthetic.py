from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticRawPhysiology:
    red: np.ndarray
    ir: np.ndarray
    sample_rate_hz: float
    true_resp_bpm: float


def make_raw_ppg(
    duration_seconds: float = 120.0,
    sample_rate_hz: float = 100.0,
    heart_rate_bpm: float = 60.0,
    respiratory_rate_bpm: float = 15.0,
    seed: int = 123,
) -> SyntheticRawPhysiology:
    rng = np.random.default_rng(seed)

    t = np.arange(
        0.0,
        duration_seconds,
        1.0 / sample_rate_hz,
    )

    heart_hz = heart_rate_bpm / 60.0
    resp_hz = respiratory_rate_bpm / 60.0

    # Smooth cardiac component. The amplitudes differ by wavelength so the
    # ratio-of-ratios has a stable synthetic value.
    pulse = (
        np.sin(2.0 * np.pi * heart_hz * t)
        + 0.25
        * np.sin(4.0 * np.pi * heart_hz * t)
    )

    respiration = np.sin(
        2.0 * np.pi * resp_hz * t
    )

    # Respiratory baseline + amplitude modulation.
    ir = (
        120_000.0
        + 1_500.0 * respiration
        + (
            18_000.0
            * (1.0 + 0.08 * respiration)
            * pulse
        )
    )

    red = (
        100_000.0
        + 1_000.0 * respiration
        + (
            10_000.0
            * (1.0 + 0.06 * respiration)
            * pulse
        )
    )

    ir += rng.normal(0.0, 180.0, t.size)
    red += rng.normal(0.0, 160.0, t.size)

    return SyntheticRawPhysiology(
        red=red,
        ir=ir,
        sample_rate_hz=sample_rate_hz,
        true_resp_bpm=respiratory_rate_bpm,
    )
