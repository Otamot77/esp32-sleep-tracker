from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from .models import RespiratoryEstimate


MIN_RESP_HZ = 0.10  # 6 breaths/min
MAX_RESP_HZ = 0.50  # 30 breaths/min
MIN_WINDOW_SECONDS = 60.0


def _respiratory_band(
    signal: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    x = x - np.median(x)

    nyquist = sample_rate_hz / 2.0

    sos = butter(
        3,
        [
            MIN_RESP_HZ / nyquist,
            MAX_RESP_HZ / nyquist,
        ],
        btype="bandpass",
        output="sos",
    )

    return sosfiltfilt(sos, x)


def estimate_respiratory_rate(
    ir: np.ndarray,
    sample_rate_hz: float,
    motion_fraction: float,
) -> RespiratoryEstimate:
    """
    Estimate breathing rate from slow respiratory modulation of the PPG.

    This is intentionally quality-gated because respiration is a much weaker
    component of wrist PPG than the cardiac pulse.
    """
    ir = np.asarray(ir, dtype=float)
    nan = float("nan")

    minimum_samples = int(
        sample_rate_hz * MIN_WINDOW_SECONDS
    )

    if ir.size < minimum_samples:
        return RespiratoryEstimate(
            breaths_per_minute=nan,
            confidence=0.0,
            usable=False,
        )

    respiratory = _respiratory_band(
        ir,
        sample_rate_hz,
    )

    # Downsample after low-frequency filtering. The respiratory band ends at
    # 0.5 Hz, so 4 Hz still has ample Nyquist margin.
    target_hz = 4.0
    step = max(
        1,
        int(round(sample_rate_hz / target_hz)),
    )

    downsampled = respiratory[::step]
    effective_hz = sample_rate_hz / step

    frequencies, power = welch(
        downsampled,
        fs=effective_hz,
        nperseg=min(512, downsampled.size),
    )

    band = (
        (frequencies >= MIN_RESP_HZ)
        & (frequencies <= MAX_RESP_HZ)
    )

    if not np.any(band):
        return RespiratoryEstimate(
            breaths_per_minute=nan,
            confidence=0.0,
            usable=False,
        )

    band_f = frequencies[band]
    band_p = power[band]

    peak_index = int(np.argmax(band_p))
    peak_hz = float(band_f[peak_index])

    total_band_power = float(np.sum(band_p))

    if total_band_power <= 0.0:
        return RespiratoryEstimate(
            breaths_per_minute=nan,
            confidence=0.0,
            usable=False,
        )

    # Fraction of respiratory-band power concentrated near the winning peak.
    resolution_hz = (
        float(np.median(np.diff(band_f)))
        if band_f.size >= 2
        else 0.02
    )

    neighborhood = np.abs(
        band_f - peak_hz
    ) <= max(0.025, 1.5 * resolution_hz)

    peak_concentration = float(
        np.sum(band_p[neighborhood])
        / total_band_power
    )

    motion_score = float(
        np.clip(
            1.0 - motion_fraction / 0.20,
            0.0,
            1.0,
        )
    )

    confidence = float(
        np.clip(
            0.80 * peak_concentration
            + 0.20 * motion_score,
            0.0,
            1.0,
        )
    )

    breaths_per_minute = peak_hz * 60.0

    usable = bool(
        math.isfinite(breaths_per_minute)
        and 6.0 <= breaths_per_minute <= 30.0
        and confidence >= 0.55
        and motion_fraction <= 0.20
    )

    return RespiratoryEstimate(
        breaths_per_minute=float(
            breaths_per_minute
        ),
        confidence=confidence,
        usable=usable,
    )
