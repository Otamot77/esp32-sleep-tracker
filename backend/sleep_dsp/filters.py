from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def robust_center(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)

    if signal.size == 0:
        return signal.copy()

    return signal - np.median(signal)


def bandpass_ppg(
    signal: np.ndarray,
    sample_rate_hz: float,
    low_hz: float = 0.5,
    high_hz: float = 5.0,
    order: int = 3,
) -> np.ndarray:
    """
    Remove slow baseline drift and high-frequency noise while preserving the
    pulse waveform.
    """
    x = robust_center(np.asarray(signal, dtype=float))

    if x.size < max(30, order * 10):
        return x.copy()

    nyquist = sample_rate_hz / 2.0

    if not (0.0 < low_hz < high_hz < nyquist):
        raise ValueError("Invalid band-pass frequencies")

    sos = butter(
        order,
        [low_hz / nyquist, high_hz / nyquist],
        btype="bandpass",
        output="sos",
    )

    return sosfiltfilt(sos, x)


def robust_sigma(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=float)

    if x.size == 0:
        return 0.0

    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return float(1.4826 * mad)
