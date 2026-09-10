from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .constants import (
    MAX_HEART_RATE_BPM,
    MIN_HEART_RATE_BPM,
    PPG_BANDPASS_HIGH_HZ,
    PPG_BANDPASS_LOW_HZ,
)
from .filters import bandpass_ppg, robust_sigma


@dataclass(frozen=True)
class BeatDetection:
    filtered: np.ndarray
    peak_indices: np.ndarray
    peak_times_s: np.ndarray
    rr_ms_raw: np.ndarray
    rr_ms_valid: np.ndarray
    rr_valid_mask: np.ndarray
    prominence_threshold: float


def clean_rr_intervals(
    rr_ms: np.ndarray,
    local_deviation_limit: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    rr = np.asarray(rr_ms, dtype=float)

    if rr.size == 0:
        return rr.copy(), np.zeros(0, dtype=bool)

    min_rr = 60_000.0 / MAX_HEART_RATE_BPM
    max_rr = 60_000.0 / MIN_HEART_RATE_BPM

    physiological = (rr >= min_rr) & (rr <= max_rr)
    local_ok = np.ones(rr.size, dtype=bool)

    if rr.size >= 3:
        for i in range(rr.size):
            lo = max(0, i - 2)
            hi = min(rr.size, i + 3)
            neighbors = rr[lo:hi]
            local_median = float(np.median(neighbors))

            if local_median > 0:
                relative_error = abs(rr[i] - local_median) / local_median
                local_ok[i] = relative_error <= local_deviation_limit

    valid_mask = physiological & local_ok
    return rr[valid_mask], valid_mask


def detect_beats(
    ir_signal: np.ndarray,
    sample_rate_hz: float,
    sample_times_s: np.ndarray | None = None,
) -> BeatDetection:
    raw = np.asarray(ir_signal, dtype=float)

    times: np.ndarray | None = None

    if sample_times_s is not None:
        times = np.asarray(sample_times_s, dtype=float)

        if times.shape != raw.shape:
            raise ValueError(
                "sample_times_s must match the PPG signal shape"
            )

    filtered = bandpass_ppg(
        raw,
        sample_rate_hz,
        PPG_BANDPASS_LOW_HZ,
        PPG_BANDPASS_HIGH_HZ,
    )

    if filtered.size < int(sample_rate_hz * 3):
        empty_i = np.zeros(0, dtype=int)
        empty_f = np.zeros(0, dtype=float)

        return BeatDetection(
            filtered=filtered,
            peak_indices=empty_i,
            peak_times_s=empty_f,
            rr_ms_raw=empty_f,
            rr_ms_valid=empty_f,
            rr_valid_mask=np.zeros(0, dtype=bool),
            prominence_threshold=0.0,
        )

    sigma = robust_sigma(filtered)
    prominence = max(0.35 * sigma, np.finfo(float).eps)

    min_peak_distance_s = 60.0 / MAX_HEART_RATE_BPM
    min_peak_distance_samples = max(
        1,
        int(sample_rate_hz * min_peak_distance_s),
    )

    peaks, _ = find_peaks(
        filtered,
        distance=min_peak_distance_samples,
        prominence=prominence,
    )

    peak_times_s = (
        times[peaks]
        if times is not None
        else peaks.astype(float) / sample_rate_hz
    )
    rr_ms_raw = np.diff(peak_times_s) * 1000.0

    rr_ms_valid, rr_valid_mask = clean_rr_intervals(rr_ms_raw)

    if times is not None and rr_valid_mask.size:
        sample_steps = np.diff(times)
        expected_step_s = 1.0 / sample_rate_hz

        broken_steps = (
            (sample_steps <= 0.0)
            | (sample_steps > expected_step_s * 1.5)
        )

        for i, (left, right) in enumerate(
            zip(peaks[:-1], peaks[1:])
        ):
            if np.any(broken_steps[left:right]):
                rr_valid_mask[i] = False

        rr_ms_valid = rr_ms_raw[rr_valid_mask]

    return BeatDetection(
        filtered=filtered,
        peak_indices=peaks,
        peak_times_s=peak_times_s,
        rr_ms_raw=rr_ms_raw,
        rr_ms_valid=rr_ms_valid,
        rr_valid_mask=rr_valid_mask,
        prominence_threshold=float(prominence),
    )
