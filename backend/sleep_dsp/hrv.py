from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch


@dataclass(frozen=True)
class TimeDomainHrv:
    mean_hr_bpm: float
    median_hr_bpm: float
    mean_rr_ms: float
    median_rr_ms: float
    rmssd_ms: float
    sdnn_ms: float
    pnn50_percent: float


@dataclass(frozen=True)
class FrequencyDomainHrv:
    lf_power_ms2: float
    hf_power_ms2: float
    lf_hf_ratio: float


def time_domain_hrv(rr_ms: np.ndarray) -> TimeDomainHrv:
    rr = np.asarray(rr_ms, dtype=float)

    if rr.size == 0:
        nan = float("nan")
        return TimeDomainHrv(
            mean_hr_bpm=nan,
            median_hr_bpm=nan,
            mean_rr_ms=nan,
            median_rr_ms=nan,
            rmssd_ms=nan,
            sdnn_ms=nan,
            pnn50_percent=nan,
        )

    instantaneous_hr = 60_000.0 / rr
    differences = np.diff(rr)

    rmssd = (
        float(np.sqrt(np.mean(differences**2)))
        if differences.size
        else float("nan")
    )

    sdnn = (
        float(np.std(rr, ddof=1))
        if rr.size >= 2
        else float("nan")
    )

    pnn50 = (
        float(np.mean(np.abs(differences) > 50.0) * 100.0)
        if differences.size
        else float("nan")
    )

    return TimeDomainHrv(
        mean_hr_bpm=float(np.mean(instantaneous_hr)),
        median_hr_bpm=float(np.median(instantaneous_hr)),
        mean_rr_ms=float(np.mean(rr)),
        median_rr_ms=float(np.median(rr)),
        rmssd_ms=rmssd,
        sdnn_ms=sdnn,
        pnn50_percent=pnn50,
    )


def frequency_domain_hrv(
    rr_ms: np.ndarray,
    minimum_duration_s: float = 240.0,
) -> FrequencyDomainHrv:
    """
    Frequency-domain HRV for longer windows.

    We intentionally do not use this for 30-second sleep epochs.
    """
    rr = np.asarray(rr_ms, dtype=float)

    if rr.size < 4:
        nan = float("nan")
        return FrequencyDomainHrv(nan, nan, nan)

    beat_times_s = np.cumsum(rr) / 1000.0
    beat_times_s -= beat_times_s[0]

    if beat_times_s[-1] < minimum_duration_s:
        nan = float("nan")
        return FrequencyDomainHrv(nan, nan, nan)

    interpolation_rate_hz = 4.0

    grid = np.arange(
        beat_times_s[0],
        beat_times_s[-1],
        1.0 / interpolation_rate_hz,
    )

    interpolator = interp1d(
        beat_times_s,
        rr,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    interpolated_rr = interpolator(grid)
    interpolated_rr -= np.mean(interpolated_rr)

    frequencies, power = welch(
        interpolated_rr,
        fs=interpolation_rate_hz,
        nperseg=min(256, interpolated_rr.size),
    )

    def band_power(low: float, high: float) -> float:
        mask = (frequencies >= low) & (frequencies < high)

        if not np.any(mask):
            return 0.0

        return float(np.trapezoid(power[mask], frequencies[mask]))

    lf = band_power(0.04, 0.15)
    hf = band_power(0.15, 0.40)
    ratio = lf / hf if hf > 0 else float("nan")

    return FrequencyDomainHrv(
        lf_power_ms2=lf,
        hf_power_ms2=hf,
        lf_hf_ratio=ratio,
    )
