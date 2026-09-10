from __future__ import annotations

import math

import numpy as np

from sleep_staging.models import NightSummary

from .models import NightPhysiology, PersonalBaseline, RecoveryScore


DEFAULT_SLEEP_TARGET_MINUTES = 480.0


def _clip100(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _baseline_relative_high_is_good(
    current: float,
    baseline: float,
    scale_fraction: float,
) -> float:
    if baseline <= 0.0:
        return 50.0

    relative = (current - baseline) / (baseline * scale_fraction)
    return _clip100(50.0 + 50.0 * math.tanh(relative))


def _baseline_relative_low_is_good(
    current: float,
    baseline: float,
    scale_fraction: float,
) -> float:
    if baseline <= 0.0:
        return 50.0

    relative = (current - baseline) / (baseline * scale_fraction)
    return _clip100(50.0 - 50.0 * math.tanh(relative))


def score_recovery(
    summary: NightSummary,
    physiology: NightPhysiology,
    baseline: PersonalBaseline,
    sleep_target_minutes: float = DEFAULT_SLEEP_TARGET_MINUTES,
) -> RecoveryScore:
    """
    Transparent product heuristic.

    Weights:
      30% sleep duration
      20% sleep efficiency
      20% HRV relative to personal baseline
      20% sleeping HR relative to personal baseline
      10% usable sensor coverage

    Before a personal baseline exists, the two physiology components are held
    at a neutral 50/100 and the result is explicitly marked provisional.
    """
    if sleep_target_minutes <= 0.0:
        raise ValueError("sleep_target_minutes must be positive")

    duration_score = _clip100(
        100.0 * summary.total_sleep_minutes / sleep_target_minutes
    )

    # 70% or lower -> 0, 90% or higher -> 100.
    efficiency_score = _clip100(
        (summary.sleep_efficiency_percent - 70.0) / 20.0 * 100.0
    )

    signal_score = _clip100(summary.usable_signal_percent)

    hrv_delta_percent: float | None = None
    sleeping_hr_delta_bpm: float | None = None

    has_valid_current = (
        math.isfinite(physiology.median_sleep_rmssd_ms)
        and math.isfinite(physiology.median_sleep_hr_bpm)
    )

    if (
        baseline.ready
        and baseline.median_sleep_rmssd_ms is not None
        and baseline.median_sleep_hr_bpm is not None
        and has_valid_current
    ):
        hrv_score = _baseline_relative_high_is_good(
            physiology.median_sleep_rmssd_ms,
            baseline.median_sleep_rmssd_ms,
            scale_fraction=0.20,
        )

        sleeping_hr_score = _baseline_relative_low_is_good(
            physiology.median_sleep_hr_bpm,
            baseline.median_sleep_hr_bpm,
            scale_fraction=0.10,
        )

        hrv_delta_percent = (
            100.0
            * (
                physiology.median_sleep_rmssd_ms
                - baseline.median_sleep_rmssd_ms
            )
            / baseline.median_sleep_rmssd_ms
        )

        sleeping_hr_delta_bpm = (
            physiology.median_sleep_hr_bpm
            - baseline.median_sleep_hr_bpm
        )

        provisional = False
    else:
        # Use neutral HR and HRV scores until a personal baseline is available.
        hrv_score = 50.0
        sleeping_hr_score = 50.0
        provisional = True

    score = (
        0.30 * duration_score
        + 0.20 * efficiency_score
        + 0.20 * hrv_score
        + 0.20 * sleeping_hr_score
        + 0.10 * signal_score
    )

    return RecoveryScore(
        score=_clip100(score),
        provisional=provisional,
        sleep_duration_score=duration_score,
        sleep_efficiency_score=efficiency_score,
        hrv_score=hrv_score,
        sleeping_hr_score=sleeping_hr_score,
        signal_quality_score=signal_score,
        hrv_delta_percent=hrv_delta_percent,
        sleeping_hr_delta_bpm=sleeping_hr_delta_bpm,
    )
