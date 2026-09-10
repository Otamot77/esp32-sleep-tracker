from __future__ import annotations

import numpy as np


def ppg_quality_score(
    beat_count: int,
    rr_valid_mask: np.ndarray,
    motion_fraction: float,
    epoch_seconds: float,
) -> float:
    """
    Prototype 0..1 signal-quality score, not a clinically validated SQI.
    """
    if epoch_seconds <= 0:
        return 0.0

    expected_min_beats = epoch_seconds * 30.0 / 60.0
    expected_max_beats = epoch_seconds * 120.0 / 60.0

    plausible_count = (
        expected_min_beats <= beat_count <= expected_max_beats
    )

    rr_valid_fraction = (
        float(np.mean(rr_valid_mask))
        if rr_valid_mask.size
        else 0.0
    )

    motion_score = float(
        np.clip(1.0 - motion_fraction / 0.35, 0.0, 1.0)
    )

    count_score = 1.0 if plausible_count else 0.35

    return float(
        np.clip(
            0.45 * rr_valid_fraction
            + 0.40 * motion_score
            + 0.15 * count_score,
            0.0,
            1.0,
        )
    )
