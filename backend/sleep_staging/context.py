from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sleep_dsp.models import EpochFeatures


@dataclass(frozen=True)
class ContextualEpoch:
    epoch: EpochFeatures
    epoch_index: int

    night_progress: float

    hr_z: float
    rmssd_z: float
    motion_z: float
    gyro_z: float
    local_hr_variability_z: float


def _robust_z(values: np.ndarray) -> np.ndarray:
    """
    Median/MAD normalization.

    Unlike mean/std, a few wake or motion epochs do not dominate the baseline.
    """
    x = np.asarray(values, dtype=float)

    finite = np.isfinite(x)

    if not np.any(finite):
        return np.zeros_like(x)

    median = float(np.median(x[finite]))
    mad = float(np.median(np.abs(x[finite] - median)))

    # 1.4826 turns MAD into a standard-deviation-like scale for Gaussian data.
    scale = 1.4826 * mad

    if scale < 1e-9:
        scale = float(np.std(x[finite]))

    if scale < 1e-9:
        scale = 1.0

    z = (x - median) / scale
    z[~finite] = 0.0

    # Extreme artifacts should not create arbitrarily huge logits.
    return np.clip(z, -4.0, 4.0)


def build_context(
    epochs: list[EpochFeatures],
    local_window_epochs: int = 5,
) -> list[ContextualEpoch]:
    if not epochs:
        return []

    hr = np.array([e.mean_hr_bpm for e in epochs], dtype=float)
    rmssd = np.array([e.rmssd_ms for e in epochs], dtype=float)
    motion = np.array([e.motion_fraction for e in epochs], dtype=float)
    gyro = np.array([e.gyro_rms_dps for e in epochs], dtype=float)

    # Local HR variability is intentionally an inter-epoch feature.
    # REM tends to be more autonomically irregular than stable deep sleep,
    # but this is only a weak heuristic, not a diagnostic rule.
    local_hr_var = np.zeros(len(epochs), dtype=float)

    half = local_window_epochs // 2

    for i in range(len(epochs)):
        lo = max(0, i - half)
        hi = min(len(epochs), i + half + 1)

        window = hr[lo:hi]
        finite = window[np.isfinite(window)]

        local_hr_var[i] = (
            float(np.std(finite))
            if finite.size >= 2
            else 0.0
        )

    hr_z = _robust_z(hr)
    rmssd_z = _robust_z(rmssd)
    motion_z = _robust_z(motion)
    gyro_z = _robust_z(gyro)
    local_hr_var_z = _robust_z(local_hr_var)

    denominator = max(1, len(epochs) - 1)

    return [
        ContextualEpoch(
            epoch=epoch,
            epoch_index=i,
            night_progress=i / denominator,
            hr_z=float(hr_z[i]),
            rmssd_z=float(rmssd_z[i]),
            motion_z=float(motion_z[i]),
            gyro_z=float(gyro_z[i]),
            local_hr_variability_z=float(local_hr_var_z[i]),
        )
        for i, epoch in enumerate(epochs)
    ]
