from __future__ import annotations

import math

import numpy as np

from sleep_dsp.models import EpochFeatures
from sleep_staging.models import StagePrediction
from sleep_staging.stages import SleepStage

from .models import NightPhysiology


def _finite(values: list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    return result[np.isfinite(result)]


def derive_night_physiology(
    epochs: list[EpochFeatures],
    predictions: list[StagePrediction],
) -> NightPhysiology:
    if len(epochs) != len(predictions):
        raise ValueError("epochs and predictions must have the same length")

    usable_sleep: list[EpochFeatures] = []

    for epoch, prediction in zip(epochs, predictions):
        if (
            prediction.stage is not SleepStage.WAKE
            and epoch.usable_for_sleep_staging
            and math.isfinite(epoch.mean_hr_bpm)
            and math.isfinite(epoch.rmssd_ms)
        ):
            usable_sleep.append(epoch)

    hr = _finite([e.mean_hr_bpm for e in usable_sleep])
    rmssd = _finite([e.rmssd_ms for e in usable_sleep])

    return NightPhysiology(
        median_sleep_hr_bpm=(
            float(np.median(hr)) if hr.size else float("nan")
        ),
        median_sleep_rmssd_ms=(
            float(np.median(rmssd)) if rmssd.size else float("nan")
        ),
        usable_sleep_epochs=len(usable_sleep),
    )
