from __future__ import annotations

import numpy as np

from .models import NightSummary, StagePrediction
from .stages import SleepStage


EPOCH_MINUTES = 0.5


def summarize_night(
    predictions: list[StagePrediction],
) -> NightSummary:
    if not predictions:
        return NightSummary(
            recording_minutes=0.0,
            total_sleep_minutes=0.0,
            sleep_efficiency_percent=0.0,
            sleep_onset_latency_minutes=0.0,
            wake_after_sleep_onset_minutes=0.0,
            wake_minutes=0.0,
            light_minutes=0.0,
            deep_minutes=0.0,
            rem_minutes=0.0,
            light_percent_of_sleep=0.0,
            deep_percent_of_sleep=0.0,
            rem_percent_of_sleep=0.0,
            usable_signal_percent=0.0,
            mean_stage_confidence=0.0,
        )

    stages = np.array([int(p.stage) for p in predictions], dtype=int)

    counts = {
        stage: int(np.sum(stages == int(stage)))
        for stage in SleepStage
    }

    recording_epochs = len(predictions)
    sleep_epochs = recording_epochs - counts[SleepStage.WAKE]

    recording_minutes = recording_epochs * EPOCH_MINUTES
    total_sleep_minutes = sleep_epochs * EPOCH_MINUTES

    sleep_efficiency = (
        100.0 * sleep_epochs / recording_epochs
        if recording_epochs
        else 0.0
    )

    sleep_indices = np.flatnonzero(stages != int(SleepStage.WAKE))

    if sleep_indices.size:
        first_sleep = int(sleep_indices[0])
        last_sleep = int(sleep_indices[-1])

        sleep_onset_latency = first_sleep * EPOCH_MINUTES

        waso_epochs = int(
            np.sum(
                stages[first_sleep : last_sleep + 1]
                == int(SleepStage.WAKE)
            )
        )
    else:
        sleep_onset_latency = recording_minutes
        waso_epochs = 0

    wake_minutes = counts[SleepStage.WAKE] * EPOCH_MINUTES
    light_minutes = counts[SleepStage.LIGHT] * EPOCH_MINUTES
    deep_minutes = counts[SleepStage.DEEP] * EPOCH_MINUTES
    rem_minutes = counts[SleepStage.REM] * EPOCH_MINUTES

    def pct_of_sleep(stage: SleepStage) -> float:
        if sleep_epochs == 0:
            return 0.0

        return 100.0 * counts[stage] / sleep_epochs

    usable = np.mean(
        [p.signal_usable for p in predictions],
        dtype=float,
    )

    confidence = np.mean(
        [p.confidence for p in predictions],
        dtype=float,
    )

    return NightSummary(
        recording_minutes=recording_minutes,
        total_sleep_minutes=total_sleep_minutes,
        sleep_efficiency_percent=float(sleep_efficiency),
        sleep_onset_latency_minutes=float(sleep_onset_latency),
        wake_after_sleep_onset_minutes=waso_epochs * EPOCH_MINUTES,
        wake_minutes=wake_minutes,
        light_minutes=light_minutes,
        deep_minutes=deep_minutes,
        rem_minutes=rem_minutes,
        light_percent_of_sleep=pct_of_sleep(SleepStage.LIGHT),
        deep_percent_of_sleep=pct_of_sleep(SleepStage.DEEP),
        rem_percent_of_sleep=pct_of_sleep(SleepStage.REM),
        usable_signal_percent=float(100.0 * usable),
        mean_stage_confidence=float(confidence),
    )
