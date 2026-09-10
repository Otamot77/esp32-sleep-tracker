from __future__ import annotations

import numpy as np

from sleep_dsp.models import EpochFeatures

from .baseline_model import ExplainableBaselineModel
from .context import build_context
from .models import NightSummary, StagePrediction
from .stages import SleepStage
from .summary import summarize_night
from .temporal import viterbi_smooth


class NightStager:
    """Estimate per-epoch stages, smooth the sequence, and summarize the night."""

    def __init__(
        self,
        model: ExplainableBaselineModel | None = None,
    ) -> None:
        self.model = model or ExplainableBaselineModel()

    def stage(
        self,
        epochs: list[EpochFeatures],
    ) -> tuple[list[StagePrediction], NightSummary]:
        if not epochs:
            return [], summarize_night([])

        contexts = build_context(epochs)

        emissions = np.vstack(
            [
                self.model.emission_probabilities(context)
                for context in contexts
            ]
        )

        raw_path = np.argmax(emissions, axis=1)
        smooth_path = viterbi_smooth(emissions)

        predictions: list[StagePrediction] = []

        for i, (context, probabilities) in enumerate(
            zip(contexts, emissions)
        ):
            stage = SleepStage(int(smooth_path[i]))
            raw_stage = SleepStage(int(raw_path[i]))

            # Confidence is the model probability assigned to the final
            # smoothed state, not simply the largest raw probability.
            confidence = float(probabilities[int(stage)])

            predictions.append(
                StagePrediction(
                    epoch_index=i,
                    start_device_time_us=(
                        context.epoch.start_device_time_us
                    ),
                    end_device_time_us=(
                        context.epoch.end_device_time_us
                    ),
                    raw_stage=raw_stage,
                    stage=stage,
                    wake_probability=float(probabilities[0]),
                    light_probability=float(probabilities[1]),
                    deep_probability=float(probabilities[2]),
                    rem_probability=float(probabilities[3]),
                    confidence=confidence,
                    signal_usable=(
                        context.epoch.usable_for_sleep_staging
                    ),
                    inferred_with_temporal_context=(
                        stage != raw_stage
                    ),
                )
            )

        return predictions, summarize_night(predictions)
