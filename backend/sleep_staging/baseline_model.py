from __future__ import annotations

import math

import numpy as np

from .context import ContextualEpoch


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)

    z = math.exp(value)
    return z / (1.0 + z)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


class ExplainableBaselineModel:
    """Heuristic four-class staging baseline used until labeled PSG data is available."""

    def emission_probabilities(
        self,
        context: ContextualEpoch,
    ) -> np.ndarray:
        e = context.epoch

        # Motion is the main wake cue; HR and gyro add context.
        wake_logit = (
            -2.30
            + 17.0 * min(max(e.motion_fraction, 0.0), 0.5)
            + 0.65 * max(context.hr_z, 0.0)
            + 0.35 * max(context.gyro_z, 0.0)
        )

        p_wake = _sigmoid(wake_logit)

        # Broad priors for sleep-only classes; no single threshold determines a stage.
        light_logit = (
            1.15
            - 0.12 * abs(context.hr_z)
            - 0.10 * abs(context.rmssd_z)
            - 0.18 * abs(context.motion_z)
        )

        deep_logit = (
            0.35
            - 0.75 * context.hr_z
            + 0.55 * context.rmssd_z
            - 0.60 * context.motion_z
            - 0.55 * context.local_hr_variability_z
            + 0.95 * (1.0 - context.night_progress)
        )

        rem_logit = (
            0.05
            + 0.25 * context.hr_z
            + 0.10 * context.rmssd_z
            - 0.50 * context.motion_z
            + 0.95 * context.local_hr_variability_z
            + 1.20 * context.night_progress
        )

        sleep_conditional = _softmax(
            np.array(
                [light_logit, deep_logit, rem_logit],
                dtype=float,
            )
        )

        p_sleep = 1.0 - p_wake

        probabilities = np.array(
            [
                p_wake,
                p_sleep * sleep_conditional[0],
                p_sleep * sleep_conditional[1],
                p_sleep * sleep_conditional[2],
            ],
            dtype=float,
        )

        # Move low-quality epochs closer to an even, uncertain distribution.
        quality = min(max(float(e.ppg_quality), 0.0), 1.0)

        if not e.usable_for_sleep_staging:
            quality *= 0.55

        uniform = np.full(4, 0.25, dtype=float)
        probabilities = (
            quality * probabilities
            + (1.0 - quality) * uniform
        )

        probabilities /= np.sum(probabilities)
        return probabilities
