from __future__ import annotations

import numpy as np

from sleep_staging.stager import NightStager
from sleep_staging.stages import SleepStage
from sleep_staging.synthetic_night import make_synthetic_night
from sleep_staging.temporal import viterbi_smooth


def test_viterbi_removes_single_deep_to_rem_flip() -> None:
    # Mostly deep, with one raw REM-like epoch in the middle.
    emissions = np.array(
        [
            [0.01, 0.04, 0.93, 0.02],
            [0.01, 0.04, 0.92, 0.03],
            [0.01, 0.10, 0.30, 0.59],
            [0.01, 0.04, 0.92, 0.03],
            [0.01, 0.04, 0.93, 0.02],
        ],
        dtype=float,
    )

    path = viterbi_smooth(emissions)

    assert np.all(path == int(SleepStage.DEEP))


def test_synthetic_night_pipeline_is_reasonable() -> None:
    night = make_synthetic_night(seed=44)
    predictions, summary = NightStager().stage(night.epochs)

    assert len(predictions) == len(night.truth)

    accuracy = np.mean(
        [
            prediction.stage == truth
            for prediction, truth in zip(
                predictions,
                night.truth,
            )
        ]
    )

    # This checks the synthetic software path, not clinical accuracy.
    assert accuracy >= 0.72

    assert summary.total_sleep_minutes > 400.0
    assert 70.0 <= summary.sleep_efficiency_percent <= 100.0
    assert summary.deep_minutes > 20.0
    assert summary.rem_minutes > 20.0


def test_probabilities_sum_to_one() -> None:
    night = make_synthetic_night(seed=7)
    predictions, _ = NightStager().stage(night.epochs[:120])

    for prediction in predictions:
        total = (
            prediction.wake_probability
            + prediction.light_probability
            + prediction.deep_probability
            + prediction.rem_probability
        )

        assert abs(total - 1.0) < 1e-9


def test_bad_signal_is_retained_but_flagged() -> None:
    night = make_synthetic_night(seed=44)
    predictions, _ = NightStager().stage(night.epochs)

    bad = [
        prediction
        for prediction in predictions
        if not prediction.signal_usable
    ]

    assert len(bad) > 0
    assert all(
        prediction.confidence < 0.85
        for prediction in bad
    )
