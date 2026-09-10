from __future__ import annotations

import json
from types import SimpleNamespace

from sleep_product.history import HistoryStore
from sleep_product.models import NightPhysiology, PersonalBaseline
from sleep_product.recovery import score_recovery
from sleep_staging.models import NightSummary


def summary() -> NightSummary:
    return NightSummary(
        recording_minutes=480.0,
        total_sleep_minutes=450.0,
        sleep_efficiency_percent=93.75,
        sleep_onset_latency_minutes=10.0,
        wake_after_sleep_onset_minutes=20.0,
        wake_minutes=30.0,
        light_minutes=220.0,
        deep_minutes=100.0,
        rem_minutes=130.0,
        light_percent_of_sleep=48.9,
        deep_percent_of_sleep=22.2,
        rem_percent_of_sleep=28.9,
        usable_signal_percent=97.0,
        mean_stage_confidence=0.78,
    )


def test_recovery_is_provisional_without_baseline() -> None:
    result = score_recovery(
        summary(),
        NightPhysiology(
            median_sleep_hr_bpm=55.0,
            median_sleep_rmssd_ms=60.0,
            usable_sleep_epochs=800,
        ),
        PersonalBaseline(
            median_sleep_hr_bpm=None,
            median_sleep_rmssd_ms=None,
            sample_nights=0,
            ready=False,
        ),
    )

    assert result.provisional
    assert result.hrv_score == 50.0
    assert result.sleeping_hr_score == 50.0
    assert 0.0 <= result.score <= 100.0


def test_better_than_baseline_physio_scores_above_neutral() -> None:
    result = score_recovery(
        summary(),
        NightPhysiology(
            median_sleep_hr_bpm=53.0,
            median_sleep_rmssd_ms=60.0,
            usable_sleep_epochs=800,
        ),
        PersonalBaseline(
            median_sleep_hr_bpm=58.0,
            median_sleep_rmssd_ms=50.0,
            sample_nights=8,
            ready=True,
        ),
    )

    assert not result.provisional
    assert result.hrv_score > 50.0
    assert result.sleeping_hr_score > 50.0
    assert result.hrv_delta_percent > 0.0
    assert result.sleeping_hr_delta_bpm < 0.0


def test_history_stores_missing_physiology_as_json_null(tmp_path) -> None:
    analysis = SimpleNamespace(
        session_id="test-session",
        physiology=SimpleNamespace(
            median_sleep_hr_bpm=float("nan"),
            median_sleep_rmssd_ms=float("nan"),
        ),
        summary=SimpleNamespace(
            total_sleep_minutes=0.0,
            sleep_efficiency_percent=0.0,
        ),
        recovery=SimpleNamespace(
            score=20.0,
            provisional=True,
        ),
    )

    path = tmp_path / "history.json"
    HistoryStore(path).append(analysis, 1, 2)
    record = json.loads(path.read_text(encoding="utf-8"))[0]

    assert record["median_sleep_hr_bpm"] is None
    assert record["median_sleep_rmssd_ms"] is None
