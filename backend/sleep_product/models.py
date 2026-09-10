from __future__ import annotations

from dataclasses import asdict, dataclass

from sleep_staging.models import NightSummary, StagePrediction


@dataclass(frozen=True)
class NightPhysiology:
    """
    Night-level values from usable sleeping epochs.

    Sleeping HR is not the same as a clinical resting-HR measurement.
    """

    median_sleep_hr_bpm: float
    median_sleep_rmssd_ms: float
    usable_sleep_epochs: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class PersonalBaseline:
    median_sleep_hr_bpm: float | None
    median_sleep_rmssd_ms: float | None
    sample_nights: int
    ready: bool

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class RecoveryScore:
    """
    Explainable, heuristic 0..100 recovery/readiness score.

    This is a product metric, not a medical diagnostic.
    """

    score: float
    provisional: bool

    sleep_duration_score: float
    sleep_efficiency_score: float
    hrv_score: float
    sleeping_hr_score: float
    signal_quality_score: float

    hrv_delta_percent: float | None
    sleeping_hr_delta_bpm: float | None

    def as_dict(self) -> dict[str, float | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class NightAnalysis:
    session_id: str
    predictions: list[StagePrediction]
    summary: NightSummary
    physiology: NightPhysiology
    baseline_before_night: PersonalBaseline
    recovery: RecoveryScore

    def summary_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "session_id": self.session_id,
        }
        result.update(self.summary.as_dict())
        result.update(self.physiology.as_dict())
        result.update({
            f"baseline_{k}": v
            for k, v in self.baseline_before_night.as_dict().items()
        })
        result.update({
            f"recovery_{k}": v
            for k, v in self.recovery.as_dict().items()
        })
        return result
