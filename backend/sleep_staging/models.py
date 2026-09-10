from __future__ import annotations

from dataclasses import asdict, dataclass

from .stages import SleepStage, stage_name


@dataclass(frozen=True)
class StagePrediction:
    epoch_index: int
    start_device_time_us: int
    end_device_time_us: int

    raw_stage: SleepStage
    stage: SleepStage

    wake_probability: float
    light_probability: float
    deep_probability: float
    rem_probability: float

    confidence: float
    signal_usable: bool
    inferred_with_temporal_context: bool

    def probability(self, stage: SleepStage) -> float:
        return (
            self.wake_probability,
            self.light_probability,
            self.deep_probability,
            self.rem_probability,
        )[int(stage)]

    @property
    def stage_name(self) -> str:
        return stage_name(self.stage)

    @property
    def raw_stage_name(self) -> str:
        return stage_name(self.raw_stage)

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["stage"] = self.stage_name
        result["raw_stage"] = self.raw_stage_name
        return result


@dataclass(frozen=True)
class NightSummary:
    recording_minutes: float
    total_sleep_minutes: float
    sleep_efficiency_percent: float

    sleep_onset_latency_minutes: float
    wake_after_sleep_onset_minutes: float

    wake_minutes: float
    light_minutes: float
    deep_minutes: float
    rem_minutes: float

    light_percent_of_sleep: float
    deep_percent_of_sleep: float
    rem_percent_of_sleep: float

    usable_signal_percent: float
    mean_stage_confidence: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)
