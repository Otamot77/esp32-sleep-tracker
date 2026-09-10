from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SpO2Calibration:
    """
    Quadratic mapping from red/IR ratio-of-ratios to SpO2:

        SpO2 = a*R^2 + b*R + c

    The coefficients must come from reference calibration data.
    """

    a: float
    b: float
    c: float

    def apply(self, ratio: float) -> float:
        return (
            self.a * ratio * ratio
            + self.b * ratio
            + self.c
        )


@dataclass(frozen=True)
class OximetryEstimate:
    ratio_of_ratios: float
    spo2_percent: float
    quality: float
    calibrated: bool
    usable: bool


@dataclass(frozen=True)
class RespiratoryEstimate:
    breaths_per_minute: float
    confidence: float
    usable: bool


@dataclass(frozen=True)
class SlowPhysiologyEstimate:
    start_device_time_us: int
    end_device_time_us: int

    ratio_of_ratios: float
    spo2_percent: float
    spo2_quality: float
    spo2_calibrated: bool
    spo2_usable: bool

    respiratory_rate_bpm: float
    respiratory_confidence: float
    respiration_usable: bool

    motion_fraction: float

    def as_dict(self) -> dict[str, float | bool | int]:
        return asdict(self)
