from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EpochFeatures:
    start_device_time_us: int
    end_device_time_us: int

    ppg_sample_count: int
    imu_sample_count: int
    beat_count: int
    valid_rr_count: int

    mean_hr_bpm: float
    median_hr_bpm: float

    rmssd_ms: float
    sdnn_ms: float
    pnn50_percent: float

    mean_rr_ms: float
    median_rr_ms: float

    accel_enmo_mean_mg: float
    accel_dynamic_rms_mg: float
    gyro_rms_dps: float
    motion_fraction: float

    ppg_quality: float
    usable_for_sleep_staging: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)
