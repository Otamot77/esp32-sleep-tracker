from __future__ import annotations

import math

import numpy as np

from .constants import IMU_SAMPLE_RATE_HZ, PPG_SAMPLE_RATE_HZ
from .hrv import time_domain_hrv
from .models import EpochFeatures
from .motion import extract_motion_features
from .ppg import detect_beats
from .quality import ppg_quality_score


def _safe(value: float) -> float:
    return float(value) if math.isfinite(value) else float("nan")


def extract_epoch_features(
    start_device_time_us: int,
    end_device_time_us: int,
    ppg_t_us: np.ndarray,
    red: np.ndarray,
    ir: np.ndarray,
    imu_t_us: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
) -> EpochFeatures:
    del red

    duration_s = (
        end_device_time_us - start_device_time_us
    ) / 1_000_000.0

    beats = detect_beats(
        ir,
        PPG_SAMPLE_RATE_HZ,
        sample_times_s=ppg_t_us.astype(float) / 1_000_000.0,
    )
    motion = extract_motion_features(ax, ay, az, gx, gy, gz)
    hrv = time_domain_hrv(beats.rr_ms_valid)

    quality = ppg_quality_score(
        beat_count=int(beats.peak_indices.size),
        rr_valid_mask=beats.rr_valid_mask,
        motion_fraction=motion.motion_fraction,
        epoch_seconds=duration_s,
    )

    expected_ppg = duration_s * PPG_SAMPLE_RATE_HZ
    expected_imu = duration_s * IMU_SAMPLE_RATE_HZ

    ppg_completeness = (
        len(ppg_t_us) / expected_ppg if expected_ppg > 0 else 0.0
    )
    imu_completeness = (
        len(imu_t_us) / expected_imu if expected_imu > 0 else 0.0
    )

    # Missing samples reduce our confidence even if the remaining waveform
    # happens to look clean. This also keeps an empty epoch from receiving a
    # misleadingly positive quality score just because no motion was seen.
    coverage = min(
        max(ppg_completeness, 0.0),
        max(imu_completeness, 0.0),
        1.0,
    )
    quality *= coverage

    usable = (
        quality >= 0.60
        and ppg_completeness >= 0.90
        and imu_completeness >= 0.90
        and beats.rr_ms_valid.size >= 8
    )

    return EpochFeatures(
        start_device_time_us=int(start_device_time_us),
        end_device_time_us=int(end_device_time_us),
        ppg_sample_count=int(len(ppg_t_us)),
        imu_sample_count=int(len(imu_t_us)),
        beat_count=int(beats.peak_indices.size),
        valid_rr_count=int(beats.rr_ms_valid.size),
        mean_hr_bpm=_safe(hrv.mean_hr_bpm),
        median_hr_bpm=_safe(hrv.median_hr_bpm),
        rmssd_ms=_safe(hrv.rmssd_ms),
        sdnn_ms=_safe(hrv.sdnn_ms),
        pnn50_percent=_safe(hrv.pnn50_percent),
        mean_rr_ms=_safe(hrv.mean_rr_ms),
        median_rr_ms=_safe(hrv.median_rr_ms),
        accel_enmo_mean_mg=motion.enmo_mean_mg,
        accel_dynamic_rms_mg=motion.dynamic_rms_mg,
        gyro_rms_dps=motion.gyro_rms_dps,
        motion_fraction=motion.motion_fraction,
        ppg_quality=quality,
        usable_for_sleep_staging=usable,
    )
