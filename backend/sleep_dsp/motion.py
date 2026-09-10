from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    GRAVITY_M_S2,
    MOTION_ENMO_THRESHOLD_G,
    MOTION_GYRO_THRESHOLD_DPS,
    RAD_S_TO_DEG_S,
)


@dataclass(frozen=True)
class MotionFeatures:
    enmo_g: np.ndarray
    dynamic_accel_g: np.ndarray
    gyro_magnitude_dps: np.ndarray
    moving_mask: np.ndarray

    enmo_mean_mg: float
    dynamic_rms_mg: float
    gyro_rms_dps: float
    motion_fraction: float


def extract_motion_features(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
) -> MotionFeatures:
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    az = np.asarray(az, dtype=float)

    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    gz = np.asarray(gz, dtype=float)

    accel_mag_g = (
        np.sqrt(ax * ax + ay * ay + az * az) / GRAVITY_M_S2
    )

    enmo_g = np.maximum(accel_mag_g - 1.0, 0.0)
    dynamic_accel_g = (
        accel_mag_g - np.median(accel_mag_g)
        if accel_mag_g.size
        else accel_mag_g.copy()
    )

    gyro_mag_dps = (
        np.sqrt(gx * gx + gy * gy + gz * gz) * RAD_S_TO_DEG_S
    )

    moving_mask = (
        (enmo_g > MOTION_ENMO_THRESHOLD_G)
        | (gyro_mag_dps > MOTION_GYRO_THRESHOLD_DPS)
    )

    return MotionFeatures(
        enmo_g=enmo_g,
        dynamic_accel_g=dynamic_accel_g,
        gyro_magnitude_dps=gyro_mag_dps,
        moving_mask=moving_mask,
        enmo_mean_mg=float(np.mean(enmo_g) * 1000.0)
        if enmo_g.size
        else 0.0,
        dynamic_rms_mg=float(
            np.sqrt(np.mean(dynamic_accel_g**2)) * 1000.0
        )
        if dynamic_accel_g.size
        else 0.0,
        gyro_rms_dps=float(np.sqrt(np.mean(gyro_mag_dps**2)))
        if gyro_mag_dps.size
        else 0.0,
        motion_fraction=float(np.mean(moving_mask))
        if moving_mask.size
        else 0.0,
    )
