from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sleep_dsp.models import EpochFeatures

from .stages import SleepStage


@dataclass(frozen=True)
class SyntheticNight:
    epochs: list[EpochFeatures]
    truth: list[SleepStage]


def _append_stage(
    stages: list[SleepStage],
    stage: SleepStage,
    minutes: float,
) -> None:
    stages.extend(
        [stage] * max(1, int(round(minutes * 2.0)))
    )


def canonical_stage_path() -> list[SleepStage]:
    """
    Generate an intentionally simplified ~8-hour sleep architecture.

    It is useful for software tests only. It is not a normative model of how
    every person's sleep should look.
    """
    stages: list[SleepStage] = []

    _append_stage(stages, SleepStage.WAKE, 12)

    cycles = [
        # light, deep, light, REM (minutes)
        (18, 32, 18, 10),
        (20, 27, 18, 15),
        (22, 22, 18, 20),
        (24, 14, 18, 27),
        (27, 8, 16, 34),
        (24, 4, 14, 37),
    ]

    for cycle_index, (light1, deep, light2, rem) in enumerate(cycles):
        _append_stage(stages, SleepStage.LIGHT, light1)
        _append_stage(stages, SleepStage.DEEP, deep)
        _append_stage(stages, SleepStage.LIGHT, light2)
        _append_stage(stages, SleepStage.REM, rem)

        # Tiny wake transitions later in the night.
        if cycle_index >= 2:
            _append_stage(stages, SleepStage.WAKE, 1.5)

    return stages


def make_synthetic_night(seed: int = 44) -> SyntheticNight:
    """
    Create feature vectors directly instead of synthesizing 8 hours of raw
    100 Hz PPG. This isolates the staging layer from the raw-signal DSP layer.
    """
    rng = np.random.default_rng(seed)
    truth = canonical_stage_path()

    # Stage signatures are deliberately separated enough to test software
    # mechanics. They are not population reference ranges.
    signatures = {
        SleepStage.WAKE: dict(
            hr=72.0,
            rmssd=32.0,
            sdnn=38.0,
            motion=0.20,
            accel=145.0,
            gyro=16.0,
        ),
        SleepStage.LIGHT: dict(
            hr=59.0,
            rmssd=47.0,
            sdnn=45.0,
            motion=0.025,
            accel=24.0,
            gyro=1.7,
        ),
        SleepStage.DEEP: dict(
            hr=51.5,
            rmssd=68.0,
            sdnn=52.0,
            motion=0.004,
            accel=8.5,
            gyro=0.45,
        ),
        SleepStage.REM: dict(
            hr=61.5,
            rmssd=54.0,
            sdnn=58.0,
            motion=0.011,
            accel=14.0,
            gyro=0.85,
        ),
    }

    epochs: list[EpochFeatures] = []

    # Correlated REM HR variability: this makes local variability meaningful.
    rem_wave_phase = 0.0

    for i, stage in enumerate(truth):
        s = signatures[stage]

        if stage is SleepStage.REM:
            rem_wave_phase += 0.85
            hr_offset = 4.2 * np.sin(rem_wave_phase)
        else:
            hr_offset = rng.normal(0.0, 0.8)

        motion = max(
            0.0,
            float(
                rng.normal(
                    s["motion"],
                    0.020 if stage is SleepStage.WAKE else 0.004,
                )
            ),
        )

        # About 1% of epochs are deliberately marked optically poor.
        poor_signal = rng.random() < 0.01
        quality = (
            float(rng.uniform(0.25, 0.50))
            if poor_signal
            else float(rng.uniform(0.88, 1.0))
        )

        start_us = i * 30_000_000

        epochs.append(
            EpochFeatures(
                start_device_time_us=start_us,
                end_device_time_us=start_us + 30_000_000,
                ppg_sample_count=3000,
                imu_sample_count=1500,
                beat_count=max(
                    12,
                    int(round((s["hr"] + hr_offset) / 2.0)),
                ),
                valid_rr_count=max(
                    10,
                    int(round((s["hr"] + hr_offset) / 2.0)) - 1,
                ),
                mean_hr_bpm=float(
                    rng.normal(s["hr"] + hr_offset, 0.8)
                ),
                median_hr_bpm=float(
                    rng.normal(s["hr"] + hr_offset, 0.7)
                ),
                rmssd_ms=float(
                    max(5.0, rng.normal(s["rmssd"], 4.0))
                ),
                sdnn_ms=float(
                    max(5.0, rng.normal(s["sdnn"], 4.0))
                ),
                pnn50_percent=float(
                    np.clip(
                        (s["rmssd"] - 25.0) * 0.9
                        + rng.normal(0.0, 4.0),
                        0.0,
                        100.0,
                    )
                ),
                mean_rr_ms=float(
                    60_000.0 / (s["hr"] + hr_offset)
                ),
                median_rr_ms=float(
                    60_000.0 / (s["hr"] + hr_offset)
                ),
                accel_enmo_mean_mg=float(
                    max(0.0, rng.normal(s["accel"] * 0.35, 3.0))
                ),
                accel_dynamic_rms_mg=float(
                    max(0.0, rng.normal(s["accel"], 5.0))
                ),
                gyro_rms_dps=float(
                    max(0.0, rng.normal(s["gyro"], 0.3))
                ),
                motion_fraction=motion,
                ppg_quality=quality,
                usable_for_sleep_staging=not poor_signal,
            )
        )

    return SyntheticNight(
        epochs=epochs,
        truth=truth,
    )
