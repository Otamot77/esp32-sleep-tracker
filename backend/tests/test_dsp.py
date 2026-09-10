from __future__ import annotations

import numpy as np

from sleep_dsp.epochs import extract_epoch_features
from sleep_dsp.hrv import time_domain_hrv
from sleep_dsp.pipeline import RealtimeDspPipeline
from sleep_dsp.ppg import clean_rr_intervals, detect_beats
from sleep_dsp.synthetic import make_epoch
from sleep_server.protocol import PpgPacket


def _features(moving: bool):
    epoch = make_epoch(
        heart_rate_bpm=60.0,
        rr_modulation_ms=45.0,
        moving=moving,
        seed=3 if not moving else 4,
    )

    return extract_epoch_features(
        0,
        30_000_000,
        epoch.ppg_t_us,
        epoch.red,
        epoch.ir,
        epoch.imu_t_us,
        epoch.ax,
        epoch.ay,
        epoch.az,
        epoch.gx,
        epoch.gy,
        epoch.gz,
    )


def test_detects_about_60_bpm() -> None:
    epoch = make_epoch(
        heart_rate_bpm=60.0,
        rr_modulation_ms=0.0,
        moving=False,
        seed=9,
    )

    detection = detect_beats(epoch.ir, 100.0)

    assert 27 <= detection.peak_indices.size <= 31

    hrv = time_domain_hrv(detection.rr_ms_valid)
    assert 57.0 <= hrv.mean_hr_bpm <= 63.0


def test_hrv_math_known_rr_intervals() -> None:
    rr = np.array([1000.0, 1050.0, 950.0, 1000.0])
    result = time_domain_hrv(rr)

    assert abs(result.mean_rr_ms - 1000.0) < 1e-9
    assert result.rmssd_ms > 0.0
    assert result.sdnn_ms > 0.0


def test_rr_artifact_is_rejected() -> None:
    rr = np.array([1000.0, 990.0, 1010.0, 300.0, 995.0, 1005.0])

    cleaned, mask = clean_rr_intervals(rr)

    assert cleaned.size < rr.size
    assert not bool(mask[3])


def test_clean_epoch_is_usable() -> None:
    features = _features(moving=False)

    assert 55.0 <= features.mean_hr_bpm <= 65.0
    assert features.motion_fraction < 0.05
    assert features.ppg_quality >= 0.60
    assert features.usable_for_sleep_staging


def test_motion_epoch_has_lower_quality() -> None:
    clean = _features(moving=False)
    moving = _features(moving=True)

    assert moving.motion_fraction > clean.motion_fraction
    assert moving.ppg_quality < clean.ppg_quality
    assert not moving.usable_for_sleep_staging


def test_missing_sample_span_is_not_used_as_an_rr_interval() -> None:
    sample_rate_hz = 100.0
    t = np.arange(0.0, 12.0, 1.0 / sample_rate_hz)
    signal = np.sin(2.0 * np.pi * t)

    keep = ~((t >= 5.0) & (t < 5.15))
    detection = detect_beats(
        signal[keep],
        sample_rate_hz,
        sample_times_s=t[keep],
    )
    peak_times = t[keep][detection.peak_indices]
    crosses_gap = (
        (peak_times[:-1] < 5.0)
        & (peak_times[1:] > 5.15)
    )

    assert detection.rr_ms_valid.size > 0
    assert np.any(crosses_gap)
    assert np.all(~detection.rr_valid_mask[crosses_gap])
    assert np.all(detection.rr_ms_valid < 1_100.0)


def test_empty_epoch_has_zero_quality() -> None:
    empty = np.zeros(0, dtype=float)

    features = extract_epoch_features(
        0,
        30_000_000,
        empty.astype(np.int64),
        empty,
        empty,
        empty.astype(np.int64),
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
    )

    assert features.ppg_quality == 0.0
    assert not features.usable_for_sleep_staging


def test_missing_sensor_stream_does_not_grow_epoch_buffer() -> None:
    pipeline = RealtimeDspPipeline(epoch_seconds=1.0)
    emitted = []

    for index in range(201):
        emitted.extend(
            pipeline.add(
                PpgPacket(
                    sequence=index,
                    device_time_us=index * 10_000,
                    red=50_000,
                    ir=60_000,
                )
            )
        )

    assert len(emitted) == 2
    assert all(epoch.imu_sample_count == 0 for epoch in emitted)
    assert all(not epoch.usable_for_sleep_staging for epoch in emitted)
    assert len(pipeline.ppg) == 1
