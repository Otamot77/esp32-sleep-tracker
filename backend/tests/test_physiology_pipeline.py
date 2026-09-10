from __future__ import annotations

from unittest.mock import patch

from sleep_physiology.models import (
    OximetryEstimate,
    RespiratoryEstimate,
)
from sleep_physiology.pipeline import SlowPhysiologyPipeline
from sleep_server.protocol import ImuPacket, PpgPacket


def ppg(time_us: int) -> PpgPacket:
    return PpgPacket(0, time_us, 100_000, 120_000)


def imu(time_us: int) -> ImuPacket:
    return ImuPacket(
        0,
        time_us,
        0.0,
        0.0,
        9.81,
        0.0,
        0.0,
        0.0,
        30.0,
    )


def test_device_uptime_is_not_mistaken_for_window_history() -> None:
    pipeline = SlowPhysiologyPipeline(
        window_seconds=4.0,
        output_seconds=1.0,
    )

    assert pipeline.add(ppg(300_000_000)) == []
    assert pipeline.add(imu(300_000_000)) == []
    assert pipeline.add(ppg(303_000_000)) == []
    assert pipeline.add(imu(303_000_000)) == []
    assert pipeline.add(imu(303_990_000)) == []
    assert len(pipeline.add(ppg(304_000_000))) == 1


def test_incomplete_window_cannot_be_marked_usable() -> None:
    pipeline = SlowPhysiologyPipeline()
    pipeline.ppg.extend([ppg(0), ppg(119_000_000)])
    pipeline.imu.extend([imu(0), imu(119_000_000)])

    optical = OximetryEstimate(
        ratio_of_ratios=0.8,
        spo2_percent=97.0,
        quality=0.9,
        calibrated=True,
        usable=True,
    )
    breathing = RespiratoryEstimate(
        breaths_per_minute=14.0,
        confidence=0.9,
        usable=True,
    )

    with patch(
        "sleep_physiology.pipeline.estimate_oximetry",
        return_value=optical,
    ), patch(
        "sleep_physiology.pipeline.estimate_respiratory_rate",
        return_value=breathing,
    ):
        estimate = pipeline._estimate(0, 120_000_000)

    assert not estimate.spo2_usable
    assert not estimate.respiration_usable


def test_missing_stream_does_not_leave_output_schedule_behind() -> None:
    pipeline = SlowPhysiologyPipeline(
        window_seconds=2.0,
        output_seconds=1.0,
    )

    results = []
    for index in range(1_001):
        results.extend(
            pipeline.add(ppg(index * 10_000))
        )

    assert results == []
    assert pipeline.next_output_us == 11_000_000
    assert len(pipeline.ppg) <= 301
