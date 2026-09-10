from __future__ import annotations

import math
import threading

from influxdb_client import InfluxDBClient, Point, WriteOptions, WritePrecision

from sleep_dsp.models import EpochFeatures
from sleep_product.models import NightAnalysis
from sleep_physiology.models import SlowPhysiologyEstimate
from sleep_staging.models import StagePrediction

from .config import settings
from .protocol import ImuPacket, PpgPacket, TelemetryPacket


class WriteHealth:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.successful_batches = 0
        self.failed_batches = 0
        self.retry_batches = 0

    def success(self, _conf, _data) -> None:
        with self._lock:
            self.successful_batches += 1

    def error(self, _conf, _data, exception) -> None:
        with self._lock:
            self.failed_batches += 1

        print(f"[influx] batch write failed: {exception}")

    def retry(self, _conf, _data, exception) -> None:
        with self._lock:
            self.retry_batches += 1

        print(f"[influx] retrying batch: {exception}")


class InfluxWriter:
    """Batch InfluxDB writes instead of sending one request per sensor sample."""

    def __init__(self) -> None:
        self.client = InfluxDBClient(
            url=settings.influx_url,
            token=settings.influx_token,
            org=settings.influx_org,
        )

        self.health = WriteHealth()

        self.write_api = self.client.write_api(
            write_options=WriteOptions(
                batch_size=500,
                flush_interval=1000,
                jitter_interval=100,
                retry_interval=1000,
                max_retries=5,
            ),
            success_callback=self.health.success,
            error_callback=self.health.error,
            retry_callback=self.health.retry,
        )

    def flush(self) -> None:
        self.write_api.flush()

    def close(self) -> None:
        self.write_api.close()
        self.client.close()

    def _write(self, point: Point) -> None:
        self.write_api.write(
            bucket=settings.influx_bucket,
            org=settings.influx_org,
            record=point,
        )

    @staticmethod
    def _finite_field(
        point: Point,
        name: str,
        value: float,
    ) -> Point:
        if math.isfinite(value):
            return point.field(name, float(value))

        return point

    def write_packet(
        self,
        packet: PpgPacket | ImuPacket,
        session_id: str,
        measurement_ns: int,
    ) -> None:
        if isinstance(packet, PpgPacket):
            point = (
                Point("raw_ppg")
                .tag("device", settings.device_id)
                .tag("session_id", session_id)
                .field("sequence", packet.sequence)
                .field(
                    "device_time_us",
                    packet.device_time_us,
                )
                .field("red", packet.red)
                .field("ir", packet.ir)
                .time(
                    measurement_ns,
                    WritePrecision.NS,
                )
            )
        else:
            point = (
                Point("raw_imu")
                .tag("device", settings.device_id)
                .tag("session_id", session_id)
                .field("sequence", packet.sequence)
                .field(
                    "device_time_us",
                    packet.device_time_us,
                )
                .time(
                    measurement_ns,
                    WritePrecision.NS,
                )
            )

            for name in (
                "ax",
                "ay",
                "az",
                "gx",
                "gy",
                "gz",
                "temperature_c",
            ):
                point = self._finite_field(
                    point,
                    name,
                    float(getattr(packet, name)),
                )

        self._write(point)

    def write_epoch_features(
        self,
        features: EpochFeatures,
        session_id: str,
        measurement_ns: int,
    ) -> None:
        point = (
            Point("sleep_epoch_features")
            .tag("device", settings.device_id)
            .tag("session_id", session_id)
            .field(
                "start_device_time_us",
                features.start_device_time_us,
            )
            .field(
                "end_device_time_us",
                features.end_device_time_us,
            )
            .field(
                "ppg_sample_count",
                features.ppg_sample_count,
            )
            .field(
                "imu_sample_count",
                features.imu_sample_count,
            )
            .field("beat_count", features.beat_count)
            .field(
                "valid_rr_count",
                features.valid_rr_count,
            )
            .field(
                "usable_for_sleep_staging",
                features.usable_for_sleep_staging,
            )
            .time(
                measurement_ns,
                WritePrecision.NS,
            )
        )

        for name in (
            "mean_hr_bpm",
            "median_hr_bpm",
            "rmssd_ms",
            "sdnn_ms",
            "pnn50_percent",
            "mean_rr_ms",
            "median_rr_ms",
            "accel_enmo_mean_mg",
            "accel_dynamic_rms_mg",
            "gyro_rms_dps",
            "motion_fraction",
            "ppg_quality",
        ):
            point = self._finite_field(
                point,
                name,
                float(getattr(features, name)),
            )

        self._write(point)

    def write_slow_physiology(
        self,
        estimate: SlowPhysiologyEstimate,
        session_id: str,
        measurement_ns: int,
    ) -> None:
        point = (
            Point("slow_physiology")
            .tag("device", settings.device_id)
            .tag("session_id", session_id)
            .field(
                "start_device_time_us",
                estimate.start_device_time_us,
            )
            .field(
                "end_device_time_us",
                estimate.end_device_time_us,
            )
            .field(
                "spo2_calibrated",
                estimate.spo2_calibrated,
            )
            .field(
                "spo2_usable",
                estimate.spo2_usable,
            )
            .field(
                "respiration_usable",
                estimate.respiration_usable,
            )
            .time(
                measurement_ns,
                WritePrecision.NS,
            )
        )

        for name in (
            "ratio_of_ratios",
            "spo2_quality",
            "spo2_percent",
            "respiratory_rate_bpm",
            "respiratory_confidence",
            "motion_fraction",
        ):
            point = self._finite_field(
                point,
                name,
                float(getattr(estimate, name)),
            )

        self._write(point)

    def write_stage_prediction(
        self,
        prediction: StagePrediction,
        session_id: str,
        measurement_ns: int,
    ) -> None:
        point = (
            Point("sleep_stage_prediction")
            .tag("device", settings.device_id)
            .tag("session_id", session_id)
            .tag("stage", prediction.stage_name)
            .field("stage_code", int(prediction.stage))
            .field(
                "epoch_index",
                prediction.epoch_index,
            )
            .field(
                "start_device_time_us",
                prediction.start_device_time_us,
            )
            .field(
                "end_device_time_us",
                prediction.end_device_time_us,
            )
            .field(
                "raw_stage",
                prediction.raw_stage_name,
            )
            .field(
                "signal_usable",
                prediction.signal_usable,
            )
            .field(
                "inferred_with_temporal_context",
                prediction.inferred_with_temporal_context,
            )
            .time(
                measurement_ns,
                WritePrecision.NS,
            )
        )

        for name in (
            "wake_probability",
            "light_probability",
            "deep_probability",
            "rem_probability",
            "confidence",
        ):
            point = self._finite_field(
                point,
                name,
                float(getattr(prediction, name)),
            )

        self._write(point)

    def write_device_health(
        self,
        telemetry: TelemetryPacket,
        session_id: str,
        measurement_ns: int,
        sequence_gaps: int,
        duplicate_packets: int,
        clock_drift_ppm: float,
        rejected_clock_anchors: int,
    ) -> None:
        point = (
            Point("device_health")
            .tag("device", settings.device_id)
            .tag("session_id", session_id)
            .field(
                "queue_depth",
                telemetry.queue_depth,
            )
            .field(
                "queue_high_watermark",
                telemetry.queue_high_watermark,
            )
            .field(
                "dropped_packets",
                telemetry.dropped_packets,
            )
            .field(
                "reconnect_count",
                telemetry.reconnect_count,
            )
            .field(
                "send_failures",
                telemetry.send_failures,
            )
            .field(
                "sequence_gaps",
                sequence_gaps,
            )
            .field(
                "duplicate_packets",
                duplicate_packets,
            )
            .field(
                "rejected_clock_anchors",
                rejected_clock_anchors,
            )
            .time(
                measurement_ns,
                WritePrecision.NS,
            )
        )

        point = self._finite_field(
            point,
            "clock_drift_ppm",
            clock_drift_ppm,
        )

        self._write(point)

    def write_night_summary(
        self,
        analysis: NightAnalysis,
        started_at_ns: int,
        ended_at_ns: int,
        sequence_gaps: int = 0,
        duplicate_packets: int = 0,
        clock_drift_ppm: float = 0.0,
        rejected_clock_anchors: int = 0,
    ) -> None:
        s = analysis.summary
        p = analysis.physiology
        r = analysis.recovery
        b = analysis.baseline_before_night

        point = (
            Point("night_summary")
            .tag("device", settings.device_id)
            .tag(
                "session_id",
                analysis.session_id,
            )
            .field(
                "recovery_provisional",
                r.provisional,
            )
            .field("baseline_ready", b.ready)
            .field(
                "baseline_sample_nights",
                b.sample_nights,
            )
            .field(
                "session_started_at_ns",
                int(started_at_ns),
            )
            .field(
                "session_ended_at_ns",
                int(ended_at_ns),
            )
            .field(
                "sequence_gaps",
                int(sequence_gaps),
            )
            .field(
                "duplicate_packets",
                int(duplicate_packets),
            )
            .field(
                "rejected_clock_anchors",
                int(rejected_clock_anchors),
            )
            .time(
                ended_at_ns,
                WritePrecision.NS,
            )
        )

        finite_fields = {
            "recording_minutes": s.recording_minutes,
            "total_sleep_minutes": s.total_sleep_minutes,
            "sleep_efficiency_percent": s.sleep_efficiency_percent,
            "sleep_onset_latency_minutes": (
                s.sleep_onset_latency_minutes
            ),
            "wake_after_sleep_onset_minutes": (
                s.wake_after_sleep_onset_minutes
            ),
            "light_minutes": s.light_minutes,
            "deep_minutes": s.deep_minutes,
            "rem_minutes": s.rem_minutes,
            "usable_signal_percent": s.usable_signal_percent,
            "mean_stage_confidence": s.mean_stage_confidence,
            "median_sleep_hr_bpm": p.median_sleep_hr_bpm,
            "median_sleep_rmssd_ms": p.median_sleep_rmssd_ms,
            "recovery_score": r.score,
            "sleep_duration_score": r.sleep_duration_score,
            "sleep_efficiency_score": r.sleep_efficiency_score,
            "hrv_score": r.hrv_score,
            "sleeping_hr_score": r.sleeping_hr_score,
            "signal_quality_score": r.signal_quality_score,
            "clock_drift_ppm": clock_drift_ppm,
        }

        for name, value in finite_fields.items():
            point = self._finite_field(
                point,
                name,
                float(value),
            )

        if r.hrv_delta_percent is not None:
            point = self._finite_field(
                point,
                "hrv_delta_percent",
                r.hrv_delta_percent,
            )

        if r.sleeping_hr_delta_bpm is not None:
            point = self._finite_field(
                point,
                "sleeping_hr_delta_bpm",
                r.sleeping_hr_delta_bpm,
            )

        if b.median_sleep_hr_bpm is not None:
            point = self._finite_field(
                point,
                "baseline_median_sleep_hr_bpm",
                b.median_sleep_hr_bpm,
            )

        if b.median_sleep_rmssd_ms is not None:
            point = self._finite_field(
                point,
                "baseline_median_sleep_rmssd_ms",
                b.median_sleep_rmssd_ms,
            )

        self._write(point)
