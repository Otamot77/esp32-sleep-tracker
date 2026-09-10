from __future__ import annotations

from dataclasses import dataclass

from sleep_dsp.epochs import extract_epoch_features
from sleep_dsp.synthetic import make_epoch
from sleep_physiology.oximetry import estimate_oximetry
from sleep_physiology.respiration import estimate_respiratory_rate
from sleep_physiology.synthetic import make_raw_ppg
from sleep_product.analysis import analyze_night
from sleep_product.models import PersonalBaseline
from sleep_server.protocol import PacketParser, PpgPacket, encode_hello, encode_ppg
from sleep_staging.synthetic_night import make_synthetic_night


@dataclass(frozen=True)
class DemoResult:
    decoded_ppg_ok: bool
    heart_rate_bpm: float
    rmssd_ms: float
    respiratory_rate_bpm: float
    ratio_of_ratios: float
    total_sleep_minutes: float
    deep_minutes: float
    rem_minutes: float
    recovery_score: float


def run_demo(verbose: bool = True) -> DemoResult:
    # Check that a packet can make a basic encode/decode round trip.
    parser = PacketParser()
    stream = (
        encode_hello(device_time_us=0, boot_id=0x123456789ABCDEF0)
        + encode_ppg(
            sequence=0,
            device_time_us=10_000,
            red=100_000,
            ir=120_000,
        )
    )
    decoded = parser.feed(stream)
    decoded_ppg_ok = (
        len(decoded) == 2
        and isinstance(decoded[1], PpgPacket)
        and decoded[1].red == 100_000
        and decoded[1].ir == 120_000
    )

    # Known synthetic inputs make it possible to sanity-check the DSP before
    # collecting repeatable overnight recordings from the wearable.
    fast = make_epoch(
        heart_rate_bpm=60.0,
        rr_modulation_ms=45.0,
        moving=False,
        seed=200,
    )
    features = extract_epoch_features(
        start_device_time_us=0,
        end_device_time_us=30_000_000,
        ppg_t_us=fast.ppg_t_us,
        red=fast.red,
        ir=fast.ir,
        imu_t_us=fast.imu_t_us,
        ax=fast.ax,
        ay=fast.ay,
        az=fast.az,
        gx=fast.gx,
        gy=fast.gy,
        gz=fast.gz,
    )

    slow = make_raw_ppg(
        duration_seconds=120.0,
        respiratory_rate_bpm=15.0,
        seed=201,
    )
    respiration = estimate_respiratory_rate(
        ir=slow.ir,
        sample_rate_hz=slow.sample_rate_hz,
        motion_fraction=0.0,
    )
    oximetry = estimate_oximetry(
        red=slow.red,
        ir=slow.ir,
        sample_rate_hz=slow.sample_rate_hz,
        motion_fraction=0.0,
        calibration=None,
    )

    night = make_synthetic_night(seed=202)
    baseline = PersonalBaseline(
        median_sleep_hr_bpm=58.0,
        median_sleep_rmssd_ms=49.0,
        sample_nights=8,
        ready=True,
    )
    analysis = analyze_night(
        session_id="software-demo",
        epochs=night.epochs,
        baseline=baseline,
    )

    result = DemoResult(
        decoded_ppg_ok=decoded_ppg_ok,
        heart_rate_bpm=features.mean_hr_bpm,
        rmssd_ms=features.rmssd_ms,
        respiratory_rate_bpm=respiration.breaths_per_minute,
        ratio_of_ratios=oximetry.ratio_of_ratios,
        total_sleep_minutes=analysis.summary.total_sleep_minutes,
        deep_minutes=analysis.summary.deep_minutes,
        rem_minutes=analysis.summary.rem_minutes,
        recovery_score=analysis.recovery.score,
    )

    if verbose:
        print("=" * 56)
        print(" SLEEP TRACKER - SOFTWARE DEMO")
        print("=" * 56)
        print(f"Packet round trip:  {'PASS' if result.decoded_ppg_ok else 'FAIL'}")
        print(f"Estimated HR:       {result.heart_rate_bpm:.1f} bpm")
        print(f"Estimated RMSSD:    {result.rmssd_ms:.1f} ms")
        print(f"Respiratory rate:   {result.respiratory_rate_bpm:.1f} breaths/min")
        print(f"Red/IR ratio:       {result.ratio_of_ratios:.3f}")
        print("SpO2:               withheld until calibration")
        print()
        print("Synthetic-night output")
        print(f"Total sleep:        {result.total_sleep_minutes:.1f} min")
        print(f"Deep sleep:         {result.deep_minutes:.1f} min")
        print(f"REM:                {result.rem_minutes:.1f} min")
        print(f"Recovery score:     {result.recovery_score:.0f}/100")
        print()
        print("Synthetic inputs are software checks, not wearable validation.")
        print("=" * 56)

    return result


def main() -> None:
    run_demo(verbose=True)


if __name__ == "__main__":
    main()
