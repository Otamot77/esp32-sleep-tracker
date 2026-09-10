from __future__ import annotations

import math

import numpy as np

from sleep_dsp.filters import bandpass_ppg, robust_sigma

from .models import OximetryEstimate, SpO2Calibration


def estimate_oximetry(
    red: np.ndarray,
    ir: np.ndarray,
    sample_rate_hz: float,
    motion_fraction: float,
    calibration: SpO2Calibration | None = None,
) -> OximetryEstimate:
    """
    Calculate the optical ratio-of-ratios used by pulse oximetry.

        R = (AC_red / DC_red) / (AC_ir / DC_ir)

    A real SpO2 percentage is only emitted if calibrated coefficients were
    supplied. The ratio itself is still useful for calibration experiments.
    """
    red = np.asarray(red, dtype=float)
    ir = np.asarray(ir, dtype=float)

    nan = float("nan")

    if (
        red.size < int(sample_rate_hz * 10.0)
        or ir.size != red.size
    ):
        return OximetryEstimate(
            ratio_of_ratios=nan,
            spo2_percent=nan,
            quality=0.0,
            calibrated=calibration is not None,
            usable=False,
        )

    dc_red = float(np.median(red))
    dc_ir = float(np.median(ir))

    if dc_red <= 0.0 or dc_ir <= 0.0:
        return OximetryEstimate(
            ratio_of_ratios=nan,
            spo2_percent=nan,
            quality=0.0,
            calibrated=calibration is not None,
            usable=False,
        )

    ac_red_signal = bandpass_ppg(
        red,
        sample_rate_hz,
        low_hz=0.5,
        high_hz=5.0,
    )
    ac_ir_signal = bandpass_ppg(
        ir,
        sample_rate_hz,
        low_hz=0.5,
        high_hz=5.0,
    )

    ac_red = float(np.sqrt(np.mean(ac_red_signal**2)))
    ac_ir = float(np.sqrt(np.mean(ac_ir_signal**2)))

    if ac_red <= 0.0 or ac_ir <= 0.0:
        return OximetryEstimate(
            ratio_of_ratios=nan,
            spo2_percent=nan,
            quality=0.0,
            calibrated=calibration is not None,
            usable=False,
        )

    ratio = (ac_red / dc_red) / (ac_ir / dc_ir)

    # Red and IR pulse waveforms should be strongly related if optical
    # contact is good. Correlation is only one quality cue, not proof that
    # the result is physiologically accurate.
    correlation = float(
        np.corrcoef(
            ac_red_signal,
            ac_ir_signal,
        )[0, 1]
    )

    if not math.isfinite(correlation):
        correlation = 0.0

    # Relative AC amplitude: reject a nearly-flat optical signal.
    relative_ir_ac = ac_ir / dc_ir
    amplitude_score = float(
        np.clip(
            relative_ir_ac / 0.005,
            0.0,
            1.0,
        )
    )

    correlation_score = float(
        np.clip(
            (abs(correlation) - 0.50) / 0.45,
            0.0,
            1.0,
        )
    )

    motion_score = float(
        np.clip(
            1.0 - motion_fraction / 0.25,
            0.0,
            1.0,
        )
    )

    quality = float(
        np.clip(
            0.45 * correlation_score
            + 0.30 * amplitude_score
            + 0.25 * motion_score,
            0.0,
            1.0,
        )
    )

    # A broad sanity range for the dimensionless ratio. This is an artifact
    # guardrail, not a medical threshold.
    ratio_plausible = 0.20 <= ratio <= 1.80

    spo2 = nan

    if calibration is not None and ratio_plausible:
        spo2 = float(
            np.clip(
                calibration.apply(ratio),
                50.0,
                100.0,
            )
        )

    usable = bool(
        ratio_plausible
        and quality >= 0.60
        and (
            calibration is None
            or math.isfinite(spo2)
        )
    )

    return OximetryEstimate(
        ratio_of_ratios=float(ratio),
        spo2_percent=spo2,
        quality=quality,
        calibrated=calibration is not None,
        usable=usable,
    )
