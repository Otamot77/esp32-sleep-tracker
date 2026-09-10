from __future__ import annotations

from sleep_dsp.models import EpochFeatures
from sleep_staging.stager import NightStager

from .models import NightAnalysis, PersonalBaseline
from .physiology import derive_night_physiology
from .recovery import score_recovery


def analyze_night(
    session_id: str,
    epochs: list[EpochFeatures],
    baseline: PersonalBaseline,
    stager: NightStager | None = None,
) -> NightAnalysis:
    stage_engine = stager or NightStager()
    predictions, summary = stage_engine.stage(epochs)

    physiology = derive_night_physiology(
        epochs,
        predictions,
    )

    recovery = score_recovery(
        summary,
        physiology,
        baseline,
    )

    return NightAnalysis(
        session_id=session_id,
        predictions=predictions,
        summary=summary,
        physiology=physiology,
        baseline_before_night=baseline,
        recovery=recovery,
    )
