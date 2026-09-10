from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .models import NightAnalysis, PersonalBaseline


MIN_BASELINE_NIGHTS = 3
BASELINE_WINDOW_NIGHTS = 14


class HistoryStore:
    """Small JSON store used to build the rolling personal baseline."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

        return value if isinstance(value, list) else []

    def baseline(
        self,
        min_nights: int = MIN_BASELINE_NIGHTS,
        window_nights: int = BASELINE_WINDOW_NIGHTS,
    ) -> PersonalBaseline:
        records = self.load()[-window_nights:]

        hr: list[float] = []
        rmssd: list[float] = []

        for record in records:
            try:
                current_hr = float(record["median_sleep_hr_bpm"])
                current_rmssd = float(record["median_sleep_rmssd_ms"])
            except (KeyError, TypeError, ValueError):
                continue

            if math.isfinite(current_hr) and math.isfinite(current_rmssd):
                hr.append(current_hr)
                rmssd.append(current_rmssd)

        sample_nights = min(len(hr), len(rmssd))
        ready = sample_nights >= min_nights

        if sample_nights == 0:
            return PersonalBaseline(
                median_sleep_hr_bpm=None,
                median_sleep_rmssd_ms=None,
                sample_nights=0,
                ready=False,
            )

        return PersonalBaseline(
            median_sleep_hr_bpm=float(np.median(hr)),
            median_sleep_rmssd_ms=float(np.median(rmssd)),
            sample_nights=sample_nights,
            ready=ready,
        )

    def append(
        self,
        analysis: NightAnalysis,
        started_at_ns: int,
        ended_at_ns: int,
    ) -> None:
        records = self.load()

        sleep_hr = analysis.physiology.median_sleep_hr_bpm
        sleep_rmssd = analysis.physiology.median_sleep_rmssd_ms

        record = {
            "session_id": analysis.session_id,
            "started_at_ns": int(started_at_ns),
            "ended_at_ns": int(ended_at_ns),
            "median_sleep_hr_bpm": (
                sleep_hr if math.isfinite(sleep_hr) else None
            ),
            "median_sleep_rmssd_ms": (
                sleep_rmssd if math.isfinite(sleep_rmssd) else None
            ),
            "total_sleep_minutes": (
                analysis.summary.total_sleep_minutes
            ),
            "sleep_efficiency_percent": (
                analysis.summary.sleep_efficiency_percent
            ),
            "recovery_score": analysis.recovery.score,
            "recovery_provisional": analysis.recovery.provisional,
        }

        records.append(record)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )
