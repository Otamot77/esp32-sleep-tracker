from __future__ import annotations

from enum import IntEnum


class SleepStage(IntEnum):
    """
    Four-class wearable-oriented stage set.

    Clinical PSG scoring separates N1 and N2. Our non-EEG wearable cannot
    reliably make that distinction, so both are represented as LIGHT.
    """

    WAKE = 0
    LIGHT = 1
    DEEP = 2
    REM = 3


STAGE_NAMES: tuple[str, ...] = (
    "wake",
    "light",
    "deep",
    "rem",
)


def stage_name(stage: SleepStage | int) -> str:
    return STAGE_NAMES[int(stage)]
