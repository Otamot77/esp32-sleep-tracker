from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SequenceStatus(Enum):
    CONTIGUOUS = "contiguous"
    GAP = "gap"
    DUPLICATE_OR_OLD = "duplicate_or_old"


@dataclass(frozen=True)
class SequenceCheck:
    status: SequenceStatus
    missing: int = 0


def classify_sequence(
    previous: int,
    current: int,
) -> SequenceCheck:
    """
    Classify a 32-bit wrapping sequence number.

    `current == expected`:
        normal next packet.

    Small forward delta:
        one or more packets were lost.

    Large modular backward delta:
        retry duplicate or stale/out-of-order packet; ignore it.
    """
    expected = (previous + 1) & 0xFFFFFFFF

    if current == expected:
        return SequenceCheck(
            SequenceStatus.CONTIGUOUS,
            0,
        )

    forward = (current - expected) & 0xFFFFFFFF

    if forward < 0x80000000:
        return SequenceCheck(
            SequenceStatus.GAP,
            max(1, forward),
        )

    return SequenceCheck(
        SequenceStatus.DUPLICATE_OR_OLD,
        0,
    )
