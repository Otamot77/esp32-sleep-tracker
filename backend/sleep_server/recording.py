from __future__ import annotations

from pathlib import Path
from typing import BinaryIO


class PacketRecorder:
    """
    Append the exact binary protocol stream to disk.

    The same recorder survives brief TCP reconnects because it belongs to the
    logical boot session, not to one socket connection.
    """

    def __init__(
        self,
        directory: str | Path,
        session_id: str,
    ) -> None:
        self.path = (
            Path(directory) /
            f"{session_id}.stbin"
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._handle: BinaryIO = self.path.open("ab")

    def append(self, data: bytes) -> None:
        self._handle.write(data)

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()
