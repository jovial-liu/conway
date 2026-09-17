from __future__ import annotations

from pathlib import Path
import os
import platform
from typing import BinaryIO


class InstanceLock:
    """Best-effort cross-platform single-instance lock for one Conway state root."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file: BinaryIO | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+b")
        if file.seek(0, os.SEEK_END) == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        try:
            if platform.system() == "Windows":
                import msvcrt

                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            file.close()
            raise RuntimeError(
                f"Another Conway process appears to be using this state directory: {self.path.parent}"
            ) from exc

        file.seek(0)
        file.truncate()
        file.write(str(os.getpid()).encode("ascii"))
        file.flush()
        self.file = file

    def release(self) -> None:
        file = self.file
        if file is None:
            return
        try:
            file.seek(0)
            if platform.system() == "Windows":
                import msvcrt

                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            file.close()
            self.file = None

    def __enter__(self) -> "InstanceLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
