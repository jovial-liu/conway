"""Small, atomic file primitives. No database and no network side effects."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile


def atomic_write(path: Path, text: str, *, private: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        if not private and path.exists():
            temp.chmod(path.stat().st_mode & 0o777)
        os.replace(temp, path)
        if os.name == 'posix':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def read_tail(path: Path, max_bytes: int = 65536) -> str:
    """Bound memory even when a journal is very large; never return a partial first line."""
    with path.open('rb') as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        start = max(0, size - max_bytes)
        file.seek(start)
        if start:
            file.readline(max_bytes)
        return file.read(max_bytes).decode('utf-8', errors='replace')
