"""Exclusive local report writes; diagnostic commands never overwrite user files."""
from __future__ import annotations
import json
from pathlib import Path
import os


def report_path(output: Path, state_root: Path) -> Path:
    path = output.expanduser().absolute()
    if path.exists() or path.is_symlink() or path.resolve().is_relative_to(state_root.resolve()):
        raise ValueError('Report path must be new and outside the state directory')
    return path


def write_report(output: Path, report: dict, state_root: Path) -> None:
    path = report_path(output, state_root)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open('x', encoding='utf-8') as file:
            created = True
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
