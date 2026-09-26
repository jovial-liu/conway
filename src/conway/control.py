"""Cooperative session controls. Never send signals to a reused/unverified PID."""
from __future__ import annotations
import json
from contextlib import contextmanager
from pathlib import Path
import signal
import threading
import time
from .storage import atomic_write


class EmergencyStop(RuntimeError):
    """A user stop/failsafe request. Must not be retried as a model error."""


@contextmanager
def termination_signals():
    """Let SIGTERM follow the same cleanup path as an explicit stop; never respawn."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)
    def stop(signum, frame):
        raise EmergencyStop('Termination requested')
    signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class SessionControl:
    def __init__(self, root: Path, session_id: str) -> None:
        self.path = root / 'control.json'
        self.session_id = session_id

    def command(self) -> str | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if data.get('session_id') == self.session_id:
                command = data.get('command')
                if command not in ('pause', 'resume', 'stop'):
                    raise ValueError('unknown command')
                return command
        except (OSError, ValueError, AttributeError):
            raise EmergencyStop('Invalid control file; stop rather than ignore it')
        return None

    def check(self) -> None:
        if self.command() == 'stop':
            raise EmergencyStop('Stop requested')

    def sleep(self, seconds: float) -> None:
        end = time.monotonic() + max(0, seconds)
        while time.monotonic() < end:
            self.check()
            time.sleep(min(0.1, max(0, end - time.monotonic())))

    def request(self, command: str) -> None:
        if command not in {'pause', 'resume', 'stop'}:
            raise ValueError('Unknown control command')
        atomic_write(self.path, json.dumps({'session_id': self.session_id, 'command': command}))
