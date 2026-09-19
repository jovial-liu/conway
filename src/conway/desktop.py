from __future__ import annotations
from dataclasses import dataclass
import ctypes
import platform
import shutil
import subprocess
import psutil


@dataclass(slots=True)
class DesktopMetadata:
    active_app: str | None = None
    active_window: str | None = None
    provider: str = 'none'

    def prompt_text(self) -> str:
        parts = []
        if self.active_app:
            parts.append(f'active app: {self.active_app}')
        if self.active_window:
            parts.append(f'active window: {self.active_window}')
        return '; '.join(parts) if parts else 'active window metadata unavailable'


def _run_text(command: list[str], timeout: float = 1.5) -> str | None:
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return p.stdout.strip()[:2000] or None if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _windows_metadata() -> DesktopMetadata:
    try:
        from ctypes import wintypes
        user = ctypes.windll.user32
        user.GetForegroundWindow.restype = wintypes.HWND
        user.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        hwnd = user.GetForegroundWindow()
        buf = ctypes.create_unicode_buffer(min(2000, max(1, user.GetWindowTextLengthW(hwnd) + 1)))
        user.GetWindowTextW(hwnd, buf, len(buf))
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        app = psutil.Process(pid.value).name() if pid.value else None
        return DesktopMetadata(app, buf.value or None, 'win32')
    except (AttributeError, OSError, psutil.Error):
        return DesktopMetadata(provider='win32')


def _macos_metadata() -> DesktopMetadata:
    script = '''tell application "System Events"
set p to first application process whose frontmost is true
set appName to name of p
set windowName to ""
try
set windowName to name of front window of p
end try
return appName & linefeed & windowName
end tell'''
    text = _run_text(['osascript', '-e', script])
    parts = text.splitlines() if text else []
    return DesktopMetadata(parts[0] if parts else None, parts[1] if len(parts) > 1 else None, 'macos-system-events')


def _linux_metadata() -> DesktopMetadata:
    exe = shutil.which('xdotool')
    if not exe:
        return DesktopMetadata(provider='linux-no-xdotool')
    window = _run_text([exe, 'getactivewindow'])
    if not window:
        return DesktopMetadata(provider='xdotool')
    title = _run_text([exe, 'getwindowname', window])
    pid = _run_text([exe, 'getwindowpid', window])
    try:
        app = psutil.Process(int(pid)).name() if pid else None
    except (ValueError, psutil.Error):
        app = None
    return DesktopMetadata(app, title, 'xdotool')


def capture_desktop_metadata() -> DesktopMetadata:
    backend = {'Darwin': _macos_metadata, 'Windows': _windows_metadata, 'Linux': _linux_metadata}.get(platform.system())
    return backend() if backend else DesktopMetadata()
