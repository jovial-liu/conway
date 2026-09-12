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
    provider: str = "none"

    def prompt_text(self) -> str:
        parts = []
        if self.active_app:
            parts.append(f"active app: {self.active_app}")
        if self.active_window:
            parts.append(f"active window: {self.active_window}")
        return "; ".join(parts) if parts else "active window metadata unavailable"


def _run_text(command: list[str], timeout: float = 1.5) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def _macos_metadata() -> DesktopMetadata:
    script = r'''
tell application "System Events"
    set p to first application process whose frontmost is true
    set appName to name of p
    set windowName to ""
    try
        set windowName to name of front window of p
    end try
    return appName & linefeed & windowName
end tell
'''.strip()
    text = _run_text(["osascript", "-e", script])
    if not text:
        return DesktopMetadata(provider="macos-system-events")
    lines = text.splitlines()
    return DesktopMetadata(
        active_app=lines[0].strip() if lines else None,
        active_window=lines[1].strip() if len(lines) > 1 and lines[1].strip() else None,
        provider="macos-system-events",
    )


def _windows_metadata() -> DesktopMetadata:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return DesktopMetadata(provider="win32")

        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(max(1, length + 1))
        user32.GetWindowTextW(hwnd, buffer, len(buffer))
        title = buffer.value.strip() or None

        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        app = None
        if pid.value:
            try:
                app = psutil.Process(pid.value).name()
            except (psutil.Error, OSError):
                pass
        return DesktopMetadata(active_app=app, active_window=title, provider="win32")
    except (AttributeError, OSError):
        return DesktopMetadata(provider="win32")


def _linux_metadata() -> DesktopMetadata:
    xdotool = shutil.which("xdotool")
    if not xdotool:
        return DesktopMetadata(provider="linux-no-xdotool")
    window_id = _run_text([xdotool, "getactivewindow"])
    if not window_id:
        return DesktopMetadata(provider="xdotool")
    title = _run_text([xdotool, "getwindowname", window_id])
    pid_text = _run_text([xdotool, "getwindowpid", window_id])
    app = None
    if pid_text:
        try:
            app = psutil.Process(int(pid_text)).name()
        except (ValueError, psutil.Error, OSError):
            pass
    return DesktopMetadata(active_app=app, active_window=title, provider="xdotool")


def capture_desktop_metadata() -> DesktopMetadata:
    """Return best-effort foreground-app/window metadata without making it mandatory.

    Screenshot perception remains the universal fallback. This helper deliberately
    uses only OS-provided interfaces or optional tools already present on the host.
    """
    system = platform.system()
    if system == "Darwin":
        return _macos_metadata()
    if system == "Windows":
        return _windows_metadata()
    if system == "Linux":
        return _linux_metadata()
    return DesktopMetadata(provider=system.lower() or "unknown")
