from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import time
from typing import Any

from .actions import validate_action


@dataclass(slots=True)
class Observation:
    screenshot_path: Path
    width: int
    height: int
    cursor_x: int = 0
    cursor_y: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "screenshot_path": str(self.screenshot_path),
            "screen_width": self.width,
            "screen_height": self.height,
            "cursor": {"x": self.cursor_x, "y": self.cursor_y},
        }


class Computer:
    """Cross-platform GUI adapter backed by PyAutoGUI.

    Conway uses only permissions already granted to the current OS user. The
    adapter intentionally keeps the loop API independent of the eventual native
    accessibility-tree implementations for macOS, Windows, and Linux.
    """

    def __init__(self, screenshot_dir: Path) -> None:
        self.screenshot_dir = screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        try:
            import pyautogui  # lazy import: `conway doctor` remains headless
        except Exception as exc:  # pragma: no cover - depends on desktop session
            raise RuntimeError(
                "GUI backend could not initialize. Run Conway in a desktop session and grant normal "
                "OS screen/input permissions."
            ) from exc
        self.gui = pyautogui
        self.gui.PAUSE = 0.08
        self.gui.FAILSAFE = True

    def observe(self, cycle: int) -> Observation:
        width, height = self.gui.size()
        cursor = self.gui.position()
        path = self.screenshot_dir / f"screen-{cycle:08d}.png"
        image = self.gui.screenshot()
        image.save(path)
        return Observation(path, int(width), int(height), int(cursor.x), int(cursor.y))

    def prune_screenshots(self, keep: int = 30) -> None:
        files = sorted(self.screenshot_dir.glob("screen-*.png"))
        for path in files[:-max(1, keep)]:
            try:
                path.unlink()
            except OSError:
                pass

    def _paste_text(self, text: str) -> bool:
        try:
            import pyperclip

            pyperclip.copy(text)
            modifier = "command" if platform.system() == "Darwin" else "ctrl"
            self.gui.hotkey(modifier, "v")
            return True
        except Exception:
            return False

    def execute(self, action: dict[str, Any], width: int | None = None, height: int | None = None) -> str:
        if width is None or height is None:
            current_width, current_height = self.gui.size()
            width, height = int(current_width), int(current_height)
        normalized = validate_action(action, width, height)
        kind = normalized["type"]
        args = normalized["args"]

        if kind == "wait":
            seconds = float(args["seconds"])
            time.sleep(seconds)
            return f"waited {seconds:.2f}s"

        if kind == "click":
            self.gui.click(x=args["x"], y=args["y"], button=args["button"])
            return f"clicked ({args['x']}, {args['y']})"

        if kind == "double_click":
            self.gui.doubleClick(x=args["x"], y=args["y"], interval=0.12, button=args["button"])
            return f"double-clicked ({args['x']}, {args['y']})"

        if kind == "move":
            self.gui.moveTo(args["x"], args["y"], duration=args["duration"])
            return f"moved pointer to ({args['x']}, {args['y']})"

        if kind == "type":
            text = args["text"]
            # PyAutoGUI's write() is ASCII-oriented. Clipboard paste gives much
            # better Unicode behavior across macOS/Windows and many Linux desktops.
            if any(ord(char) > 127 for char in text) and self._paste_text(text):
                return f"pasted {len(text)} characters"
            self.gui.write(text, interval=args["interval"])
            return f"typed {len(text)} characters"

        if kind == "press":
            self.gui.press(args["key"])
            return f"pressed {args['key']}"

        if kind == "hotkey":
            self.gui.hotkey(*args["keys"])
            return f"pressed hotkey {'+'.join(args['keys'])}"

        if kind == "scroll":
            self.gui.scroll(args["amount"])
            return f"scrolled {args['amount']}"

        if kind == "drag":
            self.gui.dragTo(
                args["x"],
                args["y"],
                duration=args["duration"],
                button=args["button"],
            )
            return f"dragged pointer to ({args['x']}, {args['y']})"

        raise ValueError(f"Unsupported action type: {kind}")
