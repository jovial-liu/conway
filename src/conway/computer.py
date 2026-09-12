from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any


@dataclass(slots=True)
class Observation:
    screenshot_path: Path
    width: int
    height: int

    def summary(self) -> dict[str, Any]:
        return {
            "screenshot_path": str(self.screenshot_path),
            "screen_width": self.width,
            "screen_height": self.height,
        }


class Computer:
    """Cross-platform GUI adapter backed by PyAutoGUI.

    It operates only with permissions already granted by the OS. Platform-native
    accessibility-tree adapters can be layered on top later without changing the
    autonomous loop API.
    """

    def __init__(self, screenshot_dir: Path) -> None:
        self.screenshot_dir = screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        try:
            import pyautogui  # imported lazily so `conway doctor` works headlessly
        except Exception as exc:  # pragma: no cover - depends on desktop session
            raise RuntimeError(
                "GUI backend could not initialize. Make sure Conway is running in a desktop session "
                "and that normal OS screen/input permissions are granted."
            ) from exc
        self.gui = pyautogui
        self.gui.PAUSE = 0.08
        self.gui.FAILSAFE = True

    def observe(self, cycle: int) -> Observation:
        width, height = self.gui.size()
        path = self.screenshot_dir / f"screen-{cycle:08d}.png"
        image = self.gui.screenshot()
        image.save(path)
        return Observation(path, int(width), int(height))

    def _point(self, x: Any, y: Any) -> tuple[int, int]:
        width, height = self.gui.size()
        xi = max(0, min(int(x), int(width) - 1))
        yi = max(0, min(int(y), int(height) - 1))
        return xi, yi

    def execute(self, action: dict[str, Any]) -> str:
        kind = str(action.get("type", "wait")).lower()
        args = action.get("args") or {}

        if kind == "wait":
            seconds = max(0.0, min(float(args.get("seconds", 1.0)), 30.0))
            time.sleep(seconds)
            return f"waited {seconds:.2f}s"

        if kind == "click":
            x, y = self._point(args["x"], args["y"])
            self.gui.click(x=x, y=y, button=str(args.get("button", "left")))
            return f"clicked ({x}, {y})"

        if kind == "double_click":
            x, y = self._point(args["x"], args["y"])
            self.gui.doubleClick(x=x, y=y, interval=0.12, button=str(args.get("button", "left")))
            return f"double-clicked ({x}, {y})"

        if kind == "move":
            x, y = self._point(args["x"], args["y"])
            self.gui.moveTo(x, y, duration=min(float(args.get("duration", 0.2)), 2.0))
            return f"moved pointer to ({x}, {y})"

        if kind == "type":
            text = str(args.get("text", ""))
            self.gui.write(text, interval=max(0.0, min(float(args.get("interval", 0.01)), 0.5)))
            return f"typed {len(text)} characters"

        if kind == "press":
            key = str(args["key"])
            self.gui.press(key)
            return f"pressed {key}"

        if kind == "hotkey":
            keys = [str(k) for k in args.get("keys", [])]
            if not keys:
                raise ValueError("hotkey requires args.keys")
            self.gui.hotkey(*keys)
            return f"pressed hotkey {'+'.join(keys)}"

        if kind == "scroll":
            amount = int(args.get("amount", 0))
            self.gui.scroll(amount)
            return f"scrolled {amount}"

        if kind == "drag":
            x, y = self._point(args["x"], args["y"])
            self.gui.dragTo(x, y, duration=min(float(args.get("duration", 0.5)), 3.0), button=str(args.get("button", "left")))
            return f"dragged pointer to ({x}, {y})"

        raise ValueError(f"Unsupported action type: {kind}")
