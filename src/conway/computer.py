from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import time
from typing import Any

from .actions import validate_action
from .desktop import capture_desktop_metadata


@dataclass(slots=True)
class Observation:
    screenshot_path: Path
    width: int
    height: int
    input_width: int
    input_height: int
    cursor_x: int = 0
    cursor_y: int = 0
    active_app: str | None = None
    active_window: str | None = None
    metadata_provider: str = "none"

    @property
    def scale_x(self) -> float:
        return self.input_width / self.width if self.width else 1.0

    @property
    def scale_y(self) -> float:
        return self.input_height / self.height if self.height else 1.0

    def summary(self) -> dict[str, Any]:
        return {
            "screenshot_path": str(self.screenshot_path),
            "screenshot_size": {"width": self.width, "height": self.height},
            "input_size": {"width": self.input_width, "height": self.input_height},
            "coordinate_scale": {"x": round(self.scale_x, 4), "y": round(self.scale_y, 4)},
            "cursor": {"x": self.cursor_x, "y": self.cursor_y},
            "active_app": self.active_app,
            "active_window": self.active_window,
            "metadata_provider": self.metadata_provider,
        }


class Computer:
    """Cross-platform screenshot/input adapter backed by PyAutoGUI.

    The screenshot coordinate space and OS input coordinate space are tracked
    separately. This matters on Retina/HiDPI and scaled displays where a screenshot
    can contain a different number of pixels than the mouse API reports.
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
        input_width, input_height = self.gui.size()
        cursor = self.gui.position()
        path = self.screenshot_dir / f"screen-{cycle:08d}.png"
        image = self.gui.screenshot()
        image.save(path)
        screenshot_width, screenshot_height = image.size

        # Convert the current cursor to the screenshot pixel space shown to the VLM.
        cursor_x = int(round(int(cursor.x) * screenshot_width / max(1, int(input_width))))
        cursor_y = int(round(int(cursor.y) * screenshot_height / max(1, int(input_height))))
        cursor_x = max(0, min(cursor_x, screenshot_width - 1))
        cursor_y = max(0, min(cursor_y, screenshot_height - 1))

        metadata = capture_desktop_metadata()
        return Observation(
            screenshot_path=path,
            width=int(screenshot_width),
            height=int(screenshot_height),
            input_width=int(input_width),
            input_height=int(input_height),
            cursor_x=cursor_x,
            cursor_y=cursor_y,
            active_app=metadata.active_app,
            active_window=metadata.active_window,
            metadata_provider=metadata.provider,
        )

    def prune_screenshots(self, keep: int = 30) -> None:
        files = sorted(self.screenshot_dir.glob("screen-*.png"))
        for path in files[:-max(1, keep)]:
            try:
                path.unlink()
            except OSError:
                pass

    @staticmethod
    def map_screenshot_point(x: int, y: int, observation: Observation) -> tuple[int, int]:
        """Map VLM screenshot pixels to the coordinate space used by mouse APIs."""
        mapped_x = int(round(x * observation.scale_x))
        mapped_y = int(round(y * observation.scale_y))
        mapped_x = max(0, min(mapped_x, observation.input_width - 1))
        mapped_y = max(0, min(mapped_y, observation.input_height - 1))
        return mapped_x, mapped_y

    def _paste_text(self, text: str) -> bool:
        try:
            import pyperclip

            previous = pyperclip.paste()
            pyperclip.copy(text)
            modifier = "command" if platform.system() == "Darwin" else "ctrl"
            self.gui.hotkey(modifier, "v")
            # Most GUI toolkits consume the paste synchronously; a short delay avoids
            # restoring the clipboard before the target application has read it.
            time.sleep(0.08)
            pyperclip.copy(previous)
            return True
        except Exception:
            return False

    def execute(self, action: dict[str, Any], observation: Observation | None = None) -> str:
        if observation is None:
            current_width, current_height = self.gui.size()
            observation = Observation(
                screenshot_path=Path("."),
                width=int(current_width),
                height=int(current_height),
                input_width=int(current_width),
                input_height=int(current_height),
            )

        normalized = validate_action(action, observation.width, observation.height)
        kind = normalized["type"]
        args = normalized["args"]

        if kind == "wait":
            seconds = float(args["seconds"])
            time.sleep(seconds)
            return f"waited {seconds:.2f}s"

        if kind in {"click", "double_click", "move", "drag"}:
            x, y = self.map_screenshot_point(args["x"], args["y"], observation)

            if kind == "click":
                self.gui.click(x=x, y=y, button=args["button"])
                return f"clicked screenshot ({args['x']}, {args['y']}) -> input ({x}, {y})"

            if kind == "double_click":
                self.gui.doubleClick(x=x, y=y, interval=0.12, button=args["button"])
                return f"double-clicked screenshot ({args['x']}, {args['y']}) -> input ({x}, {y})"

            if kind == "move":
                self.gui.moveTo(x, y, duration=args["duration"])
                return f"moved pointer to input ({x}, {y})"

            self.gui.dragTo(x, y, duration=args["duration"], button=args["button"])
            return f"dragged pointer to input ({x}, {y})"

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

        raise ValueError(f"Unsupported action type: {kind}")
