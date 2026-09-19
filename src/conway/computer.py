from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import platform
import time
from typing import Any
from .actions import validate_action
from .control import EmergencyStop
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
    metadata_provider: str = 'none'
    ui_tree: dict[str, Any] = field(default_factory=dict)

    @property
    def scale_x(self) -> float:
        return self.input_width / self.width if self.width else 1.0

    @property
    def scale_y(self) -> float:
        return self.input_height / self.height if self.height else 1.0

    def summary(self) -> dict[str, Any]:
        return {'screenshot_path': str(self.screenshot_path),
                'screenshot_size': {'width': self.width, 'height': self.height},
                'input_size': {'width': self.input_width, 'height': self.input_height},
                'coordinate_scale': {'x': round(self.scale_x, 4), 'y': round(self.scale_y, 4)},
                'cursor': {'x': self.cursor_x, 'y': self.cursor_y},
                'active_app': self.active_app, 'active_window': self.active_window,
                'metadata_provider': self.metadata_provider, 'ui_tree': self.ui_tree}


class Computer:
    """Primary-display GUI adapter. Requires a logged-in desktop and normal OS grants."""
    def __init__(self, screenshot_dir: Path, *, ui_tree: bool = False, ui_timeout: float = 2, ui_max_nodes: int = 80) -> None:
        self.screenshot_dir = screenshot_dir
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.ui_enabled, self.ui_timeout, self.ui_max_nodes = ui_tree, ui_timeout, ui_max_nodes
        try:
            import pyautogui
        except Exception as exc:
            raise RuntimeError('Desktop unavailable. Use a logged-in desktop with screen/input permissions, or --mock for an offline smoke test.') from exc
        self.gui = pyautogui
        self.gui.PAUSE, self.gui.FAILSAFE = 0.08, True

    def observe(self, cycle: int) -> Observation:
        iw, ih = self.gui.size()
        cursor = self.gui.position()
        image = self.gui.screenshot()
        path = self.screenshot_dir / f'screen-{cycle:08d}.png'
        image.save(path)
        width, height = image.size
        metadata = capture_desktop_metadata()
        ui = {}
        if self.ui_enabled:
            from .accessibility import capture_ui_tree
            ui = capture_ui_tree(timeout=self.ui_timeout, max_nodes=self.ui_max_nodes)
        return Observation(path, width, height, int(iw), int(ih),
                           min(width - 1, max(0, round(cursor.x * width / iw))),
                           min(height - 1, max(0, round(cursor.y * height / ih))),
                           metadata.active_app, metadata.active_window, metadata.provider, ui)

    def prune_screenshots(self, keep: int = 30) -> None:
        for path in sorted(self.screenshot_dir.glob('screen-*.png'))[:-max(1, keep)]:
            path.unlink(missing_ok=True)

    @staticmethod
    def map_screenshot_point(x: int, y: int, observation: Observation) -> tuple[int, int]:
        return (max(0, min(round(x * observation.scale_x), observation.input_width - 1)),
                max(0, min(round(y * observation.scale_y), observation.input_height - 1)))

    def _paste_text(self, text: str) -> None:
        import pyperclip
        previous = pyperclip.paste()
        pyperclip.copy(text)
        try:
            self.gui.hotkey('command' if platform.system() == 'Darwin' else 'ctrl', 'v')
            time.sleep(0.2)
        finally:
            # Do not overwrite a clipboard value changed by the user/app meanwhile.
            if pyperclip.paste() == text:
                pyperclip.copy(previous)

    def execute(self, action: dict, observation: Observation | None = None) -> str:
        iw, ih = self.gui.size()
        obs = observation or Observation(Path('.'), iw, ih, iw, ih)
        action = validate_action(action, obs.width, obs.height)
        kind, args = action['type'], action['args']
        if kind == 'wait':
            time.sleep(args['seconds'])
            return f"waited {args['seconds']:.2f}s"
        if (iw, ih) != (obs.input_width, obs.input_height):
            raise RuntimeError('Display geometry changed after observation; obtain a fresh screenshot')
        current = capture_desktop_metadata()
        if obs.active_window and current.active_window and obs.active_window != current.active_window:
            raise RuntimeError('Foreground window changed after observation; obtain a fresh screenshot')
        try:
            self.gui.failSafeCheck()
            if kind in {'click', 'double_click', 'move', 'drag'}:
                x, y = self.map_screenshot_point(args['x'], args['y'], obs)
                if kind == 'click':
                    self.gui.click(x=x, y=y, button=args['button'])
                elif kind == 'double_click':
                    self.gui.doubleClick(x=x, y=y, button=args['button'], interval=args['interval'])
                elif kind == 'move':
                    self.gui.moveTo(x, y, duration=args['duration'])
                else:
                    self.gui.dragTo(x, y, duration=args['duration'], button=args['button'])
            elif kind == 'type':
                text = args['text']
                if any(ord(c) > 127 for c in text):
                    self._paste_text(text)  # Fail explicitly; never claim Unicode was typed when it was not.
                else:
                    self.gui.write(text, interval=args['interval'])
            elif kind in {'press', 'hotkey'}:
                keys = [args['key']] if kind == 'press' else args['keys']
                if any(key not in self.gui.KEYBOARD_KEYS for key in keys):
                    raise ValueError('Unsupported keyboard key')
                self.gui.press(keys[0]) if kind == 'press' else self.gui.hotkey(*keys)
            elif kind == 'scroll':
                self.gui.scroll(args['amount'])
            else:
                raise ValueError(f'Unsupported GUI action: {kind}')
        except self.gui.FailSafeException as exc:
            raise EmergencyStop('Mouse-corner failsafe activated') from exc
        return f'{kind} dispatched; verify the new observation before assuming task success'


class MockComputer:
    """Synthetic observation source. Never imports GUI libraries or touches the desktop."""
    def __init__(self, screenshot_dir: Path) -> None:
        self.screenshot_dir = screenshot_dir
        screenshot_dir.mkdir(parents=True, exist_ok=True)

    def observe(self, cycle: int) -> Observation:
        from PIL import Image
        path = self.screenshot_dir / f'screen-{cycle:08d}.png'
        Image.new('RGB', (320, 200)).save(path)
        return Observation(path, 320, 200, 320, 200, metadata_provider='synthetic')

    def execute(self, action: dict, observation: Observation | None = None) -> str:
        if action.get('type') != 'wait':
            raise RuntimeError('MockComputer never executes real actions')
        return 'synthetic wait'

    prune_screenshots = Computer.prune_screenshots
