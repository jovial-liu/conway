from __future__ import annotations

from pathlib import Path
import subprocess
import webbrowser
from typing import Any

from .actions import GUI_ACTIONS, validate_action
from .computer import Computer, Observation
from .config import ToolSettings


class ToolExecutor:
    """Route validated actions to GUI or local current-user tools.

    System tools are intentionally ordinary user-space operations. Conway does not
    add privilege escalation or attempt to bypass operating-system controls.
    """

    def __init__(self, computer: Computer, settings: ToolSettings) -> None:
        self.computer = computer
        self.settings = settings

    def enabled_actions(self) -> set[str]:
        enabled: set[str] = set()
        if self.settings.gui:
            enabled.update(GUI_ACTIONS)
        else:
            enabled.add("wait")
        if self.settings.shell:
            enabled.add("shell")
        if self.settings.filesystem:
            enabled.update({"read_file", "write_file", "list_dir"})
        if self.settings.open_url:
            enabled.add("open_url")
        return enabled

    def manifest_text(self) -> str:
        enabled = self.enabled_actions()
        lines: list[str] = []
        if "click" in enabled:
            lines.extend(
                [
                    '- click: {"x": screenshot_x, "y": screenshot_y, "button": "left|middle|right"}',
                    '- double_click: {"x": screenshot_x, "y": screenshot_y, "button": "left|middle|right"}',
                    '- move: {"x": screenshot_x, "y": screenshot_y, "duration": 0.2}',
                    '- drag: {"x": screenshot_x, "y": screenshot_y, "duration": 0.5, "button": "left"}',
                    '- type: {"text": "...", "interval": 0.01}',
                    '- press: {"key": "enter"}',
                    '- hotkey: {"keys": ["ctrl", "l"]}',
                    '- scroll: {"amount": -5}',
                ]
            )
        lines.append('- wait: {"seconds": 1}')
        if "shell" in enabled:
            lines.append('- shell: {"command": "...", "cwd": null, "timeout": 30}')
        if "read_file" in enabled:
            lines.extend(
                [
                    '- read_file: {"path": "...", "max_chars": 50000}',
                    '- write_file: {"path": "...", "content": "...", "append": false}',
                    '- list_dir: {"path": ".", "limit": 200}',
                ]
            )
        if "open_url" in enabled:
            lines.append('- open_url: {"url": "https://..."}')
        return "\n".join(lines)

    def _bounded_output(self, text: str) -> str:
        limit = self.settings.max_output_chars
        if len(text) <= limit:
            return text
        return "[output truncated]\n" + text[-limit:]

    @staticmethod
    def _path(value: str) -> Path:
        return Path(value).expanduser()

    def execute(self, action: dict[str, Any], observation: Observation) -> str:
        normalized = validate_action(action, observation.width, observation.height)
        kind = normalized["type"]
        args = normalized["args"]
        if kind not in self.enabled_actions():
            raise PermissionError(f"Conway tool is disabled by config: {kind}")

        if kind in GUI_ACTIONS:
            return self.computer.execute(normalized, observation)

        if kind == "read_file":
            path = self._path(args["path"])
            text = path.read_text(encoding="utf-8", errors="replace")
            max_chars = int(args["max_chars"])
            if len(text) > max_chars:
                text = "[file truncated]\n" + text[-max_chars:]
            return f"read {path}\n{text}"

        if kind == "write_file":
            path = self._path(args["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if args["append"] else "w"
            with path.open(mode, encoding="utf-8") as file:
                file.write(args["content"])
            verb = "appended" if args["append"] else "wrote"
            return f"{verb} {len(args['content'])} characters to {path}"

        if kind == "list_dir":
            path = self._path(args["path"])
            entries = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            rendered: list[str] = []
            for item in entries[: int(args["limit"])]:
                marker = "dir" if item.is_dir() else "file"
                rendered.append(f"{marker}\t{item.name}")
            if len(entries) > int(args["limit"]):
                rendered.append(f"... {len(entries) - int(args['limit'])} more entries")
            return f"listed {path}\n" + "\n".join(rendered)

        if kind == "shell":
            timeout = min(float(args["timeout"]), self.settings.max_shell_seconds)
            cwd = self._path(args["cwd"]) if args.get("cwd") else None
            try:
                result = subprocess.run(
                    args["command"],
                    cwd=cwd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                output = self._bounded_output(f"stdout:\n{stdout}\nstderr:\n{stderr}")
                return f"shell timed out after {timeout:.1f}s\n{output}"
            output = self._bounded_output(f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            return f"shell exit code {result.returncode}\n{output}"

        if kind == "open_url":
            opened = webbrowser.open(args["url"], new=2, autoraise=True)
            return f"requested browser open for {args['url']} (accepted={opened})"

        raise ValueError(f"Unsupported tool action: {kind}")
