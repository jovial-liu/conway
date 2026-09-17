from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


class ActionValidationError(ValueError):
    pass


GUI_ACTIONS = frozenset(
    {
        "wait",
        "click",
        "double_click",
        "move",
        "type",
        "press",
        "hotkey",
        "scroll",
        "drag",
    }
)
SYSTEM_ACTIONS = frozenset({"shell", "read_file", "write_file", "list_dir", "open_url"})
ALL_ACTIONS = GUI_ACTIONS | SYSTEM_ACTIONS


def _number(args: dict[str, Any], key: str) -> float:
    try:
        return float(args[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ActionValidationError(f"action requires numeric args.{key}") from exc


def _point(args: dict[str, Any], width: int, height: int) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ActionValidationError("invalid screen dimensions")
    x = max(0, min(int(_number(args, "x")), width - 1))
    y = max(0, min(int(_number(args, "y")), height - 1))
    return x, y


def _text(args: dict[str, Any], key: str, *, max_len: int, allow_empty: bool = False) -> str:
    value = str(args.get(key, ""))
    if not allow_empty and not value.strip():
        raise ActionValidationError(f"action requires args.{key}")
    if len(value) > max_len:
        raise ActionValidationError(f"args.{key} exceeds {max_len} characters")
    return value


def validate_action(action: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Validate model output and return a bounded, normalized action.

    Conway deliberately validates both GUI and local system actions before routing
    them to an executor. This is schema validation and resource bounding, not an
    OS-permission bypass: all actions still execute as the user who started Conway.
    """
    if not isinstance(action, dict):
        raise ActionValidationError("action must be an object")

    kind = str(action.get("type", "")).strip().lower()
    if kind not in ALL_ACTIONS:
        raise ActionValidationError(f"unsupported action type: {kind or '<missing>'}")

    raw_args = action.get("args") or {}
    if not isinstance(raw_args, dict):
        raise ActionValidationError("action.args must be an object")
    args = dict(raw_args)

    if kind == "wait":
        seconds = max(0.0, min(float(args.get("seconds", 1.0)), 30.0))
        return {"type": kind, "args": {"seconds": seconds}}

    if kind in {"click", "double_click", "move"}:
        x, y = _point(args, width, height)
        out: dict[str, Any] = {"x": x, "y": y}
        if kind in {"click", "double_click"}:
            button = str(args.get("button", "left")).lower()
            if button not in {"left", "middle", "right"}:
                raise ActionValidationError("button must be left, middle, or right")
            out["button"] = button
        if kind == "move":
            out["duration"] = max(0.0, min(float(args.get("duration", 0.2)), 2.0))
        return {"type": kind, "args": out}

    if kind == "drag":
        x, y = _point(args, width, height)
        button = str(args.get("button", "left")).lower()
        if button not in {"left", "middle", "right"}:
            raise ActionValidationError("button must be left, middle, or right")
        duration = max(0.0, min(float(args.get("duration", 0.5)), 3.0))
        return {"type": kind, "args": {"x": x, "y": y, "button": button, "duration": duration}}

    if kind == "type":
        text = _text(args, "text", max_len=4000, allow_empty=True)
        interval = max(0.0, min(float(args.get("interval", 0.01)), 0.5))
        return {"type": kind, "args": {"text": text, "interval": interval}}

    if kind == "press":
        key = _text(args, "key", max_len=32).strip().lower()
        return {"type": kind, "args": {"key": key}}

    if kind == "hotkey":
        keys = args.get("keys")
        if not isinstance(keys, list) or not 1 <= len(keys) <= 5:
            raise ActionValidationError("hotkey requires 1-5 args.keys")
        normalized = [str(key).strip().lower() for key in keys]
        if any(not key or len(key) > 32 for key in normalized):
            raise ActionValidationError("invalid hotkey key")
        return {"type": kind, "args": {"keys": normalized}}

    if kind == "scroll":
        amount = max(-50, min(int(_number(args, "amount")), 50))
        return {"type": kind, "args": {"amount": amount}}

    if kind == "shell":
        command = _text(args, "command", max_len=16000)
        cwd = args.get("cwd")
        cwd_text = str(cwd) if cwd not in (None, "") else None
        if cwd_text and len(cwd_text) > 4096:
            raise ActionValidationError("args.cwd is too long")
        timeout = max(1.0, min(float(args.get("timeout", 30.0)), 120.0))
        return {"type": kind, "args": {"command": command, "cwd": cwd_text, "timeout": timeout}}

    if kind == "read_file":
        path = _text(args, "path", max_len=4096)
        max_chars = max(100, min(int(args.get("max_chars", 50000)), 200000))
        return {"type": kind, "args": {"path": path, "max_chars": max_chars}}

    if kind == "write_file":
        path = _text(args, "path", max_len=4096)
        content = _text(args, "content", max_len=1_000_000, allow_empty=True)
        append = bool(args.get("append", False))
        return {"type": kind, "args": {"path": path, "content": content, "append": append}}

    if kind == "list_dir":
        path = str(args.get("path", "."))
        if not path or len(path) > 4096:
            raise ActionValidationError("invalid args.path")
        limit = max(1, min(int(args.get("limit", 200)), 500))
        return {"type": kind, "args": {"path": path, "limit": limit}}

    if kind == "open_url":
        url = _text(args, "url", max_len=8192)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ActionValidationError("open_url requires an http(s) URL")
        return {"type": kind, "args": {"url": url}}

    raise ActionValidationError(f"unhandled action type: {kind}")
