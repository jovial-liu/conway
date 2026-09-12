from __future__ import annotations

from typing import Any


class ActionValidationError(ValueError):
    pass


_ALLOWED = {
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


def validate_action(action: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Validate model output and return a bounded, normalized GUI action."""
    if not isinstance(action, dict):
        raise ActionValidationError("action must be an object")

    kind = str(action.get("type", "")).strip().lower()
    if kind not in _ALLOWED:
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
        text = str(args.get("text", ""))
        if len(text) > 4000:
            raise ActionValidationError("type action is limited to 4000 characters")
        interval = max(0.0, min(float(args.get("interval", 0.01)), 0.5))
        return {"type": kind, "args": {"text": text, "interval": interval}}

    if kind == "press":
        key = str(args.get("key", "")).strip().lower()
        if not key or len(key) > 32:
            raise ActionValidationError("press requires a short args.key")
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

    raise ActionValidationError(f"unhandled action type: {kind}")
