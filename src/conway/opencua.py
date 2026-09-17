from __future__ import annotations

import ast
from math import ceil, floor, sqrt
import re
from typing import Any


class OpenCUAParseError(ValueError):
    pass


_OPENCUA_CALL = re.compile(
    r"(?:pyautogui\.)?(click|doubleClick|moveTo|dragTo|write|typewrite|press|hotkey|scroll|sleep)\s*\(",
    re.IGNORECASE,
)


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 12_845_056,
) -> tuple[int, int]:
    """Reproduce the Qwen2.5-VL smart-resize geometry used by OpenCUA.

    OpenCUA-7B emits coordinates in the smart-resized image, so Conway maps them
    back to the original screenshot before its normal HiDPI conversion.
    """
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("image aspect ratio is too extreme")

    resized_h = max(factor, round(height / factor) * factor)
    resized_w = max(factor, round(width / factor) * factor)
    area = resized_h * resized_w

    if area > max_pixels:
        beta = sqrt((height * width) / max_pixels)
        resized_h = max(factor, floor(height / beta / factor) * factor)
        resized_w = max(factor, floor(width / beta / factor) * factor)
    elif area < min_pixels:
        beta = sqrt(min_pixels / (height * width))
        resized_h = max(factor, ceil(height * beta / factor) * factor)
        resized_w = max(factor, ceil(width * beta / factor) * factor)
    return int(resized_h), int(resized_w)


def model_point_to_screenshot(
    model_x: float,
    model_y: float,
    original_width: int,
    original_height: int,
) -> tuple[int, int]:
    resized_h, resized_w = smart_resize(original_height, original_width)
    x = int(round(float(model_x) / resized_w * original_width))
    y = int(round(float(model_y) / resized_h * original_height))
    x = max(0, min(x, original_width - 1))
    y = max(0, min(y, original_height - 1))
    return x, y


def _extract_call_source(text: str) -> str:
    match = _OPENCUA_CALL.search(text)
    if not match:
        raise OpenCUAParseError("no supported pyautogui call found")
    start = match.start()
    depth = 0
    quote: str | None = None
    escape = False
    for index in range(match.end() - 1, len(text)):
        char = text[index]
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise OpenCUAParseError("unterminated pyautogui call")


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise OpenCUAParseError("OpenCUA action contains a non-literal argument") from exc


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "pyautogui":
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    raise OpenCUAParseError("unsupported call target")


def _args(call: ast.Call) -> tuple[list[Any], dict[str, Any]]:
    positional = [_literal(node) for node in call.args]
    keywords: dict[str, Any] = {}
    for item in call.keywords:
        if item.arg is None:
            raise OpenCUAParseError("**kwargs are not supported")
        keywords[item.arg] = _literal(item.value)
    return positional, keywords


def _value(positional: list[Any], keywords: dict[str, Any], index: int, key: str, default: Any = None) -> Any:
    if key in keywords:
        return keywords[key]
    if index < len(positional):
        return positional[index]
    return default


def parse_opencua_action(text: str, width: int, height: int) -> dict[str, Any]:
    """Safely translate one OpenCUA pyautogui-style call into Conway's action schema.

    The parser never evaluates model-generated Python. It accepts only one literal
    call from a small allowlist and converts OpenCUA's smart-resized coordinates
    into the original screenshot coordinate space.
    """
    source = _extract_call_source(text.strip())
    try:
        expression = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise OpenCUAParseError(f"invalid OpenCUA action syntax: {exc}") from exc
    if not isinstance(expression.body, ast.Call):
        raise OpenCUAParseError("OpenCUA output is not a function call")

    call = expression.body
    name = _call_name(call)
    positional, keywords = _args(call)
    lowered = name.lower()

    if lowered in {"click", "doubleclick", "moveto", "dragto"}:
        model_x = _value(positional, keywords, 0, "x")
        model_y = _value(positional, keywords, 1, "y")
        if model_x is None or model_y is None:
            raise OpenCUAParseError(f"{name} requires x and y")
        x, y = model_point_to_screenshot(float(model_x), float(model_y), width, height)
        if lowered == "click":
            return {
                "type": "click",
                "args": {"x": x, "y": y, "button": str(keywords.get("button", "left"))},
            }
        if lowered == "doubleclick":
            return {
                "type": "double_click",
                "args": {"x": x, "y": y, "button": str(keywords.get("button", "left"))},
            }
        if lowered == "moveto":
            return {
                "type": "move",
                "args": {"x": x, "y": y, "duration": float(keywords.get("duration", 0.2))},
            }
        return {
            "type": "drag",
            "args": {
                "x": x,
                "y": y,
                "duration": float(keywords.get("duration", 0.5)),
                "button": str(keywords.get("button", "left")),
            },
        }

    if lowered in {"write", "typewrite"}:
        text_value = _value(positional, keywords, 0, "message", "")
        if "text" in keywords:
            text_value = keywords["text"]
        interval = float(keywords.get("interval", 0.01))
        return {"type": "type", "args": {"text": str(text_value), "interval": interval}}

    if lowered == "press":
        key = _value(positional, keywords, 0, "key")
        if key is None:
            raise OpenCUAParseError("press requires a key")
        return {"type": "press", "args": {"key": str(key)}}

    if lowered == "hotkey":
        if not positional:
            raise OpenCUAParseError("hotkey requires keys")
        return {"type": "hotkey", "args": {"keys": [str(key) for key in positional]}}

    if lowered == "scroll":
        amount = _value(positional, keywords, 0, "clicks")
        if amount is None:
            amount = keywords.get("amount")
        if amount is None:
            raise OpenCUAParseError("scroll requires an amount")
        return {"type": "scroll", "args": {"amount": int(amount)}}

    if lowered == "sleep":
        seconds = _value(positional, keywords, 0, "seconds", 1.0)
        return {"type": "wait", "args": {"seconds": float(seconds)}}

    raise OpenCUAParseError(f"unsupported OpenCUA call: {name}")
