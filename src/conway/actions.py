from __future__ import annotations

import math
import json
from typing import Any
from urllib.parse import urlsplit


class ActionValidationError(ValueError):
    pass


GUI_ACTIONS = frozenset({'wait', 'click', 'double_click', 'move', 'type', 'press', 'hotkey', 'scroll', 'drag'})
SYSTEM_ACTIONS = frozenset({'shell', 'read_file', 'write_file', 'list_dir', 'open_url', 'read_skill', 'mcp_call'})
CONTROL_ACTIONS = frozenset({'finish'})
ALL_ACTIONS = GUI_ACTIONS | SYSTEM_ACTIONS | CONTROL_ACTIONS
SIDE_EFFECT_ACTIONS = (GUI_ACTIONS - {'wait'}) | {'shell', 'write_file', 'open_url', 'mcp_call'}


def _number(args: dict[str, Any], key: str) -> float:
    val = args.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        raise ActionValidationError(f'args.{key} must be a finite number')
    return float(val)


def _bounded(args: dict, key: str, default: float, low: float, high: float) -> float:
    return min(high, max(low, _number({key: args.get(key, default)}, key)))


def _point(args: dict, width: int, height: int) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ActionValidationError('invalid screen dimensions')
    return max(0, min(int(_number(args, 'x')), width - 1)), max(0, min(int(_number(args, 'y')), height - 1))


def _text(args: dict, key: str, *, max_len: int, allow_empty: bool = False) -> str:
    val = args.get(key, '')
    if not isinstance(val, str) or '\x00' in val:
        raise ActionValidationError(f'args.{key} must be text without NUL')
    if (not allow_empty and not val.strip()) or len(val) > max_len:
        raise ActionValidationError(f'args.{key} must contain {0 if allow_empty else 1}..{max_len} characters')
    return val


def validate_action(action: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    if not isinstance(action, dict) or not isinstance(action.get('type'), str):
        raise ActionValidationError('action must be an object with a string type')
    kind = action['type'].strip().lower()
    if kind not in ALL_ACTIONS:
        raise ActionValidationError(f'unsupported action type: {kind}')
    args = action.get('args', {})
    if not isinstance(args, dict):
        raise ActionValidationError('action.args must be an object')
    out: dict[str, Any]
    if kind == 'wait':
        out = {'seconds': _bounded(args, 'seconds', 1, 0, 30)}
    elif kind == 'finish':
        outcome = args.get('outcome', 'completed')
        if not isinstance(outcome, str) or outcome not in {'completed', 'failed'}:
            raise ActionValidationError('finish outcome must be completed or failed')
        out = {'reason': _text(args, 'reason', max_len=2000, allow_empty=True), 'outcome': outcome}
    elif kind in {'click', 'double_click', 'move', 'drag'}:
        x, y = _point(args, width, height)
        out = {'x': x, 'y': y}
        if kind != 'move':
            button = args.get('button', 'left')
            if not isinstance(button, str) or button not in {'left', 'middle', 'right'}:
                raise ActionValidationError('button must be left, middle, or right')
            out['button'] = button
        if kind == 'double_click':
            out['interval'] = _bounded(args, 'interval', 0.12, 0, 0.5)
        if kind in {'move', 'drag'}:
            out['duration'] = _bounded(args, 'duration', 0.2 if kind == 'move' else 0.5, 0, 3)
    elif kind == 'type':
        out = {'text': _text(args, 'text', max_len=4000, allow_empty=True),
               'interval': _bounded(args, 'interval', 0.01, 0, 0.1)}
    elif kind == 'press':
        out = {'key': _text(args, 'key', max_len=32).strip().lower()}
    elif kind == 'hotkey':
        keys = args.get('keys')
        if not isinstance(keys, list) or not 1 <= len(keys) <= 5:
            raise ActionValidationError('hotkey requires 1-5 keys')
        out = {'keys': [_text({'key': key}, 'key', max_len=32).strip().lower() for key in keys]}
    elif kind == 'scroll':
        out = {'amount': int(_bounded(args, 'amount', 0, -50, 50))}
    elif kind == 'shell':
        cwd = args.get('cwd')
        if cwd is not None:
            cwd = _text(args, 'cwd', max_len=4096)
        out = {'command': _text(args, 'command', max_len=16000), 'cwd': cwd,
               'timeout': _bounded(args, 'timeout', 30, 1, 120)}
    elif kind == 'read_file':
        out = {'path': _text(args, 'path', max_len=4096),
               'max_chars': int(_bounded(args, 'max_chars', 50000, 100, 200000))}
    elif kind == 'write_file':
        append = args.get('append', False)
        if type(append) is not bool:
            raise ActionValidationError('append must be a boolean')
        out = {'path': _text(args, 'path', max_len=4096),
               'content': _text(args, 'content', max_len=1000000, allow_empty=True), 'append': append}
    elif kind == 'list_dir':
        out = {'path': _text({'path': args.get('path', '.')}, 'path', max_len=4096),
               'limit': int(_bounded(args, 'limit', 200, 1, 500))}
    elif kind == 'open_url':
        url = _text(args, 'url', max_len=8192)
        try:
            parsed = urlsplit(url)
            _ = parsed.port
            if parsed.scheme not in {'https', 'http'} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError()
        except ValueError as exc:
            raise ActionValidationError('open_url requires http(s) without embedded credentials') from exc
        out = {'url': url}
    elif kind == 'read_skill':
        out = {'name': _text(args, 'name', max_len=64),
               'resource': _text({'resource': args.get('resource', 'SKILL.md')}, 'resource', max_len=4096),
               'offset': int(_bounded(args, 'offset', 0, 0, 64000))}
    elif kind == 'mcp_call':
        arguments = args.get('arguments', {})
        if not isinstance(arguments, dict):
            raise ActionValidationError('MCP arguments must be a JSON object')
        try:
            encoded = json.dumps(arguments, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ActionValidationError('Invalid MCP JSON arguments') from exc
        if len(encoded) > 16000:
            raise ActionValidationError('MCP arguments exceed 16000 characters')
        out = {'server': _text(args, 'server', max_len=64), 'tool': _text(args, 'tool', max_len=256),
               'arguments': json.loads(encoded)}
    else:
        raise ActionValidationError(f'Unhandled action: {kind}')
    return {'type': kind, 'args': out}
