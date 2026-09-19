"""Literal OpenCUA translation, never Python execution.

Geometry follows the OpenCUA-7B model card. Processor pixel limits may be
explicitly overridden to match a differently configured inference server.
"""
from __future__ import annotations
import ast
from math import ceil, floor, sqrt, isfinite
import re
from typing import Any
from .actions import validate_action


class OpenCUAParseError(ValueError):
    pass


def smart_resize(height: int, width: int, *, factor: int = 28, min_pixels: int = 3136,
                 max_pixels: int = 12845056) -> tuple[int, int]:
    if any(type(v) is not int or v <= 0 for v in (height, width, factor, min_pixels, max_pixels)):
        raise ValueError('resize parameters must be positive integers')
    if min_pixels > max_pixels or max_pixels < factor * factor:
        raise ValueError('invalid pixel limits')
    if max(height, width) / min(height, width) > 200:
        raise ValueError('image aspect ratio exceeds 200')
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = sqrt(height * width / max_pixels)
        h, w = max(factor, floor(height / beta / factor) * factor), max(factor, floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = sqrt(min_pixels / (height * width))
        h, w = ceil(height * beta / factor) * factor, ceil(width * beta / factor) * factor
    return h, w


def model_point_to_screenshot(model_x: float, model_y: float, original_width: int,
                              original_height: int, **resize) -> tuple[int, int]:
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not isfinite(v) for v in (model_x, model_y)):
        raise OpenCUAParseError('coordinates must be finite numbers')
    h, w = smart_resize(original_height, original_width, **resize)
    # Model-card mapping uses truncation, then the normal input adapter handles HiDPI.
    return max(0, min(int(model_x / w * original_width), original_width - 1)), max(0, min(int(model_y / h * original_height), original_height - 1))


def _source(text: str) -> str:
    if not isinstance(text, str) or len(text) > 32768:
        raise OpenCUAParseError('invalid or oversized model output')
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    if '```' in text:
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.S)
        if len(blocks) != 1:
            raise OpenCUAParseError('expected exactly one Python code block')
        outside = re.sub(r'```(?:python)?\s*\n.*?```', '', text, flags=re.S)
        if re.search(r'(pyautogui\.|time\.|\bexec\(|\beval\()', outside):
            raise OpenCUAParseError('additional calls outside code block')
        return blocks[0].strip()
    return text


def parse_opencua_action(text: str, width: int, height: int, *, min_pixels: int = 3136,
                         max_pixels: int = 12845056) -> dict[str, Any]:
    source = _source(text)
    if source.upper() in {'DONE', 'FAIL'}:
        return {'type': 'finish', 'args': {'reason': source.upper(), 'outcome': 'failed' if source.upper() == 'FAIL' else 'completed'}}
    if source.upper() == 'WAIT':
        return {'type': 'wait', 'args': {'seconds': 1.0}}
    try:
        module = ast.parse(source, mode='exec')
        if len(module.body) != 1 or not isinstance(module.body[0], ast.Expr) or not isinstance(module.body[0].value, ast.Call):
            raise OpenCUAParseError('exactly one literal call is required')
        if sum(1 for _ in ast.walk(module)) > 150:
            raise OpenCUAParseError('action is too complex')
        call = module.body[0].value
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            owner, name = call.func.value.id, call.func.attr
            if owner not in {'pyautogui', 'time'} or (owner == 'time' and name != 'sleep'):
                raise OpenCUAParseError('unsupported call target')
        elif isinstance(call.func, ast.Name):
            name = call.func.id
        else:
            raise OpenCUAParseError('unsupported call target')
        pos = [ast.literal_eval(v) for v in call.args]
        kw = {}
        for v in call.keywords:
            if v.arg is None or v.arg in kw:
                raise OpenCUAParseError('expanded or duplicate keywords are unsupported')
            kw[v.arg] = ast.literal_eval(v.value)
        def bind(names: tuple[str, ...]) -> dict:
            if len(pos) > len(names) or set(kw) - set(names):
                raise OpenCUAParseError('unsupported arguments')
            out = dict(zip(names, pos))
            if set(out) & set(kw):
                raise OpenCUAParseError('argument specified twice')
            return {**out, **kw}
        if name in {'click', 'doubleClick', 'moveTo', 'dragTo'}:
            names = {'moveTo': ('x', 'y', 'duration'), 'dragTo': ('x', 'y', 'duration', 'button'),
                     'click': ('x', 'y', 'clicks', 'interval', 'button'), 'doubleClick': ('x', 'y', 'interval', 'button')}[name]
            args = bind(names)
            x, y = model_point_to_screenshot(args['x'], args['y'], width, height,
                                             min_pixels=min_pixels, max_pixels=max_pixels)
            kind = {'click': 'click', 'doubleClick': 'double_click', 'moveTo': 'move', 'dragTo': 'drag'}[name]
            if 'clicks' in args:
                if type(args['clicks']) is not int or args['clicks'] not in {1, 2}:
                    raise OpenCUAParseError('only single/double clicks are supported')
                if args['clicks'] == 2:
                    kind = 'double_click'
            if kind == 'click' and 'interval' in args and args['interval'] != 0:
                raise OpenCUAParseError('custom click interval unsupported; use doubleClick')
            out = {'x': x, 'y': y}
            if kind == 'double_click':
                out['interval'] = args.get('interval', 0.12)
            if kind != 'move':
                out['button'] = args.get('button', 'left')
            if kind in {'move', 'drag'}:
                out['duration'] = args.get('duration', 0.2 if kind == 'move' else 0.5)
        elif name in {'write', 'typewrite'}:
            args = bind(('message', 'interval'))
            kind, out = 'type', {'text': args['message'], 'interval': args.get('interval', 0.01)}
        elif name == 'press':
            args = bind(('keys',))
            kind, out = 'press', {'key': args['keys']}
        elif name == 'hotkey':
            if kw:
                raise OpenCUAParseError('hotkey keyword arguments unsupported')
            kind, out = 'hotkey', {'keys': pos}
        elif name == 'scroll':
            args = bind(('clicks',))
            kind, out = 'scroll', {'amount': args['clicks']}
        elif name == 'sleep':
            args = bind(('seconds',))
            kind, out = 'wait', {'seconds': args['seconds']}
        else:
            raise OpenCUAParseError(f'unsupported call: {name}')
        return validate_action({'type': kind, 'args': out}, width, height)
    except (ValueError, TypeError, KeyError, SyntaxError, RecursionError) as exc:
        if isinstance(exc, OpenCUAParseError):
            raise
        raise OpenCUAParseError(f'Invalid literal action: {exc}') from exc
