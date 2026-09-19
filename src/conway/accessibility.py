"""Optional, read-only native UI snapshots. Native calls run in a disposable process.

Labels are untrusted observations, never instructions. This layer does not request
permissions or retrieve text values/passwords. Screenshot perception is fallback.
"""
from __future__ import annotations
from collections import deque
import json
import platform
import subprocess
import sys
from typing import Any


def _label(value: Any) -> str:
    return str(value or '').replace('\x00', '')[:180]


def _windows(limit: int) -> list[dict]:
    import ctypes
    from ctypes import wintypes
    from pywinauto import Desktop
    user = ctypes.windll.user32
    user.GetForegroundWindow.restype = wintypes.HWND
    root = Desktop(backend='uia').window(handle=user.GetForegroundWindow()).wrapper_object()
    queue, result = deque([root]), []
    while queue and len(result) < limit:
        node = queue.popleft()
        info = node.element_info
        rect = info.rectangle
        result.append({'role': _label(info.control_type), 'name': _label(info.name),
                       'bounds': [rect.left, rect.top, rect.right, rect.bottom]})
        # children(), not descendants(), so traversal is explicitly bounded.
        queue.extend(node.children()[:max(0, limit - len(result) - len(queue))])
    return result


def _macos(limit: int) -> list[dict]:
    import ApplicationServices as AX
    from AppKit import NSWorkspace
    pid = NSWorkspace.sharedWorkspace().frontmostApplication().processIdentifier()
    root = AX.AXUIElementCreateApplication(pid)
    def attr(node, key):
        error, value = AX.AXUIElementCopyAttributeValue(node, key, None)
        return value if error == 0 else None
    root = attr(root, 'AXFocusedWindow') or root
    queue, result = deque([root]), []
    while queue and len(result) < limit:
        node = queue.popleft()
        role, subrole = attr(node, 'AXRole'), attr(node, 'AXSubrole')
        secure = 'secure' in str(subrole).lower()
        name = '[secure field]' if secure else attr(node, 'AXTitle') or attr(node, 'AXDescription')
        result.append({'role': _label(role), 'name': _label(name)})
        if not secure:
            children = attr(node, 'AXChildren') or []
            queue.extend(children[:max(0, limit - len(result) - len(queue))])
    return result


def _linux(limit: int) -> list[dict]:
    import pyatspi
    desktop = pyatspi.Registry.getDesktop(0)
    active = None
    for app in list(desktop)[:100]:
        for window in list(app)[:100]:
            if window.getState().contains(pyatspi.STATE_ACTIVE):
                active = window
                break
        if active is not None:
            break
    if active is None:
        return []
    queue, result = deque([active]), []
    while queue and len(result) < limit:
        node = queue.popleft()
        role = node.getRoleName()
        secure = 'password' in role.lower()
        result.append({'role': _label(role), 'name': '[secure field]' if secure else _label(node.name)})
        if not secure:
            remaining = max(0, limit - len(result) - len(queue))
            queue.extend(node[i] for i in range(min(node.childCount, remaining)))
    return result


def capture_ui_tree(*, timeout: float = 2.0, max_nodes: int = 80) -> dict:
    try:
        process = subprocess.run([sys.executable, '-m', 'conway.accessibility', str(max_nodes)],
                                 capture_output=True, text=True, timeout=timeout, check=False)
        if process.returncode != 0 or len(process.stdout) > 300000:
            return {'status': 'unavailable', 'nodes': []}
        data = json.loads(process.stdout)
        if not isinstance(data, dict) or not isinstance(data.get('nodes'), list):
            raise ValueError('invalid UI snapshot')
        return data
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'nodes': []}
    except (OSError, ValueError):
        return {'status': 'unavailable', 'nodes': []}


def _worker() -> None:
    try:
        limit = min(500, max(1, int(sys.argv[1])))
        provider = {'Darwin': _macos, 'Windows': _windows, 'Linux': _linux}[platform.system()]
        nodes = provider(limit)
        data = {'status': 'ok', 'provider': platform.system(), 'coordinate_space': 'native', 'nodes': nodes}
    except Exception as exc:
        data = {'status': 'unavailable', 'detail': type(exc).__name__, 'nodes': []}
    print(json.dumps(data, ensure_ascii=True))


if __name__ == '__main__':
    _worker()
