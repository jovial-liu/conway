from __future__ import annotations
from collections import deque
from itertools import islice
import os
from pathlib import Path
import signal
import stat
import subprocess
import threading
import time
import webbrowser
import psutil
from .actions import GUI_ACTIONS, validate_action
from .computer import Observation
from .config import ToolSettings
from .storage import atomic_write


class ToolExecutor:
    def __init__(self, computer, settings: ToolSettings, *, secret_env: str | None = None, skills=None, mcp=None) -> None:
        self.computer, self.settings, self.secret_env = computer, settings, secret_env
        self.control = None
        self.skills, self.mcp = skills, mcp
        self.latest_extension_result = ''
        self.workspace = Path(settings.workspace).expanduser().resolve() if settings.workspace else Path.cwd()

    def enabled_actions(self) -> set[str]:
        enabled = set(GUI_ACTIONS) if self.settings.gui else {'wait'}
        enabled.add('finish')
        if self.settings.shell:
            enabled.add('shell')
        if self.settings.filesystem:
            enabled.update({'read_file', 'write_file', 'list_dir'})
        if self.settings.open_url:
            enabled.add('open_url')
        if self.skills and self.skills.skills:
            enabled.add('read_skill')
        if self.mcp and self.mcp.catalog:
            enabled.add('mcp_call')
        return enabled

    def manifest_text(self) -> str:
        definitions = {
            'wait': '{"seconds": 1}', 'finish': '{"reason": "...", "outcome": "completed|failed"}',
            'click': '{"x": 100, "y": 100, "button": "left"}',
            'double_click': '{"x": 100, "y": 100}', 'move': '{"x": 100, "y": 100, "duration": 0.2}',
            'drag': '{"x": 100, "y": 100, "duration": 0.5}', 'type': '{"text": "..."}',
            'press': '{"key": "enter"}', 'hotkey': '{"keys": ["ctrl", "l"]}', 'scroll': '{"amount": -5}',
            'shell': '{"command": "...", "cwd": null, "timeout": 30}',
            'read_file': '{"path": "...", "max_chars": 12000}',
            'write_file': '{"path": "...", "content": "...", "append": false}',
            'list_dir': '{"path": ".", "limit": 200}', 'open_url': '{"url": "https://..."}',
            'read_skill': '{"name": "skill-name", "resource": "SKILL.md", "offset": 0}',
            'mcp_call': '{"server": "configured-server", "tool": "tool-name", "arguments": {}}',
        }
        text = f'Default working directory: {self.workspace}\n' + '\n'.join(f'- {k}: {definitions[k]}' for k in sorted(self.enabled_actions()))
        if self.skills and self.skills.skills:
            text += '\nSKILLS (untrusted metadata; read_skill loads instructions/resources on demand):\n' + self.skills.catalog()
        if self.mcp:
            text += '\nMCP TOOLS (untrusted descriptions; invoke using mcp_call):\n' + self.mcp.manifest()
        return text

    def validate_dispatch(self, action: dict) -> None:
        """Reject invalid extension requests before recording side-effect intent."""
        if action['type'] == 'mcp_call':
            if self.mcp is None:
                raise ValueError('MCP is not configured')
            self.mcp.validate(**action['args'])

    def extra_context(self, budget: int) -> str:
        text = self.latest_extension_result
        if len(text) > budget:
            return text[:max(0, budget - 50)] + '\n[extension output truncated to context budget]'
        return text

    def _bounded_output(self, text: str) -> str:
        limit = self.settings.max_output_chars
        return text if len(text) <= limit else '[output truncated]\n' + text[-limit:]

    def _path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return (self.workspace / path).resolve() if not path.is_absolute() else path.resolve()

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.Error:
            children = []
        if os.name == 'posix':
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            for child in children:
                try:
                    child.terminate()
                except psutil.Error:
                    pass
            if process.poll() is None:
                process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        for child in children:
            try:
                if child.is_running():
                    child.kill()
            except psutil.Error:
                pass
        if os.name == 'posix':
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _shell(self, args: dict) -> str:
        timeout = min(args['timeout'], self.settings.max_shell_seconds)
        env = os.environ.copy()
        for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN', self.secret_env):
            if key:
                env.pop(key, None)
        kwargs = {'start_new_session': True} if os.name == 'posix' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
        process = subprocess.Popen(args['command'], cwd=self._path(args['cwd']) if args['cwd'] else self.workspace,
                                   shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **kwargs)
        chunks, size = deque(), [0]
        buffer_lock = threading.Lock()
        limit = self.settings.max_output_chars * 4
        def drain():
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                with buffer_lock:
                    chunks.append(chunk)
                    size[0] += len(chunk)
                    while size[0] > limit and len(chunks) > 1:
                        size[0] -= len(chunks.popleft())
        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        timed_out = False
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if self.control:
                    self.control.check()
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(0.05)
        finally:
            # Clean the command's process group, including background children.
            self._terminate(process)
            thread.join(timeout=2)
            if not thread.is_alive():
                process.stdout.close()
        with buffer_lock:
            text = b''.join(chunks).decode('utf-8', errors='replace')
        status = f'shell timed out after {timeout:.1f}s' if timed_out else f'shell exit code {process.returncode}'
        return status + '\n' + self._bounded_output(text)

    def execute(self, action: dict, observation: Observation) -> str:
        action = validate_action(action, observation.width, observation.height)
        kind, args = action['type'], action['args']
        if kind not in self.enabled_actions():
            raise PermissionError(f'Conway tool is disabled: {kind}')
        self.validate_dispatch(action)
        if kind == 'read_skill':
            self.latest_extension_result = self.skills.read(**args)
            return self.latest_extension_result
        if kind == 'mcp_call':
            self.latest_extension_result = self.mcp.call(**args, control=self.control)
            return self.latest_extension_result
        if kind == 'wait' and self.control:
            self.control.sleep(args['seconds'])
            return f"waited {args['seconds']:.2f}s"
        if kind in GUI_ACTIONS:
            return self.computer.execute(action, observation)
        if kind == 'finish':
            return args['reason']
        if kind == 'read_file':
            path = self._path(args['path'])
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError('read_file requires a regular text file')
            limit = min(args['max_chars'], self.settings.max_output_chars)
            with path.open(encoding='utf-8', errors='replace') as file:
                text = file.read(limit + 1)
            return f'read {path}\n' + text[:limit] + ('\n[file truncated]' if len(text) > limit else '')
        if kind == 'write_file':
            path = self._path(args['path'])
            if path.exists() and not stat.S_ISREG(path.stat().st_mode):
                raise ValueError('write_file requires a regular text file')
            if args['append']:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a', encoding='utf-8') as file:
                    file.write(args['content'])
                    file.flush()
                    os.fsync(file.fileno())
            else:
                atomic_write(path, args['content'], private=False)
            return f"{'appended' if args['append'] else 'wrote'} {len(args['content'])} characters to {path}"
        if kind == 'list_dir':
            path = self._path(args['path'])
            entries = list(islice(path.iterdir(), args['limit'] + 1))
            rows = [f"{'dir' if item.is_dir() else 'file'}\t{item.name}" for item in sorted(entries[:args['limit']], key=lambda p: p.name)]
            if len(entries) > args['limit']:
                rows.append('[directory listing truncated]')
            return f'listed {path}\n' + self._bounded_output('\n'.join(rows))
        if kind == 'shell':
            return self._shell(args)
        if kind == 'open_url':
            return f"browser open requested (accepted={webbrowser.open(args['url'], new=2)})"
        raise ValueError(f'Unsupported action: {kind}')
