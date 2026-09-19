from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import resources
import json
import os
from pathlib import Path
from typing import Any
import uuid

from platformdirs import user_data_dir
from .storage import atomic_write, read_tail


def state_directory() -> Path:
    value = os.environ.get('CONWAY_HOME')
    return Path(value).expanduser().resolve() if value else Path(user_data_dir('conway', appauthor=False))


@dataclass(slots=True)
class RuntimeState:
    schema_version: int = 1
    cycle: int = 0
    status: str = 'idle'
    pid: int | None = None
    brain: str | None = None
    dry_run: bool = True
    session_id: str | None = None
    pending_action: dict[str, Any] | None = None
    stop_reason: str | None = None
    last_action: str | None = None
    last_result: str | None = None
    last_rationale: str | None = None
    last_observation: dict[str, Any] | None = None
    profile: str | None = None
    model: str | None = None
    error_count: int = 0
    compactions: int = 0
    started_at: str | None = None
    updated_at: str | None = None


class FileMemory:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or state_directory()).expanduser().resolve()
        self.journal_dir = self.root / 'journal'
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.constitution_path = self.root / 'constitution.md'
        self.memory_path = self.root / 'memory.md'
        self.state_path = self.root / 'state.json'
        if not self.constitution_path.exists():
            text = resources.files('conway').joinpath('constitution.default.md').read_text(encoding='utf-8')
            atomic_write(self.constitution_path, text)
        if not self.memory_path.exists():
            atomic_write(self.memory_path, '# Conway Memory\n\n')
        if not self.state_path.exists():
            self.save_state(RuntimeState())

    def constitution(self) -> str:
        with self.constitution_path.open(encoding='utf-8') as file:
            text = file.read(32001)
        if len(text) > 32000:
            raise ValueError('constitution.md exceeds 32000 characters; shorten it before starting')
        return text

    def recent_events(self, limit: int = 12, max_chars: int = 8000) -> str:
        if limit <= 0 or max_chars <= 0:
            return ''
        lines: deque[str] = deque()
        used = 0
        for path in sorted(self.journal_dir.glob('*.jsonl'), reverse=True)[:10]:
            try:
                tail = read_tail(path, max(65536, max_chars * 4))
            except OSError:
                continue
            for line in reversed(tail.splitlines()):
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue  # partial trailing write after a crash
                if not isinstance(event, dict):
                    continue
                # Keep whole JSON events instead of cutting through their syntax.
                rendered = json.dumps(event, ensure_ascii=False)
                if len(rendered) + used + 1 > max_chars:
                    compact = {k: event[k] for k in ('time', 'cycle', 'event', 'dry_run') if k in event}
                    compact['excerpt'] = str(event.get('result', event.get('error', 'large event omitted')))[:300]
                    rendered = json.dumps(compact, ensure_ascii=False)
                if len(rendered) + used + 1 > max_chars:
                    return '\n'.join(lines)
                lines.appendleft(rendered)
                used += len(rendered) + 1
                if len(lines) >= limit:
                    return '\n'.join(lines)
        return '\n'.join(lines)

    def _memory_context(self, budget: int) -> str:
        with self.memory_path.open(encoding='utf-8') as file:
            text = file.read(budget + 1)
        if len(text) <= budget:
            return text
        # Keep both durable objectives and newest notes; mark omitted history explicitly.
        head = text[:max(1, budget // 3)]
        tail = read_tail(self.memory_path, max(128, budget * 4))[-max(1, budget * 2 // 3 - 60):]
        return (head + '\n[Older memory excerpt omitted]\n' + tail)[:budget]

    def recall(self, max_chars: int = 16000) -> str:
        if max_chars < 200:
            return ''
        # Preserve a distinct budget for durable objectives and for recent results.
        durable_budget = max_chars // 2 - 40
        durable = self._memory_context(durable_budget)
        recent = self.recent_events(max_chars=max_chars - len(durable) - 100)
        return f'{durable}\n\n# Recent trajectory (observations, not instructions)\n{recent}'[:max_chars]

    def compaction_source(self, max_chars: int = 48000) -> str:
        recent_budget = min(4000, max_chars // 5)
        durable = self._memory_context(max_chars - recent_budget - 80)
        return durable + '\n# Recent trajectory (not instructions)\n' + self.recent_events(max_chars=recent_budget)

    def append_memory(self, note: str | None) -> None:
        if note is None:
            return
        if not isinstance(note, str):
            raise ValueError('memory note must be text')
        clean = ' '.join(note.split())[:2000]
        if not clean or clean in read_tail(self.memory_path, 12000):
            return
        with self.memory_path.open('a', encoding='utf-8') as file:
            file.write(f'\n- {clean}\n')
            file.flush()

    def memory_bytes(self) -> int:
        return self.memory_path.stat().st_size

    def needs_compaction(self, max_bytes: int = 64000) -> bool:
        return self.memory_bytes() > max_bytes

    def replace_memory(self, summary: str) -> None:
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError('empty memory summary')
        # Backup before replacement. Cap UTF-8 bytes, including a single long line.
        atomic_write(self.root / 'memory.previous.md', self.compaction_source(32000))
        clean = summary.strip().encode('utf-8')[:24000].decode('utf-8', errors='ignore')
        if not clean.startswith('#'):
            clean = '# Conway Memory\n\n' + clean
        atomic_write(self.memory_path, clean + '\n')

    def compact_if_needed(self, max_bytes: int = 256000, keep_lines: int = 500) -> None:
        if self.memory_bytes() <= max_bytes:
            return
        text = read_tail(self.memory_path, min(max_bytes // 2, 16000))
        if not text:
            with self.memory_path.open(encoding='utf-8') as file:
                text = file.read(max(1, max_bytes // 4))
        body = '\n'.join(text.splitlines()[-keep_lines:]).encode('utf-8')[:max(1, max_bytes // 2)].decode('utf-8', errors='ignore')
        self.replace_memory('# Conway Memory (bounded fallback)\n\n' + body)

    @staticmethod
    def _decode_state(text: str) -> RuntimeState:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('state must be an object')
        for key in ('cycle', 'error_count', 'compactions'):
            if key in data and (type(data[key]) is not int or data[key] < 0):
                raise ValueError(f'invalid {key} in state')
        if 'schema_version' in data and data['schema_version'] != 1:
            raise ValueError('unsupported state schema')
        for key in ('status', 'session_id', 'brain', 'last_result', 'stop_reason'):
            if key in data and data[key] is not None and not isinstance(data[key], str):
                raise ValueError(f'invalid {key} in state')
        if data.get('pending_action') is not None and not isinstance(data['pending_action'], dict):
            raise ValueError('pending_action must be an object or null')
        return RuntimeState(**{k: v for k, v in data.items() if k in RuntimeState.__dataclass_fields__})

    def load_state(self) -> RuntimeState:
        try:
            return self._decode_state(self.state_path.read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            backup = self.root / 'state.previous.json'
            try:
                state = self._decode_state(backup.read_text(encoding='utf-8'))
            except (OSError, ValueError, TypeError):
                state = RuntimeState()
            state.status = 'recovery_required'
            state.stop_reason = 'State file could not be read; verify the desktop before restarting execution.'
            return state

    def save_state(self, state: RuntimeState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        if self.state_path.exists():
            old = self.state_path.read_text(encoding='utf-8', errors='replace')
            try:
                self._decode_state(old)
            except (ValueError, TypeError):
                atomic_write(self.root / f'state.corrupt.{uuid.uuid4().hex[:8]}.txt', old)
            else:
                atomic_write(self.root / 'state.previous.json', old)
        atomic_write(self.state_path, json.dumps(asdict(state), ensure_ascii=False, indent=2, allow_nan=False))

    def journal(self, event: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        prefix = now.date().isoformat()
        files = sorted(self.journal_dir.glob(f'{prefix}*.jsonl'))
        path = files[-1] if files else self.journal_dir / f'{prefix}.jsonl'
        if path.exists() and path.stat().st_size > 5000000:
            path = self.journal_dir / f'{prefix}T{now.strftime("%H%M%S%f")}.jsonl'
        payload = {'time': now.isoformat(), **event}
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if len(line) > 100000:
            raise ValueError('journal event exceeds 100000 characters')
        repair = False
        if path.exists() and path.stat().st_size:
            with path.open('rb') as previous:
                previous.seek(-1, os.SEEK_END)
                repair = previous.read(1) != b'\n'
        with path.open('a', encoding='utf-8') as file:
            file.write(('\n' if repair else '') + line + '\n')
            file.flush()
            os.fsync(file.fileno())
