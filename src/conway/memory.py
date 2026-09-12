from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from importlib import resources

from platformdirs import user_data_dir


@dataclass(slots=True)
class RuntimeState:
    cycle: int = 0
    status: str = "idle"
    last_action: str | None = None
    last_result: str | None = None
    last_rationale: str | None = None
    profile: str | None = None
    model: str | None = None
    error_count: int = 0
    compactions: int = 0
    started_at: str | None = None
    updated_at: str | None = None


class FileMemory:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(user_data_dir("conway", appauthor=False))
        self.journal_dir = self.root / "journal"
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_dir.mkdir(parents=True, exist_ok=True)

        self.constitution_path = self.root / "constitution.md"
        self.memory_path = self.root / "memory.md"
        self.state_path = self.root / "state.json"

        if not self.constitution_path.exists():
            default = resources.files("conway").joinpath("constitution.default.md").read_text(encoding="utf-8")
            self.constitution_path.write_text(default, encoding="utf-8")
        if not self.memory_path.exists():
            self.memory_path.write_text("# Conway Memory\n\n", encoding="utf-8")
        if not self.state_path.exists():
            self.save_state(RuntimeState())

    def constitution(self) -> str:
        return self.constitution_path.read_text(encoding="utf-8")

    def recent_events(self, limit: int = 12, max_chars: int = 8000) -> str:
        lines: deque[str] = deque(maxlen=max(1, limit))
        for path in sorted(self.journal_dir.glob("*.jsonl"), reverse=True)[:3]:
            try:
                file_lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(file_lines):
                lines.appendleft(line)
                if len(lines) >= limit:
                    break
            if len(lines) >= limit:
                break
        text = "\n".join(lines)
        return text[-max_chars:]

    def recall(self, max_chars: int = 16000) -> str:
        durable = self.memory_path.read_text(encoding="utf-8")
        recent = self.recent_events()
        text = f"{durable}\n\n# Recent trajectory\n{recent}" if recent else durable
        if len(text) <= max_chars:
            return text
        return "# Conway Context (recent excerpt)\n\n" + text[-max_chars:]

    def compaction_source(self, max_chars: int = 48000) -> str:
        durable = self.memory_path.read_text(encoding="utf-8")
        recent = self.recent_events(limit=30, max_chars=16000)
        text = f"{durable}\n\n# Recent trajectory\n{recent}" if recent else durable
        return text[-max_chars:]

    def append_memory(self, note: str | None) -> None:
        if not note or not note.strip():
            return
        clean = " ".join(note.strip().split())
        tail = self.memory_path.read_text(encoding="utf-8")[-8000:]
        if clean in tail:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(f"\n- [{timestamp}] {clean}\n")

    def memory_bytes(self) -> int:
        try:
            return self.memory_path.stat().st_size
        except OSError:
            return 0

    def needs_compaction(self, max_bytes: int = 64_000) -> bool:
        return self.memory_bytes() > max_bytes

    def replace_memory(self, summary: str) -> None:
        clean = summary.strip()
        if not clean:
            return
        if not clean.lstrip().startswith("#"):
            clean = "# Conway Memory\n\n" + clean
        self.memory_path.write_text(clean.rstrip() + "\n", encoding="utf-8")

    def compact_if_needed(self, max_bytes: int = 256_000, keep_lines: int = 500) -> None:
        """Last-resort bounded compaction when model-driven summarization is unavailable."""
        try:
            if self.memory_path.stat().st_size <= max_bytes:
                return
            lines = self.memory_path.read_text(encoding="utf-8").splitlines()
            kept = lines[-keep_lines:]
            self.memory_path.write_text(
                "# Conway Memory (fallback compacted)\n\n" + "\n".join(kept) + "\n",
                encoding="utf-8",
            )
        except OSError:
            return

    def load_state(self) -> RuntimeState:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return RuntimeState()
            allowed = RuntimeState.__dataclass_fields__.keys()
            filtered = {key: value for key, value in data.items() if key in allowed}
            return RuntimeState(**filtered)
        except (OSError, json.JSONDecodeError, TypeError):
            return RuntimeState()

    def save_state(self, state: RuntimeState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self.state_path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")

    def journal(self, event: dict) -> None:
        now = datetime.now(timezone.utc)
        path = self.journal_dir / f"{now.date().isoformat()}.jsonl"
        payload = {"time": now.isoformat(), **event}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
