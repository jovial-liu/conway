from __future__ import annotations

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

    def recall(self, max_chars: int = 12000) -> str:
        text = self.memory_path.read_text(encoding="utf-8")
        if len(text) <= max_chars:
            return text
        return "# Conway Memory (recent excerpt)\n\n" + text[-max_chars:]

    def append_memory(self, note: str | None) -> None:
        if not note or not note.strip():
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(f"\n- [{timestamp}] {note.strip()}\n")

    def load_state(self) -> RuntimeState:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return RuntimeState(**data)
        except (OSError, json.JSONDecodeError, TypeError):
            return RuntimeState()

    def save_state(self, state: RuntimeState) -> None:
        self.state_path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")

    def journal(self, event: dict) -> None:
        now = datetime.now(timezone.utc)
        path = self.journal_dir / f"{now.date().isoformat()}.jsonl"
        payload = {"time": now.isoformat(), **event}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
