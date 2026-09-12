from pathlib import Path

from conway.memory import FileMemory


def test_file_memory_keeps_notes_and_recent_events(tmp_path: Path):
    memory = FileMemory(tmp_path)
    memory.append_memory("durable fact")
    memory.journal({"cycle": 1, "result": "ok"})
    context = memory.recall()
    assert "durable fact" in context
    assert '"cycle": 1' in context
