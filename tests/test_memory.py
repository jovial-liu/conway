from pathlib import Path

from conway.memory import FileMemory


def test_file_memory_keeps_notes_and_recent_events(tmp_path: Path):
    memory = FileMemory(tmp_path)
    memory.append_memory("durable fact")
    memory.journal({"cycle": 1, "result": "ok"})
    context = memory.recall()
    assert "durable fact" in context
    assert '"cycle": 1' in context


def test_memory_can_be_replaced_by_rolling_summary(tmp_path: Path):
    memory = FileMemory(tmp_path)
    memory.memory_path.write_text("x" * 200, encoding="utf-8")
    assert memory.needs_compaction(max_bytes=100)
    memory.replace_memory("## Active work\nContinue task A")
    text = memory.memory_path.read_text(encoding="utf-8")
    assert "Continue task A" in text
    assert not memory.needs_compaction(max_bytes=100)
