from pathlib import Path

from conway.computer import Observation
from conway.config import ToolSettings
from conway.tools import ToolExecutor


class DummyComputer:
    def execute(self, action, observation):
        return "gui"


def observation(tmp_path: Path) -> Observation:
    return Observation(
        screenshot_path=tmp_path / "screen.png",
        width=100,
        height=100,
        input_width=100,
        input_height=100,
    )


def test_file_tools_round_trip(tmp_path: Path):
    executor = ToolExecutor(DummyComputer(), ToolSettings())  # type: ignore[arg-type]
    target = tmp_path / "nested" / "note.txt"
    result = executor.execute(
        {"type": "write_file", "args": {"path": str(target), "content": "hello", "append": False}},
        observation(tmp_path),
    )
    assert "wrote 5 characters" in result

    read = executor.execute(
        {"type": "read_file", "args": {"path": str(target)}},
        observation(tmp_path),
    )
    assert "hello" in read

    listing = executor.execute(
        {"type": "list_dir", "args": {"path": str(target.parent)}},
        observation(tmp_path),
    )
    assert "note.txt" in listing


def test_disabled_shell_is_not_enabled(tmp_path: Path):
    settings = ToolSettings(shell=False)
    executor = ToolExecutor(DummyComputer(), settings)  # type: ignore[arg-type]
    assert "shell" not in executor.enabled_actions()
