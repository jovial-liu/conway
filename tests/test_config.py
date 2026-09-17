from pathlib import Path

from conway.config import ensure_config, load_config


def test_default_config_is_created(tmp_path: Path):
    path = ensure_config(tmp_path)
    assert path.exists()
    config = load_config(tmp_path)
    assert config.profile == "auto"
    assert config.tools.gui is True
    assert config.tools.filesystem is True


def test_config_overrides_are_loaded(tmp_path: Path):
    path = ensure_config(tmp_path)
    path.write_text(
        "profile: tiny\ninterval: 1.25\ntools:\n  shell: false\n  max_output_chars: 4321\n",
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.profile == "tiny"
    assert config.interval == 1.25
    assert config.tools.shell is False
    assert config.tools.max_output_chars == 4321
