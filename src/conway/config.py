from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ToolSettings:
    gui: bool = True
    shell: bool = True
    filesystem: bool = True
    open_url: bool = True
    max_shell_seconds: float = 45.0
    max_output_chars: int = 12000


@dataclass(slots=True)
class ConwayConfig:
    profile: str = "auto"
    brain: str = "auto"
    endpoint: str | None = None
    model: str | None = None
    port: int = 8042
    interval: float = 0.5
    screenshot_keep: int = 30
    memory_compaction_bytes: int = 64000
    tools: ToolSettings = field(default_factory=ToolSettings)


def _merge_dataclass(config: ConwayConfig, raw: dict[str, Any]) -> ConwayConfig:
    if "profile" in raw:
        config.profile = str(raw["profile"])
    if "brain" in raw:
        config.brain = str(raw["brain"])
    if "endpoint" in raw:
        config.endpoint = str(raw["endpoint"]) if raw["endpoint"] else None
    if "model" in raw:
        config.model = str(raw["model"]) if raw["model"] else None
    if "port" in raw:
        config.port = max(1, min(int(raw["port"]), 65535))
    if "interval" in raw:
        config.interval = max(0.0, min(float(raw["interval"]), 60.0))
    if "screenshot_keep" in raw:
        config.screenshot_keep = max(1, min(int(raw["screenshot_keep"]), 500))
    if "memory_compaction_bytes" in raw:
        config.memory_compaction_bytes = max(8000, min(int(raw["memory_compaction_bytes"]), 2_000_000))

    tools = raw.get("tools")
    if isinstance(tools, dict):
        for key in ("gui", "shell", "filesystem", "open_url"):
            if key in tools:
                setattr(config.tools, key, bool(tools[key]))
        if "max_shell_seconds" in tools:
            config.tools.max_shell_seconds = max(1.0, min(float(tools["max_shell_seconds"]), 300.0))
        if "max_output_chars" in tools:
            config.tools.max_output_chars = max(1000, min(int(tools["max_output_chars"]), 200_000))
    return config


def config_path(root: Path) -> Path:
    return root / "config.yaml"


def ensure_config(root: Path) -> Path:
    path = config_path(root)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(asdict(ConwayConfig()), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return path


def load_config(root: Path) -> ConwayConfig:
    path = ensure_config(root)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Could not read Conway config at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"Conway config must contain a YAML mapping: {path}")
    return _merge_dataclass(ConwayConfig(), raw)
