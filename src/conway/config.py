from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

import yaml
from .storage import atomic_write


class ConfigError(ValueError):
    """An invalid setting; never silently coerce a permission or ignore a typo."""


@dataclass(slots=True)
class ToolSettings:
    gui: bool = True
    shell: bool = True
    filesystem: bool = True
    open_url: bool = True
    max_shell_seconds: float = 45.0
    max_output_chars: int = 12000
    workspace: str | None = None


@dataclass(slots=True)
class MCPServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    env_keys: list[str] = field(default_factory=list)
    timeout: float = 30.0


@dataclass(slots=True)
class ConwayConfig:
    profile: str = 'auto'
    brain: str = 'auto'
    endpoint: str | None = None
    model: str | None = None
    port: int = 8042
    interval: float = 0.5
    screenshot_keep: int = 30
    memory_compaction_bytes: int = 64000
    max_errors: int = 5
    context_chars: int = 16000
    request_timeout: float = 180.0
    startup_timeout: float = 900.0
    context_size: int = 8192
    api_key_env: str | None = None
    runtime_path: str | None = None
    ui_tree: bool = False
    ui_timeout: float = 2.0
    ui_max_nodes: int = 80
    opencua_min_pixels: int = 3136
    opencua_max_pixels: int = 12845056
    idle_initial_seconds: float = 2.0
    idle_max_seconds: float = 60.0
    recovery_initial_seconds: float = 5.0
    recovery_max_seconds: float = 300.0
    repeat_action_limit: int = 3
    tools: ToolSettings = field(default_factory=ToolSettings)
    skill_dirs: list[str] = field(default_factory=list)
    mcp_servers: dict[str, MCPServerConfig] = field(default_factory=dict)


def endpoint_url(value: str) -> str:
    if not isinstance(value, str):
        raise ConfigError('endpoint must be an http(s) URL')
    p = urlsplit(value)
    try:
        _ = p.port
    except ValueError as exc:
        raise ConfigError('endpoint contains an invalid port') from exc
    if p.scheme not in {'http', 'https'} or not p.hostname or p.username or p.password or p.query or p.fragment:
        raise ConfigError('endpoint must be an http(s) URL without credentials, query or fragment')
    if any(c.isspace() for c in value):
        raise ConfigError('endpoint must not contain whitespace')
    base = value.rstrip('/')
    # Accept a server root as a convenience. Preserve custom API prefixes.
    return base + '/v1' if p.path in {'', '/'} else base


def _number(value: Any, name: str, low: float, high: float, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f'{name} must be a finite number')
    if not low <= value <= high or (integer and not float(value).is_integer()):
        raise ConfigError(f'{name} must be {"an integer " if integer else ""}between {low:g} and {high:g}')
    return int(value) if integer else float(value)


def _merge_dataclass(config: ConwayConfig, raw: dict[str, Any]) -> ConwayConfig:
    unknown = set(raw) - {f.name for f in fields(config)}
    if unknown:
        raise ConfigError(f'Unknown settings: {", ".join(sorted(map(str, unknown)))}')
    if 'skill_dirs' in raw:
        value = raw['skill_dirs']
        if not isinstance(value, list) or len(value) > 16 or not all(isinstance(p, str) and p.strip() and '\x00' not in p for p in value):
            raise ConfigError('skill_dirs must be a list of at most 16 directory paths')
        config.skill_dirs = value
    if 'mcp_servers' in raw:
        servers = raw['mcp_servers']
        if not isinstance(servers, dict) or len(servers) > 8:
            raise ConfigError('mcp_servers must be a mapping of at most 8 named stdio servers')
        parsed = {}
        for name, value in servers.items():
            if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', name) or not isinstance(value, dict):
                raise ConfigError('Invalid MCP server name or settings')
            if set(value) - {f.name for f in fields(MCPServerConfig)}:
                raise ConfigError(f'Unknown MCP settings for {name}')
            command = value.get('command')
            if not isinstance(command, str) or not command.strip() or '\x00' in command or len(command) > 4096:
                raise ConfigError('MCP command must be a non-empty executable path or name')
            server = MCPServerConfig(command)
            for key in ('args', 'allowed_tools', 'env_keys'):
                vals = value.get(key, [])
                if not isinstance(vals, list) or len(vals) > 64 or not all(isinstance(v, str) and '\x00' not in v and len(v) <= 4096 for v in vals):
                    raise ConfigError(f'MCP {key} must be a bounded list of strings')
                if key != 'args' and (len(set(vals)) != len(vals) or any(not v.strip() or v == '*' for v in vals)):
                    raise ConfigError(f'MCP {key} requires distinct, explicit names')
                setattr(server, key, vals)
            if not server.allowed_tools:
                raise ConfigError('Every MCP server needs at least one allowed_tools entry')
            if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k) for k in server.env_keys):
                raise ConfigError('MCP env_keys must contain environment variable names')
            server.timeout = _number(value.get('timeout', 30), 'MCP timeout', 1, 300)
            parsed[name] = server
        config.mcp_servers = parsed
    for name in ('profile', 'brain'):
        if name in raw:
            choices = {'auto', 'tiny', 'small', 'standard', 'computer-use'} if name == 'profile' else {'auto', 'generic', 'opencua'}
            if not isinstance(raw[name], str) or raw[name] not in choices:
                raise ConfigError(f'{name} must be one of {sorted(choices)}')
            setattr(config, name, raw[name])
    for name in ('endpoint', 'model', 'api_key_env', 'runtime_path'):
        if name in raw:
            val = raw[name]
            if val is not None and (not isinstance(val, str) or not val.strip() or len(val) > 4096):
                raise ConfigError(f'{name} must be a non-empty string or null')
            setattr(config, name, endpoint_url(val) if name == 'endpoint' and val else val)
    ranges = {
        'port': (1, 65535, True), 'interval': (0, 60, False),
        'screenshot_keep': (1, 500, True), 'memory_compaction_bytes': (8000, 2000000, True),
        'max_errors': (1, 100, True), 'context_chars': (2000, 100000, True),
        'request_timeout': (1, 1800, False), 'startup_timeout': (1, 7200, False),
        'context_size': (2048, 131072, True), 'ui_timeout': (0.1, 10, False),
        'ui_max_nodes': (1, 500, True), 'opencua_min_pixels': (784, 12845056, True),
        'opencua_max_pixels': (784, 12845056, True),
        'idle_initial_seconds': (0.1, 3600, False), 'idle_max_seconds': (0.1, 3600, False),
        'recovery_initial_seconds': (0.1, 3600, False), 'recovery_max_seconds': (0.1, 3600, False),
        'repeat_action_limit': (1, 50, True),
    }
    for name, (low, high, integer) in ranges.items():
        if name in raw:
            setattr(config, name, _number(raw[name], name, low, high, integer))
    if config.opencua_min_pixels > config.opencua_max_pixels:
        raise ConfigError('opencua_min_pixels must not exceed opencua_max_pixels')
    for prefix in ('idle', 'recovery'):
        if getattr(config, prefix + '_initial_seconds') > getattr(config, prefix + '_max_seconds'):
            raise ConfigError(f'{prefix}_initial_seconds must not exceed {prefix}_max_seconds')
    if 'ui_tree' in raw:
        if type(raw['ui_tree']) is not bool:
            raise ConfigError('ui_tree must be true or false, not a quoted string')
        config.ui_tree = raw['ui_tree']
    if 'tools' in raw:
        tools = raw['tools']
        if not isinstance(tools, dict):
            raise ConfigError('tools must be a mapping')
        unknown = set(tools) - {f.name for f in fields(config.tools)}
        if unknown:
            raise ConfigError(f'Unknown tool settings: {sorted(map(str, unknown))}')
        for key in ('gui', 'shell', 'filesystem', 'open_url'):
            if key in tools:
                if type(tools[key]) is not bool:
                    raise ConfigError(f'tools.{key} must be true or false, not a quoted string')
                setattr(config.tools, key, tools[key])
        if 'max_shell_seconds' in tools:
            config.tools.max_shell_seconds = _number(tools['max_shell_seconds'], 'max_shell_seconds', 1, 300)
        if 'max_output_chars' in tools:
            config.tools.max_output_chars = _number(tools['max_output_chars'], 'max_output_chars', 1000, 200000, True)
        if 'workspace' in tools:
            val = tools['workspace']
            if val is not None and (not isinstance(val, str) or not val.strip()):
                raise ConfigError('tools.workspace must be a path or null')
            config.tools.workspace = val
    return config


def config_path(root: Path) -> Path:
    return root / 'config.yaml'


def ensure_config(root: Path) -> Path:
    path = config_path(root)
    if not path.exists():
        atomic_write(path, yaml.safe_dump(asdict(ConwayConfig()), sort_keys=False, allow_unicode=True))
    return path


class StrictLoader(yaml.SafeLoader):
    """Reject duplicate config keys instead of silently accepting the last value."""


def _unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError('Config keys must be strings')
        if key in result:
            raise ConfigError(f'Duplicate config key: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_config(root: Path) -> ConwayConfig:
    path = ensure_config(root)
    try:
        if path.stat().st_size > 100000:
            raise ConfigError('Config exceeds 100 KB')
        raw = yaml.load(path.read_text(encoding='utf-8'), Loader=StrictLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f'Could not read config {path}: {exc}') from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError('Config must be a YAML mapping')
    return _merge_dataclass(ConwayConfig(), raw)
