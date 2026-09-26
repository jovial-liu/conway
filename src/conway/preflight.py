"""Deployment checks without downloading a model or dispatching desktop actions."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from . import __version__
from .config import ConwayConfig
from .hardware import detect_hardware, choose_model_profile
from .memory import FileMemory
from .runtime import connect_external, find_runtime


def desktop_worker(config: dict) -> dict:
    from .computer import Computer
    with TemporaryDirectory(prefix='conway-desktop-check-') as directory:
        computer = Computer(Path(directory), ui_tree=config['ui_tree'],
                            ui_timeout=config['ui_timeout'], ui_max_nodes=config['ui_max_nodes'])
        observation = computer.observe(0)
        if min(observation.width, observation.height, observation.input_width, observation.input_height) <= 0:
            raise ValueError('Invalid display geometry')
        # Deliberately omit window titles, UI labels, cursor positions and temporary screenshot paths.
        return {'screenshot_size': [observation.width, observation.height],
                'input_size': [observation.input_width, observation.input_height],
                'coordinate_scale': [observation.scale_x, observation.scale_y],
                'metadata_provider': observation.metadata_provider,
                'ui_tree_status': observation.ui_tree.get('status', 'not_requested'),
                'input_control_verified': False}


def check_desktop(config: ConwayConfig) -> dict:
    settings = {key: getattr(config, key) for key in ('ui_tree', 'ui_timeout', 'ui_max_nodes')}
    try:
        result = subprocess.run([sys.executable, '-m', 'conway.preflight', json.dumps(settings)],
                                capture_output=True, text=True, timeout=15 + config.ui_timeout)
        if result.returncode:
            raise RuntimeError('Desktop capture failed')
        data = json.loads(result.stdout)
        if not isinstance(data, dict) or 'screenshot_size' not in data:
            raise ValueError('Invalid desktop worker response')
        return {'status': 'passed', 'details': data}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return {'status': 'failed', 'error_type': type(exc).__name__,
                'hint': 'Use a logged-in desktop, install GUI dependencies, and grant screen permissions. Native Wayland is unsupported.'}


def preflight(config: ConwayConfig, state_root: Path, *, desktop: bool = False) -> dict:
    checks = []
    memory = FileMemory(state_root)
    hardware = detect_hardware()
    def add(name, status, **details):
        checks.append({'name': name, 'status': status, **details})
    try:
        if not memory.constitution().strip():
            raise ValueError('Constitution is empty')
        add('constitution', 'passed')
    except (ValueError, OSError) as exc:
        add('constitution', 'failed', error_type=type(exc).__name__, hint='Check constitution.md; use 1–32000 characters.')
    state = memory.load_state()
    add('recovery', 'warning' if state.pending_action or state.status == 'recovery_required' else 'passed',
        previous_action_uncertain=state.pending_action is not None)
    try:
        key = os.environ.get(config.api_key_env) if config.api_key_env else None
        if config.api_key_env and not key:
            raise ValueError('Configured API-key environment variable is empty')
        if config.endpoint:
            handle = connect_external(config.endpoint, config.model, api_key=key)
            try:
                add('model_service', 'passed', model=handle.served_model, vision_tested=False)
            finally:
                handle.close()
        else:
            profile, _ = choose_model_profile(hardware, config.profile)
            if profile == 'computer-use':
                raise ValueError('computer-use requires an external endpoint')
            binary = find_runtime(config.runtime_path)
            if os.name != 'nt' and not os.access(binary, os.X_OK):
                raise OSError('Runtime file is not executable')
            add('local_runtime', 'passed', profile=profile, weights_loaded=False,
                hint='Runtime path and estimated memory checked; run probe or vision-check to load the model.')
    except (ValueError, RuntimeError, OSError) as exc:
        add('model_service' if config.endpoint else 'local_runtime', 'failed', error_type=type(exc).__name__,
            hint='Check endpoint/model ID/API-key environment variable, or llama-server installation and available memory.')
    add('gui_dependency', 'passed' if importlib.util.find_spec('pyautogui') else 'failed',
        hint='Install Conway dependencies in this Python environment.')
    if not desktop:
        add('desktop', 'not_checked', hint='Run preflight --desktop on the target computer to test capture.')
    elif os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland':
        add('desktop', 'failed', hint='Native Wayland control is unsupported. Use an X11 session.')
    else:
        add('desktop', **check_desktop(config))
    blocked = any(item['status'] == 'failed' for item in checks)
    return {'schema_version': 1, 'conway_version': __version__,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'status': 'blocked' if blocked else 'passed', 'checks': checks, 'hardware': asdict(hardware),
            'ready_for_observation': desktop and not blocked,
            'actions_executed': 0, 'model_inference_tested': False, 'input_control_verified': False}


if __name__ == '__main__':
    try:
        print(json.dumps(desktop_worker(json.loads(sys.argv[1]))))
    except Exception:
        raise SystemExit(1)
