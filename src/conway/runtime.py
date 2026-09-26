from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import socket
import subprocess
import time
from typing import Any
import httpx
from .config import endpoint_url
from .hardware import HardwareProfile, choose_model_profile


@dataclass(slots=True)
class RuntimeHandle:
    profile_name: str
    model_repo: str
    base_url: str
    process: subprocess.Popen | None = None
    log_file: Any = None
    served_model: str | None = None

    def close(self) -> None:
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None


def model_ids(base_url: str, *, api_key: str | None = None, transport=None) -> list[str]:
    with httpx.Client(timeout=10, trust_env=False, transport=transport) as client:
        response = client.get(endpoint_url(base_url) + '/models',
                              headers={'Authorization': f'Bearer {api_key}'} if api_key else {})
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get('data'), list):
        raise ValueError('Endpoint /models did not return a model list')
    ids = [item['id'] for item in data['data'] if isinstance(item, dict) and isinstance(item.get('id'), str) and item['id']]
    if not ids:
        raise ValueError('Endpoint reports no loaded models')
    return ids


def _ready(base_url: str, expected_model: str | None = None) -> bool:
    try:
        ids = model_ids(base_url)
        return expected_model in ids if expected_model else bool(ids)
    except (httpx.HTTPError, ValueError):
        return False


def connect_external(base_url: str, model: str | None = None, *, api_key: str | None = None, transport=None) -> RuntimeHandle:
    url = endpoint_url(base_url)
    try:
        ids = model_ids(url, api_key=api_key, transport=transport)
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError('Model endpoint is not ready; check /v1/models, server status and authentication') from exc
    if model and model not in ids:
        raise ValueError(f'Served model {model!r} not found. Available IDs: {ids}')
    if model is None and len(ids) != 1:
        raise ValueError(f'Endpoint exposes multiple models; set --model to one of {ids}')
    selected = model or ids[0]
    return RuntimeHandle('external', selected, url, served_model=selected)


def find_runtime(path: str | None = None) -> str:
    candidate = str(Path(path).expanduser()) if path else shutil.which('llama-server')
    if not candidate or not Path(candidate).is_file():
        raise RuntimeError('llama-server not found. Install llama.cpp or set runtime_path in config.yaml. See docs/INSTALL.md.')
    return candidate


def start_local_runtime(hardware: HardwareProfile, state_root: Path, requested_profile: str = 'auto', port: int = 8042,
                        *, runtime_path: str | None = None, startup_timeout: float = 900, context_size: int = 8192) -> RuntimeHandle:
    profile_name, profile = choose_model_profile(hardware, requested_profile)
    if profile['backend'] == 'external_openai':
        raise RuntimeError('This profile requires --endpoint with an already-served multimodal model')
    binary = find_runtime(runtime_path)
    if not 1 <= port <= 65535:
        raise ValueError('Invalid port')
    with socket.socket() as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'Port {port} is occupied; use --endpoint to explicitly reuse a server, or --port to choose another')
    repo = profile['repo']
    alias = 'conway-local'
    command = [binary, '-hf', f"{repo}:{profile['quant']}", '--alias', alias, '--host', '127.0.0.1',
               '--port', str(port), '--ctx-size', str(context_size), '--parallel', '1',
               '--n-gpu-layers', '0' if hardware.accelerator == 'cpu' else '999', '--jinja', '--no-webui']
    state_root.mkdir(parents=True, exist_ok=True)
    log_path = state_root / 'llama-runtime.log'
    if log_path.exists() and log_path.stat().st_size > 2000000:
        os.replace(log_path, state_root / 'llama-runtime.previous.log')
    log = log_path.open('a', encoding='utf-8')
    handle = RuntimeHandle(profile_name, repo, f'http://127.0.0.1:{port}/v1', log_file=log, served_model=alias)
    env = os.environ.copy()
    if hardware.accelerator == 'cuda' and hardware.gpu_index is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(hardware.gpu_index)
    try:
        handle.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if handle.process.poll() is not None:
                raise RuntimeError(f'llama-server exited during startup. Inspect {log_path}')
            if _ready(handle.base_url, alias):
                return handle
            time.sleep(1)
        raise RuntimeError(f'Model startup timed out. First download can take longer; inspect {log_path} and startup_timeout')
    except BaseException:
        handle.close()
        raise
