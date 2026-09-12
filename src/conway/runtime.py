from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time

import httpx

from .hardware import HardwareProfile, choose_model_profile


@dataclass(slots=True)
class RuntimeHandle:
    profile_name: str
    model_repo: str
    base_url: str
    process: subprocess.Popen | None = None
    log_file: object | None = None

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass


def _ready(base_url: str) -> bool:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=2.0)
        return response.status_code < 500
    except httpx.HTTPError:
        return False


def connect_external(base_url: str, model: str) -> RuntimeHandle:
    if not _ready(base_url):
        raise RuntimeError(f"No OpenAI-compatible model server responded at {base_url}")
    return RuntimeHandle("external", model, base_url)


def start_local_runtime(
    hardware: HardwareProfile,
    state_root: Path,
    requested_profile: str = "auto",
    port: int = 8042,
) -> RuntimeHandle:
    profile_name, profile = choose_model_profile(hardware, requested_profile)
    if profile.get("backend") == "external_openai":
        raise RuntimeError(
            f"Profile '{profile_name}' expects an externally served model. "
            "Start a compatible OpenAI-style multimodal server and use --endpoint."
        )

    llama_server = shutil.which("llama-server")
    llama = shutil.which("llama")
    if not llama_server and not llama:
        raise RuntimeError(
            "llama.cpp was not found. Install a current llama.cpp build, then run Conway again. "
            "On macOS it is commonly available through Homebrew; on Windows through WinGet; "
            "Linux users can use an official binary or build llama.cpp."
        )

    repo = str(profile["repo"])
    quant = profile.get("quant")
    hf_model = f"{repo}:{quant}" if quant else repo
    base_url = f"http://127.0.0.1:{port}/v1"

    if _ready(base_url):
        return RuntimeHandle(profile_name, repo, base_url)

    state_root.mkdir(parents=True, exist_ok=True)
    log_path = state_root / "llama-runtime.log"
    log_file = log_path.open("a", encoding="utf-8")

    if llama_server:
        command = [llama_server, "-hf", hf_model, "--host", "127.0.0.1", "--port", str(port)]
    else:
        command = [llama, "serve", "-hf", hf_model, "--host", "127.0.0.1", "--port", str(port)]

    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    handle = RuntimeHandle(profile_name, repo, base_url, process, log_file)

    # First launch may include a model download, so poll rather than assuming a fixed boot time.
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            handle.close()
            raise RuntimeError(f"llama.cpp exited during startup. See {log_path}")
        if _ready(base_url):
            return handle
        time.sleep(2)

    handle.close()
    raise RuntimeError(f"Model server did not become ready. See {log_path}")
