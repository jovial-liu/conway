from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import platform
import shutil
import subprocess

import psutil
import yaml


@dataclass(slots=True)
class HardwareProfile:
    os_name: str
    architecture: str
    ram_gb: float
    accelerator: str
    vram_gb: float | None
    effective_memory_gb: float


def _nvidia_vram_gb() -> float | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        values = [float(x.strip()) / 1024 for x in result.stdout.splitlines() if x.strip()]
        return max(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def detect_hardware() -> HardwareProfile:
    os_name = platform.system()
    arch = platform.machine()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    vram = _nvidia_vram_gb()

    if os_name == "Darwin" and arch in {"arm64", "aarch64"}:
        accelerator = "metal"
        effective = ram_gb
    elif vram:
        accelerator = "cuda"
        effective = vram
    else:
        accelerator = "cpu"
        effective = ram_gb

    return HardwareProfile(
        os_name=os_name,
        architecture=arch,
        ram_gb=round(ram_gb, 1),
        accelerator=accelerator,
        vram_gb=round(vram, 1) if vram else None,
        effective_memory_gb=round(effective, 1),
    )


def load_model_manifest() -> dict:
    text = resources.files("conway").joinpath("models.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def choose_model_profile(hardware: HardwareProfile, requested: str = "auto") -> tuple[str, dict]:
    manifest = load_model_manifest()
    profiles: dict[str, dict] = manifest["profiles"]

    if requested != "auto":
        if requested not in profiles:
            raise ValueError(f"Unknown model profile: {requested}")
        return requested, profiles[requested]

    ordered = manifest.get("selection", {}).get("default_order", list(profiles))
    compatible: list[tuple[float, str, dict]] = []
    for name in ordered:
        cfg = profiles[name]
        minimum = float(cfg.get("min_effective_memory_gb", 0))
        if hardware.effective_memory_gb >= minimum and cfg.get("backend") != "external_openai":
            compatible.append((minimum, name, cfg))

    if compatible:
        _, name, cfg = max(compatible, key=lambda item: item[0])
        return name, cfg

    # Lowest-memory local profile as a best-effort fallback.
    local = [(float(v.get("min_effective_memory_gb", 0)), k, v) for k, v in profiles.items() if v.get("backend") != "external_openai"]
    _, name, cfg = min(local, key=lambda item: item[0])
    return name, cfg


def doctor_report() -> str:
    hw = detect_hardware()
    profile_name, model = choose_model_profile(hw)
    runtime = "llama.cpp found" if (shutil.which("llama-server") or shutil.which("llama")) else "llama.cpp not found"
    vram = f"{hw.vram_gb:.1f} GB" if hw.vram_gb else "shared / not detected"
    return "\n".join(
        [
            "Conway hardware detection",
            f"OS: {hw.os_name}",
            f"Architecture: {hw.architecture}",
            f"RAM: {hw.ram_gb:.1f} GB",
            f"Accelerator: {hw.accelerator}",
            f"VRAM: {vram}",
            f"Effective model memory: {hw.effective_memory_gb:.1f} GB",
            f"Recommended profile: {profile_name}",
            f"Recommended model: {model['repo']} ({model.get('quant') or 'server default'})",
            f"Runtime: {runtime}",
        ]
    )
