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
    available_ram_gb: float | None = None
    available_vram_gb: float | None = None
    gpu_index: int | None = None


class InsufficientMemoryError(RuntimeError):
    pass


def _nvidia_memory() -> tuple[int, float, float] | None:
    exe = shutil.which('nvidia-smi')
    if not exe:
        return None
    try:
        p = subprocess.run([exe, '--query-gpu=index,memory.total,memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=3, check=False)
        if p.returncode:
            return None
        rows = []
        for line in p.stdout.splitlines():
            index, total, free = line.split(',')
            rows.append((int(index), float(total) / 1024, float(free) / 1024))
        return max(rows, key=lambda row: row[2]) if rows else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def detect_hardware() -> HardwareProfile:
    system, arch = platform.system(), platform.machine()
    mem = psutil.virtual_memory()
    total, available = mem.total / 1024**3, mem.available / 1024**3
    gpu = _nvidia_memory()
    if system == 'Darwin' and arch in {'arm64', 'aarch64'}:
        accelerator, vram, free, index = 'metal', None, None, None
        effective = max(0, min(available - 2, total * 0.70))
    elif gpu:
        index, vram, free = gpu
        accelerator = 'cuda'
        effective = max(0, min(free - 1, vram * 0.90)) if available >= 2 else 0
    else:
        accelerator, vram, free, index = 'cpu', None, None, None
        effective = max(0, min(available - 2, total * 0.75))
    return HardwareProfile(system, arch, round(total, 2), accelerator, vram, round(effective, 2),
                           round(available, 2), free, index)


def load_model_manifest() -> dict:
    return yaml.safe_load(resources.files('conway').joinpath('models.yaml').read_text(encoding='utf-8'))


def choose_model_profile(hardware: HardwareProfile, requested: str = 'auto') -> tuple[str, dict]:
    profiles = load_model_manifest()['profiles']
    if requested != 'auto':
        if requested not in profiles:
            raise ValueError(f'Unknown model profile: {requested}')
        cfg = profiles[requested]
        if cfg['backend'] != 'external_openai' and hardware.effective_memory_gb < cfg['min_effective_memory_gb']:
            raise InsufficientMemoryError(f'{requested} needs an estimated {cfg["min_effective_memory_gb"]} GB model budget; '
                                          f'only {hardware.effective_memory_gb:.1f} GB is available')
        return requested, cfg
    candidates = [(cfg['min_effective_memory_gb'], name, cfg) for name, cfg in profiles.items()
                  if cfg['backend'] != 'external_openai' and hardware.effective_memory_gb >= cfg['min_effective_memory_gb']]
    if not candidates:
        raise InsufficientMemoryError('No bundled local VLM fits the estimated available memory. '
                                      'Close other applications or configure an external multimodal endpoint.')
    _, name, cfg = max(candidates, key=lambda item: item[0])
    return name, cfg


def doctor_report() -> str:
    hw = detect_hardware()
    try:
        name, model = choose_model_profile(hw)
        selection = f"{name}: {model['repo']} ({model.get('quant') or 'external'})"
    except InsufficientMemoryError as exc:
        selection = str(exc)
    return '\n'.join(['Conway hardware detection', f'OS: {hw.os_name} / {hw.architecture}',
        f'RAM total/available: {hw.ram_gb:.1f} / {hw.available_ram_gb or 0:.1f} GiB',
        f'Accelerator candidate: {hw.accelerator} (runtime build must support it)',
        f'VRAM total/free: {hw.vram_gb} / {hw.available_vram_gb} GiB',
        f'Estimated model memory budget: {hw.effective_memory_gb:.1f} GiB',
        f'Recommended profile: {selection}',
        f'llama-server: {shutil.which("llama-server") or "not found; configure runtime_path or --endpoint"}',
        'Memory estimates include headroom, not a guarantee; image tokens and KV cache also consume memory.',
        'AMD/Intel GPU acceleration is not auto-verified; CPU fallback or an external server is supported.'])
