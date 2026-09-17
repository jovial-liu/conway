from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
import sys

import yaml

from .brain import MockBrain, OpenAICompatibleVLM, OpenCUABrain
from .computer import Computer
from .config import ensure_config, load_config
from .hardware import choose_model_profile, detect_hardware, doctor_report, load_model_manifest
from .lock import InstanceLock
from .loop import ConwayLoop
from .memory import FileMemory
from .runtime import RuntimeHandle, connect_external, start_local_runtime
from .tools import ToolExecutor


_PROFILE_CHOICES = ["auto", "tiny", "small", "standard", "computer-use"]
_BRAIN_CHOICES = ["auto", "generic", "opencua"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conway", description="Local autonomous computer-use agent harness")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Initialize Conway's local config/state directory")
    sub.add_parser("doctor", help="Detect hardware and recommend a model profile")
    sub.add_parser("status", help="Show Conway's local runtime state and recent trajectory")
    sub.add_parser("config", help="Print the active config.yaml")
    sub.add_parser("models", help="List bundled model profiles and the hardware recommendation")

    start = sub.add_parser("start", help="Start Conway's autonomous loop")
    start.add_argument("--profile", choices=_PROFILE_CHOICES, default=None)
    start.add_argument("--brain", choices=_BRAIN_CHOICES, default=None)
    start.add_argument("--endpoint", help="Existing OpenAI-compatible /v1 endpoint")
    start.add_argument("--model", help="Model name sent to an external endpoint")
    start.add_argument("--port", type=int, default=None)
    start.add_argument("--execute", action="store_true", help="Enable actions; otherwise Conway observes and plans only")
    start.add_argument("--gui-only", action="store_true", help="Disable shell/filesystem/URL tools for this run")
    start.add_argument("--mock", action="store_true", help="Use a no-op mock brain")
    start.add_argument("--max-steps", type=int, default=0, help="Cycles to run; 0 runs until stopped")
    start.add_argument("--interval", type=float, default=None)
    return parser


def initialize() -> int:
    memory = FileMemory()
    path = ensure_config(memory.root)
    print(f"Conway state directory: {memory.root}")
    print(f"Config: {path}")
    print(f"Constitution: {memory.constitution_path}")
    print(f"Memory: {memory.memory_path}")
    return 0


def show_status() -> int:
    memory = FileMemory()
    state = memory.load_state()
    print(f"Conway state directory: {memory.root}")
    print(json.dumps(asdict(state), ensure_ascii=False, indent=2))
    recent = memory.recent_events(limit=5, max_chars=6000)
    if recent:
        print("\nRecent trajectory:")
        print(recent)
    return 0


def show_config() -> int:
    memory = FileMemory()
    path = ensure_config(memory.root)
    print(f"# {path}")
    print(path.read_text(encoding="utf-8").rstrip())
    return 0


def show_models() -> int:
    hardware = detect_hardware()
    recommended, _ = choose_model_profile(hardware)
    manifest = load_model_manifest()
    print(f"Recommended for this machine: {recommended}\n")
    for name, profile in manifest["profiles"].items():
        marker = "*" if name == recommended else " "
        minimum = profile.get("min_effective_memory_gb", "-")
        brain = profile.get("brain", "generic")
        print(f"{marker} {name:12} {brain:8} min={minimum!s:>4} GB  {profile['repo']}")
    return 0


def _brain_type(requested: str, profile_name: str, model_name: str) -> str:
    if requested != "auto":
        return requested
    profile = load_model_manifest().get("profiles", {}).get(profile_name, {})
    configured = str(profile.get("brain", "generic"))
    if configured in {"generic", "opencua"}:
        return configured
    if "opencua" in model_name.lower():
        return "opencua"
    return "generic"


def start_conway(args: argparse.Namespace) -> int:
    memory = FileMemory()
    config = load_config(memory.root)
    profile_request = args.profile or config.profile
    brain_request = args.brain or config.brain
    endpoint = args.endpoint or config.endpoint
    configured_model = args.model or config.model
    port = args.port if args.port is not None else config.port
    interval = args.interval if args.interval is not None else config.interval

    tool_settings = config.tools
    if args.gui_only:
        tool_settings = replace(tool_settings, shell=False, filesystem=False, open_url=False)

    runtime: RuntimeHandle | None = None
    brain = None
    lock = InstanceLock(memory.root / "instance.lock")
    lock.acquire()
    try:
        computer = Computer(memory.root / "screenshots")
        executor = ToolExecutor(computer, tool_settings)

        if args.mock:
            brain = MockBrain()
            profile_name = "mock"
            model_name = "mock"
            brain_name = "mock"
        else:
            if endpoint:
                model_name = configured_model or "local-model"
                runtime = connect_external(endpoint, model_name)
                profile_name = profile_request if profile_request != "auto" else "external"
            else:
                runtime = start_local_runtime(
                    detect_hardware(),
                    memory.root,
                    requested_profile=profile_request,
                    port=port,
                )
                profile_name = runtime.profile_name
                model_name = configured_model or runtime.model_repo

            brain_name = _brain_type(brain_request, profile_name, model_name)
            if brain_name == "opencua":
                brain = OpenCUABrain(runtime.base_url, model_name)
            else:
                brain = OpenAICompatibleVLM(
                    runtime.base_url,
                    model_name,
                    tool_manifest=executor.manifest_text(),
                )

        state = memory.load_state()
        state.pid = os.getpid()
        state.profile = profile_name
        state.model = model_name
        state.brain = brain_name
        state.dry_run = not args.execute
        state.status = "starting"
        memory.save_state(state)

        runner = ConwayLoop(
            brain,
            executor,
            memory,
            dry_run=not args.execute,
            interval=interval,
            max_steps=args.max_steps,
            screenshot_keep=config.screenshot_keep,
            memory_compaction_bytes=config.memory_compaction_bytes,
        )
        print(f"Conway state: {memory.root}")
        print(f"Profile: {profile_name}")
        print(f"Brain: {brain_name}")
        print(f"Model: {model_name}")
        print(f"Enabled actions: {', '.join(sorted(executor.enabled_actions()))}")
        print("Mode: execution enabled" if args.execute else "Mode: observation-only")
        runner.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        state = memory.load_state()
        state.status = "error"
        state.pid = None
        state.last_result = f"startup/runtime error: {type(exc).__name__}: {exc}"
        memory.save_state(state)
        print(f"Conway failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if brain is not None:
            try:
                brain.close()
            except Exception:
                pass
        if runtime:
            runtime.close()
        lock.release()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        return initialize()
    if args.command == "doctor":
        print(doctor_report())
        return 0
    if args.command == "status":
        return show_status()
    if args.command == "config":
        return show_config()
    if args.command == "models":
        return show_models()
    return start_conway(args)


if __name__ == "__main__":
    raise SystemExit(main())
