from __future__ import annotations

import argparse
import sys

from .brain import MockBrain, OpenAICompatibleVLM
from .computer import Computer
from .hardware import detect_hardware, doctor_report
from .loop import ConwayLoop
from .memory import FileMemory
from .runtime import RuntimeHandle, connect_external, start_local_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conway", description="Local computer-use agent harness")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Detect hardware and recommend a model profile")

    start = sub.add_parser("start", help="Start Conway")
    start.add_argument("--profile", default="auto", choices=["auto", "small", "standard", "computer-use"])
    start.add_argument("--endpoint", help="Existing OpenAI-compatible /v1 endpoint")
    start.add_argument("--model", help="Model name sent to an external endpoint")
    start.add_argument("--port", type=int, default=8042)
    start.add_argument("--execute", action="store_true", help="Enable GUI actions; otherwise Conway runs in observation-only mode")
    start.add_argument("--mock", action="store_true", help="Use a no-op mock brain")
    start.add_argument("--max-steps", type=int, default=0, help="Cycles to run; 0 runs until stopped")
    start.add_argument("--interval", type=float, default=0.5)
    return parser


def start_conway(args: argparse.Namespace) -> int:
    memory = FileMemory()
    runtime: RuntimeHandle | None = None
    try:
        if args.mock:
            brain = MockBrain()
        else:
            if args.endpoint:
                model_name = args.model or "local-model"
                runtime = connect_external(args.endpoint, model_name)
            else:
                runtime = start_local_runtime(
                    detect_hardware(), memory.root, requested_profile=args.profile, port=args.port
                )
                model_name = args.model or runtime.model_repo
            brain = OpenAICompatibleVLM(runtime.base_url, model_name)

        computer = Computer(memory.root / "screenshots")
        runner = ConwayLoop(
            brain,
            computer,
            memory,
            dry_run=not args.execute,
            interval=args.interval,
            max_steps=args.max_steps,
        )
        print(f"Conway state: {memory.root}")
        print("Mode: GUI execution enabled" if args.execute else "Mode: observation-only")
        runner.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Conway failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if runtime:
            runtime.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print(doctor_report())
        return 0
    return start_conway(args)


if __name__ == "__main__":
    raise SystemExit(main())
