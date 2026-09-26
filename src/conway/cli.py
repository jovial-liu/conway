from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import uuid
from . import __version__
from .brain import MockBrain, OpenAICompatibleVLM, OpenCUABrain
from .computer import Computer, MockComputer
from .config import ensure_config, load_config, _merge_dataclass
from .control import EmergencyStop, SessionControl, termination_signals
from .autonomy import AutonomyPolicy
from .hardware import choose_model_profile, detect_hardware, doctor_report, load_model_manifest, InsufficientMemoryError
from .lock import InstanceLock
from .loop import ConwayLoop
from .memory import FileMemory
from .runtime import connect_external, start_local_runtime
from .tools import ToolExecutor


_PROFILE_CHOICES = ['auto', 'tiny', 'small', 'standard', 'computer-use']
_BRAIN_CHOICES = ['auto', 'generic', 'opencua']


def _positive(value: str) -> float:
    import math
    num = float(value)
    if not math.isfinite(num) or num < 0:
        raise argparse.ArgumentTypeError('value must be finite and non-negative')
    return num


def _steps(value: str) -> int:
    num = int(value)
    if num < 0:
        raise argparse.ArgumentTypeError('steps must be non-negative')
    return num


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='conway', description='Local autonomous computer-use harness')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--home', type=Path, help='Override CONWAY_HOME for this invocation')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init', help='Create local constitution/goals/config/state, without starting a model')
    doctor = sub.add_parser('doctor', help='Hardware and installation diagnostics; no GUI import')
    doctor.add_argument('--json', action='store_true')
    sub.add_parser('status', help='Read stored state and recent trajectory')
    config = sub.add_parser('config', help='Print or validate config.yaml')
    config.add_argument('--check', action='store_true')
    sub.add_parser('models', help='Show local profiles and a memory-based recommendation')
    probe = sub.add_parser('probe', help='Test a model with a synthetic PNG; never reads or controls the desktop')
    probe.add_argument('--endpoint')
    probe.add_argument('--model')
    probe.add_argument('--brain', choices=_BRAIN_CHOICES)
    preflight = sub.add_parser('preflight', help='Check deployment readiness without loading weights or executing actions')
    preflight.add_argument('--desktop', action='store_true', help='Temporarily capture the local desktop; no image is sent to a model')
    preflight.add_argument('--output', type=Path, help='Save JSON to a new file outside the state directory')
    vision = sub.add_parser('vision-check', help='Score synthetic visual target localization without desktop actions')
    vision.add_argument('--endpoint')
    vision.add_argument('--model')
    vision.add_argument('--brain', choices=_BRAIN_CHOICES)
    vision.add_argument('--samples', type=int, default=8)
    vision.add_argument('--seed', type=int, default=0)
    vision.add_argument('--output', type=Path, help='Save JSON to a new file outside the state directory')
    acceptance = sub.add_parser('acceptance-check', help='Run bounded real-model file/copy-and-launch component checks')
    acceptance.add_argument('--endpoint')
    acceptance.add_argument('--model')
    acceptance.add_argument('--max-steps', type=int, default=12)
    acceptance.add_argument('--max-seconds', type=float, default=300)
    acceptance.add_argument('--output', type=Path)
    for name in ('stop', 'pause', 'resume'):
        sub.add_parser(name, help=f'Cooperatively {name} the current session')
    export = sub.add_parser('export', help='Export local trajectory events, without screenshots or uploading')
    export.add_argument('--output', required=True, type=Path)
    export.add_argument('--include-dry-run', action='store_true')
    sub.add_parser('skills', help='List locally installed Agent Skills without running scripts')
    sub.add_parser('mcp-check', help='Start configured MCP servers and list allowed tools; never call a tool')
    dataset = sub.add_parser('dataset', help='Export independently reviewed visual episodes for policy training')
    dataset.add_argument('--episodes', type=Path, nargs='+', required=True)
    dataset.add_argument('--output', type=Path, required=True)
    dataset.add_argument('--validation-percent', type=int, default=20)
    run = sub.add_parser('run', help='Continuously act from goals and observations; no conversation prompt')
    run.set_defaults(execute=True)
    run.add_argument('--observe', dest='execute', action='store_false', help='Continuous observation and planning without actions')
    start = sub.add_parser('start', help='Single session / acceptance test; finish ends this session')
    for command in (run, start):
        command.add_argument('--profile', choices=_PROFILE_CHOICES)
        command.add_argument('--brain', choices=_BRAIN_CHOICES)
        command.add_argument('--endpoint')
        command.add_argument('--model')
        command.add_argument('--port', type=int)
        command.add_argument('--gui-only', action='store_true')
        command.add_argument('--mock', action='store_true', help='Offline synthetic desktop; no model, GUI or system actions')
        command.add_argument('--max-steps', type=_steps, default=0)
        command.add_argument('--max-seconds', type=_positive, default=0)
        command.add_argument('--interval', type=_positive)
        command.add_argument('--quiet', action='store_true')
        command.add_argument('--record-episode', type=Path, help='Opt in to local screenshot/prompt/action recording in a NEW directory')
    start.add_argument('--execute', action='store_true', help='Enable current-user actions for this process only')
    task = start.add_mutually_exclusive_group()
    task.add_argument('--task', help='Goal for this session, subject to constitution.md')
    task.add_argument('--task-file', type=Path, help='UTF-8 file containing this session goal (at most 16000 characters)')
    return parser


def _brain_type(requested: str, profile_name: str, model_name: str) -> str:
    if requested != 'auto':
        return requested
    if 'opencua' in model_name.lower():
        return 'opencua'
    profile = load_model_manifest()['profiles'].get(profile_name, {})
    return profile.get('brain', 'generic')


def initialize(root=None) -> int:
    memory = FileMemory(root)
    print(f'Conway state: {memory.root}\nConfig: {ensure_config(memory.root)}\nConstitution: {memory.constitution_path}\nGoals: {memory.goals_path}')
    return 0


def show_status(root=None) -> int:
    memory = FileMemory(root)
    state = memory.load_state()
    lock = InstanceLock(memory.root / 'instance.lock')
    try:
        lock.acquire()
        active = False
    except RuntimeError:
        active = True
    finally:
        lock.release()
    print(json.dumps({'state_directory': str(memory.root), 'instance_lock_held': active,
                      'state': asdict(state)}, ensure_ascii=False, indent=2))
    print(memory.recent_events(limit=5, max_chars=6000))
    return 0


def show_config(root=None, check=False) -> int:
    memory = FileMemory(root)
    load_config(memory.root)
    print('Config valid' if check else ensure_config(memory.root).read_text(encoding='utf-8'))
    return 0


def show_models() -> int:
    try:
        name, _ = choose_model_profile(detect_hardware())
    except InsufficientMemoryError:
        name = 'none fits the estimated current budget'
    print(f'Recommended: {name}')
    for key, profile in load_model_manifest()['profiles'].items():
        print(f"{key:14} {profile['brain']:8} {profile['repo']}")
    return 0


def start_conway(args: argparse.Namespace) -> int:
    task = getattr(args, 'task', None)
    task_file = getattr(args, 'task_file', None)
    if task_file:
        with task_file.expanduser().open(encoding='utf-8') as file:
            task = file.read(16001)
    from .loop import validate_task
    task = validate_task(task)
    memory = FileMemory(args.home)
    config = load_config(memory.root)
    overrides = {name: getattr(args, name) for name in ('profile', 'brain', 'endpoint', 'model', 'port', 'interval')
                 if getattr(args, name) is not None}
    config = _merge_dataclass(config, overrides)
    settings = config.tools
    if args.gui_only or args.mock:
        settings = replace(settings, shell=False, filesystem=False, open_url=False)
    if args.mock:
        settings = replace(settings, gui=False)
    runtime, brain, mcp, recorder = None, None, None, None
    with InstanceLock(memory.root / 'instance.lock'), termination_signals():
        try:
            computer = MockComputer(memory.root / 'screenshots') if args.mock else Computer(
                memory.root / 'screenshots', ui_tree=config.ui_tree, ui_timeout=config.ui_timeout, ui_max_nodes=config.ui_max_nodes)
            executor = ToolExecutor(computer, settings, secret_env=config.api_key_env)
            if args.mock:
                brain, profile_name, model_name, brain_name = MockBrain(), 'mock', 'mock', 'mock'
            else:
                api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
                if config.api_key_env and not api_key:
                    raise ValueError(f'Configured API-key environment variable {config.api_key_env} is empty')
                if config.endpoint:
                    runtime = connect_external(config.endpoint, config.model, api_key=api_key)
                    profile_name = config.profile if config.profile != 'auto' else 'external'
                else:
                    runtime = start_local_runtime(detect_hardware(), memory.root, config.profile, config.port,
                        runtime_path=config.runtime_path, startup_timeout=config.startup_timeout, context_size=config.context_size)
                    profile_name = runtime.profile_name
                model_name = runtime.served_model or runtime.model_repo
                brain_name = _brain_type(config.brain, profile_name, model_name)
                if brain_name == 'generic' and not args.gui_only:
                    from .skills import SkillRegistry
                    executor.skills = SkillRegistry([memory.root / 'skills'] + [Path(p) for p in config.skill_dirs],
                        page_chars=max(256, min(4000, config.context_chars // 2 - 500)))
                    if config.mcp_servers and args.execute:
                        from .mcp_bridge import MCPBridge
                        mcp = MCPBridge(config.mcp_servers, output_limit=config.tools.max_output_chars)
                        mcp.start()
                        executor.mcp = mcp
                options = {'timeout': config.request_timeout, 'api_key': api_key}
                brain = (OpenCUABrain(runtime.base_url, model_name, min_pixels=config.opencua_min_pixels,
                                     max_pixels=config.opencua_max_pixels, **options) if brain_name == 'opencua'
                         else OpenAICompatibleVLM(runtime.base_url, model_name, tool_manifest=executor.manifest_text(), **options))
            if args.record_episode:
                if brain_name == 'opencua':
                    raise ValueError('Episode recording currently supports the generic JSON policy only')
                from .episodes import EpisodeRecorder
                recorder = EpisodeRecorder(args.record_episode, model=model_name, brain=brain_name,
                    dry_run=not args.execute, synthetic=args.mock, tool_manifest=executor.manifest_text(), state_root=memory.root)
            state = memory.load_state()
            state.pid, state.session_id = os.getpid(), uuid.uuid4().hex
            state.profile, state.model, state.brain = profile_name, model_name, brain_name
            state.dry_run, state.status = not args.execute, 'starting'
            memory.save_state(state)
            if not args.quiet:
                print(f'Conway {__version__} | {memory.root}\nModel: {model_name} | Brain: {brain_name}\n'
                      f'Mode: {"execution" if args.execute else "observation-only"}', flush=True)
            runner = ConwayLoop(brain, executor, memory, dry_run=not args.execute,
                interval=config.interval, max_steps=args.max_steps, screenshot_keep=config.screenshot_keep,
                memory_compaction_bytes=config.memory_compaction_bytes, max_errors=config.max_errors,
                context_chars=config.context_chars, max_seconds=args.max_seconds, quiet=args.quiet, task=task,
                recorder=recorder,
                continuous=args.command == 'run', policy=AutonomyPolicy(**{name: getattr(config, name) for name in
                    ('idle_initial_seconds', 'idle_max_seconds', 'recovery_initial_seconds', 'recovery_max_seconds', 'repeat_action_limit')}))
            return 1 if runner.run() == 'error' else 0
        except EmergencyStop as exc:
            state = memory.load_state()
            state.status, state.pid, state.next_wake_at = 'stopped', None, None
            state.stop_reason = str(exc)
            memory.save_state(state)
            return 0
        except Exception as exc:
            state = memory.load_state()
            state.status, state.pid = 'error', None
            state.last_result = f'{type(exc).__name__}: {exc}'[:4000]
            memory.save_state(state)
            raise
        finally:
            try:
                if recorder is not None:
                    recorder.close(memory.load_state().status)
            finally:
                try:
                    if mcp is not None:
                        mcp.close()
                finally:
                    try:
                        if brain is not None:
                            brain.close()
                    finally:
                        if runtime is not None:
                            runtime.close()


def request_control(root, command) -> int:
    memory = FileMemory(root)
    state = memory.load_state()
    if state.status not in {'starting', 'running', 'paused', 'idle', 'recovering'} or not state.session_id:
        print('No active session recorded', file=sys.stderr)
        return 1
    SessionControl(memory.root, state.session_id).request(command)
    print(f'{command} requested; applied at the next cooperative checkpoint')
    return 0


def export_trajectory(root, output: Path, include_dry_run: bool = False) -> int:
    memory = FileMemory(root)
    output = output.expanduser().resolve()
    # Do not permit export to overwrite any input/state/source data.
    if output.exists() or output.is_relative_to(memory.root):
        raise ValueError('Export path must be new and outside the state directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    created = False
    try:
        with output.open('x', encoding='utf-8') as out:
            created = True
            for path in sorted(memory.journal_dir.glob('*.jsonl')):
                with path.open(encoding='utf-8') as inp:
                    for line in inp:
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(event, dict) or event.get('event') != 'action_result':
                            continue
                        if event.get('dry_run') and not include_dry_run:
                            continue
                        # No screenshot bytes; preserve explicit provenance and unknown task success.
                        record = {k: event.get(k) for k in ('time', 'cycle', 'action', 'result', 'dry_run')}
                        record['task_success'] = None
                        out.write(json.dumps(record, ensure_ascii=False) + '\n')
                        count += 1
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise
    print(f'Exported {count} events to {output}; review for private data before sharing')
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'init':
            return initialize(args.home)
        if args.command == 'doctor':
            print(json.dumps(asdict(detect_hardware()), indent=2) if args.json else doctor_report())
            return 0
        if args.command == 'status':
            return show_status(args.home)
        if args.command == 'config':
            return show_config(args.home, args.check)
        if args.command == 'probe':
            from .diagnostics import probe_model
            memory = FileMemory(args.home)
            config = load_config(memory.root)
            overrides = {name: getattr(args, name) for name in ('endpoint', 'model', 'brain') if getattr(args, name) is not None}
            config = _merge_dataclass(config, overrides)
            with InstanceLock(memory.root / 'instance.lock'):
                print(json.dumps(probe_model(config, memory.root), indent=2))
            return 0
        if args.command in {'preflight', 'vision-check', 'acceptance-check'}:
            from .reports import report_path, write_report
            memory = FileMemory(args.home)
            config = load_config(memory.root)
            if args.output:
                report_path(args.output, memory.root)  # Reject invalid destinations before model startup.
            with InstanceLock(memory.root / 'instance.lock'):
                if args.command == 'preflight':
                    from .preflight import preflight
                    report = preflight(config, memory.root, desktop=args.desktop)
                elif args.command == 'acceptance-check':
                    from .acceptance import acceptance_check
                    overrides = {name: getattr(args, name) for name in ('endpoint', 'model') if getattr(args, name) is not None}
                    config = _merge_dataclass(config, overrides)
                    report = acceptance_check(config, memory.root, max_steps=args.max_steps, max_seconds=args.max_seconds)
                else:
                    from .evaluation import vision_check
                    overrides = {name: getattr(args, name) for name in ('endpoint', 'model', 'brain') if getattr(args, name) is not None}
                    config = _merge_dataclass(config, overrides)
                    report = vision_check(config, memory.root, samples=args.samples, seed=args.seed)
                if args.output:
                    write_report(args.output, report, memory.root)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report['status'] == 'passed' else 2
        if args.command == 'models':
            return show_models()
        if args.command == 'skills':
            from .skills import SkillRegistry
            memory = FileMemory(args.home)
            config = load_config(memory.root)
            print(SkillRegistry([memory.root / 'skills'] + [Path(p) for p in config.skill_dirs]).catalog())
            return 0
        if args.command == 'mcp-check':
            from .mcp_bridge import MCPBridge
            memory = FileMemory(args.home)
            config = load_config(memory.root)
            bridge = MCPBridge(config.mcp_servers)
            with InstanceLock(memory.root / 'instance.lock'):
                try:
                    bridge.start()
                    print(bridge.manifest())
                finally:
                    bridge.close()
            return 0
        if args.command == 'dataset':
            from .episodes import export_dataset
            print(json.dumps(export_dataset(args.episodes, args.output, args.validation_percent), indent=2))
            return 0
        if args.command in {'stop', 'pause', 'resume'}:
            return request_control(args.home, args.command)
        if args.command == 'export':
            return export_trajectory(args.home, args.output, args.include_dry_run)
        return start_conway(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'Conway: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
