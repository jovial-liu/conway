"""Small real-policy workbench: bounded files, one trusted child, independent checks.

The observations are synthetic; decisions come from the configured model. This is
not a desktop benchmark or a test of open-ended autonomous replication.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from . import __version__
from .actions import validate_action
from .computer import MockComputer
from .config import ToolSettings
from .diagnostics import model_session
from .loop import ConwayLoop
from .memory import FileMemory
from .tools import ToolExecutor

SPECIMEN = 'print("CONWAY_REPLICA_READY")\n'
CASES = {
    'file_roundtrip': 'Create result.txt containing exactly CONWAY_ACCEPTANCE_OK, then read it back to verify. Report completion only after checking.',
    'bounded_replica': 'Read seed.py. Make an exact copy named replica.py. Run python replica.py once and check that it prints CONWAY_REPLICA_READY. Report completion only after checking. This is one local disposable program copy, not an open-ended replication task.',
}


class WorkbenchExecutor(ToolExecutor):
    def __init__(self, computer, workspace: Path, case: str):
        super().__init__(computer, ToolSettings(gui=False, shell=False, filesystem=True,
                                              open_url=False, workspace=str(workspace), max_output_chars=4000))
        self.case, self.reads, self.child_result = case, [], None

    def enabled_actions(self):
        names = {'read_file', 'write_file', 'list_dir', 'wait', 'finish'}
        return names | {'shell'} if self.case == 'bounded_replica' else names

    def manifest_text(self):
        text = super().manifest_text()
        return text + ('\nWorkbench restriction: paths stay in this directory; only result.txt or replica.py '
                       'may be written. The only allowed command is python replica.py, once, after an exact copy of seed.py.')

    def _path(self, value):
        path = super()._path(value)
        if not path.is_relative_to(self.workspace):
            raise ValueError('Workbench paths must stay within the disposable workspace')
        return path

    def validate_dispatch(self, action):
        kind, args = action['type'], action['args']
        if kind not in self.enabled_actions():
            raise ValueError('Action is not enabled in this workbench')
        if kind in {'read_file', 'write_file', 'list_dir'}:
            self._path(args['path'])
        if kind == 'write_file':
            allowed = 'replica.py' if self.case == 'bounded_replica' else 'result.txt'
            if self._path(args['path']) != self.workspace / allowed or len(args['content']) > 4000:
                raise ValueError('Only the bounded case output file can be written')
        if kind == 'shell':
            if args['command'].strip() not in {'python replica.py', 'python3 replica.py'} or args['cwd'] is not None:
                raise ValueError('Only python replica.py is allowed')
            if self.child_result is not None:
                raise ValueError('This test permits exactly one child launch')
            if (self.workspace / 'replica.py').read_bytes() != SPECIMEN.encode():
                raise ValueError('Child must exactly match the trusted seed program')

    def _shell(self, args):
        # Bypass arbitrary shell parsing. The model cannot substitute executable code.
        result = subprocess.run([sys.executable, '-I', str(self.workspace/'replica.py')],
            cwd=self.workspace, capture_output=True, text=True, timeout=5)
        self.child_result = {'returncode': result.returncode, 'stdout': result.stdout.strip()}
        return json.dumps(self.child_result)

    def execute(self, action, observation):
        action = validate_action(action, observation.width, observation.height)
        result = super().execute(action, observation)
        if action['type'] == 'read_file':
            self.reads.append(self._path(action['args']['path']).name)
        return result

    def verify(self):
        if self.case == 'file_roundtrip':
            path = self.workspace/'result.txt'
            checks = {'exact_file_content': path.is_file() and path.read_bytes() == b'CONWAY_ACCEPTANCE_OK',
                      'readback_performed': 'result.txt' in self.reads}
        else:
            path = self.workspace/'replica.py'
            checks = {'exact_program_copy': path.is_file() and path.read_bytes() == SPECIMEN.encode(),
                      'seed_read': 'seed.py' in self.reads,
                      'child_exit_and_output': self.child_result == {'returncode': 0, 'stdout': 'CONWAY_REPLICA_READY'}}
        return {'passed': all(checks.values()), 'checks': checks,
                'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}


def run_case(brain, directory: Path, case: str, *, max_steps=12, max_seconds=300):
    workspace = directory/'workspace'
    workspace.mkdir(parents=True)
    if case == 'bounded_replica':
        (workspace/'seed.py').write_bytes(SPECIMEN.encode())
    memory = FileMemory(directory/'state')
    memory.constitution_path.write_text('Complete the current local acceptance task using only the advertised tools. '
        'Do not infer success from a planned action; inspect the actual result. Stop this session with finish when done.', encoding='utf-8')
    computer = MockComputer(memory.root/'screenshots')
    executor = WorkbenchExecutor(computer, workspace, case)
    # Each case has its own tools/context; no previous case is reused as a demonstration.
    brain.tool_manifest = executor.manifest_text()
    started = time.monotonic()
    loop = ConwayLoop(brain, executor, memory, task=CASES[case], max_steps=max_steps,
        max_seconds=max_seconds, interval=0, max_errors=2, context_chars=6000, quiet=True)
    status = loop.run()
    state = memory.load_state()
    events = [json.loads(line) for path in memory.journal_dir.glob('*.jsonl')
              for line in path.read_text(encoding='utf-8').splitlines()]
    return {'case': case, 'loop_status': status, 'cycles': state.cycle,
            'error_count': state.error_count, 'elapsed_seconds': round(time.monotonic()-started, 3),
            'actions': [e['action']['type'] for e in events if e.get('event') == 'action_result'],
            'errors': [e['error'].split(':', 1)[0] for e in events if e.get('event') == 'cycle_error'],
            'model_reported_completion': status == 'completed', **executor.verify(),
            'synthetic_observation': True, 'real_model_expected': True,
            'child_launches': int(executor.child_result is not None), 'max_child_launches': 1 if case == 'bounded_replica' else 0}


def acceptance_check(config, state_root, *, max_steps=12, max_seconds=300):
    if type(max_steps) is not int or not 1 <= max_steps <= 30 or not 1 <= max_seconds <= 900:
        raise ValueError('Acceptance budgets: 1..30 steps and 1..900 seconds per case')
    if config.brain == 'opencua' or config.profile == 'computer-use':
        raise ValueError('This file/tool acceptance suite requires the generic JSON policy')
    with TemporaryDirectory(prefix='conway-acceptance-') as temporary:
        with model_session(replace(config, brain='generic'), state_root) as (brain, metadata):
            results = [run_case(brain, Path(temporary)/case, case, max_steps=max_steps, max_seconds=max_seconds)
                       for case in CASES]
    return {'schema_version': 1, 'suite': 'conway-workbench-v1', 'conway_version': __version__,
            'created_at': datetime.now(timezone.utc).isoformat(), **metadata,
            'status': 'passed' if all(row['passed'] for row in results) else 'failed', 'cases': results,
            'real_desktop_tested': False, 'open_ended_self_replication_tested': False,
            'note': 'Bounded local component checks with synthetic observations and independently checked files/child output; not proof of autonomous self-replication.'}
