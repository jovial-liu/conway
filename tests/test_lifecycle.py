"""No real desktop interaction. Includes actual bounded user-space file/shell I/O."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import pytest
from conway.brain import Brain, Decision, MockBrain
from conway.computer import MockComputer, Observation, Computer
from conway.config import ToolSettings
from conway.control import SessionControl, EmergencyStop
from conway.tools import ToolExecutor
from conway.memory import FileMemory
from conway.loop import ConwayLoop
from conway.cli import main, export_trajectory


class OnceBrain(Brain):
    def __init__(self, action, note=None):
        self.action, self.note, self.calls = action, note, 0
    def decide(self, *args):
        self.calls += 1
        return Decision(self.action, 'Test decision', self.note)


def setup(tmp_path, brain=None, **kwargs):
    mem = FileMemory(tmp_path)
    executor = ToolExecutor(MockComputer(tmp_path/'screenshots'), ToolSettings(workspace=str(tmp_path)))
    runner = ConwayLoop(brain or MockBrain(), executor, mem, interval=0, quiet=True, **kwargs)
    return mem, executor, runner


def test_offline_cli_smoke_and_compat_commands(tmp_path, capsys):
    assert main(['--home',str(tmp_path),'init']) == 0
    assert main(['--home',str(tmp_path),'start','--mock','--max-steps','3','--interval','0','--quiet']) == 0
    state = FileMemory(tmp_path).load_state()
    assert state.cycle == 3 and state.status == 'stopped' and state.pid is None
    assert state.model == 'mock' and state.dry_run is True
    assert main(['--home',str(tmp_path),'config','--check']) == 0
    assert main(['--home',str(tmp_path),'status']) == 0
    assert main(['doctor']) == 0


def test_dry_run_never_writes_or_adds_fictional_memory(tmp_path):
    brain = OnceBrain({'type':'write_file','args':{'path':'note.txt','content':'hello'}}, 'I completed the write')
    mem, executor, loop = setup(tmp_path, brain, dry_run=True, max_steps=1)
    loop.run()
    assert not (tmp_path/'note.txt').exists()
    assert 'I completed' not in mem.memory_path.read_text()
    assert mem.load_state().pending_action is None
    assert 'planned only' in mem.load_state().last_result


def test_real_file_action_has_durable_intent_and_result(tmp_path):
    brain = OnceBrain({'type':'write_file','args':{'path':'note.txt','content':'hello'}})
    mem, executor, loop = setup(tmp_path, brain, max_steps=1)
    assert loop.run() == 'stopped'
    assert (tmp_path/'note.txt').read_text() == 'hello'
    events = [json.loads(line) for p in mem.journal_dir.glob('*.jsonl') for line in p.read_text().splitlines()]
    assert [e['event'] for e in events] == ['action_intent', 'action_result']
    assert events[0]['id'] == events[1]['id']
    assert mem.load_state().pending_action is None


def test_ambiguous_side_effect_stops_without_retry(tmp_path):
    brain = OnceBrain({'type':'write_file','args':{'path':'x','content':'x'}})
    mem, executor, loop = setup(tmp_path, brain, max_steps=3)
    def partial(*args):
        (tmp_path/'partial').write_text('side effect')
        raise OSError('failed after dispatch')
    executor.execute = partial
    assert loop.run() == 'error'
    assert brain.calls == 1
    assert mem.load_state().pending_action is not None


def test_recovery_never_replays_pending_action(tmp_path):
    mem, executor, loop = setup(tmp_path, max_steps=1)
    state = mem.load_state()
    state.pending_action = {'id':'previous','action':{'type':'write_file','args':{'path':'nope','content':'x'}}}
    mem.save_state(state)
    loop.run()
    assert not (tmp_path/'nope').exists()
    assert 'Previous action outcome is uncertain' in mem.recent_events()


def test_error_budget_stops(tmp_path, monkeypatch):
    brain = OnceBrain({'type':'unknown','args':{}})
    mem, executor, loop = setup(tmp_path, brain, max_steps=20, max_errors=2)
    monkeypatch.setattr(SessionControl, 'sleep', lambda *_: None)
    assert loop.run() == 'error'
    assert brain.calls == 2 and mem.load_state().error_count == 2


def test_emergency_stop_not_retried(tmp_path):
    brain = OnceBrain({'type':'wait','args':{}})
    mem, executor, loop = setup(tmp_path, brain, max_steps=20)
    class FailSafeException(Exception):
        pass
    def fail(*args):
        raise FailSafeException('corner')
    executor.execute = fail
    assert loop.run() == 'stopped'
    assert brain.calls == 1 and 'failsafe' in mem.load_state().stop_reason


def test_stop_arriving_during_inference_prevents_dispatch(tmp_path):
    mem, executor, loop = setup(tmp_path, max_steps=5)
    class StopBrain(Brain):
        def decide(self, *args):
            executor.control.request('stop')
            return Decision({'type':'write_file','args':{'path':'bad','content':'x'}})
    loop.brain = StopBrain()
    assert loop.run() == 'stopped'
    assert not (tmp_path/'bad').exists()


def test_time_budget_checked_after_slow_inference(tmp_path):
    mem, executor, loop = setup(tmp_path, max_steps=10, max_seconds=.01)
    class SlowBrain(Brain):
        def decide(self,*args):
            time.sleep(.03)
            return Decision({'type':'write_file','args':{'path':'bad','content':'x'}})
    loop.brain = SlowBrain()
    assert loop.run() == 'stopped'
    assert not (tmp_path/'bad').exists()


def test_session_scoped_controls(tmp_path):
    old = SessionControl(tmp_path,'old')
    current = SessionControl(tmp_path,'new')
    old.request('stop')
    current.check()
    current.request('pause')
    assert current.command() == 'pause'
    current.request('resume')
    current.check()
    current.request('stop')
    with pytest.raises(EmergencyStop):
        current.check()


def test_finish_reports_model_declared_completion(tmp_path):
    mem, executor, loop = setup(tmp_path, OnceBrain({'type':'finish','args':{'reason':'done'}}), max_steps=3)
    assert loop.run() == 'completed'
    assert mem.load_state().stop_reason == 'done'


def test_export_excludes_dryrun_and_does_not_fabricate_scores(tmp_path):
    root=tmp_path/'state'
    mem=FileMemory(root)
    for dry in (True,False):
        mem.journal({'event':'action_result','cycle':1,'action':{'type':'wait','args':{}},'result':'ok','dry_run':dry})
    out=tmp_path/'export.jsonl'
    export_trajectory(root,out)
    rows=[json.loads(v) for v in out.read_text().splitlines()]
    assert len(rows)==1 and rows[0]['task_success'] is None and rows[0]['dry_run'] is False
    with pytest.raises(ValueError):
        export_trajectory(root,out)
    assert len(out.read_text().splitlines())==1


def test_file_tools_roundtrip_and_bounded_read(tmp_path):
    mem, executor, _ = setup(tmp_path)
    obs=executor.computer.observe(1)
    assert 'wrote 5 characters' in executor.execute({'type':'write_file','args':{'path':'sub/x','content':'hello'}},obs)
    assert 'hello' in executor.execute({'type':'read_file','args':{'path':'sub/x'}},obs)
    assert 'x' in executor.execute({'type':'list_dir','args':{'path':'sub'}},obs)
    (tmp_path/'big').write_text('x'*100000)
    result=executor.execute({'type':'read_file','args':{'path':'big','max_chars':100}},obs)
    assert 'file truncated' in result and len(result)<500


def python_command(code):
    parts=[sys.executable,'-c',code]
    return subprocess.list2cmdline(parts) if os.name=='nt' else shlex.join(parts)


def test_shell_output_and_secret_env_not_inherited(tmp_path,monkeypatch):
    monkeypatch.setenv('HF_TOKEN','test-fixture-only')
    mem,executor,_=setup(tmp_path)
    obs=executor.computer.observe(1)
    code="import os; print('HELLO'); print('HF_TOKEN' in os.environ)"
    result=executor.execute({'type':'shell','args':{'command':python_command(code),'timeout':5}},obs)
    assert 'shell exit code 0' in result and 'HELLO' in result and 'False' in result


def test_shell_timeout_and_output_ring(tmp_path):
    mem,executor,_=setup(tmp_path)
    executor.settings.max_shell_seconds=.2
    executor.settings.max_output_chars=1000
    result=executor.execute({'type':'shell','args':{'command':python_command("import time; print('x'*30000,flush=True); time.sleep(5)"),'timeout':1}},executor.computer.observe(1))
    assert 'timed out' in result and len(result)<1200


def test_disabled_tool_enforced(tmp_path):
    mem,executor,_=setup(tmp_path)
    executor.settings.shell=False
    with pytest.raises(PermissionError):
        executor.execute({'type':'shell','args':{'command':'echo hello'}},executor.computer.observe(1))


def test_hidpi_coordinate_mapping():
    obs=Observation(Path('x'),2880,1800,1440,900)
    assert Computer.map_screenshot_point(1440,900,obs)==(720,450)


def test_model_declared_failure_is_not_success(tmp_path):
    mem, executor, loop = setup(tmp_path, OnceBrain({'type':'finish','args':{'reason':'cannot finish','outcome':'failed'}}), max_steps=3)
    assert loop.run() == 'error'
    assert mem.load_state().stop_reason == 'cannot finish'
