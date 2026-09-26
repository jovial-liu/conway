"""Unattended lifecycle tests with synthetic observations; no real desktop or VLM."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import httpx
import pytest

from conway.autonomy import AutonomyPolicy
from conway.brain import Brain, Decision, OpenAICompatibleVLM, OpenCUABrain
from conway.cli import build_parser, main, request_control
from conway.computer import MockComputer
from conway.config import ConwayConfig, ConfigError, ToolSettings, _merge_dataclass
from conway.control import EmergencyStop, termination_signals
from conway.loop import ConwayLoop
from conway.memory import FileMemory
from conway.tools import ToolExecutor


WAIT = {'type': 'wait', 'args': {'seconds': 0}}
FINISH = {'type': 'finish', 'args': {'reason': 'subgoal reported', 'outcome': 'completed'}}


class ScriptedBrain(Brain):
    def __init__(self, actions):
        self.actions, self.contexts = iter(actions), []
    def decide(self, constitution, memory, observation):
        self.contexts.append((constitution, memory))
        value = next(self.actions)
        if isinstance(value, Exception):
            raise value
        return Decision(value, 'fixture action')


def setup(tmp_path, actions, **options):
    memory = FileMemory(tmp_path / 'state')
    executor = ToolExecutor(MockComputer(memory.root / 'screenshots'),
                            ToolSettings(workspace=str(tmp_path), shell=False, open_url=False))
    brain = ScriptedBrain(actions)
    defaults = dict(continuous=True, max_steps=len(actions), interval=0, quiet=True,
                    policy=AutonomyPolicy(.001, .002, .001, .002))
    defaults.update(options)
    loop = ConwayLoop(brain, executor, memory, **defaults)
    return memory, executor, brain, loop


def events(memory):
    return [json.loads(line) for path in memory.journal_dir.glob('*.jsonl')
            for line in path.read_text(encoding='utf-8').splitlines()]


def await_state(memory, predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = memory.load_state()
        if predicate(state):
            return state
        time.sleep(.01)
    pytest.fail('State transition timed out')


@pytest.mark.parametrize('outcome', ['completed', 'failed'])
def test_finish_is_a_subgoal_boundary_not_a_process_exit(tmp_path, outcome):
    finish = {'type':'finish','args':{'reason':'first goal','outcome':outcome}}
    actions = [finish, {'type':'write_file','args':{'path':'next.txt','content':'next autonomous goal'}}, FINISH]
    memory, executor, brain, loop = setup(tmp_path, actions)
    assert loop.run() == 'stopped'
    assert (tmp_path/'next.txt').read_text() == 'next autonomous goal'
    state = memory.load_state()
    assert state.cycle == 3 and state.loop_mode == 'continuous' and state.goal_reports == 2
    assert state.last_goal_report['verified'] is False
    assert all('no conversation window' in c[0].lower() for c in brain.contexts)
    assert [e['event'] for e in events(memory)].count('goal_report') == 2


def test_real_model_protocol_spans_two_independent_goals(tmp_path):
    memory, executor, _, loop = setup(tmp_path, [WAIT] * 6)
    actions = [{'type':'write_file','args':{'path':f'{name}.txt','content':name}} for name in ('first','second')]
    sequence = [actions[0], {'type':'read_file','args':{'path':'first.txt'}}, FINISH,
                actions[1], {'type':'read_file','args':{'path':'second.txt'}}, FINISH]
    calls = []
    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'action':sequence[len(calls)-1]})}}]})
    loop.brain = OpenAICompatibleVLM(transport=httpx.MockTransport(handler),tool_manifest=executor.manifest_text())
    try:
        assert loop.run() == 'stopped'
    finally:
        loop.brain.close()
    assert len(calls) == 6 and (tmp_path/'second.txt').read_text() == 'second'
    assert 'first' in calls[2]['messages'][1]['content'][1]['text']
    assert 'second' in calls[5]['messages'][1]['content'][1]['text']


def test_inference_recovers_after_old_error_limit(tmp_path):
    actions = [RuntimeError('service temporarily down')]*3 + [
        {'type':'write_file','args':{'path':'recovered.txt','content':'ready'}}, FINISH]
    memory, executor, brain, loop = setup(tmp_path,actions,max_errors=1)
    assert loop.run() == 'stopped'
    assert (tmp_path/'recovered.txt').read_text() == 'ready'
    state = memory.load_state()
    assert state.error_count == 3 and state.goal_reports == 1 and state.pending_action is None
    assert all(e['action_outcome_uncertain'] is False for e in events(memory) if e['event']=='cycle_error')


@pytest.mark.parametrize('terminal',['DONE','FAIL'])
def test_opencua_terminal_response_continues_next_cycle(tmp_path,terminal):
    memory,executor,brain,loop=setup(tmp_path,[WAIT]*2)
    calls=[]
    def handler(request):
        payload=json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200,json={'choices':[{'message':{'content':terminal if len(calls)==1 else 'WAIT'}}]})
    loop.brain=OpenCUABrain(transport=httpx.MockTransport(handler))
    try:
        assert loop.run()=='stopped'
    finally:
        loop.brain.close()
    assert len(calls)==2 and memory.load_state().goal_reports==1
    assert 'does not stop this process' in calls[1]['messages'][0]['content']


def test_read_failure_is_not_an_uncertain_side_effect(tmp_path):
    memory, executor, brain, loop = setup(tmp_path,[{'type':'read_file','args':{'path':'absent'}}, FINISH],max_errors=1)
    assert loop.run() == 'stopped'
    assert memory.load_state().goal_reports == 1
    assert not any(e['event']=='action_intent' for e in events(memory))


def test_partial_write_still_stops_without_replaying(tmp_path):
    action = {'type':'write_file','args':{'path':'partial.txt','content':'part'}}
    memory, executor, brain, loop = setup(tmp_path,[action]*4)
    def partial(*args):
        (tmp_path/'partial.txt').write_text('part')
        raise OSError('write interrupted')
    executor.execute = partial
    assert loop.run() == 'error' and len(brain.contexts) == 1
    assert memory.load_state().pending_action is not None


def test_repeated_append_suppressed_even_across_idle(tmp_path):
    action = {'type':'write_file','args':{'path':'repeat.txt','content':'x','append':True}}
    memory, executor, brain, loop = setup(tmp_path,[action]*3+[WAIT,action,action])
    assert loop.run() == 'stopped'
    assert (tmp_path/'repeat.txt').read_text() == 'xxx'
    rows = events(memory)
    assert sum(e['event']=='action_suppressed' for e in rows) == 2
    assert sum(e['event']=='action_intent' for e in rows) == 3
    assert 'suppressed' in brain.contexts[-1][1]
    assert memory.load_state().error_count == 0


def test_changed_observation_releases_repeat_guard(tmp_path):
    from PIL import Image
    action = {'type':'click','args':{'x':10,'y':10}}
    memory, executor, brain, loop = setup(tmp_path,[action]*5)
    executor.computer.execute = lambda action, observation: 'fixture click dispatched'
    original = executor.computer.observe
    def observe(cycle):
        observation = original(cycle)
        if cycle == 5:
            Image.new('RGB',(320,200),'white').save(observation.screenshot_path)
        return observation
    executor.computer.observe = observe
    assert loop.run() == 'stopped'
    assert sum(e['event'] == 'action_intent' for e in events(memory)) == 4


def test_unrelated_screen_changes_do_not_release_file_repeat_guard(tmp_path):
    from PIL import Image
    action = {'type':'write_file','args':{'path':'repeat.txt','content':'x','append':True}}
    memory, executor, brain, loop = setup(tmp_path,[action]*5)
    original = executor.computer.observe
    def observe(cycle):
        observation = original(cycle)
        Image.new('RGB',(320,200),(cycle,0,0)).save(observation.screenshot_path)
        return observation
    executor.computer.observe = observe
    assert loop.run() == 'stopped'
    assert (tmp_path/'repeat.txt').read_text() == 'xxx'
    assert sum(e['event'] == 'action_suppressed' for e in events(memory)) == 2


def test_idle_backoff_is_capped_and_resets_on_changed_image(tmp_path):
    computer = MockComputer(tmp_path)
    policy = AutonomyPolicy(idle_initial_seconds=2,idle_max_seconds=5)
    policy.observe(computer.observe(1))
    assert [policy.idle_delay() for _ in range(4)] == [2,4,5,5]
    policy.observe(computer.observe(2))  # Different path is not an environmental change.
    assert policy.idle_delay() == 5
    observation = computer.observe(3)
    observation.active_window = 'another window'
    policy.observe(observation)
    assert policy.idle_delay() == 2
    assert policy.recovery_delay(1000) == 300


def test_long_model_wait_obeys_session_time_budget(tmp_path):
    memory, executor, brain, loop = setup(tmp_path,[{'type':'wait','args':{'seconds':30}}]*2,max_seconds=.08)
    executor.execute = lambda *args:pytest.fail('wait must be scheduled, not dispatched')
    before = time.monotonic()
    assert loop.run() == 'stopped' and time.monotonic()-before < 2
    assert memory.load_state().stop_reason == 'Time budget reached'
    assert memory.load_state().next_wake_at is None


@pytest.mark.parametrize('phase', ['idle','recovering'])
def test_pause_resume_and_stop_during_long_backoff(tmp_path,phase):
    action = WAIT if phase=='idle' else RuntimeError('service down')
    memory, executor, brain, loop = setup(tmp_path,[action]*20,max_steps=0,max_errors=1,
        policy=AutonomyPolicy(30,60,30,60))
    failures=[]
    def run():
        try:loop.run()
        except BaseException as exc:failures.append(exc)
    worker=threading.Thread(target=run,daemon=True)
    worker.start()
    try:
        state=await_state(memory,lambda s:s.cycle>=1 and s.status==phase)
        assert state.next_wake_at
        assert request_control(memory.root,'pause') == 0
        paused=await_state(memory,lambda s:s.status=='paused')
        time.sleep(.05)
        assert memory.load_state().cycle == paused.cycle
        assert request_control(memory.root,'resume') == 0
        await_state(memory,lambda s:s.cycle>paused.cycle and s.status==phase)
        assert request_control(memory.root,'stop') == 0
        worker.join(timeout=3)
        assert not worker.is_alive() and not failures
        assert memory.load_state().stop_reason == 'Stop requested'
    finally:
        if worker.is_alive():
            loop.control.request('stop')
            worker.join(timeout=3)


def test_run_has_no_task_prompt_and_defaults_to_execution(monkeypatch,tmp_path):
    args=build_parser().parse_args(['run'])
    assert args.execute is True and not hasattr(args,'task')
    assert build_parser().parse_args(['run','--observe']).execute is False
    monkeypatch.setattr('builtins.input',lambda *args:pytest.fail('No conversation input is allowed'))
    assert main(['--home',str(tmp_path),'run','--mock','--max-steps','1','--quiet']) == 0
    state=FileMemory(tmp_path).load_state()
    assert state.loop_mode=='continuous' and state.task is None


def test_observe_only_continuous_never_writes_or_claims_verified_success(tmp_path):
    actions=[{'type':'write_file','args':{'path':'absent','content':'no'}},FINISH,FINISH]
    memory,executor,brain,loop=setup(tmp_path,actions,dry_run=True)
    assert loop.run()=='stopped' and not (tmp_path/'absent').exists()
    assert memory.load_state().last_goal_report['dry_run'] is True
    assert memory.load_state().last_goal_report['verified'] is False


@pytest.mark.parametrize('settings',[
    {'idle_initial_seconds':0}, {'idle_max_seconds':float('inf')},
    {'idle_initial_seconds':10,'idle_max_seconds':1},
    {'recovery_initial_seconds':20,'recovery_max_seconds':10},
    {'repeat_action_limit':True}, {'repeat_action_limit':0},
])
def test_invalid_pacing_config_rejected(settings):
    with pytest.raises(ConfigError):
        _merge_dataclass(ConwayConfig(),settings)


def test_sigterm_handler_is_restored():
    before=signal.getsignal(signal.SIGTERM)
    with pytest.raises(EmergencyStop,match='Termination'):
        with termination_signals():
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    assert signal.getsignal(signal.SIGTERM)==before


def test_stop_during_memory_compaction_is_not_swallowed(tmp_path):
    memory,executor,brain,loop=setup(tmp_path,[WAIT]*3)
    memory.memory_path.write_text('large memory '*10000,encoding='utf-8')
    def stop(*args):
        raise EmergencyStop('Termination requested')
    brain.compact_memory=stop
    assert loop.run()=='stopped'
    assert memory.load_state().stop_reason=='Termination requested'
    assert len(brain.contexts)==1


@pytest.mark.skipif(os.name!='posix',reason='Windows terminate is not a cooperative SIGTERM')
def test_unattended_process_without_stdin_stops_cleanly_on_sigterm(tmp_path):
    root=tmp_path/'state'
    memory=FileMemory(root)
    process=subprocess.Popen([sys.executable,'-m','conway','--home',str(root),'run','--mock','--quiet'],
                             stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        await_state(memory,lambda s:s.cycle>=1 and s.status=='idle')
        process.terminate()
        stdout,stderr=process.communicate(timeout=5)
        assert process.returncode==0,stderr.decode()
        state=memory.load_state()
        assert state.status=='stopped' and state.pid is None and state.next_wake_at is None
        assert state.stop_reason=='Termination requested'
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
