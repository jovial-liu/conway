from pathlib import Path

import pytest

from conway.brain import Brain, Decision
from conway.computer import MockComputer
from conway.config import ToolSettings
from conway.control import EmergencyStop
from conway.loop import ConwayLoop
from conway.memory import FileMemory
from conway.tools import ToolExecutor


WAIT = {'type': 'wait', 'args': {'seconds': 0}}


def setup(tmp_path, brain, **kwargs):
    memory = FileMemory(tmp_path/'state')
    executor = ToolExecutor(MockComputer(memory.root/'screens'), ToolSettings(gui=False, shell=False, open_url=False))
    return memory, ConwayLoop(brain, executor, memory, interval=0, max_steps=2, quiet=True, **kwargs)


def test_goals_are_reread_and_existing_markdown_is_preserved(tmp_path):
    contexts = []
    class Policy(Brain):
        def decide(self, constitution, memory, observation):
            contexts.append(constitution)
            if len(contexts) == 1:
                store.goals_path.write_text('Second objective: inspect result B.', encoding='utf-8')
            return Decision({'type': 'read_file', 'args': {'path': str(store.goals_path)}})
    store, loop = setup(tmp_path, Policy(), continuous=True)
    store.goals_path.write_text('First objective: inspect result A.', encoding='utf-8')
    original = store.constitution()
    assert FileMemory(store.root).goals() == 'First objective: inspect result A.'
    loop.run()
    assert 'First objective: inspect result A.' in contexts[0]
    assert 'Second objective: inspect result B.' in contexts[1]
    assert store.constitution() == original


def test_session_task_does_not_inherit_ongoing_goals(tmp_path):
    store = FileMemory(tmp_path)
    store.goals_path.write_text('Ongoing goal only', encoding='utf-8')
    assert 'Ongoing goal only' not in store.instructions(task='Single acceptance goal')
    assert 'Single acceptance goal' in store.instructions(task='Single acceptance goal')
    store.goals_path.write_text('x'*8001, encoding='utf-8')
    with pytest.raises(ValueError, match='8000'):
        store.instructions(continuous=True)
    store.goals_path.unlink()
    assert store.goals() == ''


def test_feedback_has_real_action_result_and_next_observation(tmp_path):
    events = []
    class Policy(Brain):
        def decide(self, *args):
            events.append('decide')
            return Decision(WAIT)
        def feedback(self, transition):
            events.append(transition)
            assert transition.observation.screenshot_path.is_file()
            assert transition.task_success is None
    _, loop = setup(tmp_path, Policy())
    loop.run()
    first, last = events[1], events[3]
    assert events[0] == events[2] == 'decide'
    assert first.next_observation is not None and not first.terminal
    assert first.cycle == 1 and first.result == 'waited 0.00s'
    assert last.next_observation is None and last.terminal
    assert last.operation_id != first.operation_id


def test_dry_run_does_not_feed_a_learner(tmp_path):
    class Policy(Brain):
        def decide(self, *args): return Decision(WAIT)
        def feedback(self, transition): pytest.fail('planned actions must not become learning transitions')
    _, loop = setup(tmp_path, Policy(), dry_run=True)
    loop.run()


def test_feedback_failure_never_reexecutes_previous_action(tmp_path):
    calls = []
    class Policy(Brain):
        def decide(self, *args):
            calls.append('decide')
            return Decision(WAIT)
        def feedback(self, transition):
            calls.append('feedback')
            raise ValueError('fixture learning error')
    memory, loop = setup(tmp_path, Policy())
    assert loop.run() == 'stopped'
    assert calls == ['decide', 'feedback']
    assert memory.load_state().pending_action is None
    assert 'feedback failed' in memory.load_state().stop_reason


def test_explicit_stop_does_not_call_terminal_learning_hook(tmp_path):
    calls = []
    class Policy(Brain):
        def decide(self, *args): return Decision(WAIT)
        def feedback(self, transition): calls.append(transition)
    memory, loop = setup(tmp_path, Policy())
    def stop(*args): raise EmergencyStop('Owner stopped')
    loop._sleep = stop
    assert loop.run() == 'stopped'
    assert calls == []
