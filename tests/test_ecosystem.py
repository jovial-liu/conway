import json
from pathlib import Path
import sys
import threading
import time

import psutil
import pytest

from conway.actions import SIDE_EFFECT_ACTIONS, validate_action
from conway.config import ConfigError, MCPServerConfig, ConwayConfig, _merge_dataclass
from conway.control import EmergencyStop
from conway.mcp_bridge import MCPBridge
from conway.skills import SkillRegistry
from conway.tools import ToolExecutor
from conway.computer import MockComputer


def write_skill(root, name='verify-output', body='Read the artifact and check its content.'):
    path = root / name / 'SKILL.md'
    path.parent.mkdir(parents=True)
    path.write_text(f'---\nname: {name}\ndescription: Verify task output before reporting success.\n---\n{body}', encoding='utf-8')
    return path


def test_skill_progressive_loading_resources_and_permissions(tmp_path):
    write_skill(tmp_path, body='BODY-ONLY: never claim success before inspection.\n' + 'x' * 13000)
    registry = SkillRegistry([tmp_path])
    executor = ToolExecutor(MockComputer(tmp_path / 'screens'), ConwayConfig().tools, skills=registry)
    assert 'verify-output' in executor.manifest_text()
    assert 'BODY-ONLY' not in executor.manifest_text()
    loaded = executor.execute({'type': 'read_skill', 'args': {'name': 'verify-output'}}, executor.computer.observe(1))
    assert 'BODY-ONLY' in loaded and 'next_offset: 4000' in loaded
    assert len(loaded) < 16000
    assert 'next_offset: None' in registry.read('verify-output', offset=12000)
    (tmp_path / 'verify-output' / 'reference.md').write_text('Supporting evidence.', encoding='utf-8')
    assert 'Supporting evidence.' in registry.read('verify-output', 'reference.md')
    assert 'read_skill' not in SIDE_EFFECT_ACTIONS
    assert 'mcp_call' in SIDE_EFFECT_ACTIONS


@pytest.mark.parametrize('name', ['Uppercase', 'bad--name', '-bad', 'bad-'])
def test_invalid_skill_names(tmp_path, name):
    write_skill(tmp_path, name)
    with pytest.raises(ValueError, match='name'):
        SkillRegistry([tmp_path])


def test_skill_escapes_duplicates_and_duplicate_yaml(tmp_path):
    write_skill(tmp_path / 'a')
    write_skill(tmp_path / 'b')
    with pytest.raises(ValueError, match='Duplicate skill'):
        SkillRegistry([tmp_path / 'a', tmp_path / 'b'])
    registry = SkillRegistry([tmp_path / 'a'])
    with pytest.raises(ValueError, match='within'):
        registry.read('verify-output', '../../secret')
    path = tmp_path / 'a/verify-output/SKILL.md'
    path.write_text('---\nname: verify-output\nname: another\ndescription: test\n---\n')
    with pytest.raises(ValueError, match='Duplicate'):
        SkillRegistry([tmp_path / 'a'])


@pytest.mark.parametrize('settings', [
    {'command': 'python'}, {'command': 'python', 'allowed_tools': ['*']},
    {'command': 'python', 'allowed_tools': ['a'], 'env': {'SECRET': 'oops'}},
    {'command': 'python', 'allowed_tools': ['a'], 'timeout': True},
    {'command': 'python', 'allowed_tools': ['a'], 'env_keys': ['BAD=KEY']},
])
def test_mcp_invalid_config(settings):
    with pytest.raises(ConfigError):
        _merge_dataclass(ConwayConfig(), {'mcp_servers': {'fixture': settings}})


def bridge(tmp_path, tools=None, timeout=5, env_keys=None):
    pytest.importorskip('mcp')
    return MCPBridge({'fixture': MCPServerConfig(sys.executable,
        [str(Path(__file__).with_name('mcp_fixture.py')), str(tmp_path / 'pid')],
        tools or ['increment'], env_keys or [], timeout)})


def assert_child_stopped(pid):
    deadline = time.monotonic() + 5
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(.02)
    assert not psutil.pid_exists(pid)


def test_real_mcp_discovery_stateful_calls_schema_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv('CONWAY_MCP_TEST_SECRET', 'not-inherited')
    client = bridge(tmp_path)
    try:
        client.start()
        pid = int((tmp_path / 'pid').read_text())
        assert 'increment' in client.manifest() and 'forbidden' not in client.manifest()
        one = json.loads(client.call('fixture', 'increment', {'amount': 1}))
        two = json.loads(client.call('fixture', 'increment', {'amount': 2}))
        assert one['structuredContent'] == {'count': 1, 'secret_present': False}
        assert two['structuredContent']['count'] == 3
        with pytest.raises(ValueError, match='schema'):
            client.call('fixture', 'increment', {'amount': 'bad'})
        with pytest.raises(ValueError, match='allowlist'):
            client.call('fixture', 'forbidden', {})
    finally:
        client.close()
    assert_child_stopped(pid)


def test_real_mcp_timeout_never_replays_and_cleans_up(tmp_path):
    client = bridge(tmp_path, ['slow'], timeout=2)
    try:
        client.start()
        pid = int((tmp_path / 'pid').read_text())
        with pytest.raises(RuntimeError, match='uncertain'):
            client.call('fixture', 'slow', {})
    finally:
        client.close()
    assert_child_stopped(pid)


def test_real_mcp_stop_interrupts_inflight_call(tmp_path):
    client = bridge(tmp_path, ['slow'], timeout=30)
    stop = threading.Event()
    class Control:
        def check(self):
            if stop.is_set():
                raise EmergencyStop('Test stop')
    try:
        client.start()
        pid = int((tmp_path / 'pid').read_text())
        timer = threading.Timer(.2, stop.set)
        timer.start()
        with pytest.raises(EmergencyStop):
            client.call('fixture', 'slow', {}, Control())
        timer.join()
    finally:
        client.close()
    assert_child_stopped(pid)


def test_real_mcp_explicit_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('CONWAY_MCP_TEST_SECRET', 'explicitly-passed')
    client = bridge(tmp_path, env_keys=['CONWAY_MCP_TEST_SECRET'])
    try:
        client.start()
        assert json.loads(client.call('fixture', 'increment', {'amount': 0}))['structuredContent']['secret_present']
    finally:
        client.close()


def test_mcp_schema_cannot_resolve_remote_refs():
    from types import SimpleNamespace
    client = MCPBridge({})
    with pytest.raises(ValueError, match='external'):
        client._register('s', SimpleNamespace(name='t', description='', input_schema={'$ref': 'https://example.com/schema'}))
    with pytest.raises(ValueError):
        validate_action({'type': 'mcp_call', 'args': {'server': 's', 'tool': 't', 'arguments': {'bad': float('nan')}}}, 10, 10)


def test_autonomous_loop_loads_skill_calls_mcp_and_continues_past_goal(tmp_path):
    from conway.autonomy import AutonomyPolicy
    from conway.brain import Brain, Decision
    from conway.loop import ConwayLoop
    from conway.memory import FileMemory
    write_skill(tmp_path/'skills', body='Check the counter after changing it.')
    client = bridge(tmp_path)
    memory = FileMemory(tmp_path/'state')
    class Policy(Brain):
        def __init__(self):
            self.contexts = []
            self.actions = iter([
                {'type': 'read_skill', 'args': {'name': 'verify-output'}},
                {'type': 'mcp_call', 'args': {'server': 'fixture', 'tool': 'increment', 'arguments': {'amount': 1}}},
                {'type': 'finish', 'args': {'reason': 'first subgoal'}},
                {'type': 'mcp_call', 'args': {'server': 'fixture', 'tool': 'increment', 'arguments': {'amount': 2}}},
            ])
        def decide(self, constitution, recalled, observation):
            self.contexts.append(recalled)
            return Decision(next(self.actions))
    policy = Policy()
    try:
        client.start()
        executor = ToolExecutor(MockComputer(memory.root/'screens'), ConwayConfig().tools,
            mcp=client, skills=SkillRegistry([tmp_path/'skills']))
        loop = ConwayLoop(policy, executor, memory, continuous=True, max_steps=4, interval=0, quiet=True,
                          policy=AutonomyPolicy(.001, .002, .001, .002))
        assert loop.run() == 'stopped'
        state = memory.load_state()
        assert state.pending_action is None and state.cycle == 4 and state.goal_reports == 1
        assert json.loads(state.last_result)['structuredContent']['count'] == 3
        assert 'Check the counter after changing it.' in policy.contexts[1]
    finally:
        client.close()
