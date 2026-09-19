"""Regression tests for configuration, persistence and fail-closed parsing."""
from pathlib import Path
import json
import math
import pytest
from conway.actions import validate_action, ActionValidationError
from conway.config import load_config, ensure_config, endpoint_url, ConfigError
from conway.memory import FileMemory, RuntimeState
from conway.storage import atomic_write, read_tail
from conway.lock import InstanceLock
from conway.opencua import parse_opencua_action, smart_resize, model_point_to_screenshot, OpenCUAParseError
from conway.brain import OpenAICompatibleVLM


@pytest.mark.parametrize('body', [
    'tools:\n  shell: "false"', 'ui_tree: "false"', 'interval: .nan', 'port: true',
    'profile: []', 'brain: unknown', 'tools: []', 'tools:\n  gui: 0',
    'port: 3.5', 'request_timeout: -1', 'max_errors: 0', 'port: 123\nport: 124',
    'tools:\n  typo: true', 'typo: true', 'endpoint: file:///tmp/server',
    'opencua_min_pixels: 10000\nopencua_max_pixels: 9000', '[1, 2]',
])
def test_invalid_config_is_rejected(tmp_path, body):
    (tmp_path / 'config.yaml').write_text(body, encoding='utf-8')
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_config_valid_and_missing_defaults(tmp_path):
    ensure_config(tmp_path)
    (tmp_path / 'config.yaml').write_text('profile: tiny\ninterval: 1.25\ntools:\n  shell: false\n  max_output_chars: 4321\n')
    c = load_config(tmp_path)
    assert c.tools.shell is False and c.tools.filesystem is True
    assert c.profile == 'tiny' and c.interval == 1.25 and c.tools.max_output_chars == 4321


@pytest.mark.parametrize('url', ['https://name:secret@host/v1', 'http://host/v1?key=x', 'http://host/#x', 'http://a:bad', 'http:// a'])
def test_reject_endpoint_credentials_and_bad_urls(url):
    with pytest.raises(ValueError):
        endpoint_url(url)


def test_endpoint_normalization():
    assert endpoint_url('http://localhost:8042/') == 'http://localhost:8042/v1'
    assert endpoint_url('https://host/prefix/v1/') == 'https://host/prefix/v1'


@pytest.mark.parametrize('kind,args', [
    ('click', {'x': math.inf, 'y': 2}), ('click', {'x': True, 'y': 2}),
    ('wait', {'seconds': math.nan}), ('type', {'text': None}),
    ('write_file', {'path': 'x', 'content': 'a', 'append': 'false'}),
    ('shell', {'command': 'echo hello', 'timeout': math.inf}),
    ('open_url', {'url': 'https://a:b@example.com'}), ('press', {'key': 'a\x00'}),
])
def test_invalid_action_arguments(kind, args):
    with pytest.raises(ActionValidationError):
        validate_action({'type': kind, 'args': args}, 100, 100)


def test_args_array_not_silently_coerced():
    with pytest.raises(ActionValidationError):
        validate_action({'type': 'wait', 'args': []}, 100, 100)


def test_original_action_contract():
    assert validate_action({'type': 'click', 'args': {'x': 999, 'y': -5}}, 100, 80)['args'] == {'x': 99, 'y': 0, 'button': 'left'}
    assert validate_action({'type': 'hotkey', 'args': {'keys': ['CTRL', 'L']}}, 100, 80)['args']['keys'] == ['ctrl', 'l']


@pytest.mark.parametrize('source', [
    'pyautogui.click(1, 2); pyautogui.click(3, 4)',
    'pyautogui.click(x=make_x(), y=3)',
    'pyautogui.click(**{"x": 1, "y": 2})',
    'other.click(1, 2)', 'import os\npyautogui.click(1, 2)',
    'pyautogui.click(1, 2, arbitrary=True)', 'pyautogui.click(x=1, x=2, y=3)',
])
def test_opencua_rejects_ambiguous_or_executable_output(source):
    with pytest.raises(OpenCUAParseError):
        parse_opencua_action(source, 100, 100)


def test_opencua_geometry_and_controls():
    h, w = smart_resize(1080, 1920)
    assert h % 28 == w % 28 == 0
    assert model_point_to_screenshot(w / 2, h / 2, 1920, 1080) == (960, 540)
    assert parse_opencua_action(f'```python\npyautogui.click({w / 2}, {h / 2})\n```', 1920, 1080)['args']['x'] == 960
    assert parse_opencua_action('time.sleep(0.1)', 100, 100)['type'] == 'wait'
    assert parse_opencua_action('DONE', 100, 100)['type'] == 'finish'
    assert parse_opencua_action("pyautogui.write('hello')", 100, 100) == {'type': 'type', 'args': {'text': 'hello', 'interval': .01}}


@pytest.mark.parametrize('source', ['{"a":1,"a":2}', '{"x":NaN}', '{} {}', '[]', '{} trailing', '{"x": 1,}'])
def test_json_ambiguity_rejected(source):
    with pytest.raises(ValueError):
        OpenAICompatibleVLM._extract_json(source)


def test_json_fence_and_think_and_prefix():
    for source in ['result: {"ok":1}', '```json\n{"ok":1}\n```', '<think>not the answer</think>{"ok":1}']:
        assert OpenAICompatibleVLM._extract_json(source) == {'ok': 1}


def test_atomic_replace_and_tail(tmp_path):
    target = tmp_path / 'nested' / 'note'
    atomic_write(target, 'before')
    atomic_write(target, 'after\nline2\nline3\n')
    assert target.read_text() == 'after\nline2\nline3\n'
    assert read_tail(target, 12).endswith('line3\n')
    assert not list(target.parent.glob('*.tmp'))


def test_state_roundtrip_and_corruption_recovery(tmp_path):
    mem = FileMemory(tmp_path)
    state = RuntimeState(cycle=42, pending_action={'id': 'uncertain'})
    mem.save_state(state)
    state.cycle = 43
    mem.save_state(state)
    mem.state_path.write_text('broken{')
    recovered = mem.load_state()
    assert recovered.cycle == 42 and recovered.status == 'recovery_required'
    assert recovered.pending_action == {'id': 'uncertain'}
    mem.save_state(recovered)
    assert list(tmp_path.glob('state.corrupt.*.txt'))
    assert mem.load_state().cycle == 42


def test_bounded_memory_and_partial_journal(tmp_path):
    mem = FileMemory(tmp_path)
    mem.append_memory('durable fact')
    mem.append_memory('durable fact')
    assert mem.memory_path.read_text().count('durable fact') == 1
    mem.journal({'event': 'action_result', 'cycle': 2, 'result': 'ok'})
    path = next(mem.journal_dir.glob('*.jsonl'))
    with path.open('a') as f:
        f.write('{unfinished')
    assert '"cycle": 2' in mem.recent_events()
    assert len(mem.recall(1000)) <= 1000
    mem.memory_path.write_text('z' * 9000)
    mem.compact_if_needed(max_bytes=8000)
    assert mem.memory_bytes() < 8000


def test_large_journal_events_still_return_complete_json(tmp_path):
    mem = FileMemory(tmp_path)
    mem.journal({'cycle': 1, 'result': 'x' * 20000})
    for line in mem.recent_events(max_chars=1000).splitlines():
        assert isinstance(json.loads(line), dict)


def test_lock_contention_and_reacquire(tmp_path):
    first = InstanceLock(tmp_path / 'instance.lock')
    second = InstanceLock(tmp_path / 'instance.lock')
    with first:
        with pytest.raises(RuntimeError):
            second.acquire()
    with second:
        assert second.file is not None


def test_opencua_fail_and_doubleclick_interval():
    assert parse_opencua_action('FAIL',100,100)['args']['outcome']=='failed'
    action=parse_opencua_action("pyautogui.doubleClick(20, 30, 0.1, 'right')",100,100)
    assert action['args']['interval']==.1 and action['args']['button']=='right'


def test_memory_keeps_both_goals_and_newest_notes_when_bounded(tmp_path):
    mem=FileMemory(tmp_path)
    mem.memory_path.write_text('ORIGINAL GOAL\n'+'middle history\n'*1000+'NEWEST FACT\n')
    context=mem.recall(2000)
    assert 'ORIGINAL GOAL' in context and 'NEWEST FACT' in context
    assert 'omitted' in context


def test_append_after_partial_journal_preserves_next_event(tmp_path):
    mem=FileMemory(tmp_path)
    mem.journal({'cycle':1,'result':'first'})
    path=next(mem.journal_dir.glob('*.jsonl'))
    with path.open('a') as f:
        f.write('{broken')
    mem.journal({'cycle':2,'result':'second'})
    assert 'second' in mem.recent_events() and 'first' in mem.recent_events()
