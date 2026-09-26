"""Protocol and scoring fixtures; these tests make no claims about real VLM accuracy."""
import base64
from contextlib import contextmanager
import io
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import httpx
from PIL import Image
import pytest

from conway.brain import Brain, Decision, OpenAICompatibleVLM, OpenCUABrain
from conway.cli import main
from conway.computer import MockComputer
from conway.config import ConwayConfig, ToolSettings
from conway.evaluation import COLORS, make_cases, run_grounding, vision_check
from conway.hardware import HardwareProfile
from conway.loop import ConwayLoop
from conway.memory import FileMemory
from conway.reports import write_report
from conway.tools import ToolExecutor


class SequenceBrain(Brain):
    def __init__(self, actions):
        self.actions = iter(actions)
    def decide(self, *args):
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        return Decision(action)


def center(case):
    l, t, r, b = case.bounds
    return {'type': 'click', 'args': {'x': (l + r) // 2, 'y': (t + b) // 2}}


def test_seeded_images_scoring_misses_and_errors(tmp_path):
    cases = make_cases(tmp_path / 'a', samples=4, seed=19)
    duplicate = make_cases(tmp_path / 'b', samples=4, seed=19)
    different = make_cases(tmp_path / 'c', samples=4, seed=20)
    assert [c.image_sha256 for c in cases] == [c.image_sha256 for c in duplicate]
    assert [c.image_sha256 for c in cases] != [c.image_sha256 for c in different]
    report = run_grounding(SequenceBrain([center(cases[0]), {'type':'click','args':{'x':0,'y':0}},
                                         {'type':'wait','args':{}}, ValueError('private fixture detail')]), cases)
    assert report['hits'] == 1 and report['hit_rate'] == .25
    assert report['valid_clicks'] == 2 and report['errors'] == 1
    assert report['actions_executed'] == 0 and report['real_desktop_tested'] is False
    assert 'private fixture detail' not in json.dumps(report)


@pytest.mark.parametrize('samples,seed', [(0,0),(101,0),(True,0),(1,-1),(1,2**32),(1,False)])
def test_case_bounds_rejected(tmp_path,samples,seed):
    with pytest.raises(ValueError):
        make_cases(tmp_path,samples=samples,seed=seed)


@pytest.mark.parametrize('brain_class',[OpenAICompatibleVLM,OpenCUABrain])
def test_image_only_fixture_through_real_adapter(tmp_path,brain_class):
    cases = make_cases(tmp_path,samples=6,seed=41)
    calls = []
    def serve(request):
        payload = json.loads(request.content)
        target = next(color for color in COLORS if f'{color} rectangle' in payload['messages'][0]['content'])
        url = payload['messages'][1]['content'][0]['image_url']['url']
        image = Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))).convert('RGB')
        rgb = tuple(bytes.fromhex(COLORS[target][1:]))
        # Independent pixel oracle in the HTTP fixture; no access to case.bounds.
        matches = [(x,y) for y in range(image.height) for x in range(image.width) if image.getpixel((x,y)) == rgb]
        x = (min(p[0] for p in matches) + max(p[0] for p in matches)) // 2
        y = (min(p[1] for p in matches) + max(p[1] for p in matches)) // 2
        calls.append(payload)
        if brain_class is OpenCUABrain:
            from conway.opencua import smart_resize
            h,w = smart_resize(image.height,image.width)
            content = f'pyautogui.click({x*w/image.width}, {y*h/image.height})'
        else:
            content = json.dumps({'action':{'type':'click','args':{'x':x,'y':y}}})
        return httpx.Response(200,json={'choices':[{'message':{'content':content}}]})
    brain = brain_class(transport=httpx.MockTransport(serve),
                        tool_manifest='- click: {"x": 100, "y": 100}')
    try:
        report = run_grounding(brain,cases)
    finally:
        brain.close()
    assert report['hits'] == 6 and len(calls) == 6
    for case,call in zip(cases,calls):
        assert 'target_bounds' not in json.dumps(call)
        assert str(list(case.bounds)) not in json.dumps(call)


def test_vision_report_metadata_and_cli_failure(tmp_path,monkeypatch,capsys):
    from conway import evaluation
    @contextmanager
    def session(*a,**kw):
        yield SequenceBrain([{'type':'wait','args':{}}]*2), {'model':'fixture-only','brain':'generic','profile':'external'}
    monkeypatch.setattr(evaluation,'model_session',session)
    root = tmp_path/'state'
    output = tmp_path/'report.json'
    assert main(['--home',str(root),'vision-check','--samples','2','--output',str(output)]) == 2
    report = json.loads(output.read_text())
    assert report['status'] == 'failed' and report['model_revision'] is None
    assert report['model'] == 'fixture-only' and report['seed'] == 0
    assert report['samples'] == 2 and not list(root.glob('screenshots/*.png'))
    assert json.loads(capsys.readouterr().out)['hits'] == 0
    assert main(['--home',str(root),'vision-check','--output',str(output)]) == 1


def test_report_never_overwrites_or_writes_state(tmp_path):
    root = tmp_path/'state'
    root.mkdir()
    target = tmp_path/'keep.json'
    target.write_text('original')
    for path in (target,root/'new.json'):
        with pytest.raises(ValueError):
            write_report(path,{'ok':True},root)
    assert target.read_text() == 'original'
    assert not (root/'new.json').exists()


def test_model_session_cleanup_on_failure(tmp_path,monkeypatch):
    from conway import diagnostics
    closed=[]
    monkeypatch.setattr(diagnostics,'connect_external',lambda *a,**kw:SimpleNamespace(
        served_model='fixture',model_repo='fixture',base_url='http://localhost/v1',profile_name='external',
        close=lambda:closed.append('runtime')))
    class BrokenBrain:
        def __init__(self,*a,**kw): pass
        def decide(self,*a): raise ValueError('bad model')
        def close(self): closed.append('brain')
    monkeypatch.setattr(diagnostics,'OpenAICompatibleVLM',BrokenBrain)
    report=vision_check(ConwayConfig(endpoint='http://localhost/v1'),tmp_path,samples=2)
    assert report['status']=='failed' and report['errors']==2
    assert closed==['brain','runtime']


def test_task_reaches_every_cycle_and_does_not_replace_constitution(tmp_path):
    memory=FileMemory(tmp_path)
    original=memory.constitution()
    seen=[]
    class TaskBrain(Brain):
        def decide(self,constitution,*args):
            seen.append(constitution)
            return Decision({'type':'wait','args':{'seconds':0}})
    executor=ToolExecutor(MockComputer(tmp_path/'screenshots'),ToolSettings())
    loop=ConwayLoop(TaskBrain(),executor,memory,dry_run=True,max_steps=2,interval=0,quiet=True,task='创建中文验收文件')
    assert loop.run()=='stopped'
    assert len(seen)==2 and all(s.startswith(original) and '创建中文验收文件' in s for s in seen)
    assert memory.constitution()==original and memory.load_state().task=='创建中文验收文件'
    assert 'session_task' in memory.recent_events()
    assert main(['--home',str(tmp_path),'start','--mock','--max-steps','1','--quiet'])==0
    assert memory.load_state().task is None


@pytest.mark.parametrize('task',['','  ','x'*16001,'x\x00y'])
def test_invalid_task_fails_before_state_creation(tmp_path,task):
    root=tmp_path/'state'
    assert main(['--home',str(root),'start','--mock','--task',task,'--max-steps','1'])==1
    assert not root.exists()


def test_utf8_task_file(tmp_path):
    path=tmp_path/'task.md'
    path.write_text('核对目标文件\n保存测试结果',encoding='utf-8')
    root=tmp_path/'state'
    assert main(['--home',str(root),'start','--mock','--task-file',str(path),'--max-steps','1','--quiet'])==0
    assert FileMemory(root).load_state().task=='核对目标文件\n保存测试结果'


def setup_preflight(monkeypatch):
    from conway import preflight
    monkeypatch.setattr(preflight,'detect_hardware',lambda:HardwareProfile('Linux','x86_64',32,'cpu',None,12))
    monkeypatch.setattr(preflight,'find_runtime',lambda path:sys.executable)
    monkeypatch.setattr(preflight.importlib.util,'find_spec',lambda name:object())
    return preflight


def test_preflight_does_not_claim_unchecked_desktop(tmp_path,monkeypatch):
    module=setup_preflight(monkeypatch)
    monkeypatch.setattr(module,'check_desktop',lambda config:pytest.fail('unexpected desktop read'))
    report=module.preflight(ConwayConfig(),tmp_path)
    assert report['status']=='passed' and report['ready_for_observation'] is False
    assert report['model_inference_tested'] is False and report['actions_executed']==0
    assert next(c for c in report['checks'] if c['name']=='desktop')['status']=='not_checked'


def test_preflight_missing_key_and_wayland_block(tmp_path,monkeypatch):
    module=setup_preflight(monkeypatch)
    monkeypatch.delenv('CONWAY_TEST_EMPTY_KEY',raising=False)
    monkeypatch.setenv('XDG_SESSION_TYPE','wayland')
    monkeypatch.setattr(module,'connect_external',lambda *a,**kw:pytest.fail('unexpected connection'))
    report=module.preflight(ConwayConfig(endpoint='http://localhost/v1',api_key_env='CONWAY_TEST_EMPTY_KEY'),tmp_path,desktop=True)
    assert report['status']=='blocked' and not report['ready_for_observation']
    assert {c['name'] for c in report['checks'] if c['status']=='failed'}=={'model_service','desktop'}


def test_desktop_worker_timeout_is_bounded(monkeypatch):
    from conway import preflight
    def timeout(*a,**kw):
        assert 15 < kw['timeout'] <= 25
        raise subprocess.TimeoutExpired(a[0],kw['timeout'])
    monkeypatch.setattr(preflight.subprocess,'run',timeout)
    assert preflight.check_desktop(ConwayConfig())['status']=='failed'


def test_desktop_worker_removes_capture_and_private_metadata(monkeypatch):
    from conway.preflight import desktop_worker
    from conway import computer
    paths=[]
    class FixtureComputer(MockComputer):
        def __init__(self,path,**kwargs):
            super().__init__(path)
        def observe(self,cycle):
            obs=super().observe(cycle)
            paths.append(obs.screenshot_path)
            obs.active_window='PRIVATE WINDOW TITLE'
            obs.ui_tree={'status':'ok','nodes':[{'name':'private ui label'}]}
            return obs
    monkeypatch.setattr(computer,'Computer',FixtureComputer)
    result=desktop_worker({'ui_tree':True,'ui_timeout':2,'ui_max_nodes':80})
    assert result['screenshot_size']==[320,200]
    assert result['input_control_verified'] is False
    assert 'PRIVATE' not in json.dumps(result) and 'private ui label' not in json.dumps(result)
    assert all(not p.exists() for p in paths)
