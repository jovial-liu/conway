import json
from pathlib import Path
from types import SimpleNamespace
import httpx
import pytest
from conway.brain import OpenAICompatibleVLM, OpenCUABrain
from conway.computer import MockComputer
from conway.runtime import connect_external, model_ids, _ready, RuntimeHandle
from conway.hardware import HardwareProfile, choose_model_profile, InsufficientMemoryError
from conway.cli import _brain_type


def transport(status=200, data=None):
    return httpx.MockTransport(lambda request: httpx.Response(status, json=data))


@pytest.mark.parametrize('code', [400, 401, 403, 404, 500])
def test_bad_status_never_ready(code):
    with pytest.raises(RuntimeError):
        connect_external('http://localhost:8042', transport=transport(code, {'error': 'not ready'}))


@pytest.mark.parametrize('data', [{}, {'data': []}, {'data': [None]}, {'data': 'wrong'}])
def test_invalid_model_list_rejected(data):
    with pytest.raises(RuntimeError):
        connect_external('http://localhost:8042', transport=transport(data=data))


def test_served_alias_is_discovered_and_checked():
    t = transport(data={'data': [{'id': 'alias'}]})
    handle = connect_external('http://localhost:8042/', transport=t)
    assert handle.served_model == 'alias' and handle.base_url.endswith('/v1')
    with pytest.raises(ValueError):
        connect_external('http://localhost:8042', 'wrong', transport=t)
    with pytest.raises(ValueError):
        connect_external('http://localhost:8042', transport=transport(data={'data': [{'id': 'a'}, {'id': 'b'}]}))


def test_404_is_not_ready(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('invalid /models')
    monkeypatch.setattr('conway.runtime.model_ids', fail)
    assert _ready('http://localhost:8042') is False


def test_no_retry_permanent_client_error():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={'error': 'private secret not echoed'})
    brain = OpenAICompatibleVLM(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError, match='HTTP 401') as exc:
            brain._chat([], max_tokens=10, temperature=0)
        assert 'private secret' not in str(exc.value)
        assert len(calls) == 1
    finally:
        brain.close()


def test_retry_503_then_success(monkeypatch):
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(503 if len(calls) == 1 else 200, json={'choices': [{'message': {'content': 'ok'}}]})
    monkeypatch.setattr('conway.brain.time.sleep', lambda _: None)
    brain = OpenAICompatibleVLM(transport=httpx.MockTransport(handler))
    try:
        assert brain._chat([], max_tokens=10, temperature=0) == 'ok'
        assert len(calls) == 2
    finally:
        brain.close()


def test_vlm_request_contains_real_png_and_constitution(tmp_path):
    observed = MockComputer(tmp_path).observe(1)
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({'action': {'type':'wait','args':{'seconds':0}}})}}]})
    brain = OpenAICompatibleVLM('http://localhost:8042', 'served-alias', api_key='test-only', transport=httpx.MockTransport(handler))
    try:
        decision = brain.decide('My constitution', 'Known fact', observed)
        payload = json.loads(requests[0].content)
        assert payload['model'] == 'served-alias'
        assert payload['messages'][0]['content'].startswith('My constitution')
        assert payload['messages'][1]['content'][0]['image_url']['url'].startswith('data:image/png;base64,iVBOR')
        assert requests[0].headers['authorization'] == 'Bearer test-only'
        assert decision.action['type'] == 'wait'
    finally:
        brain.close()


def test_opencua_endpoint_output_adapter(tmp_path):
    brain = OpenCUABrain(transport=transport(data={'choices': [{'message': {'content': 'pyautogui.click(140, 84)'}}]}))
    try:
        result = brain.decide('goal', '', MockComputer(tmp_path).observe(1))
        assert result.action['type'] == 'click'
    finally:
        brain.close()


@pytest.mark.parametrize('budget,expected', [(6,'tiny'), (12,'small'), (24,'standard')])
def test_hardware_profiles(budget, expected):
    assert choose_model_profile(HardwareProfile('Linux','x86_64',32,'cpu',None,budget))[0] == expected


def test_memory_too_small_does_not_force_download():
    with pytest.raises(InsufficientMemoryError):
        choose_model_profile(HardwareProfile('Linux','x86_64',2,'cpu',None,1))


def test_brain_auto_detects_served_opencua_alias():
    assert _brain_type('auto', 'external', 'opencua-7b') == 'opencua'
    assert _brain_type('generic', 'computer-use', 'opencua-7b') == 'generic'


def test_local_runtime_command_and_cleanup(tmp_path, monkeypatch):
    from conway import runtime
    binary=tmp_path/'llama-server'
    binary.write_text('fixture executable, not launched')
    commands=[]
    class Process:
        def __init__(self,*args,**kwargs):
            commands.append((args,kwargs))
            self.alive=True
        def poll(self):
            return None if self.alive else 0
        def terminate(self):
            self.alive=False
        def wait(self,timeout=None):
            return 0
    monkeypatch.setattr(runtime.subprocess,'Popen',Process)
    monkeypatch.setattr(runtime,'_ready',lambda base,model:model=='conway-local')
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    hw=HardwareProfile('Linux','x86_64',32,'cpu',None,12)
    handle=runtime.start_local_runtime(hw,tmp_path,runtime_path=str(binary),port=port)
    command=commands[0][0][0]
    assert '--alias' in command and 'conway-local' in command
    assert command[command.index('--n-gpu-layers')+1]=='0'
    assert command[command.index('--host')+1]=='127.0.0.1'
    handle.close()
    assert not handle.process.alive and handle.log_file is None


def test_occupied_local_port_is_not_silently_reused(tmp_path,monkeypatch):
    from conway import runtime
    import socket
    monkeypatch.setattr(runtime,'find_runtime',lambda path:'fixture')
    hw=HardwareProfile('Linux','x86_64',32,'cpu',None,12)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));sock.listen()
        with pytest.raises(RuntimeError,match='occupied'):
            runtime.start_local_runtime(hw,tmp_path,port=sock.getsockname()[1])


def test_probe_never_dispatches_and_reports_protocol_not_vision(tmp_path,monkeypatch):
    from conway import diagnostics
    from conway.config import ConwayConfig
    from conway.brain import Decision
    closed=[]
    monkeypatch.setattr(diagnostics,'connect_external',lambda *a,**kw:SimpleNamespace(served_model='fixture',model_repo='fixture',base_url='http://localhost/v1',close=lambda:closed.append('runtime')))
    class ProbeBrain:
        def __init__(self,*args,**kwargs):
            pass
        def decide(self,constitution,memory,observation):
            assert observation.screenshot_path.is_file()
            return Decision({'type':'click','args':{'x':1,'y':1}})
        def close(self):
            closed.append('brain')
    monkeypatch.setattr(diagnostics,'OpenAICompatibleVLM',ProbeBrain)
    report=diagnostics.probe_model(ConwayConfig(endpoint='http://localhost/v1'),tmp_path)
    assert report['actions_executed']==0 and report['vision_grounding_verified'] is False
    assert report['valid_action_type']=='click'
    assert closed==['brain','runtime']
