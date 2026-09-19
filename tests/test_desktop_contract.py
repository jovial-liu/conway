from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
from unittest.mock import Mock
import pytest
from PIL import Image
from conway.accessibility import capture_ui_tree
from conway.computer import Computer, Observation
from conway.control import EmergencyStop
from conway.desktop import DesktopMetadata


class FakeFailsafe(Exception):
    pass


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    gui = SimpleNamespace(size=lambda:(100,80), position=lambda:SimpleNamespace(x=20,y=30),
        screenshot=lambda:Image.new('RGB',(200,160)), failSafeCheck=Mock(),
        click=Mock(), doubleClick=Mock(), moveTo=Mock(), dragTo=Mock(), write=Mock(),
        press=Mock(), hotkey=Mock(), scroll=Mock(), FailSafeException=FakeFailsafe,
        KEYBOARD_KEYS=['ctrl','command','enter','a','v'])
    monkeypatch.setitem(sys.modules,'pyautogui',gui)
    monkeypatch.setattr('conway.computer.capture_desktop_metadata',lambda:DesktopMetadata('Editor','Note','fake'))
    computer=Computer(tmp_path)
    return computer,gui


def test_capture_and_input_mapping(desktop):
    computer,gui=desktop
    observation=computer.observe(1)
    assert observation.width==200 and observation.input_width==100
    assert observation.cursor_x==40 and observation.cursor_y==60
    computer.execute({'type':'click','args':{'x':100,'y':80}},observation)
    gui.click.assert_called_once_with(x=50,y=40,button='left')


@pytest.mark.parametrize('action,call',[
    ({'type':'double_click','args':{'x':20,'y':20}},'doubleClick'),
    ({'type':'move','args':{'x':20,'y':20}},'moveTo'),
    ({'type':'drag','args':{'x':20,'y':20}},'dragTo'),
    ({'type':'type','args':{'text':'hello'}},'write'),
    ({'type':'press','args':{'key':'enter'}},'press'),
    ({'type':'hotkey','args':{'keys':['ctrl','a']}},'hotkey'),
    ({'type':'scroll','args':{'amount':2}},'scroll'),
])
def test_gui_dispatch_contract(desktop,action,call):
    computer,gui=desktop
    result=computer.execute(action,computer.observe(1))
    assert getattr(gui,call).call_count==1
    assert 'verify' in result


def test_geometry_change_prevents_stale_click(desktop):
    computer,gui=desktop
    obs=computer.observe(1)
    gui.size=lambda:(200,160)
    with pytest.raises(RuntimeError,match='geometry changed'):
        computer.execute({'type':'click','args':{'x':10,'y':10}},obs)
    gui.click.assert_not_called()


def test_foreground_change_prevents_stale_click(desktop,monkeypatch):
    computer,gui=desktop
    obs=computer.observe(1)
    monkeypatch.setattr('conway.computer.capture_desktop_metadata',lambda:DesktopMetadata('Mail','Inbox','fake'))
    with pytest.raises(RuntimeError,match='Foreground window'):
        computer.execute({'type':'click','args':{'x':10,'y':10}},obs)
    gui.click.assert_not_called()


def test_failsafe_propagates_as_stop(desktop):
    computer,gui=desktop
    gui.failSafeCheck.side_effect=FakeFailsafe()
    with pytest.raises(EmergencyStop):
        computer.execute({'type':'click','args':{'x':10,'y':10}},computer.observe(1))


def test_unicode_paste_restores_clipboard(desktop,monkeypatch):
    computer,gui=desktop
    clipboard=['original']
    fake=SimpleNamespace(paste=lambda:clipboard[0],copy=lambda value:clipboard.__setitem__(0,value))
    monkeypatch.setitem(sys.modules,'pyperclip',fake)
    monkeypatch.setattr('conway.computer.time.sleep',lambda _:None)
    computer.execute({'type':'type','args':{'text':'你好'}},computer.observe(1))
    assert clipboard==['original']
    assert gui.hotkey.call_count==1 and gui.write.call_count==0


def test_unicode_paste_failure_is_not_silent(desktop,monkeypatch):
    computer,gui=desktop
    def failed():
        raise RuntimeError('clipboard unavailable')
    monkeypatch.setitem(sys.modules,'pyperclip',SimpleNamespace(paste=failed))
    with pytest.raises(RuntimeError,match='clipboard unavailable'):
        computer.execute({'type':'type','args':{'text':'你好'}},computer.observe(1))
    gui.write.assert_not_called()


def test_ui_tree_timeout_degrades_to_screenshot(monkeypatch):
    def timeout(*args,**kwargs):
        raise subprocess.TimeoutExpired(args[0],1)
    monkeypatch.setattr('conway.accessibility.subprocess.run',timeout)
    assert capture_ui_tree(timeout=.1)['status']=='timeout'


def test_ui_tree_valid_payload(monkeypatch):
    monkeypatch.setattr('conway.accessibility.subprocess.run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps({'status':'ok','nodes':[{'name':'Save'}]})))
    assert capture_ui_tree()['nodes'][0]['name']=='Save'


def test_ui_tree_invalid_payload(monkeypatch):
    monkeypatch.setattr('conway.accessibility.subprocess.run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='not-json'))
    assert capture_ui_tree()['status']=='unavailable'
