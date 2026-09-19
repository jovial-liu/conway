"""Local HTTP fixture is NOT an actual VLM; it verifies the complete tool protocol."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import pytest
from conway.brain import OpenAICompatibleVLM
from conway.computer import MockComputer
from conway.config import ToolSettings
from conway.loop import ConwayLoop
from conway.memory import FileMemory
from conway.tools import ToolExecutor
from conway.runtime import connect_external


def test_http_brain_file_tools_memory_roundtrip(tmp_path):
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def reply(self,data):
            content=json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        def do_GET(self):
            assert self.path=='/v1/models'
            self.reply({'data':[{'id':'fixture-vlm'}]})
        def do_POST(self):
            assert self.path=='/v1/chat/completions'
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(payload)
            actions=[{'type':'write_file','args':{'path':'output.txt','content':'verified fixture'}},
                     {'type':'read_file','args':{'path':'output.txt'}},
                     {'type':'finish','args':{'reason':'fixture done'}}]
            action=actions[len(seen)-1]
            self.reply({'choices':[{'message':{'content':json.dumps({'action':action})}}]})
        def log_message(self,*args):
            pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    endpoint=f'http://127.0.0.1:{server.server_port}/v1'
    brain=None
    try:
        handle=connect_external(endpoint)
        mem=FileMemory(tmp_path/'state')
        executor=ToolExecutor(MockComputer(mem.root/'screenshots'),ToolSettings(shell=False,open_url=False,workspace=str(tmp_path)))
        brain=OpenAICompatibleVLM(endpoint,handle.served_model,tool_manifest=executor.manifest_text())
        loop=ConwayLoop(brain,executor,mem,max_steps=5,interval=0,quiet=True)
        assert loop.run()=='completed'
        assert (tmp_path/'output.txt').read_text()=='verified fixture'
        assert len(seen)==3
        assert 'verified fixture' in seen[2]['messages'][1]['content'][1]['text']
        assert mem.load_state().pending_action is None
        assert mem.load_state().cycle==3
    finally:
        if brain:
            brain.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_installed_style_cli_in_subprocess(tmp_path):
    result=subprocess.run([sys.executable,'-m','conway','--home',str(tmp_path),'start','--mock',
                           '--max-steps','2','--interval','0','--quiet'],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert FileMemory(tmp_path).load_state().cycle==2
