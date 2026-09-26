"""Optional official MCP SDK client. One owner task keeps stdio sessions alive."""
from __future__ import annotations

import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextlib import AsyncExitStack
import json
import os
import threading

from .config import MCPServerConfig


class MCPBridge:
    def __init__(self, servers: dict[str, MCPServerConfig], *, output_limit: int = 12000) -> None:
        self.servers, self.output_limit = servers, output_limit
        self.catalog: dict[tuple[str, str], dict] = {}
        self._validators = {}
        self._ready: Future = Future()
        self._thread = None
        self._loop = self._task = self._queue = None
        self._closed = False

    def start(self) -> None:
        if self._thread is not None or self._closed:
            raise RuntimeError('MCP bridge cannot be restarted')
        try:
            import mcp  # noqa: F401
            import jsonschema  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('Install MCP support with pip install "conway-agent[mcp]"') from exc
        self._thread = threading.Thread(target=self._run, name='conway-mcp', daemon=True)
        self._thread.start()
        try:
            self._ready.result(timeout=sum(s.timeout for s in self.servers.values()) + 10)
        except BaseException:
            self.close()
            raise

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(RuntimeError(f'MCP startup failed: {type(exc).__name__}'))

    async def _serve(self) -> None:
        from mcp import Client, StdioServerParameters, stdio_client
        self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
        self._queue = asyncio.Queue()
        clients = {}
        # stderr can contain credentials; never copy it into the agent's context/journal.
        with open(os.devnull, 'w') as errlog:
            async with AsyncExitStack() as stack:
                for name, settings in self.servers.items():
                    missing = [key for key in settings.env_keys if not os.environ.get(key)]
                    if missing:
                        raise ValueError(f'Missing configured MCP environment keys for {name}')
                    params = StdioServerParameters(command=settings.command, args=settings.args,
                        env={key: os.environ[key] for key in settings.env_keys})
                    async with asyncio.timeout(settings.timeout):
                        client = await stack.enter_async_context(Client(stdio_client(params, errlog=errlog),
                            read_timeout_seconds=settings.timeout, input_required_max_rounds=0))
                        clients[name] = client
                        cursor, cursors, found = None, set(), set()
                        for _ in range(32):
                            listed = await client.list_tools(cursor=cursor)
                            for tool in listed.tools:
                                if tool.name not in settings.allowed_tools:
                                    continue
                                if tool.name in found:
                                    raise ValueError('Duplicate MCP tool name')
                                self._register(name, tool)
                                found.add(tool.name)
                            cursor = listed.next_cursor
                            if cursor is None:
                                break
                            if cursor in cursors:
                                raise ValueError('MCP pagination cycle')
                            cursors.add(cursor)
                        else:
                            raise ValueError('MCP tool pagination exceeds 32 pages')
                        if set(settings.allowed_tools) != found:
                            raise ValueError(f'Configured MCP tool not advertised by {name}')
                if len(self.manifest()) > 48000:
                    raise ValueError('MCP catalog exceeds 48000 characters; expose fewer tools')
                self._ready.set_result(None)
                while True:
                    request = await self._queue.get()
                    if request is None:
                        break
                    server, tool, arguments, future = request
                    try:
                        async with asyncio.timeout(self.servers[server].timeout):
                            result = await clients[server].call_tool(tool, arguments)
                        # Do not fetch resource links or forward inline binary data automatically.
                        payload = result.model_dump(mode='json', by_alias=True, exclude_none=True)
                        if payload.get('isError') or payload.get('resultType', 'complete') != 'complete':
                            raise RuntimeError('MCP error or additional input required; outcome uncertain')
                        blocks = payload.get('content', [])
                        payload['content'] = [b if b.get('type') == 'text' else
                                              {'type': b.get('type'), 'omitted': True} for b in blocks]
                        text = json.dumps(payload, ensure_ascii=False)
                        if len(text) > self.output_limit:
                            text = text[:self.output_limit] + '\n[MCP result truncated]'
                        future.set_result(text)
                    except asyncio.CancelledError:
                        if not future.done():
                            future.set_exception(RuntimeError('MCP call interrupted; outcome uncertain'))
                        raise
                    except Exception as exc:
                        # Timeout/disconnect after dispatch is NOT retried. The loop keeps pending intent.
                        future.set_exception(RuntimeError(f'MCP call failed ({type(exc).__name__}); outcome uncertain'))

    def _register(self, server, tool) -> None:
        from jsonschema import Draft202012Validator
        from referencing import Registry
        schema = tool.input_schema
        if len(json.dumps(schema)) > 16000:
            raise ValueError('MCP tool schema too large')

        def inspect(value, depth=0):
            if depth > 32:
                raise ValueError('MCP schema nesting exceeds 32')
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {'$ref', '$dynamicRef'} and (not isinstance(item, str) or not item.startswith('#')):
                        raise ValueError('MCP schema must not fetch external references')
                    inspect(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    inspect(item, depth + 1)
        inspect(schema)
        Draft202012Validator.check_schema(schema)
        key = (server, tool.name)
        self._validators[key] = Draft202012Validator(schema, registry=Registry())
        self.catalog[key] = {'server': server, 'tool': tool.name,
            'description': (tool.description or '')[:1500], 'input_schema': schema}

    def manifest(self) -> str:
        return json.dumps(list(self.catalog.values()), ensure_ascii=False)

    def validate(self, server: str, tool: str, arguments: dict) -> None:
        validator = self._validators.get((server, tool))
        if validator is None:
            raise ValueError('MCP tool is not in the configured allowlist')
        try:
            validator.validate(arguments)
        except Exception as exc:
            raise ValueError('MCP arguments do not match the advertised input schema') from exc

    def call(self, server: str, tool: str, arguments: dict, control=None) -> str:
        self.validate(server, tool, arguments)
        if self._closed or self._thread is None or not self._thread.is_alive():
            raise RuntimeError('MCP connection is closed')
        future: Future = Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (server, tool, arguments, future))
        try:
            while True:
                if control:
                    control.check()
                try:
                    return future.result(timeout=0.05)
                except FutureTimeout:
                    if not self._thread.is_alive():
                        raise RuntimeError('MCP worker exited; action outcome uncertain')
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._task.cancel)
        if self._thread:
            self._thread.join(timeout=8)
            if self._thread.is_alive():
                raise RuntimeError('MCP worker did not finish shutdown')
