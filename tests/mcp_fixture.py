"""Actual stdio server for protocol/lifecycle tests; never imports a desktop adapter."""
import asyncio
import os
from pathlib import Path
import sys
from mcp.server import MCPServer
from pydantic import BaseModel

server = MCPServer('Conway test fixture')
count = 0
marker = Path(sys.argv[1])
marker.write_text(str(os.getpid()))


class Counter(BaseModel):
    count: int
    secret_present: bool


@server.tool()
def increment(amount: int) -> Counter:
    global count
    count += amount
    return Counter(count=count, secret_present='CONWAY_MCP_TEST_SECRET' in os.environ)


@server.tool()
async def slow() -> str:
    await asyncio.sleep(60)
    return 'done'


@server.tool()
def forbidden() -> str:
    raise AssertionError('must not be called')


if __name__ == '__main__':
    server.run()
