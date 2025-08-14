from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool
def greet(name: str) -> str:
    """Returns a personalized greeting."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()  # Runs using STDIO by default


import asyncio
from fastmcp import Client

client = Client("my_server.py")  # Local .py file (uses stdio transport)

async def call_tool(name: str):
    async with client:
        result = await client.call_tool("greet", {"name": name})
        print(result)  # Should print: Hello, <name>!

if __name__ == "__main__":
    asyncio.run(call_tool("Ford"))

==============

from fastmcp import FastMCP

mcp = FastMCP(name="My MCP Server")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

if __name__ == "__main__":
    mcp.run()


import asyncio
from fastmcp import Client

client = Client("server.py")

async def run():
    async with client:
        result = await client.call_tool("add", {"a": 5, "b": 7})
        print("Result from 'add':", result)

if __name__ == "__main__":
    asyncio.run(run())

======
