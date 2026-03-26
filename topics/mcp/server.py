"""
Create the MCP server 
"""

from fastmcp import FastMCP, Context
from pathlib import Path

mcp = FastMCP("Demo Server")


# --- Pure computation tools ---


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


@mcp.tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b


# --- External state tool ---


@mcp.tool
def read_text_file(path: str) -> str:
    """Read a UTF-8 text file from disk and return its contents."""
    return Path(path).read_text(encoding="utf-8")


# --- Context-aware tool ---


@mcp.tool
def greet(name: str, ctx: Context) -> str:
    """Greet a user by name and log the action."""
    ctx.info(f"Greeting {name}")
    return f"Hello, {name}! Welcome to the MCP demo."


if __name__ == "__main__":
    mcp.run(transport="sse", host="localhost", port=8000)
