# MCP with FastMCP

Build a simple MCP (Model Context Protocol) server using FastMCP and connect to it with the OpenAI Agents SDK.

## What is MCP?

MCP is a standard protocol for exposing tools and context to language models. Instead of every framework inventing its own tool format, MCP gives clients and servers a shared contract for listing capabilities, describing inputs, and calling them.

- Your MCP server exposes capabilities (tools)
- Your model client or agent connects to that server
- The model sees a consistent tool schema
- The model can call the tool without knowing your implementation details

## What we build

A FastMCP server with three types of tools:

1. **Pure computation** - `add` and `multiply` (deterministic input/output)
2. **External state** - `read_text_file` (reads from the local filesystem)
3. **Context-aware** - `greet` (uses MCP Context for logging)

Then a client using the OpenAI Agents SDK that connects to the server and uses the tools.

## Setup

```bash
# Navigate to this topic
cd topics/mcp

# Create virtual environment
uv venv .venv --python 3.11

# Activate the venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt
```

## Running

### Run the server standalone (for testing)

```bash
python server.py
```

### Run the client (starts the server automatically)

Copy the example env file and add your key:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
python client.py
```

The client starts the FastMCP server as a subprocess via stdio, discovers the available tools, and lets the model decide which ones to call.

## How it works

FastMCP turns Python functions into MCP tools automatically:

- The **function name** becomes the tool name
- The **docstring** becomes the description
- The **type annotations** become the JSON Schema
- The **return value** is serialized automatically

So this function:

```python
@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b
```

Becomes this tool schema that the model sees:

```json
{
  "name": "add",
  "description": "Add two integers together.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "a": {"type": "integer"},
      "b": {"type": "integer"}
    },
    "required": ["a", "b"]
  }
}
```

## Next steps

- Add a Claude/Anthropic client example
- Add a Claude Code MCP integration example
