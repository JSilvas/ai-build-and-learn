# Tavily Web Agent

An agentic web research assistant powered by [Tavily](https://tavily.com), [FastMCP](https://github.com/jlowin/fastmcp), and Anthropic's Claude. The agent uses a ReAct loop to reason about which web tools to use, call them in sequence, and iterate until it produces a thorough, well-sourced answer.

---

## Project Design

### Architecture

The project has three main components: the language model, a set of web tools, and a system prompt.

```
┌─────────────────────────────────────────────────┐
│                   agent.py                      │
│                                                 │
│   Claude  ──►  ReAct Loop (Anthropic SDK)       │
│                     │                           │
│              FastMCP Client (SSE)               │
└─────────────────────┼───────────────────────────┘
                      │ http://localhost:8001/sse
┌─────────────────────▼───────────────────────────┐
│                   server.py                     │
│                                                 │
│   FastMCP Server  ──►  Tavily API               │
│   ├── tavily_search                             │
│   ├── tavily_extract                            │
│   └── tavily_crawl                             │
└─────────────────────────────────────────────────┘
```

### Components

**Language Model — Claude (Anthropic SDK)**
The LLM serves as the agent's brain. It decides which tools to call, in what order, and synthesizes the results into a final answer. Configured in `agent.py` via `model="claude-opus-4-6"`.

**Web Tools — FastMCP Server (`server.py`)**
A FastMCP server running on port `8001` exposes three Tavily-powered tools over SSE transport:

| Tool | Purpose | When to use |
|------|---------|-------------|
| `tavily_search` | Search the web and return ranked results with snippets | First — for discovery and finding URLs |
| `tavily_extract` | Extract full page content from specific URLs | After search — when snippets aren't enough |
| `tavily_crawl` | Crawl a site from a root URL, following internal links | For site-wide or documentation research |

**System Prompt (`system_prompt.py`)**
A detailed prompt that instructs the agent on:
- Which tool to use in each situation
- How to chain tools together (search → extract → crawl)
- Quality standards: cite sources, cross-verify claims, iterate on poor results

**ReAct Loop**
The ReAct (Reason + Act) loop is implemented manually in `agent.py` using the Anthropic SDK and FastMCP's `Client`:

1. Send the query and available tools to Claude
2. If Claude responds with `tool_use`, call each tool via the FastMCP client and feed the results back
3. Repeat until Claude responds with a final text answer (no more tool calls)

Each tool call is printed to the console so you can follow the agent's reasoning in real time.

### File Structure

```
topics/tavily/
├── server.py          # FastMCP server with 3 Tavily tools (port 8001)
├── agent.py           # Claude agent with manual ReAct loop
├── system_prompt.py   # System prompt constant imported by agent.py
├── requirements.txt   # Python dependencies
└── .env.example       # API key template
```

---

## Installation

### 1. Create and activate a virtual environment

```bash
cd topics/tavily
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows (bash shell)
source .venv/Scripts/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=your-anthropic-key-here
TAVILY_API_KEY=your-tavily-key-here
```

- Get an Anthropic API key at [console.anthropic.com](https://console.anthropic.com)
- Get a Tavily API key at [app.tavily.com](https://app.tavily.com)

---

## Running the Agent

The server and agent run in separate terminals.

### Terminal 1 — Start the MCP server

```bash
python server.py
```

FastMCP will print the registered tools on startup:

```
Tavily Web Agent Server
Tools: tavily_search, tavily_extract, tavily_crawl
Running on http://localhost:8001/sse
```

### Terminal 2 — Run the agent

Run the demo query:

```bash
python agent.py
```

Or pass your own query as a command-line argument:

```bash
python agent.py "What are the best Python web frameworks in 2025?"
python agent.py "Summarize the latest research on LLM reasoning benchmarks."
```

If no argument is provided, the agent falls back to the built-in demo query (MCP adoption research). Quotes are optional for single-word queries but required for multi-word questions.

As the agent runs, each tool call is printed so you can follow the ReAct loop:

```
Connecting to: http://localhost:8001/sse
Calling tool: tavily_search({"query": "Model Context Protocol adoption 2025"})
Calling tool: tavily_extract({"urls": ["https://...", "https://..."]})
Calling tool: tavily_crawl({"url": "https://modelcontextprotocol.io"})

## Summary
...
```

---

## Verification

To confirm all three tools are working end-to-end:

1. **Server starts cleanly** — Terminal 1 shows all 3 tools listed with no errors.
2. **Agent connects** — Terminal 2 prints `Connecting to: http://localhost:8001/sse`.
3. **Tools are invoked** — The console shows `Calling tool:` lines for `tavily_search`, `tavily_extract`, and `tavily_crawl`.
4. **Final output** — The agent prints a structured summary with URLs cited as sources.

---

## Tool Reference

### `tavily_search`

```python
tavily_search(
    query: str,
    max_results: int = 5,            # 1–20
    search_depth: str = "basic",     # "basic" | "advanced"
    include_domains: list[str] = [], # restrict to these domains
    exclude_domains: list[str] = [], # exclude these domains
) -> dict
```

Returns `{"query": ..., "results": [{"title", "url", "content", "score"}, ...]}`.

### `tavily_extract`

```python
tavily_extract(
    urls: list[str],                 # 1–20 URLs
    extract_depth: str = "basic",    # "basic" | "advanced"
) -> dict
```

Returns `{"results": [{"url", "raw_content"}, ...], "failed_results": [...]}`.

### `tavily_crawl`

```python
tavily_crawl(
    url: str,                        # root URL to start from
    max_depth: int = 1,              # link-hops to follow
    max_breadth: int = 10,           # links per page
    limit: int = 10,                 # max total pages
    instructions: str = None,        # natural-language focus hint
) -> dict
```

Returns `{"root_url": ..., "results": [{"url", "raw_content"}, ...]}`.
