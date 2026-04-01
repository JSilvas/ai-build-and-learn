"""
Claude agent using the Tavily MCP server via Anthropic SDK and FastMCP Client.

This file implements the agent side of the system. It connects to the running
FastMCP server (server.py), discovers the available Tavily tools, and runs a
manual ReAct (Reason + Act) loop using the Anthropic SDK.

How it works:
  1. FastMCP Client connects to the server and fetches the tool list
  2. Tools are converted to the format the Anthropic API expects
  3. The query and tools are sent to Claude
  4. If Claude responds with tool calls, each tool is executed via the MCP client
     and the results are fed back into the conversation
  5. Steps 3-4 repeat until Claude produces a final text answer with no tool calls

Requires server.py to be running first:
    python server.py

Then in a separate terminal:
    python agent.py
    python agent.py "Your custom research question here"
"""

import asyncio
import json     # Used to pretty-print tool call arguments in the console
import os
import sys      # Used to read the optional command-line query argument
from dotenv import load_dotenv      # Reads .env file into os.environ
from anthropic import Anthropic     # Anthropic SDK for calling Claude
from fastmcp import Client          # FastMCP client for connecting to the MCP server
from system_prompt import SYSTEM_PROMPT  # Imported from system_prompt.py

# Load environment variables from .env so ANTHROPIC_API_KEY is available
# to the Anthropic SDK (it reads it automatically from os.environ).
load_dotenv()

# The URL of the running FastMCP server. Defaults to localhost:8001 but can
# be overridden via the TAVILY_MCP_SERVER_URL environment variable to point
# at a remotely deployed server.
TAVILY_MCP_SERVER_URL = os.getenv(
    "TAVILY_MCP_SERVER_URL", "http://localhost:8001/sse"
)
print(f"Connecting to: {TAVILY_MCP_SERVER_URL}")

# Default query used when no command-line argument is provided.
# Exercises all three tools: search (discovery), extract (full content),
# and crawl (site-wide documentation).
DEFAULT_QUERY = (
    "Research the current state of Model Context Protocol (MCP) adoption. "
    "First, search for recent news and articles about MCP. "
    "Then extract the full content from the top 3 results to get details. "
    "Finally, crawl the official MCP documentation site (modelcontextprotocol.io) "
    "to understand what tools and SDKs are officially supported. "
    "Give me a structured summary with sources cited."
)


async def main(query: str):
    # Create the FastMCP client pointed at the running server.
    # The client communicates over SSE (Server-Sent Events) HTTP transport.
    mcp_client = Client(TAVILY_MCP_SERVER_URL)

    # async with opens the connection to the MCP server and keeps it open
    # for the duration of the block. The connection is closed automatically
    # when the block exits.
    async with mcp_client:

        # Ask the MCP server which tools are available. This returns the tool
        # names, descriptions, and parameter schemas defined in server.py.
        tools = await mcp_client.list_tools()

        # Convert the MCP tool schema format into the format the Anthropic API
        # expects. The key difference is that Anthropic uses "input_schema"
        # while MCP uses "inputSchema".
        anthropic_tools = [
            {
                "name": tool.name,                  # e.g. "tavily_search"
                "description": tool.description,    # shown to Claude so it knows when to use the tool
                "input_schema": tool.inputSchema,   # JSON schema for the tool's parameters
            }
            for tool in tools
        ]

        # Create the Anthropic client. It automatically reads ANTHROPIC_API_KEY
        # from os.environ (loaded from .env above).
        client = Anthropic()

        # Initialize the conversation with the user's query.
        # Messages is a list that grows as the conversation progresses —
        # each tool call and result is appended to maintain the full history.
        messages = [{"role": "user", "content": query}]

        # -----------------------------------------------------------------------
        # ReAct Loop (Reason + Act)
        # -----------------------------------------------------------------------
        # Each iteration of this loop is one "step" of the agent:
        #   - Claude reasons about what to do next
        #   - If it wants to use a tool, we call the tool and feed back the result
        #   - If it has enough information, it produces a final answer and we stop
        #
        # This loop replaces what Runner.run() did automatically in the
        # OpenAI Agents SDK — here we manage it ourselves for full control.

        while True:
            # Send the current conversation history to Claude along with the
            # available tools. Claude will either respond with tool calls
            # (stop_reason="tool_use") or a final text answer (stop_reason="end_turn").
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,        # generous limit for long research responses
                system=SYSTEM_PROMPT,   # passed separately from messages (Anthropic convention)
                tools=anthropic_tools,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                # Claude wants to call one or more tools.
                # First, add Claude's response (including its tool call requests)
                # to the message history so it has context in the next iteration.
                messages.append({"role": "assistant", "content": response.content})

                # Process each tool call Claude requested.
                # A single response can contain multiple tool calls, so we
                # collect all results before sending them back.
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        # Print the tool call to the console so the user can
                        # follow along with the agent's reasoning in real time.
                        print(f"Calling tool: {block.name}({json.dumps(block.input)})")

                        # Execute the tool on the MCP server and get the result.
                        # block.name is the tool name (e.g. "tavily_search")
                        # block.input is a dict of the tool's arguments
                        result = await mcp_client.call_tool(block.name, block.input)

                        # Package the result in the format Anthropic expects:
                        # tool_use_id links this result back to the specific tool call.
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,    # must match the tool_use block id
                                "content": str(result),     # convert result to string for the API
                            }
                        )

                # Add all tool results to the message history as a user turn.
                # This is how Anthropic's API receives tool results — they are
                # sent back as a "user" message containing tool_result blocks.
                messages.append({"role": "user", "content": tool_results})

            else:
                # stop_reason is "end_turn" — Claude has finished reasoning
                # and is returning a final text response with no more tool calls.
                # Extract and print the text content from the response.
                for block in response.content:
                    if hasattr(block, "text"):
                        print(block.text)
                break  # Exit the ReAct loop


if __name__ == "__main__":
    # Accept an optional query as a command-line argument.
    # " ".join handles multi-word queries passed without quotes.
    # Falls back to DEFAULT_QUERY if no argument is provided.
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUERY
    asyncio.run(main(query))
