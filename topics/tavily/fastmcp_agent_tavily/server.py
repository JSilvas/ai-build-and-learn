"""
FastMCP server exposing Tavily web intelligence tools.

This file defines the MCP server that the agent connects to. It wraps the
Tavily Python SDK and exposes three tools — search, extract, and crawl —
that the Claude agent can call during its ReAct loop.

To start the server:
    python server.py

The server listens on http://localhost:8001/sse using SSE (Server-Sent Events)
transport, which allows the FastMCP client in agent.py to connect and discover
the available tools.
"""

import os
from typing import Optional
from dotenv import load_dotenv  # Reads key=value pairs from .env into os.environ
from fastmcp import FastMCP     # High-level framework for building MCP servers
from tavily import TavilyClient # Official Tavily Python SDK

# Load environment variables from the .env file in the current directory.
# This must be called before os.environ is accessed below, so TAVILY_API_KEY
# is available when TavilyClient is instantiated.
load_dotenv()

# Create the FastMCP server instance. The name appears in logs and the
# FastMCP startup banner when the server is launched.
mcp = FastMCP("Tavily Web Agent Server")

# Instantiate the Tavily client once at module level so it is shared across
# all tool calls. Creating it here (rather than inside each tool function)
# avoids the overhead of re-authenticating on every request.
# TAVILY_API_KEY is loaded from .env by load_dotenv() above.
_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ---------------------------------------------------------------------------
# Tool 1: Web Search
# ---------------------------------------------------------------------------
# The @mcp.tool decorator registers this function as an MCP tool. FastMCP
# automatically uses the function name as the tool name, the docstring as
# the tool description (which Claude reads when deciding which tool to use),
# and the type annotations to build the JSON schema for the tool's parameters.

@mcp.tool
def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
) -> dict:
    """Search the web using Tavily and return ranked results with snippets.

    Use this tool first when you need to discover relevant URLs or get an
    overview of a topic. Returns titles, URLs, content snippets, and
    relevance scores.

    Args:
        query: The search query string.
        max_results: Number of results to return (default: 5, max: 20).
        search_depth: "basic" for fast overview, "advanced" for deeper
                      coverage (default: "basic").
        include_domains: Restrict results to these domains only, e.g.
                         ["reuters.com", "bbc.com"].
        exclude_domains: Exclude results from these domains.
    """
    try:
        # Call the Tavily search API. Optional list params default to []
        # rather than None because the Tavily SDK expects a list.
        response = _client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_domains or [],
            exclude_domains=exclude_domains or [],
        )

        # Normalize the response into a consistent dict structure.
        # .get() with a default value protects against missing keys in the
        # API response if Tavily changes their response shape.
        return {
            "query": query,
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),   # snippet, not full page
                    "score": r.get("score", 0.0),       # relevance score 0–1
                }
                for r in response.get("results", [])
            ],
        }
    except Exception as e:
        # Return a structured error dict instead of raising so the agent
        # receives a readable message and can decide how to recover.
        return {"query": query, "error": str(e), "results": []}


# ---------------------------------------------------------------------------
# Tool 2: Content Extraction
# ---------------------------------------------------------------------------
# Use this tool when the agent already has specific URLs from a search and
# needs the full page content rather than just snippets.

@mcp.tool
def tavily_extract(
    urls: list[str],
    extract_depth: str = "basic",
) -> dict:
    """Extract structured full-page content from specific URLs.

    Use this tool when you already have URLs (e.g. from tavily_search) and
    need the full readable content of those pages rather than just snippets.
    More precise than crawling when you have a known list of pages to read.

    Args:
        urls: List of URLs to extract content from (1-20 URLs).
        extract_depth: "basic" for main content, "advanced" for deeper
                       extraction including tables and structured data
                       (default: "basic").
    """
    try:
        # Tavily extract accepts a batch of URLs in a single API call,
        # which is more efficient than calling it once per URL.
        response = _client.extract(
            urls=urls,
            extract_depth=extract_depth,
        )

        return {
            # Successfully extracted pages with their full text content
            "results": [
                {
                    "url": r.get("url", ""),
                    "raw_content": r.get("raw_content", ""),  # full page text
                }
                for r in response.get("results", [])
            ],
            # URLs that could not be extracted (e.g. paywalled, broken links)
            "failed_results": response.get("failed_results", []),
        }
    except Exception as e:
        return {"error": str(e), "results": []}


# ---------------------------------------------------------------------------
# Tool 3: Web Crawl
# ---------------------------------------------------------------------------
# Use this tool when the agent needs to explore an entire site or docs section
# by following internal links, rather than extracting specific known pages.

@mcp.tool
def tavily_crawl(
    url: str,
    max_depth: int = 1,
    max_breadth: int = 10,
    limit: int = 10,
    instructions: Optional[str] = None,
) -> dict:
    """Crawl a website starting from a root URL to gather site-wide content.

    Use this tool when you need comprehensive information from an entire site
    or a documentation section, not just a single page. Follows internal links
    up to max_depth levels deep. More expensive than extract — prefer extract
    when you have specific URLs.

    Args:
        url: The root URL to begin crawling from.
        max_depth: How many link-hops deep to follow (default: 1).
        max_breadth: Max number of links to follow per page (default: 10).
        limit: Total maximum pages to return (default: 10).
        instructions: Optional natural-language guidance to focus the crawl,
                      e.g. "focus on API reference pages only".
    """
    try:
        # Build kwargs dict with required parameters first.
        # instructions is injected only when provided because some versions
        # of the Tavily SDK do not accept instructions=None gracefully.
        kwargs = dict(
            url=url,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
        )
        if instructions:
            kwargs["instructions"] = instructions

        response = _client.crawl(**kwargs)

        return {
            "root_url": url,
            # Each result is one crawled page with its URL and full text
            "results": [
                {
                    "url": r.get("url", ""),
                    "raw_content": r.get("raw_content", ""),
                }
                for r in response.get("results", [])
            ],
        }
    except Exception as e:
        return {"root_url": url, "error": str(e), "results": []}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# When run directly, start the FastMCP server using SSE transport on port 8001.
# Port 8001 is used to avoid collision with the demo MCP server in topics/mcp/
# which runs on port 8000.
#
# SSE (Server-Sent Events) is an HTTP-based transport that keeps a persistent
# connection open so the client can receive streamed responses from the server.

if __name__ == "__main__":
    mcp.run(transport="sse", host="localhost", port=8001)
