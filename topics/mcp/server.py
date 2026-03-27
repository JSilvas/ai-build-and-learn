"""
Create the MCP server 
"""

from fastmcp import FastMCP, Context
from pathlib import Path
import httpx
from typing import Optional
from ddgs import DDGS
from bs4 import BeautifulSoup

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


# --- External API tool ---


@mcp.tool
async def get_weather(location: str) -> dict[str, str]:
    """Get current weather for a location using the wttr.in API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://wttr.in/{location}?format=j1",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

        if not data or "current_condition" not in data:
            return {"location": location, "error": "Weather data unavailable for this location"}

        current = data["current_condition"][0]
        return {
            "location": location,
            "temperature_c": current["temp_C"],
            "temperature_f": current["temp_F"],
            "condition": current["weatherDesc"][0]["value"],
            "feels_like_c": current["FeelsLikeC"],
            "feels_like_f": current["FeelsLikeF"],
            "humidity": current["humidity"],
            "wind_speed_kmph": current["windspeedKmph"],
        }
    except Exception as e:
        return {"location": location, "error": f"Weather API unavailable: {e}"}


# --- Web search tools ---


@mcp.tool
def duck_duck_go(
    query: str,
    max_results: int = 10,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: Optional[str] = None,
) -> list[dict[str, str]]:
    """Search DuckDuckGo for web results.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default: 10).
        region: Region code, e.g. "us-en", "uk-en", "de-de" (default: "us-en").
        safesearch: "off", "moderate", or "strict" (default: "moderate").
        timelimit: Time filter - "d" (day), "w" (week), "m" (month), "y" (year), or None.
    """
    ddgs = DDGS()
    results = ddgs.text(
        query=query,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        max_results=max_results,
    )
    return [
        {
            "title": r.get("title", ""),
            "href": r.get("href", ""),
            "body": r.get("body", ""),
        }
        for r in results
    ]


@mcp.tool
async def fetch_webpage(url: str, max_length: int = 5000) -> dict[str, str]:
    """Fetch and extract text content from a webpage.

    Args:
        url: The URL of the webpage to fetch.
        max_length: Maximum length of content to return (default: 5000 chars).
    """
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string if soup.title else "No title"
    text = soup.get_text(separator="\n", strip=True)

    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"

    return {"url": url, "title": title, "content": text}


# --- Context-aware tool ---


@mcp.tool
def greet(name: str, ctx: Context) -> str:
    """Greet a user by name and log the action."""
    ctx.info(f"Greeting {name}")
    return f"Hello, {name}! Welcome to the MCP demo."


if __name__ == "__main__":
    mcp.run(transport="sse", host="localhost", port=8000)
