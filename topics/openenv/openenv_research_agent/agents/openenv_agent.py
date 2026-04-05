"""
OpenEnv Research Agent — Claude via OpenEnv AnthropicClient.

This is the "after" side of the demo comparison. Unlike the traditional
agent, this agent:

  1. Discovers available tools dynamically at runtime (no fixed action space)
  2. Reasons about which tool to use based on prior observations
  3. Chains tools in sequence: search → extract → crawl as needed
  4. Earns high llm_judge_reward scores because it actually researches well

The ReAct loop is handled by OpenEnv's AnthropicClient.complete_with_tools()
rather than being written manually (unlike fastmcp_agent_tavily/agent.py).

This agent can also be run in "race mode" — multiple instances of this
class run concurrently against the same ResearchEnvironment, which supports
concurrent sessions via SUPPORTS_CONCURRENT_SESSIONS = True.

Usage (single agent):
    agent = OpenEnvAgent(query="What is MCP?")
    for step_result in agent.run(env):
        print(step_result)

Usage (race — 3 agents, see app.py):
    agents = [OpenEnvAgent(query, agent_id=i) for i in range(3)]
    # run concurrently via asyncio
"""

import os
import json
from typing import Generator, Optional
from dotenv import load_dotenv
from anthropic import Anthropic

from env.models import ResearchAction
from env.research_env import ResearchEnvironment
from reward import llm_judge_reward
from system_prompt import SYSTEM_PROMPT

load_dotenv()

# Tool schemas passed to Claude so it knows how to call each Tavily action
_TOOL_SCHEMAS = [
    {
        "name": "tavily_search",
        "description": "Search the web using Tavily. Use first for any research task to discover relevant URLs and get an overview.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "default": 5},
                "search_depth": {"type": "string", "enum": ["basic", "advanced"], "default": "basic"},
                "include_domains": {"type": "array", "items": {"type": "string"}},
                "exclude_domains": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
    },
    {
        "name": "tavily_extract",
        "description": "Extract full page content from specific URLs. Use after search when you need full content, not just snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}, "description": "URLs to extract"},
                "extract_depth": {"type": "string", "enum": ["basic", "advanced"], "default": "basic"},
            },
            "required": ["urls"],
        },
    },
    {
        "name": "tavily_crawl",
        "description": "Crawl a website from a root URL to gather site-wide content. Use for docs or when you need many pages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Root URL to crawl from"},
                "max_depth": {"type": "integer", "default": 1},
                "max_breadth": {"type": "integer", "default": 10},
                "limit": {"type": "integer", "default": 10},
                "instructions": {"type": "string"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "finish",
        "description": "Signal that you have gathered enough information to answer the research question. Call this when done.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Brief summary of what you found"},
            },
            "required": ["summary"],
        },
    },
]


class OpenEnvAgent:
    """
    Claude-powered research agent running inside an OpenEnv environment.

    Uses the Anthropic SDK directly (matching the fastmcp_agent_tavily
    pattern) rather than OpenEnv's AnthropicClient wrapper, so the ReAct
    loop is explicit and inspectable for demo purposes.
    """

    def __init__(
        self,
        query: str,
        agent_id: int = 0,
        max_steps: int = 10,
    ):
        self.query = query
        self.agent_id = agent_id
        self.max_steps = max_steps
        self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def run(self, env: ResearchEnvironment) -> Generator[dict, None, None]:
        """
        Run one research episode, yielding a status dict after each step.

        Yields dicts with: step, tool_name, tool_args, llm_score, done, agent_id
        so the Gradio UI can update reward charts and step logs in real time.
        """
        env.reset(query=self.query)

        # Conversation history — grows with each tool call and result
        messages = [{"role": "user", "content": self.query}]

        for _ in range(self.max_steps):
            # Ask Claude what to do next given the conversation so far
            response = self._client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=_TOOL_SCHEMAS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                # Claude wants to call tools — add its response to history
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_name = block.name
                    tool_args = block.input

                    # Special case: finish action ends the episode
                    if tool_name == "finish":
                        action = ResearchAction(tool_name="finish", tool_args=tool_args)
                        step_result = env.step(action)
                        yield {
                            "step": step_result.observation.step,
                            "tool_name": "finish",
                            "tool_args": tool_args,
                            "llm_score": step_result.observation.reward,
                            "done": True,
                            "agent_id": self.agent_id,
                            "agent": "openenv",
                        }
                        return

                    # Execute the tool via the environment
                    action = ResearchAction(tool_name=tool_name, tool_args=tool_args)
                    step_result = env.step(action)
                    obs = step_result.observation

                    # Score with LLM judge
                    llm_score = llm_judge_reward(
                        query=self.query,
                        tool_name=tool_name,
                        result=obs.result,
                        step=obs.step,
                    )

                    yield {
                        "step": obs.step,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "llm_score": llm_score,
                        "result_preview": _preview(obs.result),
                        "done": obs.done,
                        "agent_id": self.agent_id,
                        "agent": "openenv",
                    }

                    # Feed the tool result back to Claude
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(obs.result)[:3000],  # truncate for token limits
                    })

                    if obs.done:
                        return

                messages.append({"role": "user", "content": tool_results})

            else:
                # Claude produced a final text answer — episode complete
                final_text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                yield {
                    "step": env.state.step,
                    "tool_name": "final_answer",
                    "tool_args": {},
                    "llm_score": env.state.total_reward / max(env.state.step, 1),
                    "result_preview": final_text[:500],
                    "done": True,
                    "agent_id": self.agent_id,
                    "agent": "openenv",
                }
                return


def _preview(result: dict, max_chars: int = 200) -> str:
    """Return a short readable preview of a tool result for the step log."""
    try:
        text = json.dumps(result)
        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    except Exception:
        return str(result)[:max_chars]
