"""
Traditional RL Agent — the "before" side of the demo comparison.

This agent deliberately mimics the limitations of classic RL approaches:

  1. Fixed discrete action space — it can only call tavily_search with
     pre-defined query templates. No ability to reason about which tool
     to use or adapt based on results.

  2. Keyword stuffing strategy — it constructs queries by appending as many
     query keywords as possible, explicitly to maximize the keyword_reward.
     This is reward hacking in action.

  3. No reasoning — actions are chosen by cycling through templates, not
     by reading or understanding the previous observation.

  4. No tool chaining — it never calls tavily_extract or tavily_crawl
     because its fixed policy doesn't include them.

The result: high keyword_reward scores (0.7-0.9), low llm_judge_reward
scores (0.2-0.4). This gap is the central demo moment.

Usage:
    agent = TraditionalAgent(query="What is MCP?")
    for step_result in agent.run(env):
        print(step_result)
"""

import re
from typing import Generator
from env.models import ResearchAction
from env.research_env import ResearchEnvironment
from reward import keyword_reward, llm_judge_reward


# Fixed action templates — the traditional agent's entire "policy"
# Each template stuffs the query keywords in different positions.
_QUERY_TEMPLATES = [
    "{query}",
    "{query} overview explanation",
    "{query} details facts sources",
    "{query} comprehensive information data",
    "{query} analysis research findings",
    "{keywords} {query} complete guide",
    "{keywords} {query} key points summary",
    "{keywords} {query} everything you need to know",
]


class TraditionalAgent:
    """
    A fixed-policy RL agent that keyword-stuffs to game the keyword reward.

    Demonstrates why naive reward functions fail for language tasks:
    the agent optimizes the metric, not the actual research goal.
    """

    def __init__(self, query: str, max_steps: int = 8):
        self.query = query
        self.max_steps = max_steps
        self._keywords = self._extract_keywords(query)

    def _extract_keywords(self, query: str) -> str:
        """Extract content words from the query for stuffing."""
        stopwords = {"the", "a", "an", "is", "are", "what", "how", "why", "of", "in", "on"}
        words = [w for w in re.findall(r'\w+', query.lower()) if w not in stopwords and len(w) > 2]
        # Repeat keywords to maximize match count — classic reward hacking
        return " ".join(words * 3)

    def _choose_action(self, step: int) -> ResearchAction:
        """
        Select the next action by cycling through templates.
        No reasoning, no adaptation — pure fixed policy.
        """
        template = _QUERY_TEMPLATES[step % len(_QUERY_TEMPLATES)]
        stuffed_query = template.format(
            query=self.query,
            keywords=self._keywords,
        )
        # Always uses tavily_search — never discovers extract or crawl
        return ResearchAction(
            tool_name="tavily_search",
            tool_args={"query": stuffed_query, "max_results": 5},
        )

    def run(self, env: ResearchEnvironment) -> Generator[dict, None, None]:
        """
        Run one episode, yielding a status dict after each step.

        Yields dicts with: step, action, keyword_score, llm_score, done
        so the Gradio UI can update the reward chart in real time.
        """
        step_result = env.reset(query=self.query)

        for step in range(self.max_steps):
            action = self._choose_action(step)

            step_result = env.step(action)
            obs = step_result.observation

            # Score the same result with BOTH reward functions
            # This is the moment we show to the audience
            kw_score = keyword_reward(
                query=self.query,
                tool_name=obs.tool_name,
                result=obs.result,
                step=obs.step,
            )
            llm_score = llm_judge_reward(
                query=self.query,
                tool_name=obs.tool_name,
                result=obs.result,
                step=obs.step,
            )

            yield {
                "step": obs.step,
                "tool_name": obs.tool_name,
                "query_used": action.tool_args.get("query", ""),
                "keyword_score": kw_score,
                "llm_score": llm_score,
                "done": obs.done,
                "agent": "traditional",
            }

            if obs.done:
                break
