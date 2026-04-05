"""
Reward functions for the Research RL Environment.

Two reward functions are provided and used side-by-side in the demo:

1. keyword_reward  — Traditional RL style. Counts keyword matches in the
                     tool result. Easy to game — an agent that stuffs
                     keywords scores high even with garbage content.

2. llm_judge_reward — OpenEnv style. Uses Claude as a judge to evaluate
                      the quality and relevance of the tool result against
                      the original query. Much harder to game, much more
                      meaningful as a training signal.

The demo's "reward hacking" moment works by running the traditional agent
with keyword_reward (scores 8-9/10), then showing the same output through
llm_judge_reward (scores 2-3/10). The gap is the story.

Both functions share the same signature:
    reward_fn(query, tool_name, result, step) -> float (0.0 - 1.0)

so they are interchangeable when injected into ResearchEnvironment.
"""

import os
import re
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 1. Keyword Match Reward (Traditional RL)
# ---------------------------------------------------------------------------
# Reward = fraction of query keywords found in the tool result text.
# Simple, fast, and completely gameable.

def keyword_reward(
    query: str,
    tool_name: str,
    result: dict,
    step: int,
) -> float:
    """
    Traditional RL reward: count how many query keywords appear in the result.

    Returns a float between 0.0 and 1.0.
    Deliberately simple and gameable — this is the point of the demo.
    """
    if not result or "error" in result:
        return 0.0

    # Flatten result dict to a single string for keyword scanning
    result_text = _flatten_result(result).lower()

    # Tokenize query into keywords (strip stopwords crudely)
    stopwords = {"the", "a", "an", "is", "are", "what", "how", "why", "of", "in", "on"}
    keywords = [
        w for w in re.findall(r'\w+', query.lower())
        if w not in stopwords and len(w) > 2
    ]

    if not keywords:
        return 0.0

    matches = sum(1 for kw in keywords if kw in result_text)
    score = matches / len(keywords)

    # Small step penalty to discourage padding — but not enough to stop gaming
    step_penalty = max(0.0, (step - 1) * 0.02)
    return max(0.0, round(score - step_penalty, 3))


# ---------------------------------------------------------------------------
# 2. LLM-as-Judge Reward (OpenEnv style)
# ---------------------------------------------------------------------------
# Reward = Claude's quality rating of the tool result vs the research query.
# Returns a normalized float 0.0-1.0 (Claude rates 1-10, we divide by 10).

_anthropic_client = None

def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def llm_judge_reward(
    query: str,
    tool_name: str,
    result: dict,
    step: int,
) -> float:
    """
    OpenEnv-style reward: use Claude to judge the quality of the tool result.

    Claude evaluates whether the result genuinely advances the research goal,
    not just whether it contains the right words. Returns 0.0-1.0.

    Falls back to 0.0 on API errors so the episode can continue.
    """
    if not result or "error" in result:
        return 0.0

    result_text = _flatten_result(result)
    if not result_text.strip():
        return 0.0

    # Truncate to avoid token limits — first 2000 chars is enough to judge
    result_preview = result_text[:2000]

    prompt = f"""You are evaluating a web research agent's tool call result.

Research question: {query}
Tool used: {tool_name}
Tool result (truncated):
{result_preview}

Rate the quality of this result for answering the research question.
Consider: relevance, information density, source credibility, and whether
it genuinely advances understanding (not just keyword matches).

Respond with ONLY a single integer from 1 to 10.
1 = completely irrelevant or empty
5 = somewhat relevant but shallow
10 = highly relevant, rich, directly addresses the question
"""

    try:
        client = _get_anthropic_client()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",   # fast + cheap for reward scoring
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        score_text = response.content[0].text.strip()
        score = int(re.search(r'\d+', score_text).group())
        score = max(1, min(10, score))   # clamp to 1-10
        return round(score / 10.0, 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _flatten_result(result: dict) -> str:
    """Recursively flatten a result dict to a single string for text scanning."""
    parts = []
    for value in result.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    parts.append(_flatten_result(item))
                elif isinstance(item, str):
                    parts.append(item)
        elif isinstance(value, dict):
            parts.append(_flatten_result(value))
    return " ".join(parts)
