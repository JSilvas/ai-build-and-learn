"""
Local-LLM agent for autoresearch Mode D.

Asks an Ollama-served model for ONE focused experimental edit to train.py
expressed as aider-style SEARCH/REPLACE blocks (NOT unified diffs — small
models can't compute correct hunk headers reliably). We substitute the
SEARCH text with the REPLACE text directly in the file, then commit.

Format the model must produce:

    DESCRIPTION: <one-line summary>
    EDITS:
    <<<<<<< SEARCH
    <exact text from train.py to find — whitespace must match>
    =======
    <replacement text>
    >>>>>>> REPLACE

    (optionally more SEARCH/REPLACE blocks)

Why search/replace and not unified diff: in our first Mode D run on
qwen3-coder-next, every iteration's *reasoning* was correct ("reduce
DEPTH from 3 to 2") but the diffs were all rejected ("corrupt patch",
"patch failed"). Computing exact `@@ -line,count @@` headers is a known
weak spot for small coder models — but matching a literal chunk of
source they just read is well within reach. Aider, Cursor, and most
local-coding tools use this format for the same reason.

No reflection / retry. If the model's output can't be parsed or any
SEARCH block doesn't have a unique match, the iteration fails with a
descriptive [TAG] and the loop moves on. Matches karpathy's
"discard and move on" ethos.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("AUTORESEARCH_LOCAL_MODEL", "qwen3-coder-next")

DEFAULT_TIMEOUT_SEC = int(os.getenv("AUTORESEARCH_AGENT_TIMEOUT", "600"))

SYSTEM_PROMPT = (
    "You are a research assistant that proposes ONE experimental edit to a "
    "Python file. You output ONLY a DESCRIPTION line and one or more "
    "SEARCH/REPLACE blocks exactly in the format specified by the user. "
    "NO prose, NO explanations, NO markdown code fences."
)

USER_PROMPT_TEMPLATE = """{instructions}

==================== CURRENT train.py ====================
{train_py}
==================== END train.py ====================

==================== EXPERIMENT HISTORY (results.tsv) ====================
{history}
==================== END HISTORY ====================

Now produce your DESCRIPTION line and SEARCH/REPLACE block(s) in the strict
format the instructions above describe. Output NOTHING ELSE."""


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

DESC_RE = re.compile(r"^\s*DESCRIPTION:\s*(.+?)\s*$", re.MULTILINE)

# Match a SEARCH/REPLACE block. Greedy-but-bounded — REPLACE block runs until
# the closing marker. Allow common variations in fence-marker length / spacing.
SR_BLOCK_RE = re.compile(
    r"<{3,}\s*SEARCH\s*\n"      # opening: <<<<<<< SEARCH
    r"(?P<search>.*?)\n"        # search text
    r"={3,}\s*\n"               # divider: =======
    r"(?P<replace>.*?)"         # replace text (may be empty)
    r"\n>{3,}\s*REPLACE",       # closing: >>>>>>> REPLACE
    re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    """Some models still wrap the whole response in ```. Strip outermost fence."""
    text = re.sub(r"^```(?:\w+)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text


def parse_response(response: str) -> tuple[Optional[str], list[tuple[str, str]]]:
    """Returns (description, [(search, replace), ...]). Either may be empty."""
    cleaned = _strip_code_fences(response)
    desc_m = DESC_RE.search(cleaned)
    description = desc_m.group(1).strip() if desc_m else None
    blocks = [(m.group("search"), m.group("replace"))
              for m in SR_BLOCK_RE.finditer(cleaned)]
    return description, blocks


# ---------------------------------------------------------------------------
# Apply blocks to the target file
# ---------------------------------------------------------------------------

def apply_blocks(file_path: Path, blocks: list[tuple[str, str]]) -> tuple[bool, str]:
    """Apply each SEARCH/REPLACE block to file_path in order.

    Each SEARCH must appear EXACTLY ONCE in the current file (after prior
    blocks have been applied). Returns (ok, reason) — reason is empty on
    success or a tagged failure string like 'SEARCH NOT FOUND: <snippet>'
    on the first block that fails.
    """
    if not blocks:
        return False, "no SEARCH/REPLACE blocks parsed"

    contents = file_path.read_text()
    for i, (search, replace) in enumerate(blocks):
        if not search:
            return False, f"block {i}: empty SEARCH"
        count = contents.count(search)
        if count == 0:
            snippet = search.strip().splitlines()[0][:60] if search.strip() else "<blank>"
            return False, f"SEARCH NOT FOUND (block {i}, near: {snippet!r})"
        if count > 1:
            snippet = search.strip().splitlines()[0][:60]
            return False, f"AMBIGUOUS MATCH (block {i}, {count}x near: {snippet!r})"
        contents = contents.replace(search, replace, 1)

    file_path.write_text(contents)
    return True, ""


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def _ollama_chat(model: str, system: str, user: str,
                 timeout: int = DEFAULT_TIMEOUT_SEC,
                 url: str = OLLAMA_URL) -> str:
    r = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_ctx": 32768,
            },
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


# ---------------------------------------------------------------------------
# git commit
# ---------------------------------------------------------------------------

def _git_commit(repo: Path, target_rel: str, message: str) -> tuple[bool, str]:
    add = subprocess.run(
        ["git", "add", target_rel], cwd=str(repo), capture_output=True, text=True,
    )
    if add.returncode != 0:
        return False, f"git add failed: {add.stderr.strip()}"
    commit = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(repo), capture_output=True, text=True,
    )
    if commit.returncode != 0:
        return False, f"git commit failed: {commit.stderr.strip()}"
    return True, ""


# ---------------------------------------------------------------------------
# Public entry point — called by driver.py
# ---------------------------------------------------------------------------

def propose_change(repo: Path, train_py_path: Path, instructions_path: Path,
                   history: str, model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    """Ask the local model for ONE edit, apply + commit it.

    Returns (success, description). On failure (model unreachable, malformed
    output, SEARCH not found / ambiguous, git rejected), success=False and
    description starts with a [TAG] explaining the failure mode. Caller
    treats it as a discard and continues the loop.
    """
    train_py = train_py_path.read_text()
    instructions = instructions_path.read_text()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        instructions=instructions,
        train_py=train_py,
        history=history,
    )

    try:
        response = _ollama_chat(model, SYSTEM_PROMPT, user_prompt)
    except requests.exceptions.RequestException as e:
        return False, f"[OLLAMA UNREACHABLE] {e}"

    description, blocks = parse_response(response)
    if description is None:
        description = "(model omitted DESCRIPTION line)"
    if not blocks:
        return False, f"[INVALID OUTPUT: no SEARCH/REPLACE blocks] {description}"

    ok, reason = apply_blocks(train_py_path, blocks)
    if not ok:
        return False, f"[{reason}] {description}"

    target_rel = str(train_py_path.relative_to(repo))
    ok, err = _git_commit(repo, target_rel, description)
    if not ok:
        return False, f"[{err}] {description}"

    return True, description
