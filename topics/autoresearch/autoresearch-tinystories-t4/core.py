"""
core.py — Single-experiment logic shared by agent.py and flyte_workflow.py.

Contains all helpers and run_single_experiment() so both orchestration
modes call the same code path.
"""

import difflib
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import anthropic

import firestore_logger
import metrics

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_SCRIPT = Path(__file__).parent / "train.py"
PROGRAM_MD   = Path(__file__).parent / "program.md"
TRAIN_BACKUP = Path(__file__).parent / "train.py.bak"

CLAUDE_MODEL      = "claude-sonnet-4-6"
_MAX_API_RETRIES  = 3
_RETRY_BASE_DELAY = 5  # seconds, doubles each attempt

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an AI research assistant running automated ML experiments.
Your job is to propose exactly ONE focused change to train.py to improve val_bpb.
You must follow the strategy in program.md exactly.
Respond with two sections:
1. REASONING: one short paragraph explaining what you are changing and why.
2. NEW_TRAIN_PY: the complete updated train.py file with your change applied.
No other text outside these two sections."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def compute_diff(before: str, after: str) -> str:
    """Return a unified diff string between two versions of train.py."""
    lines_before = before.splitlines(keepends=True)
    lines_after  = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        lines_before, lines_after,
        fromfile="train.py (before)",
        tofile="train.py (after)",
    )
    return "".join(diff)


def _find_section(response: str, marker: str) -> int:
    """
    Find the position of a section marker in the response, case-insensitively.

    Handles variations:
      - "REASONING:" / "NEW_TRAIN_PY:"
      - "## REASONING" / "## NEW_TRAIN_PY"  (markdown headings)
      - "1. REASONING:" / "**REASONING:**"  (numbered or bold)
    Returns the index just after the matched marker, or -1 if not found.
    """
    word = marker.rstrip(":")
    pattern = re.compile(
        r"^[ \t]*(?:#{1,3}\s+|\d+\.\s+|\*{1,2})?" + re.escape(word) + r"(?:\*{1,2})?:?[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(response)
    return match.end() if match else -1


def parse_llm_response(response: str) -> tuple[str, str]:
    """
    Extract REASONING and NEW_TRAIN_PY from Claude's response.

    Returns:
        (reasoning, new_train_py) — both as strings.
    Raises:
        ValueError if NEW_TRAIN_PY section is not found.
    """
    reasoning    = ""
    new_train_py = ""

    reasoning_pos = _find_section(response, "REASONING:")
    new_train_pos = _find_section(response, "NEW_TRAIN_PY:")

    if reasoning_pos != -1:
        end = new_train_pos if new_train_pos != -1 else len(response)
        reasoning = response[reasoning_pos:end].strip()

    if new_train_pos != -1:
        code_block = response[new_train_pos:].strip()
        if code_block.startswith("```"):
            code_block = code_block.split("\n", 1)[1]
        if code_block.endswith("```"):
            code_block = code_block.rsplit("```", 1)[0]
        new_train_py = code_block.strip()

    if not new_train_py:
        raise ValueError("Claude response did not contain NEW_TRAIN_PY section")

    return reasoning, new_train_py


def build_dynamic_prompt(current_train: str, experiment_history: list[dict]) -> str:
    """Build the per-experiment portion of the user prompt (uncached)."""
    history_lines = []
    for exp in experiment_history[-10:]:
        status = "KEPT" if exp.get("kept") else "REVERTED"
        history_lines.append(
            f"  Exp {exp['experiment_number']}: {exp['change_description']} "
            f"→ delta={exp['delta']:+.4f} [{status}]"
        )
    history_text = "\n".join(history_lines) if history_lines else "  (no experiments yet)"

    return f"""## Current train.py
```python
{current_train}
```

## Experiment History (most recent 10)
{history_text}

Propose ONE change to improve val_bpb. Follow the strategy guide."""


def call_claude(
    client: anthropic.Anthropic,
    program: str,
    current_train: str,
    experiment_history: list[dict],
) -> str:
    """
    Call Claude with prompt caching and retry on transient errors.

    System prompt and program.md are cached. Dynamic content is sent uncached.
    Retries up to _MAX_API_RETRIES times with exponential backoff.
    """
    for attempt in range(_MAX_API_RETRIES):
        try:
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                system=[
                    {"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
                ],
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"## Strategy Guide (program.md)\n{program}\n\n",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": build_dynamic_prompt(current_train, experiment_history),
                        },
                    ],
                }],
            )
            return message.content[0].text
        except anthropic.APIError as e:
            if attempt < _MAX_API_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                print(f"Claude API error (attempt {attempt + 1}/{_MAX_API_RETRIES}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def run_training() -> tuple[str, int]:
    """Run train.py as a subprocess and return (stdout+stderr, returncode)."""
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result.stdout + "\n" + result.stderr, result.returncode


def propose_change(
    client: anthropic.Anthropic,
    program: str,
    experiment_history: list[dict],
) -> tuple[str, str, str]:
    """Read current train.py, call Claude, parse the proposed change.

    Returns (reasoning, new_train_py, current_train).
    Raises ValueError if the response cannot be parsed.
    Raises anthropic.APIError after _MAX_API_RETRIES failed attempts.
    """
    current_train = read_file(TRAIN_SCRIPT)
    response_text = call_claude(client, program, current_train, experiment_history)
    reasoning, new_train_py = parse_llm_response(response_text)
    return reasoning, new_train_py, current_train


def apply_and_train(new_train_py: str, current_train: str) -> tuple[str, int, str]:
    """Back up train.py, write the new version, and run training.

    Returns (training_output, returncode, change_diff).
    returncode -2 signals a training timeout; train.py is already reverted.
    Any other non-zero returncode means training failed; train.py still needs revert.
    """
    shutil.copy(TRAIN_SCRIPT, TRAIN_BACKUP)
    write_file(TRAIN_SCRIPT, new_train_py)
    change_diff = compute_diff(current_train, new_train_py)
    try:
        training_output, returncode = run_training()
        return training_output, returncode, change_diff
    except subprocess.TimeoutExpired:
        shutil.copy(TRAIN_BACKUP, TRAIN_SCRIPT)
        return "", -2, change_diff


def evaluate_and_log(
    current_val_bpb: float,
    training_output: str,
    returncode: int,
    change_diff: str,
    reasoning: str,
    run_id: str | None,
    experiment_number: int,
    exp_start: str,
    exp_start_time: float,
    gcp_project: str | None,
    dry_run: bool = False,
) -> "ExperimentOutcome":
    """Keep/revert decision, build the experiment record, and log to Firestore.

    returncode -2 means training timed out and train.py is already reverted.
    Any other non-zero returncode means training failed; this function reverts.
    """
    duration = round(time.time() - exp_start_time, 1)

    if returncode != 0:
        if returncode != -2:
            shutil.copy(TRAIN_BACKUP, TRAIN_SCRIPT)
        print(f"Training failed (code={returncode}) — skipping.")
        return ExperimentOutcome(skipped=True, new_val_bpb=current_val_bpb)

    new_val_bpb = metrics.parse_val_bpb(training_output)
    if new_val_bpb is None:
        print("Could not parse val_bpb — reverting.")
        shutil.copy(TRAIN_BACKUP, TRAIN_SCRIPT)
        return ExperimentOutcome(skipped=True, new_val_bpb=current_val_bpb)

    result = metrics.build_experiment_result(current_val_bpb, new_val_bpb, training_output)

    if result.kept:
        print(f"KEPT   val_bpb {current_val_bpb:.6f} → {new_val_bpb:.6f} (delta={result.delta:+.6f})")
        updated_val_bpb = new_val_bpb
    else:
        print(f"REVERT val_bpb {current_val_bpb:.6f} → {new_val_bpb:.6f} (delta={result.delta:+.6f})")
        shutil.copy(TRAIN_BACKUP, TRAIN_SCRIPT)
        updated_val_bpb = current_val_bpb

    exp_record = {
        "experiment_number":  experiment_number,
        "started_at":         exp_start,
        "duration_seconds":   duration,
        "change_description": reasoning[:2000],
        "change_diff":        change_diff,
        "val_bpb_before":     result.val_bpb_before,
        "val_bpb_after":      result.val_bpb_after,
        "delta":              result.delta,
        "kept":               result.kept,
        "train_loss":         result.train_loss,
        "step_count":         result.step_count,
    }

    if run_id is not None and not dry_run:
        try:
            firestore_logger.log_experiment(
                run_id=run_id,
                project_id=gcp_project,
                experiment_number=exp_record["experiment_number"],
                started_at=exp_record["started_at"],
                duration_seconds=exp_record["duration_seconds"],
                change_description=exp_record["change_description"],
                change_diff=exp_record["change_diff"],
                val_bpb_before=exp_record["val_bpb_before"],
                val_bpb_after=exp_record["val_bpb_after"],
                kept=exp_record["kept"],
                train_loss=exp_record["train_loss"],
                step_count=exp_record["step_count"],
            )
        except Exception as e:
            print(f"WARNING: Firestore log_experiment failed: {e}. Continuing.")

    return ExperimentOutcome(skipped=False, new_val_bpb=updated_val_bpb, exp_record=exp_record)


# ── Single experiment ─────────────────────────────────────────────────────────

@dataclass
class ExperimentOutcome:
    skipped:      bool
    new_val_bpb:  float
    exp_record:   dict = field(default_factory=dict)


def run_single_experiment(
    client: anthropic.Anthropic,
    program: str,
    run_id: str | None,
    experiment_number: int,
    current_val_bpb: float,
    experiment_history: list[dict],
    gcp_project: str | None,
    dry_run: bool = False,
) -> ExperimentOutcome:
    """
    Run one full experiment cycle: propose → train → keep/revert → log.

    Returns ExperimentOutcome. If skipped=True the experiment could not
    complete (API error, training failure, parse error) and new_val_bpb
    is unchanged.
    """
    exp_start      = datetime.now(timezone.utc).isoformat()
    exp_start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Experiment {experiment_number} | val_bpb={current_val_bpb:.6f}")
    print(f"{'='*60}")

    # Propose
    try:
        reasoning, new_train_py, current_train = propose_change(client, program, experiment_history)
    except Exception as e:
        print(f"Proposal failed: {e}. Skipping.")
        return ExperimentOutcome(skipped=True, new_val_bpb=current_val_bpb)

    print(f"Change proposed: {reasoning[:200]}")

    # Apply + train
    training_output, returncode, change_diff = apply_and_train(new_train_py, current_train)

    # Evaluate + log
    return evaluate_and_log(
        current_val_bpb=current_val_bpb,
        training_output=training_output,
        returncode=returncode,
        change_diff=change_diff,
        reasoning=reasoning,
        run_id=run_id,
        experiment_number=experiment_number,
        exp_start=exp_start,
        exp_start_time=exp_start_time,
        gcp_project=gcp_project,
        dry_run=dry_run,
    )
