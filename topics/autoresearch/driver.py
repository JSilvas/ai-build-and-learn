"""
AutoResearch iteration driver.

One iteration =
    1. Ask Claude Code to make ONE experimental edit to train.py and commit.
    2. Run `uv run train.py` for the fixed time budget.
    3. Parse val_bpb / peak VRAM from run.log.
    4. Keep the commit (if better) or `git reset --hard` (if worse / crashed).
    5. Append a row to results.tsv.

The driver is intentionally subprocess-shaped so the same code path works
whether you call it from a plain `python driver.py` loop or from the
Flyte workflow in workflow.py.

Karpathy's repo lives in ./upstream and is its own git repo — all the
research-branch churn happens there, never in the parent build-learn repo.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

UPSTREAM = Path(__file__).parent / "upstream"
RESULTS_TSV = UPSTREAM / "results.tsv"
RUN_LOG = UPSTREAM / "run.log"
RESULTS_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"

# How long to allow `uv run train.py` to live before we kill it.
# Karpathy: 5 min training + a bit of compile/eval overhead. Cap hard at 10 min.
TRAINING_TIMEOUT_SEC = 10 * 60

# Cap how long Claude has to propose a single edit.
AGENT_TIMEOUT_SEC = 5 * 60


@dataclasses.dataclass
class IterationResult:
    iteration: int
    commit: str
    description: str
    val_bpb: Optional[float]
    memory_gb: Optional[float]
    status: str  # "keep" | "discard" | "crash" | "baseline"
    elapsed_sec: float
    log_tail: str  # last few lines of run.log for the report


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path, timeout: Optional[int] = None,
         capture: bool = True, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run a subprocess with sane defaults. Raises on timeout but not on nonzero exit."""
    return subprocess.run(
        cmd, cwd=str(cwd), timeout=timeout,
        capture_output=capture, text=True, env=env,
    )


def _git(args: list[str], cwd: Path = UPSTREAM) -> str:
    r = _run(["git", *args], cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def current_commit() -> str:
    return _git(["rev-parse", "--short=7", "HEAD"])


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def ensure_branch(tag: str) -> str:
    """Create or switch to autoresearch/<tag>. Returns the branch name."""
    branch = f"autoresearch/{tag}"
    branches = _git(["branch", "--list", branch])
    if branches:
        _git(["checkout", branch])
    else:
        _git(["checkout", "-b", branch])
    return branch


def ensure_results_tsv() -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER + "\n")


def append_result(row: IterationResult) -> None:
    line = "\t".join([
        row.commit,
        f"{row.val_bpb:.6f}" if row.val_bpb is not None else "0.000000",
        f"{row.memory_gb:.1f}" if row.memory_gb is not None else "0.0",
        row.status,
        row.description.replace("\t", " ").replace("\n", " "),
    ])
    with RESULTS_TSV.open("a") as f:
        f.write(line + "\n")


def read_history(max_rows: int = 30) -> str:
    """Return the last N rows of results.tsv as a string for the agent prompt."""
    if not RESULTS_TSV.exists():
        return RESULTS_HEADER + "\n(no experiments yet)\n"
    lines = RESULTS_TSV.read_text().splitlines()
    if len(lines) <= max_rows + 1:
        return "\n".join(lines)
    return "\n".join([lines[0], *lines[-max_rows:]])


# ---------------------------------------------------------------------------
# Claude Code invocation
# ---------------------------------------------------------------------------

AGENT_PROMPT_TEMPLATE = """You are running ONE iteration of an autonomous LLM pretraining research loop.

Working directory: {repo}
Current branch: {branch}

Rules (from program.md):
- You may ONLY edit train.py. Do NOT edit prepare.py. Do NOT add dependencies.
- You will NOT run training — that happens externally after you exit.
- Make exactly ONE focused experimental change you have not tried before.
- Commit the change with `git add train.py && git commit -m "<short description>"`.

Previous experiment history (results.tsv, lower val_bpb is better):
```
{history}
```

Now: read train.py, pick one focused change, edit train.py, commit it, then stop.

Your final stdout line MUST be:
CHANGE: <one-line description of what you changed>
"""


def propose_change(branch: str, model: str = "sonnet") -> str:
    """Spawn Claude Code in the upstream/ dir to make one edit + commit. Returns description."""
    history = read_history()
    prompt = AGENT_PROMPT_TEMPLATE.format(repo=UPSTREAM, branch=branch, history=history)

    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--add-dir", str(UPSTREAM),
        "--allowedTools", "Read,Edit,Write,Bash,Glob,Grep",
    ]
    print(f"[agent] launching: {shlex.join(cmd[:6])} ...", flush=True)
    r = _run(cmd, cwd=UPSTREAM, timeout=AGENT_TIMEOUT_SEC)
    if r.returncode != 0:
        raise RuntimeError(f"claude -p failed (rc={r.returncode}):\n{r.stderr[-2000:]}")

    # Extract "CHANGE: ..." line if the agent followed instructions.
    desc = ""
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("CHANGE:"):
            desc = line[len("CHANGE:"):].strip()
            break
    if not desc:
        # Fall back to the latest commit subject — the agent should have committed.
        try:
            desc = _git(["log", "-1", "--format=%s"])
        except Exception:
            desc = "(no description)"
    return desc


# ---------------------------------------------------------------------------
# Training run + result parsing
# ---------------------------------------------------------------------------

VAL_BPB_RE = re.compile(r"^val_bpb:\s+([0-9.]+)", re.MULTILINE)
VRAM_RE = re.compile(r"^peak_vram_mb:\s+([0-9.]+)", re.MULTILINE)


def run_training() -> tuple[Optional[float], Optional[float], str, bool]:
    """Run `uv run train.py > run.log 2>&1` with a hard timeout.

    Returns (val_bpb, memory_gb, log_tail, timed_out).
    val_bpb / memory_gb are None if the run crashed or was killed.
    """
    if RUN_LOG.exists():
        RUN_LOG.unlink()
    timed_out = False
    with RUN_LOG.open("w") as f:
        # Use whichever python is active (build-learn venv by default). Falls
        # back nicely if the user instead `uv sync`'d inside upstream/ and
        # activated upstream/.venv.
        proc = subprocess.Popen(
            [sys.executable, "train.py"], cwd=str(UPSTREAM),
            stdout=f, stderr=subprocess.STDOUT,
        )
        try:
            proc.wait(timeout=TRAINING_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True

    log = RUN_LOG.read_text() if RUN_LOG.exists() else ""
    val_bpb = None
    memory_gb = None
    m = VAL_BPB_RE.search(log)
    if m:
        val_bpb = float(m.group(1))
    m = VRAM_RE.search(log)
    if m:
        memory_gb = round(float(m.group(1)) / 1024.0, 1)
    log_tail = "\n".join(log.splitlines()[-30:])
    return val_bpb, memory_gb, log_tail, timed_out


# ---------------------------------------------------------------------------
# Iteration orchestration
# ---------------------------------------------------------------------------

def baseline(branch: str) -> IterationResult:
    """First run: no agent edit, just measure baseline at HEAD."""
    t0 = time.time()
    val_bpb, memory_gb, log_tail, timed_out = run_training()
    elapsed = time.time() - t0
    commit = current_commit()
    if val_bpb is None:
        status = "crash"
    else:
        status = "baseline"
    result = IterationResult(
        iteration=0, commit=commit, description="baseline (no edits)",
        val_bpb=val_bpb, memory_gb=memory_gb, status=status,
        elapsed_sec=elapsed, log_tail=log_tail,
    )
    append_result(result)
    return result


def iterate(iteration: int, branch: str, prev_best_bpb: float,
            model: str = "sonnet") -> IterationResult:
    """One agent-driven experiment. Reverts on no improvement / crash."""
    start_commit = current_commit()
    t0 = time.time()

    description = propose_change(branch=branch, model=model)
    new_commit = current_commit()
    if new_commit == start_commit:
        # Agent didn't commit anything — skip this iteration cleanly.
        elapsed = time.time() - t0
        return IterationResult(
            iteration=iteration, commit=start_commit,
            description=description or "(no commit produced)",
            val_bpb=None, memory_gb=None, status="discard",
            elapsed_sec=elapsed, log_tail="agent did not commit",
        )

    val_bpb, memory_gb, log_tail, timed_out = run_training()
    elapsed = time.time() - t0

    if val_bpb is None:
        status = "crash"
        if timed_out:
            description = f"[TIMEOUT] {description}"
        _git(["reset", "--hard", start_commit])
    elif val_bpb < prev_best_bpb:
        status = "keep"
    else:
        status = "discard"
        _git(["reset", "--hard", start_commit])

    result = IterationResult(
        iteration=iteration,
        commit=new_commit if status == "keep" else start_commit,
        description=description,
        val_bpb=val_bpb, memory_gb=memory_gb, status=status,
        elapsed_sec=elapsed, log_tail=log_tail,
    )
    append_result(result)
    return result


def loop(tag: str, iterations: int, model: str = "sonnet") -> list[IterationResult]:
    """Standalone loop runner. workflow.py calls iterate() directly per Flyte task."""
    branch = ensure_branch(tag)
    ensure_results_tsv()

    results: list[IterationResult] = []
    base = baseline(branch)
    results.append(base)
    print(f"[iter 0] baseline val_bpb={base.val_bpb} status={base.status}", flush=True)

    if base.val_bpb is None:
        print("[loop] baseline crashed — aborting", flush=True)
        return results

    best = base.val_bpb
    for i in range(1, iterations + 1):
        r = iterate(i, branch=branch, prev_best_bpb=best, model=model)
        results.append(r)
        bpb_str = f"{r.val_bpb:.6f}" if r.val_bpb is not None else "crash"
        print(f"[iter {i}] {r.status:8s} val_bpb={bpb_str}  {r.description[:80]}", flush=True)
        if r.status == "keep" and r.val_bpb is not None:
            best = r.val_bpb
    return results


def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Run the autoresearch loop locally.")
    p.add_argument("--tag", required=True, help="Run tag, e.g. 'demo'. Branch will be autoresearch/<tag>.")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--model", default="sonnet", help="Claude model alias for the agent.")
    args = p.parse_args()
    loop(tag=args.tag, iterations=args.iterations, model=args.model)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
