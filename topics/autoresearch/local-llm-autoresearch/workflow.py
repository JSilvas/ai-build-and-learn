"""
Flyte workflow wrapping the autoresearch experiment loop.

Each iteration is its own Flyte task, so when you `flyte run --local` this
file you get a per-iteration node in the Flyte TUI with a live HTML report
showing the change description, val_bpb, status (keep/discard/crash), and
the log tail. Iterations run sequentially because each one depends on the
git state (and best-so-far val_bpb) from the previous one.

Usage:
    flyte run --local workflow.py run_autoresearch \\
        --tag demo --iterations 3

Or kick off a longer overnight run:
    flyte run --local workflow.py run_autoresearch \\
        --tag overnight --iterations 100
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from dataclasses import dataclass

import flyte
import flyte.report
from flyte import Link

import driver

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom links — show up in the task-detail panel of the Flyte TUI/web UI
# ---------------------------------------------------------------------------

REPORT_SERVER_PORT = int(os.getenv("AUTORESEARCH_REPORT_PORT", "8080"))


@dataclass
class ReportLink(Link):
    """Clickable link that opens the task's HTML report in a browser.

    Requires a tiny HTTP server running on the report dir:
        python -m http.server 8080 --directory /tmp/flyte/metadata &

    VSCode tunnels auto-forward the port, so localhost:8080 reaches the
    DGX even when you're remoting in.
    """
    name: str = "Report"
    icon_uri: str = ""
    port: int = REPORT_SERVER_PORT

    def get_link(self, run_name, action_name, **kwargs):
        return f"http://localhost:{self.port}/{run_name}/{action_name}/report.html"


@dataclass
class UpstreamLink(Link):
    """Static link to karpathy's upstream repo."""
    name: str = "Upstream"
    icon_uri: str = ""
    url: str = "https://github.com/karpathy/autoresearch"

    def get_link(self, **kwargs):
        return self.url


report_link = ReportLink()
upstream_link = UpstreamLink()

env = flyte.TaskEnvironment(
    name="autoresearch-env",
    image=flyte.Image.from_debian_base(python_version=(3, 11)).with_pip_packages(
        "flyte>=2.1.2",
        "python-dotenv",
        "markdown",
    ),
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _row_to_html(r: driver.IterationResult) -> str:
    badge_color = {
        "baseline": "#6c757d",
        "keep": "#1a7f37",
        "discard": "#d97706",
        "crash": "#cf222e",
    }.get(r.status, "#6c757d")
    val_bpb_str = f"{r.val_bpb:.6f}" if r.val_bpb is not None else "—"
    mem_str = f"{r.memory_gb:.1f} GB" if r.memory_gb is not None else "—"
    return f"""
<h2>Iteration {r.iteration}</h2>
<p>
  <span style="background:{badge_color};color:white;padding:2px 8px;border-radius:4px;">
    {r.status.upper()}
  </span>
  &nbsp;<b>commit:</b> <code>{r.commit}</code>
  &nbsp;<b>elapsed:</b> {r.elapsed_sec:.1f}s
</p>
<p><b>Change:</b> {r.description}</p>
<p><b>val_bpb:</b> {val_bpb_str} &nbsp;|&nbsp; <b>peak VRAM:</b> {mem_str}</p>
<details><summary>Log tail (last 30 lines)</summary>
<pre style="background:#f6f8fa;padding:8px;font-size:12px;overflow:auto;">{r.log_tail}</pre>
</details>
"""


# ---------------------------------------------------------------------------
# Task: baseline run (no agent, just measure HEAD)
# ---------------------------------------------------------------------------

@env.task(report=True, links=(report_link, upstream_link))
async def run_baseline(tag: str) -> str:
    branch = driver.ensure_branch(tag)
    driver.ensure_results_tsv()

    await flyte.report.replace.aio(
        f"<h2>Iteration 0 — Baseline</h2>"
        f"<p>Branch: <code>{branch}</code></p>"
        f"<p>Running <code>uv run train.py</code> at HEAD (5 min budget)…</p>"
    )
    await flyte.report.flush.aio()

    result = driver.baseline(branch)
    await flyte.report.replace.aio(_row_to_html(result))
    await flyte.report.flush.aio()
    return json.dumps(_to_dict(result))


# ---------------------------------------------------------------------------
# Task: one experiment iteration
# ---------------------------------------------------------------------------

@env.task(report=True, links=(report_link, upstream_link))
async def run_iteration(tag: str, iteration: int, prev_best_bpb: float,
                         agent: str = "claude", model: str = "",
                         instructions: str = "") -> str:
    branch = f"autoresearch/{tag}"
    instructions_path = pathlib.Path(instructions) if instructions else None
    model_arg = model or None

    pretty_model = model or ("sonnet" if agent == "claude" else "qwen3-coder-next")
    await flyte.report.replace.aio(
        f"<h2>Iteration {iteration}</h2>"
        f"<p>Branch: <code>{branch}</code> &nbsp; "
        f"prev best val_bpb: <b>{prev_best_bpb:.6f}</b></p>"
        f"<p>Agent: <code>{agent}</code> ({pretty_model}). "
        f"Asking for a change, then running training…</p>"
    )
    await flyte.report.flush.aio()

    result = driver.iterate(
        iteration=iteration, branch=branch, prev_best_bpb=prev_best_bpb,
        agent=agent, model=model_arg, instructions_path=instructions_path,
    )
    await flyte.report.replace.aio(_row_to_html(result))
    await flyte.report.flush.aio()
    return json.dumps(_to_dict(result))


# ---------------------------------------------------------------------------
# Workflow: orchestrate sequential iterations
# ---------------------------------------------------------------------------

@env.task(report=True, links=(report_link, upstream_link))
async def run_autoresearch(tag: str, iterations: int = 3,
                            agent: str = "claude", model: str = "",
                            instructions: str = "") -> str:
    """Run baseline + N agent-driven iterations, sequentially."""
    pretty_model = model or ("sonnet" if agent == "claude" else "qwen3-coder-next")
    await flyte.report.replace.aio(
        f"<h2>AutoResearch — {tag}</h2>"
        f"<p>Iterations: {iterations} (+ baseline) &nbsp;|&nbsp; "
        f"agent: <code>{agent}</code> ({pretty_model})</p>"
        f"<p>Starting…</p>"
    )
    await flyte.report.flush.aio()

    base_json = await run_baseline(tag=tag)
    base = json.loads(base_json)
    history = [base]
    best = base["val_bpb"] if base["val_bpb"] is not None else float("inf")

    if base["val_bpb"] is None:
        summary = _summary_html(tag, history, "Baseline crashed — aborting.")
        await flyte.report.replace.aio(summary)
        await flyte.report.flush.aio()
        return json.dumps({"results": history, "best_val_bpb": None})

    for i in range(1, iterations + 1):
        r_json = await run_iteration(
            tag=tag, iteration=i, prev_best_bpb=best,
            agent=agent, model=model, instructions=instructions,
        )
        r = json.loads(r_json)
        history.append(r)
        if r["status"] == "keep" and r["val_bpb"] is not None:
            best = r["val_bpb"]
        await flyte.report.replace.aio(_summary_html(tag, history))
        await flyte.report.flush.aio()

    await flyte.report.replace.aio(_summary_html(tag, history, "Done."))
    await flyte.report.flush.aio()
    return json.dumps({"results": history, "best_val_bpb": best})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(r: driver.IterationResult) -> dict:
    return {
        "iteration": r.iteration,
        "commit": r.commit,
        "description": r.description,
        "val_bpb": r.val_bpb,
        "memory_gb": r.memory_gb,
        "status": r.status,
        "elapsed_sec": r.elapsed_sec,
    }


def _history_to_plot_rows(history: list[dict]) -> list[dict]:
    """Convert the workflow's in-memory history dicts to the format plot_progress expects."""
    return [
        {
            "experiment": i,
            "commit": r.get("commit", ""),
            "val_bpb": r.get("val_bpb"),
            "memory_gb": r.get("memory_gb"),
            "status": r.get("status", ""),
            "description": r.get("description", ""),
        }
        for i, r in enumerate(history)
    ]


def _summary_html(tag: str, history: list[dict], suffix: str = "") -> str:
    rows = ""
    for r in history:
        bpb = f"{r['val_bpb']:.6f}" if r['val_bpb'] is not None else "—"
        mem = f"{r['memory_gb']:.1f}" if r['memory_gb'] is not None else "—"
        rows += (
            f"<tr><td>{r['iteration']}</td><td>{r['status']}</td>"
            f"<td><code>{r['commit']}</code></td>"
            f"<td>{bpb}</td><td>{mem}</td>"
            f"<td>{r['description'][:80]}</td></tr>"
        )

    chart_html = ""
    try:
        from plot_progress import build_chart
        plot_rows = _history_to_plot_rows(history)
        title = f"AutoResearch — {tag}: {len(history)} experiments"
        fig = build_chart(plot_rows, title=title)
        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as e:
        chart_html = f"<p><i>Chart unavailable: {e}</i></p>"

    return f"""
<h2>AutoResearch — {tag}</h2>
<p>{suffix}</p>
{chart_html}
<table border="1" cellpadding="6" style="border-collapse:collapse;">
  <tr style="background:#f6f8fa;">
    <th>#</th><th>Status</th><th>Commit</th>
    <th>val_bpb</th><th>VRAM (GB)</th><th>Description</th>
  </tr>
  {rows}
</table>
"""


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    print("Use: flyte run --local workflow.py run_autoresearch --tag demo --iterations 3")
