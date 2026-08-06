"""Train a quadruped to walk in a Flyte pod, and put the gait in the report.

The GPU task is `walk_task` below. It does two things in ONE pod, deliberately:

    train   Isaac Lab + rsl_rl PPO, thousands of envs on the GB10
    record  replay the trained policy through the RTX renderer -> mp4 -> report

The MuJoCo demo next door splits these into two tasks because MJX has no renderer, so
the replay has to happen somewhere else. Isaac Sim renders perfectly well, and this
box has exactly one GPU, so splitting would mean shipping a checkpoint through blob
storage and then queueing for the same GPU again. One task is both simpler and faster.

── Everything here runs Isaac Lab as a CHILD PROCESS ───────────────────────────
Same reason as the smoke test, and it is not about imports: Kit's shutdown cancels
every asyncio task in the process, including Flyte's own runtime. See pipeline.py.
It also happens to be how Isaac Lab is designed to be driven; its RL entry points are
scripts, not libraries.

The upside is that the reward curve can be streamed: we read rsl_rl's stdout line by
line and push the curve into the live Flyte report as it trains, so you can watch the
number climb instead of waiting half an hour to find out it learned to sit down.
"""

from __future__ import annotations

import base64
import logging
import re
import subprocess
import sys
from pathlib import Path

import flyte.report

log = logging.getLogger(__name__)

# Where Isaac Lab lives in the training image (see Dockerfile.train).
ISAACLAB = Path("/isaac-lab")
RSL_RL = ISAACLAB / "scripts" / "reinforcement_learning" / "rsl_rl"

# rsl_rl writes checkpoints to ./logs/rsl_rl/<experiment>/<timestamp>/ RELATIVE TO CWD.
# /isaac-lab is chmod a+rX in the image (readable, not writable), so running from
# there would fail on the first checkpoint save. Everything runs from here instead.
WORKDIR = Path("/tmp/isaac-run")

_REWARD_RE = re.compile(r"Mean reward:\s*(-?[\d.]+)")
_ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")


def _run_streaming(argv: list[str], on_line) -> int:
    """Run a child process, calling on_line for each stdout line. Returns exit code."""
    proc = subprocess.Popen(
        argv,
        cwd=str(WORKDIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        on_line(line.rstrip())
    return proc.wait()


def train(task_id: str, num_envs: int, iterations: int) -> tuple[list[float], list[str]]:
    """Run rsl_rl PPO. Returns (reward curve, tail of the log)."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    rewards: list[float] = []
    tail: list[str] = []
    state = {"iter": 0, "total": iterations, "flushed": 0}

    def on_line(line: str) -> None:
        tail.append(line)
        del tail[:-40]

        if m := _ITER_RE.search(line):
            state["iter"], state["total"] = int(m.group(1)), int(m.group(2))
        if m := _REWARD_RE.search(line):
            rewards.append(float(m.group(1)))
            # Repaint every 25 points. Flushing every iteration would spend more time
            # writing HTML than training; never flushing means an empty report for
            # half an hour, which is the failure mode this is here to avoid.
            if len(rewards) - state["flushed"] >= 25:
                state["flushed"] = len(rewards)
                report_progress(task_id, num_envs, rewards, state["iter"], state["total"])

    rc = _run_streaming(
        [sys.executable, str(RSL_RL / "train.py"), f"--task={task_id}", "--headless",
         "--num_envs", str(num_envs), "--max_iterations", str(iterations)],
        on_line,
    )
    if rc != 0:
        raise RuntimeError(f"training exited {rc}. log tail:\n" + "\n".join(tail))
    return rewards, tail


def record(task_id: str, video_length: int = 300) -> Path | None:
    """Replay the trained policy through the RTX renderer. Returns the mp4, if any.

    --num_envs 1 on purpose: the default viewport camera frames the whole grid, and at
    4096 envs the robots are ant-sized specks. With one robot you can actually see the
    gait, which is the entire point of putting a video in the report.
    """
    tail: list[str] = []

    def on_line(line: str) -> None:
        tail.append(line)
        del tail[:-25]

    rc = _run_streaming(
        [sys.executable, str(RSL_RL / "play.py"), f"--task={task_id}", "--headless",
         "--num_envs", "1", "--video", "--video_length", str(video_length)],
        on_line,
    )
    if rc != 0:
        log.warning("play.py exited %s; continuing without video. tail:\n%s", rc, "\n".join(tail))
        return None

    # Newest mp4 under the run's log tree. play.py names it rl-video-step-0.mp4 and
    # buries it a few levels down, so glob rather than guessing the path.
    clips = sorted(WORKDIR.glob("logs/**/*.mp4"), key=lambda p: p.stat().st_mtime)
    return clips[-1] if clips else None


# ── Report ──────────────────────────────────────────────────────────────────────

def _curve_svg(rewards: list[float], w: int = 760, h: int = 240) -> str:
    """Reward curve as a hand-rolled SVG polyline.

    Hand-rolled because the training image has no matplotlib and adding it to a 25 GB
    image for one line chart is a poor trade. An SVG polyline needs no dependency and
    scales in the browser.
    """
    if len(rewards) < 2:
        return "<p><i>not enough points yet</i></p>"

    lo, hi = min(rewards), max(rewards)
    span = (hi - lo) or 1.0
    pad = 34
    pts = " ".join(
        f"{pad + i * (w - 2 * pad) / (len(rewards) - 1):.1f},"
        f"{h - pad - (r - lo) / span * (h - 2 * pad):.1f}"
        for i, r in enumerate(rewards)
    )
    zero = ""
    if lo < 0 < hi:
        y = h - pad - (0 - lo) / span * (h - 2 * pad)
        zero = f"<line x1='{pad}' y1='{y:.1f}' x2='{w - pad}' y2='{y:.1f}' stroke='#888' stroke-dasharray='4 4'/>"
    return (
        f"<svg viewBox='0 0 {w} {h}' style='width:100%;max-width:{w}px;background:#111'>"
        f"{zero}"
        f"<polyline points='{pts}' fill='none' stroke='#5cf' stroke-width='2'/>"
        f"<text x='{pad}' y='16' fill='#aaa' font-size='12'>max {hi:.2f}</text>"
        f"<text x='{pad}' y='{h - 10}' fill='#aaa' font-size='12'>min {lo:.2f}</text>"
        f"</svg>"
    )


def report_progress(task_id: str, num_envs: int, rewards: list[float], it: int, total: int) -> None:
    flyte.report.log(
        f"<h2>{task_id}</h2>"
        f"<p>{num_envs:,} parallel envs &middot; iteration <b>{it}</b>/{total} "
        f"&middot; mean reward <b>{rewards[-1]:.2f}</b> (from {rewards[0]:.2f})</p>"
        f"{_curve_svg(rewards)}",
        do_flush=True,
    )


def report_final(
    task_id: str, num_envs: int, iterations: int, rewards: list[float],
    clip: Path | None, secs: float, tail: list[str],
) -> None:
    """Final report: the curve says the number went up, the clip says it went up for the right reason."""
    if clip and clip.exists():
        b64 = base64.b64encode(clip.read_bytes()).decode()
        # Base64 straight into a <video> tag: no JS, no external file, and the clip
        # survives in the report after the pod is gone.
        video = (
            f"<video controls autoplay loop muted style='width:100%;max-width:760px'>"
            f"<source src='data:video/mp4;base64,{b64}' type='video/mp4'></video>"
            f"<p style='color:#888;font-size:12px'>{clip.stat().st_size / 1024:.0f} KB, "
            f"rendered headless through the RTX renderer in the pod</p>"
        )
    else:
        video = "<p><b>No clip.</b> Training succeeded; the replay render did not. See the log tail.</p>"

    # env-steps/sec is the number that compares to the MuJoCo demo. rsl_rl collects
    # num_steps_per_env (24 for this config) steps per env per iteration.
    steps = num_envs * 24 * max(len(rewards), 1)
    rate = steps / secs if secs else 0

    flyte.report.log(
        f"<h2>{task_id}</h2>"
        f"<p>{num_envs:,} parallel envs &middot; {iterations} iterations &middot; {secs / 60:.1f} min</p>"
        f"<p>mean reward <b>{rewards[0]:.2f}</b> &rarr; <b>{rewards[-1]:.2f}</b> "
        f"&middot; ~<b>{rate:,.0f}</b> env-steps/sec</p>"
        f"{_curve_svg(rewards)}"
        f"<h3>The trained policy</h3>{video}"
        f"<h3>log tail</h3><pre style='font-size:11px'>{chr(10).join(tail[-20:])}</pre>",
        do_flush=True,
    )
