"""DreamerV3 on Flyte: learn a world model of MuJoCo, in a pod, with a report.

    flyte run pipeline.py train
    flyte run pipeline.py train --steps 20000        # is the plumbing alive?

Runs to the `world-models` project (.flyte/config.yaml), the one that will also hold
the Cosmos and V-JEPA 2 events. The robotics demos stay in `physical-ai`.

── Why the agent runs in a CHILD PROCESS ───────────────────────────────────────
Same reason as the Isaac Sim demo, and again not about imports. DreamerV3's entry
point is `dreamerv3/main.py`, a script that owns its own process: it parses flags,
builds a logger, spawns environment workers through `portal`, and runs a blocking
training loop. Importing it and calling `main()` inside a Flyte task would put those
env worker subprocesses and Flyte's asyncio runtime in the same process, which is the
shape that cost the Isaac Sim demo a 28-minute hung pod.

Spawning it means the task can also stream stdout and repaint the report while
training runs, instead of producing nothing for half an hour.

── Reading results ─────────────────────────────────────────────────────────────
Dreamer writes `metrics.jsonl` in its logdir, one JSON object per log flush, and that
is the honest source for the report: the curves come from the agent's own logger
rather than from anything scraped out of stdout.

Two things to know, both measured the hard way on the host:

  * `run.log_every` is a WALL CLOCK timer, not a step count, and it defaults to
    minutes. A short run finishes before it ever fires and writes an empty logdir,
    which looks exactly like a run that did nothing. This task sets it explicitly.

  * Training only starts once the replay buffer holds batch_size * batch_length
    transitions. Below that the loop is pure data collection, so a very short run can
    legitimately show no loss curves at all.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import flyte
import flyte.report

# Top level so Flyte bundles it into the pod.
import replay
import reports
from config import DREAMER_ROOT, gpu_env, orch_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Repaint the report at most this often. Dreamer's own log flush is the upper bound on
# how much new data there is to draw, so anything faster is just re-rendering HTML.
_REPAINT_SECS = 20


def _read_metrics(path: Path) -> tuple[list, dict, str]:
    """Parse Dreamer's metrics.jsonl into (score curve, loss curves, params blurb)."""
    if not path.exists():
        return [], {}, ""
    score: list[tuple[float, float]] = []
    losses: dict[str, list[tuple[float, float]]] = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = row.get("step")
        if step is None:
            continue
        if "episode/score" in row:
            score.append((float(step), float(row["episode/score"])))
        for key, value in row.items():
            if key.startswith("train/loss/") and isinstance(value, (int, float)):
                losses.setdefault(key.rsplit("/", 1)[-1], []).append(
                    (float(step), float(value))
                )
    return score, losses, ""


@gpu_env.task(report=True)
async def train(
    task_id: str = "dmc_walker_walk",
    config: str = "dmc_proprio",
    steps: int = 100_000,
    replay_steps: int = 500,
) -> dict:
    """Train DreamerV3 on a DMC task and report what the world model learned.

    Defaults are proprio rather than pixels: it is the cheaper of the two and it is
    the one whose numbers compare to the MJX + Brax run in topics/rl-mujoco, which is
    also state-based. `--config dmc_vision` switches to learning from pixels, which is
    the more impressive demo and needs the RTX/EGL path working in the pod.
    """
    logdir = Path("/tmp/dreamer") / task_id
    logdir.mkdir(parents=True, exist_ok=True)
    metrics = logdir / "metrics.jsonl"

    argv = [
        sys.executable, f"{DREAMER_ROOT}/dreamerv3/main.py",
        "--configs", config,
        "--task", task_id,
        "--logdir", str(logdir),
        "--run.steps", str(steps),
        # See the module docstring: this is a wall-clock timer and its default is
        # minutes, which is how a working run produces an empty logdir.
        "--run.log_every", "10",
    ]
    log.info("launching: %s", " ".join(argv))

    t0 = time.monotonic()
    tail: list[str] = []
    last_paint = 0.0
    env = dict(os.environ, PYTHONPATH=DREAMER_ROOT)
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, cwd=str(logdir), env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        tail.append(line.rstrip())
        del tail[:-60]
        now = time.monotonic()
        if now - last_paint >= _REPAINT_SECS:
            last_paint = now
            score, losses, _ = _read_metrics(metrics)
            done = int(score[-1][0]) if score else 0
            flyte.report.replace(
                reports.progress_html(task_id, done, steps, score, losses), do_flush=True
            )
    rc = proc.wait()
    secs = time.monotonic() - t0

    score, losses, params = _read_metrics(metrics)

    # rc alone is not trusted, for the same reason the Isaac Sim demo stopped trusting
    # it: a trainer that dies early can still exit 0, and an empty metrics file with a
    # clean exit is the signature. Fail here, while the log tail still explains it.
    if rc != 0:
        raise RuntimeError(f"dreamer exited {rc}. log tail:\n" + "\n".join(tail[-25:]))
    if not losses and not score:
        raise RuntimeError(
            "dreamer exited 0 but logged no metrics at all, which means it did not "
            "train. log tail:\n" + "\n".join(tail[-25:])
        )

    # Film the trained policy. Never allowed to fail the run: training is the
    # expensive part and a camera problem must not throw it away. The clip is probed
    # for luminance before it goes in the report, because a black clip is a valid mp4
    # that embeds as a black rectangle (see topics/isaac-sim/README.md).
    video_html, clip_probe = "", ""
    try:
        frames, ep_score = replay.record(logdir, steps=replay_steps)
        mp4 = replay.encode(frames)
        clip_probe = replay.probe(mp4)
        if mp4:
            video_html = replay.video_html(
                mp4,
                f"{len(mp4) / 1024:.0f} KB &middot; trained policy, episode return "
                f"{ep_score:.1f} &middot; {clip_probe}",
            )
        log.info("replay: %s", clip_probe)
    except Exception as exc:  # noqa: BLE001
        log.warning("replay failed, reporting without video: %s", exc)
        clip_probe = f"replay failed: {exc}"

    flyte.report.replace(
        reports.final_html(
            task_id, steps, secs, score, losses, params, tail, video_html, clip_probe
        ),
        do_flush=True,
    )
    best = max((y for _, y in score), default=0.0)
    log.info("trained %s for %s steps, best score %.1f", task_id, steps, best)
    return {
        "task": task_id,
        "config": config,
        "steps": steps,
        "minutes": round(secs / 60, 1),
        "episodes": len(score),
        "score_first": round(score[0][1], 2) if score else 0.0,
        "score_final": round(score[-1][1], 2) if score else 0.0,
        "score_best": round(best, 2),
        "loss_keys": sorted(losses),
        "clip": clip_probe,
    }


@orch_env.task(report=True)
async def dream(
    task_id: str = "dmc_walker_walk",
    config: str = "dmc_proprio",
    steps: int = 100_000,
    replay_steps: int = 500,
) -> dict:
    """Entry point. CPU-only orchestrator so it cannot deadlock its own GPU child."""
    result = await train(task_id=task_id, config=config, steps=steps,
                         replay_steps=replay_steps)
    log.info("result: %s", result)
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(dream))
