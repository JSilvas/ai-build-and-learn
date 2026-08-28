"""The orchestrators: train a G1 policy and show it walking, or compare reward presets.

Two entry points:

    # One policy, one clip. The main event.
    flyte run pipeline.py walk --num_timesteps 20000000 --preset baseline

    # Several reward presets, one report, one chart with every curve on it.
    flyte run pipeline.py compare_presets --presets '["baseline","high-step"]'

── A note on "parallel" ────────────────────────────────────────────────────────
`compare_presets` submits its training tasks concurrently, but this box has exactly
one GPU, so Flyte runs them one at a time and the rest sit Pending. That is fine and
intended: the point of the fan-out is that you launch once and collect one comparison
report, not that the wall clock shrinks. Budget N x the single-run time.

The orchestrator itself is deliberately CPU-only (see config.orch_env): an
orchestrator pod holds its resources for as long as its children run, so a GPU-holding
orchestrator would deadlock its own GPU child on "Insufficient nvidia.com/gpu".
"""

from __future__ import annotations

import json
import logging
import pickle

import flyte
import flyte.report

from config import orch_env
import envs
import reports
from render import render_replay
from train import train_policy

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@orch_env.task(report=True)
async def walk(
    num_timesteps: int = 20_000_000,
    num_envs: int = 4096,
    num_evals: int = 10,
    preset: str = envs.DEFAULT_PRESET,
    seed: int = 0,
    domain_randomization: bool = True,
    replay_steps: int = 500,
    snapshot_every_pct: float = 15.0,
    terrain: str = envs.DEFAULT_TERRAIN,
) -> str:
    """Train one G1 policy, then replay it into the report.

    `snapshot_every_pct` controls the mid-training clips that appear in the TRAINING
    task's live report (not this one): the current policy filmed every N% of the run,
    overwritten each time, with a filmstrip of stills showing the gait developing.
    Set 0 to turn it off.
    """
    import asyncio

    envs.get_preset(preset)   # fail fast on a typo, before burning an hour of GPU
    envs.env_name(terrain)    # same, for terrain

    await flyte.report.replace.aio(
        "<h2>Unitree G1 - learning to walk (MJX + Brax PPO)</h2>"
        f"<p>Training preset <b>{preset}</b> on <b>{terrain}</b> terrain for {num_timesteps:,} steps "
        f"across {num_envs:,} parallel environments...</p>"
    )
    await flyte.report.flush.aio()

    checkpoint = await train_policy(
        num_timesteps=num_timesteps,
        num_envs=num_envs,
        num_evals=num_evals,
        preset=preset,
        seed=seed,
        domain_randomization=domain_randomization,
        snapshot_every_pct=snapshot_every_pct,
        terrain=terrain,
    )

    # Two clips, not one. The trained policy on its own is unevaluable: "is that good
    # walking?" has no answer without seeing what untrained looks like on the same
    # command and the same camera. They're independent CPU rollouts, so run both at once.
    trained_f, random_f = await asyncio.gather(
        render_replay(checkpoint=checkpoint, steps=replay_steps, policy="trained"),
        render_replay(checkpoint=checkpoint, steps=replay_steps, policy="random"),
    )

    with open(await checkpoint.download(), "rb") as f:
        ckpt = pickle.load(f)
    with open(await trained_f.download()) as f:
        trained = json.load(f)
    with open(await random_f.download()) as f:
        random_run = json.load(f)

    # Both clips go on the MAIN page, side by side, and nowhere else. An earlier
    # version also gave each clip its own tab, which embedded every mp4 twice: in a
    # measured run that was 179KB of a 296KB report, ~30% pure duplication, and it
    # scales with clip length and resolution. The tabs bought full-width playback,
    # which `<video controls>` already gives for free via native fullscreen. The
    # side-by-side cannot be replaced, because trained-vs-random IS the evaluation and
    # splitting it across tabs means comparing from memory.
    await flyte.report.replace.aio(
        reports.final_html(ckpt, reports.before_after_html(random_run, trained))
    )
    await flyte.report.flush.aio()

    history = ckpt.get("history", [])
    best = max((h["reward"] for h in history), default=0.0)
    log.info(
        f"done. best eval reward {best:.1f} | trained survived "
        f"{trained['steps']}/{replay_steps} steps vs random {random_run['steps']}"
    )

    return json.dumps(
        {
            "preset": preset,
            "num_timesteps": num_timesteps,
            "num_envs": num_envs,
            "best_reward": best,
            "trained_steps": trained["steps"],
            "trained_reward": trained["reward"],
            "random_steps": random_run["steps"],
            "random_reward": random_run["reward"],
        }
    )


@orch_env.task(report=True)
async def compare_presets(
    presets: list[str] | None = None,
    num_timesteps: int = 20_000_000,
    num_envs: int = 4096,
    num_evals: int = 10,
    seed: int = 0,
    replay_steps: int = 400,
) -> str:
    """Train one policy per reward preset and put every curve on one chart."""
    import asyncio

    keys = presets or ["baseline", "high-step"]
    for key in keys:
        envs.get_preset(key)

    await flyte.report.replace.aio(
        "<h2>G1 reward presets</h2>"
        f"<p>Training {len(keys)} presets at {num_timesteps:,} steps each: "
        f"{', '.join(keys)}. One GPU, so these serialize.</p>"
    )
    await flyte.report.flush.aio()

    checkpoints = await asyncio.gather(
        *[
            train_policy(
                num_timesteps=num_timesteps,
                num_envs=num_envs,
                num_evals=num_evals,
                preset=key,
                seed=seed,
            )
            for key in keys
        ]
    )

    replays = await asyncio.gather(
        *[render_replay(checkpoint=c, steps=replay_steps) for c in checkpoints]
    )

    # Here the duplication tradeoff flips relative to `walk`: N presets do not fit
    # side by side, so each clip lives in ONE place, its own tab. The main page gets
    # the overlaid reward curves, which is the comparison that actually needs every
    # run visible at once. No mp4 is embedded twice.
    runs, replays_data = [], []
    for key, ckpt_file, replay_file in zip(keys, checkpoints, replays):
        with open(await ckpt_file.download(), "rb") as f:
            ckpt = pickle.load(f)
        with open(await replay_file.download()) as f:
            replay_data = json.load(f)
        runs.append(ckpt)
        replays_data.append(replay_data)

        tab = flyte.report.get_tab(key)
        tab.log(reports.summary_html(ckpt, replay_data) + "<br/>" + replay_data.get("html", ""))

    await flyte.report.replace.aio(
        "<h2>G1 reward presets</h2>"
        + reports.compare_html(runs)
        + "<br/>"
        + reports.preset_index_html(runs, replays_data)
    )
    await flyte.report.flush.aio()

    summary = [
        {
            "preset": r.get("preset"),
            "best_reward": max((h["reward"] for h in r.get("history", [])), default=0.0),
        }
        for r in runs
    ]
    log.info(f"done: {summary}")
    return json.dumps(summary)
