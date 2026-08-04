"""Gradio studio for the G1 walking demo: a thin Flyte *app* that launches runs.

Same architecture as the video-gen studio next door: this app is a LAUNCHER, not a
trainer. It has no jax, no mujoco, holds no GPU and trains nothing. Every button just
submits a `flyte.run(...)` against the already-registered tasks and hands back a link
to that run's Report tab, where the reward curve updates live and the replay clip
appears at the end.

Why it's built this way: a Gradio app pod stays alive for as long as the app is up. If
it trained in-process it would hold the Spark's only GPU forever and every pipeline
task would sit Unschedulable behind it. Launching runs means the GPU is held only
while a policy is actually training.

Deploy (from the devbox):
    python app.py

Env:
    RUN_MODE=local     call the tasks directly instead of via remote refs
    GRADIO_SHARE=1     expose a public gradio.live URL
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import flyte
import flyte.app
import flyte.remote as remote

from config import APP_NAME, APP_PORT, studio_app_image
from envs import DEFAULT_PRESET, PRESETS

PROJECT = os.environ.get("FLYTE_PROJECT", "physical-ai")
DOMAIN = os.environ.get("FLYTE_DOMAIN", "development")
FLYTE_UI_URL = os.environ.get("FLYTE_UI_URL", "http://localhost:30080")
RUN_MODE = os.environ.get("RUN_MODE", "remote")

_here = Path(__file__).parent

# Bake ONLY the import-light preset registry into the slim app image, so the preset
# picker can be built without dragging jax/mujoco into the app pod. envs.py keeps every
# heavy import inside a function for exactly this reason.
_bundled = studio_app_image.with_source_file(_here / "envs.py")

env = flyte.app.AppEnvironment(
    name=APP_NAME,
    image=_bundled,
    resources=flyte.Resources(cpu=1, memory="1Gi"),   # a launcher, not a trainer
    port=APP_PORT,
    requires_auth=False,
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    env_vars=(
        {"GRADIO_SHARE": os.environ["GRADIO_SHARE"]} if "GRADIO_SHARE" in os.environ else {}
    ),
)

# Rough guide from the measured 2048-env rate (59,757 env-steps/sec of pure simulation;
# PPO's gradient updates make the real figure lower). Used only to set expectations in
# the UI, so it is deliberately pessimistic.
_STEPS_PER_SEC = 40_000


def _task(name: str):
    """Resolve a task: the live import locally, a remote ref in the cluster.

    Lazy on purpose, so importing this module without a cluster still works.
    """
    if RUN_MODE == "local":
        import pipeline

        return getattr(pipeline, name)
    return remote.Task.get(
        f"g1-orch.{name}", project=PROJECT, domain=DOMAIN, auto_version="latest"
    )


def _external_url(url: str) -> str:
    """Rewrite an in-cluster run URL to one the browser can actually reach."""
    try:
        return FLYTE_UI_URL.rstrip("/") + urlparse(str(url)).path
    except Exception:
        return str(url)


def _eta(num_timesteps: int, runs: int = 1) -> str:
    minutes = runs * num_timesteps / _STEPS_PER_SEC / 60
    return f"~{minutes:.0f} min" if minutes < 90 else f"~{minutes / 60:.1f} h"


def _launch(task_name: str, blurb: str, **kwargs):
    """Submit a run and stream status. Yields (status, link) on submit and on finish."""
    run = flyte.run(_task(task_name), **kwargs)
    link = (
        f'<a href="{_external_url(run.url)}" target="_blank">Open the report for '
        f"<code>{run.name}</code></a>"
    )
    yield f"🤖 Submitted **{run.name}**.\n\n{blurb}", link
    run.wait()
    yield f"✅ **{run.name}** finished. The replay is in the report.", link


def create_demo():
    import gradio as gr

    preset_choices = [f"{k} · {s.notes}" for k, s in PRESETS.items()]
    key_of = dict(zip(preset_choices, PRESETS))
    default_choice = next(c for c in preset_choices if key_of[c] == DEFAULT_PRESET)

    with gr.Blocks(title="G1 Walk Studio") as demo:
        gr.Markdown(
            "# 🦿 G1 Walk Studio\n"
            "Train a Unitree G1 humanoid to walk on the Spark's GPU, using MJX "
            "(MuJoCo compiled to the GPU) and Brax PPO. Thousands of simulations step "
            "in parallel inside one program.\n\n"
            "The reward curve updates **live** in the run's Report tab, and a replay "
            "clip lands there when training finishes."
        )

        with gr.Tab("Train one policy"):
            w_preset = gr.Dropdown(
                choices=preset_choices, value=default_choice, label="Reward preset",
                info="baseline is DeepMind's tuned config, untouched. The others are "
                     "small, legible diffs on top of it.",
            )
            w_steps = gr.Slider(
                2_000_000, 200_000_000, 20_000_000, step=2_000_000,
                label="Environment steps",
                info="DeepMind's recipe for this env is 200M. Start at 20M to confirm "
                     "the curve is climbing before committing to the full run.",
            )
            w_envs = gr.Dropdown(
                choices=[1024, 2048, 4096, 8192], value=4096, label="Parallel environments",
                info="Throughput scales close to linearly. 8192 is what the tuned "
                     "config uses; it also needs the most memory.",
            )
            with gr.Row():
                w_dr = gr.Checkbox(
                    value=True, label="Domain randomization",
                    info="Randomize friction, mass and motor gains per env. Slower to "
                         "learn, far more robust.",
                )
                w_seed = gr.Slider(0, 9999, 0, step=1, label="Seed")
            w_go = gr.Button("Train", variant="primary")
            w_status = gr.Markdown()
            w_link = gr.HTML()

            w_go.click(
                lambda p, s, n, dr, sd: (
                    yield from _launch(
                        "walk",
                        f"Training **{key_of[p]}** for {int(s):,} steps across "
                        f"{int(n):,} envs. Rough estimate: {_eta(int(s))}. Watch the "
                        f"reward curve in the Report tab.",
                        preset=key_of[p], num_timesteps=int(s), num_envs=int(n),
                        domain_randomization=bool(dr), seed=int(sd),
                    )
                ),
                [w_preset, w_steps, w_envs, w_dr, w_seed],
                [w_status, w_link],
            )

        with gr.Tab("Compare rewards"):
            gr.Markdown(
                "Train several reward presets and put every curve on one chart. "
                "**These serialize**: the box has one GPU, so N presets take about N "
                "times as long as one."
            )
            c_presets = gr.CheckboxGroup(
                choices=preset_choices,
                value=[c for c in preset_choices if key_of[c] in ("baseline", "high-step")],
                label="Presets",
            )
            c_steps = gr.Slider(
                2_000_000, 100_000_000, 20_000_000, step=2_000_000,
                label="Environment steps (each)",
            )
            c_envs = gr.Dropdown(choices=[1024, 2048, 4096, 8192], value=4096,
                                 label="Parallel environments")
            c_go = gr.Button("Compare", variant="primary")
            c_status = gr.Markdown()
            c_link = gr.HTML()

            c_go.click(
                lambda ps, s, n: (
                    yield from _launch(
                        "compare_presets",
                        f"Training {len(ps)} presets at {int(s):,} steps each. These "
                        f"run one at a time on the single GPU: {_eta(int(s), len(ps))} "
                        f"in total.",
                        presets=[key_of[x] for x in ps],
                        num_timesteps=int(s), num_envs=int(n),
                    )
                ),
                [c_presets, c_steps, c_envs],
                [c_status, c_link],
            )

    return demo


@env.server
def studio_server():
    demo = create_demo()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=APP_PORT,
        share=os.environ.get("GRADIO_SHARE") == "1",
    )


if __name__ == "__main__":
    if RUN_MODE == "local":
        create_demo().queue().launch(
            server_name="0.0.0.0",
            server_port=APP_PORT,
            share=os.environ.get("GRADIO_SHARE") == "1",
        )
    else:
        flyte.init_from_config()
        print(flyte.app.deploy(env)[0].url)
