"""Open TTS Studio: a Flyte 2 app that LAUNCHES compare runs.

The studio is a thin CPU app. It loads no model and touches no GPU: every
"Compare" submits the `compare` pipeline as a Flyte run and links the report (the
script x voice grid, with each clip playable inline). All GPU work happens inside
the pipeline's per-adapter tasks, so the app pod stays tiny and can never pin a TTS
model in memory.

That is the opposite trade from `voice_app.py` next door, and deliberately so: the
voice chat HOLDS the GPU because resident models are its latency feature, while this
one must not, because it exists to launch the very GPU tasks it would otherwise be
blocking. On a one-GPU box only one of those two can be up at a time.

Development progression:
  1. Local app + remote pipeline:  RUN_MODE=local python app.py
     (the fast loop: the UI runs on the devbox host and imports `compare_pipeline`
      directly, so no deploy is needed. Runs still execute in the cluster.)
  2. Deploy the pipeline, then the app:
       flyte deploy compare_pipeline.py orch_env   # see the note below
       python app.py                               # deploy the studio itself

`orch_env` is the argument to pass, and it is enough: `flyte deploy` wants ONE
environment and `orch_env` declares `depends_on=[cpu_env, *GPU_ENVS.values()]`, so
deploying it registers the fetch task and all seven adapter tasks too. The GPU envs
are dict values rather than module-level names, so the CLI cannot name them directly
anyway. The app needs this because `remote.Task.get` below resolves a REGISTERED
task: without the deploy the studio comes up fine and then fails at launch time.

── One thing to know before you press the button ────────────────────────────────
The grid is scripts x VOICE COLUMNS, not scripts x models. With `Voices = all`,
every model with named M/F voices becomes two columns, so the default 7 models
expand to 11 columns. Each column is a separate GPU task and the box has one GPU,
so they run one after another. `default` (one column per model) is the cheap setting
and the one to demo with.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import flyte
import flyte.app
import flyte.remote as remote

from config import APP_NAME, APP_PORT, studio_app_image
from models import DEFAULT_MODELS, MODELS, jobs_for
from prompts import SUITES, describe

RUN_MODE = os.environ.get("RUN_MODE", "remote")

# The compare task lives in project/domain text-to-speech/development on the devbox
# (see .flyte/config.yaml); override via env if you register it elsewhere.
PROJECT = os.environ.get("FLYTE_PROJECT", "text-to-speech")
DOMAIN = os.environ.get("FLYTE_DOMAIN", "development")
# run.url can come back as an in-cluster address; rewrite it to the console URL.
FLYTE_UI_URL = os.environ.get("FLYTE_UI_URL", "http://localhost:30080")

# Bundle the model + script registries so the app image can build its pickers without
# pulling torch or a single TTS package. Both modules are import-light on purpose:
# models.py is frozen dataclasses, prompts.py is strings.
_here = Path(__file__).parent
_bundled = studio_app_image.with_source_file([_here / "models.py", _here / "prompts.py"])

env = flyte.app.AppEnvironment(
    name=APP_NAME,
    image=_bundled,
    resources=flyte.Resources(cpu=1, memory="1Gi"),   # a launcher, not a GPU box
    port=APP_PORT,
    requires_auth=False,
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    env_vars=(
        {"GRADIO_SHARE": os.environ["GRADIO_SHARE"]} if "GRADIO_SHARE" in os.environ else {}
    ),
)

# Pre-registered compare task, fetched from the control plane at run time. Deploy the
# pipeline first (`flyte deploy compare_pipeline.py`) so this resolves. The ref is lazy,
# so importing this module without a live cluster is fine.
_compare_ref = remote.Task.get(
    "tts-orch.compare", project=PROJECT, domain=DOMAIN, auto_version="latest",
)


def _external_url(url) -> str:
    """Rewrite an in-cluster run URL to the browser-reachable console URL."""
    if not url:
        return ""
    s = str(url)
    if s.startswith("http") and "flyte-binary" not in s and "flyte:" not in s:
        return s
    return f"{FLYTE_UI_URL}{urlparse(s).path}"


def _compare_task():
    """The compare entrypoint: imported directly for local dev, else the ref."""
    if RUN_MODE == "local":
        from compare_pipeline import compare
        return compare
    return _compare_ref


def _columns(model_keys, voices: str) -> list[str]:
    """The voice columns a (models, voices) selection expands into.

    The same expansion `compare` does server-side, duplicated here only so the UI can
    tell you the size of what you are about to launch BEFORE you launch it.
    """
    cols: list[str] = []
    for k in model_keys:
        spec = MODELS[k]
        if voices == "default":
            cols.append(spec.key)
        else:
            cols.extend(vk for vk, _, _ in jobs_for(spec, voices))
    return cols


def run_compare(text, model_keys, voices):
    """Submit a compare run, stream the run link, then confirm on completion."""
    # One script per line, NOT comma-split: the scripts are full of commas (the
    # normalization torture test is nothing but punctuation). The grid is scripts x
    # voice columns, and each column's task loads its model once and reads every line.
    texts = [t.strip() for t in (text or "").splitlines() if t.strip()]
    if not texts:
        yield "⚠️ Enter at least one script (one per line).", ""
        return
    if not model_keys:
        yield "⚠️ Tick at least one model.", ""
        return

    cols = _columns(model_keys, voices)
    try:
        run = flyte.run(
            _compare_task(),
            texts=texts,
            models=list(model_keys),
            voices=voices,
        )
    except Exception as e:
        yield f"❌ Could not launch run: {type(e).__name__}: {e}", ""
        return

    url = _external_url(getattr(run, "url", None))
    link = (
        f'<a href="{url}" target="_blank" rel="noopener">🔗 Open run '
        f'<code>{run.name}</code> on Flyte</a> — the <b>Report</b> tab has the grid.'
        if url else f"Running as <code>{run.name}</code>…"
    )
    yield (
        f"🚀 Launched: {len(texts)} script(s) × {len(cols)} voice column(s) "
        f"({len(model_keys)} model(s)). Weights are fetched once per model and cached "
        f"forever, so a model's first appearance is slow and every later run is not. "
        f"The report fills in column by column, so it's worth opening now.",
        link,
    )

    try:
        run.wait()
        yield (
            "✅ Done. Open the run's **Report** tab: every cell has a player, the "
            "waveform + spectrogram, and its real-time factor. Play a row "
            "left-to-right to hear the same line across models.",
            link,
        )
    except Exception as e:
        yield f"⚠️ Launched, but couldn't confirm completion here: {e}", link


def create_demo():
    import gradio as gr

    def _label(k: str) -> str:
        s = MODELS[k]
        voices = f" · {len(s.voices)} voices" if s.voices else ""
        return (f"{k} · {s.params} · {s.family}{voices} · {s.license}"
                f"{' 🔒 gated' if s.gated else ''}")

    choices = [(_label(k), k) for k in MODELS]
    # The clone-only checkpoints have no named speakers: they exist for
    # clone_pipeline.py and read the script in a model-chosen voice here. Left in the
    # list (you may well want to hear that), just not ticked by default.
    quick = "\n".join(s.text for s in SUITES["quick"])

    with gr.Blocks(title="Open TTS Studio") as demo:
        gr.Markdown(
            "# 🗣️ Open TTS Studio\n"
            "Give it a few lines, tick the models, and every model reads **the same "
            "script** so you can compare them by ear. This app launches a Flyte run and "
            "links it; it holds no GPU and loads no model. The grid lands in the run's "
            "**Report** tab, with a player per cell.\n\n"
            "🔒 = gated, needs an accepted HF license on the token."
        )

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Scripts, one per line",
                    lines=6,
                    placeholder="one script per line; the grid is scripts × voice columns",
                    value=quick,
                )
                with gr.Row():
                    suite = gr.Dropdown(
                        choices=list(SUITES), value="quick",
                        label="Load a suite into the box",
                        info="Curated scripts, each targeting one failure mode.",
                        scale=2,
                    )
                    load = gr.Button("Load", scale=1)
                model_sel = gr.CheckboxGroup(
                    choices=choices, value=DEFAULT_MODELS, label="Models to compare",
                )
            with gr.Column(scale=1):
                voices = gr.Radio(
                    ["default", "all", "female", "male"], value="default",
                    label="Voices",
                    info="'default' = one column per model (cheapest). 'all' splits "
                         "every model with named voices into M + F columns.",
                )
                size = gr.Markdown()
                run_btn = gr.Button("Compare (launch Flyte run)", variant="primary")

        status = gr.Markdown("_idle_")
        run_link = gr.HTML()

        with gr.Accordion("What to listen for in each script", open=False):
            gr.Markdown(f"```\n{describe('full')}\n```")

        # ── Wiring ────────────────────────────────────────────────────────────────
        def _estimate(text_val, keys, voices_val):
            lines = len([t for t in (text_val or "").splitlines() if t.strip()])
            cols = _columns(keys or [], voices_val)
            if not lines or not cols:
                return "_pick at least one script and one model_"
            gb = sum(MODELS[k].download_gb for k in (keys or []))
            return (f"**{lines} × {len(cols)} = {lines * len(cols)} clips**  \n"
                    f"{len(cols)} GPU task(s), one at a time (one GPU).  \n"
                    f"_{gb:.1f}GB of weights, cached after the first run._")

        for ctl in (text, model_sel, voices):
            ctl.change(_estimate, inputs=[text, model_sel, voices], outputs=size)
        demo.load(_estimate, inputs=[text, model_sel, voices], outputs=size)

        # Loading a suite REPLACES the box: these scripts are chosen as a set, and
        # appending would silently double the run's cost.
        load.click(lambda name: "\n".join(s.text for s in SUITES[name]),
                   inputs=suite, outputs=text)

        run_btn.click(run_compare, inputs=[text, model_sel, voices],
                      outputs=[status, run_link])
    return demo


@env.server
def studio_server():
    """Serve the launcher UI from the app pod."""
    flyte.init_in_cluster(project=PROJECT, domain=DOMAIN)
    create_demo().launch(
        server_name="0.0.0.0", server_port=APP_PORT,
        share=os.environ.get("GRADIO_SHARE") == "1",
    )


if __name__ == "__main__":
    flyte.init_from_config(root_dir=_here)
    if RUN_MODE == "local":
        # Local app + remote pipeline: the UI runs here on the devbox and submits
        # against the cluster, so there is nothing to deploy while iterating on the UI.
        create_demo().launch(
            server_name="0.0.0.0", server_port=APP_PORT,
            share=os.environ.get("GRADIO_SHARE") == "1",
        )
    else:
        app = flyte.with_servecontext(interactive_mode=True).serve(env)
        print(f"Open TTS Studio deployed: {app.url}")
