"""Open TTS Studio: a Flyte 2 app that LAUNCHES compare and clone runs.

Two tabs, both launchers:
  - **Compare models**: one script across the TTS models, side by side.
  - **Voice clone**: drag in a recording, cut a reference clip out of it, and clone
    that voice across the cloning models with SIM/WER scores.

The studio is a thin CPU app. It loads no model and touches no GPU: every button
submits a pipeline as a Flyte run and links the report (the script x voice grid, with
each clip playable inline). All GPU work happens inside the pipeline's per-adapter
tasks, so the app pod stays tiny and can never pin a TTS model in memory.

That is the opposite trade from `voice_app.py` next door, and deliberately so: the
voice chat HOLDS the GPU because resident models are its latency feature, while this
one must not, because it exists to launch the very GPU tasks it would otherwise be
blocking. On a one-GPU box only one of those two can be up at a time.

Development progression:
  1. Local app + remote pipeline:  RUN_MODE=local python app.py
     (the fast loop: the UI runs on the devbox host and imports `compare_pipeline`
      directly, so no deploy is needed. Runs still execute in the cluster.)
  2. Deploy the pipelines, then the app:
       flyte deploy compare_pipeline.py orch_env         # see the note below
       flyte deploy clone_pipeline.py clone_orch_env     # only for the Voice clone tab
       python app.py                                     # deploy the studio itself

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

── And one about the clone tab ──────────────────────────────────────────────────
Dragging in an hour of stream audio is fine: the app never uploads the whole thing to
a run. It scans the file for the best few seconds of continuous speech (`ref_clip.py`,
plain RMS framing, no GPU), trims that window locally, and uploads only the clip. The
scan is not diarization, so it cannot tell your voice from a guest's; that is what the
preview player is for. Listen before you launch.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import flyte
import flyte.app
import flyte.io
import flyte.remote as remote

import ref_clip
from config import APP_NAME, APP_PORT, studio_app_image
from models import CLONE_MODELS, DEFAULT_MODELS, MODELS, jobs_for
from prompts import SUITES, describe

RUN_MODE = os.environ.get("RUN_MODE", "remote")

# The compare task lives in project/domain text-to-speech/development on the devbox
# (see .flyte/config.yaml); override via env if you register it elsewhere.
PROJECT = os.environ.get("FLYTE_PROJECT", "text-to-speech")
DOMAIN = os.environ.get("FLYTE_DOMAIN", "development")
# run.url can come back as an in-cluster address; rewrite it to the console URL.
FLYTE_UI_URL = os.environ.get("FLYTE_UI_URL", "http://localhost:30080")

# Bundle the model + script registries and the clip cutter so the app image can build
# its pickers and trim a reference without pulling torch or a single TTS package. All
# three are import-light on purpose: models.py is frozen dataclasses, prompts.py is
# strings, ref_clip.py is numpy + soundfile.
_here = Path(__file__).parent
_bundled = studio_app_image.with_source_file(
    [_here / "models.py", _here / "prompts.py", _here / "ref_clip.py",
     _here / "tts_core.py"]   # only for RefVoice.warnings(); adapters stay unimported
)

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
# The clone tab's two entrypoints (`flyte deploy clone_pipeline.py clone_orch_env`).
# transcribe_ref lives in metrics_env, hence the different task-name prefix.
_clone_ref = remote.Task.get(
    "tts-clone-orch.clone", project=PROJECT, domain=DOMAIN, auto_version="latest",
)
_transcribe_ref = remote.Task.get(
    "tts-metrics.transcribe_ref", project=PROJECT, domain=DOMAIN, auto_version="latest",
)


def _external_url(url) -> str:
    """Rewrite an in-cluster run URL to the browser-reachable console URL."""
    if not url:
        return ""
    s = str(url)
    if s.startswith("http") and "flyte-binary" not in s and "flyte:" not in s:
        return s
    return f"{FLYTE_UI_URL}{urlparse(s).path}"


def _run_link(run, tail: str = "the <b>Report</b> tab has the grid.") -> str:
    url = _external_url(getattr(run, "url", None))
    if not url:
        return f"Running as <code>{run.name}</code>…"
    return (f'<a href="{url}" target="_blank" rel="noopener">🔗 Open run '
            f'<code>{run.name}</code> on Flyte</a> — {tail}')


def _compare_task():
    """The compare entrypoint: imported directly for local dev, else the ref."""
    if RUN_MODE == "local":
        from compare_pipeline import compare
        return compare
    return _compare_ref


def _clone_task():
    """The clone entrypoint. Imported lazily: `clone_pipeline` pulls in metrics.py and
    the task envs, which the slim app image has no business importing at module load."""
    if RUN_MODE == "local":
        from clone_pipeline import clone
        return clone
    return _clone_ref


def _transcribe_task():
    if RUN_MODE == "local":
        from clone_pipeline import transcribe_ref
        return transcribe_ref
    return _transcribe_ref


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


# ── The clone tab's handlers ─────────────────────────────────────────────────────

REF_SECS = 12.0     # inside the 8-15s window every cloner prefers

# Where a cut reference clip is parked in the blob store. Only used by the in-cluster
# path below; `runs.storagePrefix` on this devbox is s3://flyte-data.
REF_PREFIX = os.environ.get("TTS_REF_PREFIX", "s3://flyte-data/tts-refs")


def _ref_file(clip) -> flyte.io.File:
    """Get the cut clip into the blob store and return a File the task can read.

    Two paths, because `File.from_local_sync` only works in one of them:

      host  (RUN_MODE=local)  from_local_sync asks the control plane's dataproxy for an
                              upload location and PUTs to the PRESIGNED URL it returns.
                              Fine here: this box IS localhost.
      pod   (deployed app)    that same presigned URL is signed for localhost:30002,
                              because that is what lets a BROWSER fetch report assets
                              (`storage.signedURL.stowConfigOverride` in the flyte-binary
                              config). Inside a pod, localhost is the pod, so the PUT can
                              never land: "All connection attempts failed", three retries,
                              done. So write straight to the S3 endpoint the pod already
                              has in FLYTE_AWS_ENDPOINT and hand the task a remote URI
                              instead, skipping the signing entirely.

    The compare tab never needed this because it uploads no files, only strings; this is
    the first path in the studio that puts bytes in the blob store.
    """
    import asyncio
    from uuid import uuid4

    import flyte.storage

    if not os.environ.get("FLYTE_AWS_ENDPOINT"):
        return flyte.io.File.from_local_sync(str(clip))
    dest = f"{REF_PREFIX}/{uuid4().hex}/reference.wav"
    asyncio.run(flyte.storage.put(str(clip), dest))
    return flyte.io.File.from_existing_remote(dest)


def describe_upload(path):
    """Report what was dropped in, and auto-cut a clip if it is already short enough."""
    if not path:
        return "_nothing loaded_", None, None, ""
    try:
        info = ref_clip.probe(path)
    except Exception as e:
        return f"⚠️ could not read that file: {e}", None, None, ""
    msg = f"**{Path(path).name}** · {info.describe()}"
    if info.seconds <= 30:
        # Already reference-sized: cutting a window out of it would be theater.
        prev, warn, state = cut_ref(path, 0.0, min(info.seconds, 30.0))
        return msg + "  \n_short file, used as-is._", prev, state, warn
    return (msg + "  \n_long file: press **Find the best 12s**, or set a start time "
            "and cut manually._"), None, None, ""


def auto_pick(path, secs):
    """Scan for the best window and cut it. The scan decodes the whole file, so an hour
    takes a minute or so; that is the price of not making you scrub through it."""
    if not path:
        return 0.0, None, None, "⚠️ Drag a recording in first."
    picks = ref_clip.best_window(path, secs=float(secs), skip_start=60.0, top=1)
    start = picks[0][0] if picks else 0.0
    prev, warn, state = cut_ref(path, start, secs)
    m, s = divmod(start, 60)
    return start, prev, state, f"picked **{int(m)}m{s:04.1f}s**  \n{warn}"


def cut_ref(path, start, secs):
    """Trim [start, start+secs) to a mono wav and report its quality problems.

    Warnings come from `tts_core.RefVoice.warnings()`, the same check the pipeline's
    report card uses, so the app cannot disagree with the run about what a bad
    reference is.
    """
    if not path:
        return None, "⚠️ Drag a recording in first.", None
    out = Path(tempfile.mkdtemp(prefix="ref_")) / "reference.wav"
    try:
        clip, peak, sr = ref_clip.trim(path, float(start), float(secs), out)
    except Exception as e:
        return None, f"⚠️ could not cut that window: {e}", None

    import tts_core
    ref = tts_core.RefVoice.from_file(clip, "")
    lines = [f"**{ref.seconds:.1f}s · {sr}Hz · peak {peak:.2f}**"]
    # warnings() also complains about the missing transcript, which is not news here:
    # the transcript box below is how you supply it.
    lines += [f"⚠️ {w}" for w in ref.warnings() if "transcript" not in w]
    return clip, "  \n".join(lines), clip


def transcribe_clip(clip):
    """Whisper-transcribe the cut clip by launching the GPU task, then fill the box."""
    if not clip:
        yield "", "⚠️ Cut a reference clip first."
        return
    yield "", "🚀 Launching Whisper on the cluster (a GPU pod, so ~a minute cold)…"
    try:
        run = flyte.run(_transcribe_task(), ref_audio=_ref_file(clip))
        run.wait()
        text = (run.outputs() or [""])[0] if hasattr(run, "outputs") else ""
    except Exception as e:
        yield "", f"❌ transcribe failed: {type(e).__name__}: {e}"
        return
    yield (str(text).strip(),
           "✅ Transcribed. **Read it and fix it**: Qwen, Dia and CSM condition on this "
           "text, so an ASR error here degrades three of the five clones.")


def run_clone(clip, ref_text, text, model_keys):
    """Submit a clone run for the cut reference, stream the link, confirm on completion."""
    texts = [t.strip() for t in (text or "").splitlines() if t.strip()]
    if not clip:
        yield "⚠️ Drag in a recording and cut a reference clip first.", ""
        return
    if not (ref_text or "").strip():
        yield ("⚠️ The reference transcript is empty. Chatterbox will cope; Qwen, Dia "
               "and CSM all condition on it. Transcribe or type it first."), ""
        return
    if not texts:
        yield "⚠️ Enter at least one script (one per line).", ""
        return
    if not model_keys:
        yield "⚠️ Tick at least one cloning model.", ""
        return

    try:
        run = flyte.run(
            _clone_task(),
            ref_audio=_ref_file(clip),
            ref_text=ref_text.strip(),
            texts=texts,
            models=list(model_keys),
        )
    except Exception as e:
        yield f"❌ Could not launch run: {type(e).__name__}: {e}", ""
        return

    link = _run_link(run, "the <b>Report</b> tab has the clips and the scores.")
    yield (
        f"🚀 Launched clone: {len(texts)} script(s) × {len(model_keys)} model(s), each "
        f"generated twice (cloned, and in the model's own voice as the control), then "
        f"scored for speaker similarity and WER.",
        link,
    )
    try:
        run.wait()
        yield ("✅ Done. The report ranks the models by similarity against the "
               "reference's own self-similarity ceiling, with WER beside it.", link)
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

    clone_choices = [(_label(k), k) for k in CLONE_MODELS]
    clone_quick = "\n".join(s.text for s in SUITES["clone-quick"])

    with gr.Blocks(title="Open TTS Studio") as demo:
        gr.Markdown(
            "# 🗣️ Open TTS Studio\n"
            "Every button here launches a Flyte run and links it; the app holds no GPU "
            "and loads no model. Results land in the run's **Report** tab, with a "
            "player per cell.\n\n"
            "🔒 = gated, needs an accepted HF license on the token."
        )

        with gr.Tabs():
            # ── Compare models ────────────────────────────────────────────────────
            with gr.Tab("Compare models"):
                gr.Markdown(
                    "Give it a few lines, tick the models, and every model reads **the "
                    "same script** so you can compare them by ear."
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
                            choices=choices, value=DEFAULT_MODELS,
                            label="Models to compare",
                        )
                    with gr.Column(scale=1):
                        voices = gr.Radio(
                            ["default", "all", "female", "male"], value="default",
                            label="Voices",
                            info="'default' = one column per model (cheapest). 'all' "
                                 "splits every model with named voices into M + F columns.",
                        )
                        size = gr.Markdown()
                        run_btn = gr.Button("Compare (launch Flyte run)",
                                            variant="primary")

                status = gr.Markdown("_idle_")
                run_link = gr.HTML()

                with gr.Accordion("What to listen for in each script", open=False):
                    gr.Markdown(f"```\n{describe('full')}\n```")

            # ── Voice clone ───────────────────────────────────────────────────────
            with gr.Tab("Voice clone"):
                gr.Markdown(
                    "Drag in **any recording** (wav or mp3, any length). The app cuts a "
                    "reference clip out of it, uploads only that clip, and clones the "
                    "voice across the cloning models. Every line is generated twice, "
                    "cloned and in the model's own voice, then scored for **speaker "
                    "similarity** and **WER**.\n\n"
                    "The window picker is energy-based, not diarization: on a recording "
                    "with more than one speaker it can hand you a guest. **Play the "
                    "preview before launching.**"
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        upload = gr.Audio(
                            sources=["upload"], type="filepath",
                            label="Recording (drag it here)",
                        )
                        up_info = gr.Markdown("_nothing loaded_")
                        with gr.Row():
                            start = gr.Number(value=0.0, label="Start (seconds)", scale=1)
                            length = gr.Slider(3, 30, value=REF_SECS, step=0.5,
                                               label="Clip length (8-15s is the sweet spot)",
                                               scale=2)
                        with gr.Row():
                            pick_btn = gr.Button("Find the best 12s")
                            cut_btn = gr.Button("Cut at this start time")
                        preview = gr.Audio(label="Reference clip (listen to this)",
                                           interactive=False)
                        ref_quality = gr.Markdown()
                        ref_text = gr.Textbox(
                            label="Reference transcript (exact words in the clip)",
                            lines=3,
                            placeholder="type what is said in the clip, or transcribe it",
                        )
                        tx_btn = gr.Button("Transcribe with Whisper (launches a GPU run)")
                        tx_status = gr.Markdown()
                    with gr.Column(scale=1):
                        clone_models = gr.CheckboxGroup(
                            choices=clone_choices, value=["chatterbox"],
                            label="Cloning models",
                            info="Only clone-capable models are listed. Kokoro and "
                                 "Parler cannot clone at all.",
                        )
                        clone_text = gr.Textbox(
                            label="Scripts to say in the voice, one per line",
                            lines=5, value=clone_quick,
                        )
                        clone_suite = gr.Dropdown(
                            choices=["clone-quick", "clone"], value="clone-quick",
                            label="Load a clone suite",
                        )
                        clone_load = gr.Button("Load")
                        clone_btn = gr.Button("Clone (launch Flyte run)",
                                              variant="primary")
                clone_status = gr.Markdown("_idle_")
                clone_link = gr.HTML()
                # A hidden Textbox rather than gr.State, deliberately: State is
                # session-side and invisible to /gradio_api, so with State the whole
                # launch path could only ever be exercised by a human in a browser.
                # This holds the same string and behaves identically in the UI.
                clip_state = gr.Textbox(visible=False, value="")

        # ── Wiring: compare ───────────────────────────────────────────────────────
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

        # ── Wiring: clone ─────────────────────────────────────────────────────────
        upload.change(describe_upload, inputs=upload,
                      outputs=[up_info, preview, clip_state, ref_quality])
        pick_btn.click(auto_pick, inputs=[upload, length],
                       outputs=[start, preview, clip_state, ref_quality])
        cut_btn.click(cut_ref, inputs=[upload, start, length],
                      outputs=[preview, ref_quality, clip_state])
        # A new clip invalidates the old transcript: leaving the previous one in the box
        # is how you end up conditioning three models on words from a different passage.
        for ctl in (pick_btn, cut_btn):
            ctl.click(lambda: ("", ""), outputs=[ref_text, tx_status])
        tx_btn.click(transcribe_clip, inputs=clip_state, outputs=[ref_text, tx_status])
        clone_load.click(lambda name: "\n".join(s.text for s in SUITES[name]),
                         inputs=clone_suite, outputs=clone_text)
        clone_btn.click(run_clone,
                        inputs=[clip_state, ref_text, clone_text, clone_models],
                        outputs=[clone_status, clone_link])
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
