"""Gradio studio for the music-generation demo: a thin Flyte *app* that launches runs.

The architectural point, same as the image and video studios next door: this app is a
LAUNCHER, not a worker. No torch, no diffusers, holds no GPU, loads no model. Every
button submits a `flyte.run(...)` against the already-registered tasks and hands back a
link to that run's Report tab, where the tracks play.

Why it is built this way: a Gradio app pod stays alive for as long as the app is up. If
it generated in-process it would hold the Spark's only GPU forever and every pipeline
task would sit Unschedulable behind it. Launching runs means the GPU is held only while
a track is actually rendering, which on this box is 5 to 13 seconds.

── The tabs, and why they are split this way ────────────────────────────────────
Not one tab per model. A comparison is only meaningful if the shared inputs are
identical and each model contributes its OWN recipe, so:

  Generate : one checkpoint, every knob it has. For working on a single track.
  Compare  : N checkpoints, SHARED knobs only. Each contributes its own steps /
             guidance / shift because those describe how it was trained, not a
             preference. This is the tab that still works when the registry grows
             past ACE-Step to YuE or MusicGen: the common denominator (prompt,
             lyrics, duration, seed) is genuinely common, and everything else comes
             from the spec.
  Sweep    : one checkpoint, one knob, a row you play left to right.
  Reuse    : paste the JSON from any report card to reload its exact settings. The
             bridge that makes a report a working surface instead of a leaderboard.

When a second model FAMILY lands it needs a `adapter` field on the spec (the TTS demo
next door does exactly this) and its own image; the UI change is `knobs_for()` below,
not a new tab.

Deploy (from the devbox):
    python app.py

Env:
    RUN_MODE=local     call the tasks directly instead of via remote refs
    GRADIO_SHARE=1     expose a public gradio.live URL
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import flyte
import flyte.app
import flyte.remote as remote

from config import APP_NAME, APP_PORT, studio_app_image
from models import DEFAULT_MODELS, MODELS, SWEEPS
from prompts import BY_KEY, DEFAULT_BRIEF, SUITES

PROJECT = os.environ.get("FLYTE_PROJECT", "music-generation")
DOMAIN = os.environ.get("FLYTE_DOMAIN", "development")
FLYTE_UI_URL = os.environ.get("FLYTE_UI_URL", "http://localhost:30080")
RUN_MODE = os.environ.get("RUN_MODE", "remote")

_here = Path(__file__).parent

# Bake ONLY the import-light registries into the slim app image, so the pickers can be
# built without dragging torch or diffusers into the app pod. Both modules import
# nothing heavier than `dataclasses`, which is what keeps this honest.
_bundled = (studio_app_image
            .with_source_file(_here / "models.py")
            .with_source_file(_here / "prompts.py"))

env = flyte.app.AppEnvironment(
    name=APP_NAME,
    image=_bundled,
    resources=flyte.Resources(cpu=1, memory="2Gi"),   # a launcher, not a GPU box
    port=APP_PORT,
    requires_auth=False,
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    env_vars=({"GRADIO_SHARE": os.environ["GRADIO_SHARE"]}
              if "GRADIO_SHARE" in os.environ else {}),
)


# ── Which knobs does a model actually have? ──────────────────────────────────────
#
# One place to extend when the registry grows past ACE-Step. Today every checkpoint is
# ACE-Step so they all answer the same, but the call site already asks per model rather
# than assuming, which is the difference between adding YuE later and rewriting the UI.

_ACESTEP_KNOBS = {"steps", "guidance", "shift", "bpm", "keyscale", "language", "lyrics"}


def knobs_for(model_key: str) -> set[str]:
    """The knobs this checkpoint understands. Unknown keys degrade to 'shared only'."""
    return _ACESTEP_KNOBS if model_key in MODELS else set()


def _task(name: str):
    """Resolve a task: the live import locally, a remote ref in the cluster.

    Lazy on purpose, so importing this module without a cluster still works.
    """
    if RUN_MODE == "local":
        import compare_pipeline

        return getattr(compare_pipeline, name)
    return remote.Task.get(f"acestep-orch.{name}", project=PROJECT, domain=DOMAIN,
                           auto_version="latest")


def _external_url(url: str) -> str:
    """Rewrite an in-cluster run URL to one the browser can actually reach."""
    try:
        return FLYTE_UI_URL.rstrip("/") + urlparse(str(url)).path
    except Exception:
        return str(url)


def _submit(task_name: str, blurb: str, **kwargs):
    """Submit a run and stream status. Yields (status, link) on submit and on finish."""
    try:
        run = flyte.run(_task(task_name), **kwargs)
    except Exception as e:                      # a bad ref, a dead cluster, bad inputs
        yield f"❌ Could not submit: `{type(e).__name__}: {e}`", ""
        return
    link = (f'<a href="{_external_url(run.url)}" target="_blank">Open the report for '
            f'<code>{run.name}</code></a>')
    yield f"🎵 Submitted **{run.name}**. {blurb}", link
    try:
        run.wait()
        yield f"✅ **{run.name}** finished. The tracks play in the report.", link
    except Exception as e:
        yield (f"⚠️ **{run.name}** submitted, but waiting on it failed "
               f"(`{type(e).__name__}`). The run itself may still be fine; open the "
               f"report.", link)


def _weights_note(keys) -> str:
    gb = sum(MODELS[k].download_gb for k in keys if k in MODELS)
    return (f"First run for a checkpoint downloads its weights "
            f"(~{gb:.0f}GB across this set), cached forever after.")


def _lyric_hint(lyrics: str, duration: float) -> str:
    """Warn when a lyric cannot physically fit the requested length.

    Budget ~4s of track per sung line. ACE-Step paces the WHOLE lyric to fit
    `audio_duration` rather than truncating the end, so an over-long lyric in a short
    render gets compressed and dropped throughout, which sounds like the model is
    skipping words at random. Cheap to catch here, confusing to diagnose by ear.
    """
    lines = [l for l in (lyrics or "").splitlines()
             if l.strip() and not l.strip().startswith("[")]
    if not lines:
        return ""
    need = len(lines) * 4
    if duration >= need:
        return ""
    return (f"\n\n⚠️ {len(lines)} sung lines needs roughly **{need}s** but you asked for "
            f"{duration:g}s. ACE-Step paces the whole lyric to fit, so expect words to "
            f"be dropped throughout rather than the end being cut.")


def create_demo():
    import gradio as gr

    model_keys = list(MODELS)
    brief_keys = list(BY_KEY)
    sweep_keys = list(SWEEPS)

    def _brief_md(key: str) -> str:
        b = BY_KEY[key]
        lyr = "instrumental" if not b.lyrics.strip() else f"{len(b.lyrics.splitlines())} lyric lines"
        return f"**{b.axis}** · {lyr}\n\n> {b.prompt}"

    with gr.Blocks(title="ACE-Step Studio") as demo:
        gr.Markdown(
            "# 🎵 ACE-Step Studio\n"
            "Launch music-generation runs on the Spark. Every button submits a Flyte "
            "run; the tracks play inline in that run's **Report** tab, alongside a "
            "waveform, a spectrogram, and the command to make each track again."
        )

        # ── Generate ─────────────────────────────────────────────────────────────
        with gr.Tab("Generate"):
            gr.Markdown("One checkpoint, one track, every knob it has.")
            with gr.Row():
                g_model = gr.Dropdown(model_keys, value="xl-turbo", label="Checkpoint")
                g_brief = gr.Dropdown(brief_keys, value=DEFAULT_BRIEF, label="Brief")
            g_info = gr.Markdown(_brief_md(DEFAULT_BRIEF))
            with gr.Accordion("Write your own instead", open=False):
                g_prompt = gr.Textbox(
                    label="Style caption", lines=3,
                    placeholder="lo-fi hip hop, dusty rhodes, vinyl crackle, boom bap drums",
                    info="Genre, mood, instrumentation, production. Not the song's story. "
                         "Leave empty to use the brief above.")
                g_lyrics = gr.Textbox(
                    label="Lyrics", lines=6,
                    placeholder="[verse]\nyour words here\n\n[chorus]\n...",
                    info="Structure tags are conditioning, not decoration. EMPTY = "
                         "instrumental; do not type the word 'instrumental'.")
            with gr.Row():
                g_duration = gr.Slider(10, 300, 60, step=5, label="Duration (s)")
                g_seed = gr.Slider(0, 99999, 42, step=1, label="Seed")
            with gr.Accordion("Advanced", open=False):
                gr.Markdown(
                    "`0` / `-1` means *use this checkpoint's own default*, which is what "
                    "keeps a comparison fair. Note **guidance is inert on `xl-turbo`**: "
                    "it is guidance-distilled and the pipeline coerces it to 1.0."
                )
                with gr.Row():
                    g_steps = gr.Slider(0, 60, 0, step=1, label="Steps (0 = default)")
                    g_guidance = gr.Slider(-1, 15, -1, step=0.5, label="Guidance (-1 = default)")
                    g_shift = gr.Slider(-1, 5, -1, step=0.5, label="Shift (-1 = default)")
                with gr.Row():
                    g_bpm = gr.Slider(0, 200, 0, step=1, label="BPM (0 = model decides)")
                    g_keyscale = gr.Textbox("", label="Key / scale", placeholder="A minor")
                    g_language = gr.Textbox("en", label="Vocal language", max_lines=1)
            g_go = gr.Button("Generate", variant="primary")
            g_status = gr.Markdown()
            g_link = gr.HTML()

            g_brief.change(lambda k: _brief_md(k), g_brief, g_info)

            def _gen(model, brief, prompt, lyrics, duration, seed, steps, guidance,
                     shift, bpm, keyscale, language):
                yield from _submit(
                    "generate_one",
                    _weights_note([model]) + _lyric_hint(
                        lyrics if prompt.strip() else BY_KEY[brief].lyrics, duration),
                    model_key=model, brief=brief, prompt=prompt.strip(),
                    lyrics=lyrics if prompt.strip() else "",
                    duration=float(duration), seed=int(seed), steps=int(steps),
                    guidance=float(guidance), shift=float(shift), bpm=int(bpm),
                    keyscale=keyscale.strip(), language=language.strip() or "en")

            g_go.click(_gen,
                       [g_model, g_brief, g_prompt, g_lyrics, g_duration, g_seed,
                        g_steps, g_guidance, g_shift, g_bpm, g_keyscale, g_language],
                       [g_status, g_link])

        # ── Compare ──────────────────────────────────────────────────────────────
        with gr.Tab("Compare"):
            gr.Markdown(
                "Several checkpoints, the **same** brief, seed and length. Each one "
                "contributes its own steps / guidance / shift, because those describe "
                "how it was trained rather than a preference: handing `xl-sft` turbo's "
                "8-step no-CFG recipe would rig the comparison, not level it.\n\n"
                "This is the tab that keeps working when the registry grows past "
                "ACE-Step: only genuinely shared inputs live here."
            )
            with gr.Row():
                c_models = gr.CheckboxGroup(
                    model_keys, value=DEFAULT_MODELS, label="Checkpoints",
                    info="xl-sft and xl-base run 50 steps with real CFG, so roughly 12x "
                         "the compute of turbo per track.")
                c_suite = gr.Dropdown(list(SUITES), value="quick", label="Suite")
            c_briefs = gr.CheckboxGroup(
                brief_keys, value=[], label="Or pick briefs (overrides the suite)")
            with gr.Row():
                c_duration = gr.Slider(10, 300, 60, step=5, label="Duration (s)")
                c_seed = gr.Slider(0, 99999, 42, step=1, label="Seed")
            c_go = gr.Button("Compare", variant="primary")
            c_status = gr.Markdown()
            c_link = gr.HTML()

            def _cmp(models, suite, briefs, duration, seed):
                if not models:
                    yield "⚠️ Pick at least one checkpoint.", ""
                    return
                n = len(briefs) if briefs else len(SUITES[suite])
                yield from _submit(
                    "compare",
                    f"{len(models)} checkpoint(s) × {n} brief(s). " + _weights_note(models),
                    briefs=list(briefs) or None, suite=suite, models=list(models),
                    duration=float(duration), seed=int(seed))

            c_go.click(_cmp, [c_models, c_suite, c_briefs, c_duration, c_seed],
                       [c_status, c_link])

        # ── Sweep ────────────────────────────────────────────────────────────────
        with gr.Tab("Sweep"):
            gr.Markdown(
                "Hold everything fixed, move **one** parameter, get a row you play left "
                "to right. All values render against a single loaded pipeline, so "
                "another column costs one render, not another 11GB model load."
            )
            with gr.Row():
                s_axis = gr.Dropdown(sweep_keys, value="seed", label="Axis")
                s_model = gr.Dropdown(model_keys, value="xl-turbo", label="Checkpoint")
                s_brief = gr.Dropdown(brief_keys, value=DEFAULT_BRIEF, label="Brief")
            s_note = gr.Markdown()
            s_values = gr.Textbox("", label="Values (comma separated, blank = the axis defaults)",
                                  placeholder="4, 8, 16, 32")
            with gr.Row():
                s_duration = gr.Slider(10, 300, 60, step=5, label="Duration (s)")
                s_seed = gr.Slider(0, 99999, 42, step=1, label="Seed")
            s_go = gr.Button("Sweep", variant="primary")
            s_status = gr.Markdown()
            s_link = gr.HTML()

            def _axis_md(axis, model):
                ax = SWEEPS[axis]
                warn = ""
                if model in MODELS and MODELS[model].distilled and not ax.turbo_ok:
                    warn = (f"\n\n⚠️ **Inert on `{model}`.** It is guidance-distilled, so "
                            f"the pipeline coerces this to 1.0 and every card will sound "
                            f"the same. Pick `xl-sft` to hear the real thing.")
                return (f"**Defaults:** `{', '.join(ax.fmt.format(v) for v in ax.values)}`"
                        f"\n\n{ax.listen_for}{warn}")

            s_axis.change(_axis_md, [s_axis, s_model], s_note)
            s_model.change(_axis_md, [s_axis, s_model], s_note)
            demo.load(_axis_md, [s_axis, s_model], s_note)

            def _sweep(axis, model, brief, values, duration, seed):
                vals = [v.strip() for v in (values or "").split(",") if v.strip()]
                yield from _submit(
                    "sweep",
                    f"axis **{axis}**, {len(vals) or len(SWEEPS[axis].values)} values. "
                    + _weights_note([model]),
                    axis=axis, model_key=model, brief=brief, values=vals or None,
                    duration=float(duration), seed=int(seed))

            s_go.click(_sweep, [s_axis, s_model, s_brief, s_values, s_duration, s_seed],
                       [s_status, s_link])

        # ── Reuse ────────────────────────────────────────────────────────────────
        with gr.Tab("Reuse from a report"):
            gr.Markdown(
                "Every report card has a **reproduce / tweak this one** fold with a JSON "
                "block. Paste it here to reload those exact settings, change what you "
                "want, and relaunch.\n\n"
                "The settings in a card are the **resolved** ones, so what you get back "
                "is what actually ran, not what was originally requested."
            )
            r_json = gr.Textbox(label="Paste a report card's JSON", lines=12,
                                placeholder='{\n  "model_key": "xl-turbo",\n  ...\n}')
            r_load = gr.Button("Load settings")
            r_summary = gr.Markdown()
            with gr.Row():
                r_duration = gr.Slider(10, 300, 60, step=5, label="Duration (s)")
                r_seed = gr.Slider(0, 99999, 42, step=1, label="Seed")
            r_go = gr.Button("Relaunch", variant="primary")
            r_status = gr.Markdown()
            r_link = gr.HTML()
            r_state = gr.State({})

            def _load(raw):
                try:
                    d = json.loads(raw)
                except Exception as e:
                    return {}, f"❌ Not valid JSON: `{e}`", gr.update(), gr.update()
                s = d.get("settings") or {}
                model, brief = d.get("model_key", ""), d.get("brief", "")
                if model not in MODELS:
                    return {}, f"❌ Unknown checkpoint `{model}`.", gr.update(), gr.update()
                where = f"brief `{brief}`" if brief else "a custom prompt"
                return (d,
                        f"Loaded **{model}** on {where} · "
                        f"{s.get('steps','?')} steps · cfg {s.get('guidance','?')} · "
                        f"shift {s.get('shift','?')}. Adjust below and relaunch.",
                        gr.update(value=float(s.get("duration", 60))),
                        gr.update(value=int(s.get("seed", 42))))

            r_load.click(_load, r_json, [r_state, r_summary, r_duration, r_seed])

            def _relaunch(d, duration, seed):
                if not d:
                    yield "⚠️ Load a card's JSON first.", ""
                    return
                s = d.get("settings") or {}
                yield from _submit(
                    "generate_one", _weights_note([d["model_key"]]),
                    model_key=d["model_key"], brief=d.get("brief") or DEFAULT_BRIEF,
                    prompt="" if d.get("brief") else d.get("prompt", ""),
                    lyrics="" if d.get("brief") else d.get("lyrics", ""),
                    duration=float(duration), seed=int(seed),
                    steps=int(s.get("steps", 0)), guidance=float(s.get("guidance", -1)),
                    shift=float(s.get("shift", -1)), bpm=int(s.get("bpm", 0)),
                    keyscale=s.get("keyscale", "") or "",
                    language=s.get("language", "en") or "en")

            r_go.click(_relaunch, [r_state, r_duration, r_seed], [r_status, r_link])

    return demo


@env.server
def studio_server():
    flyte.init_in_cluster(project=PROJECT, domain=DOMAIN)
    create_demo().launch(
        server_name="0.0.0.0", server_port=APP_PORT,
        share=os.environ.get("GRADIO_SHARE") == "1",
    )


if __name__ == "__main__":
    flyte.init_from_config(root_dir=_here)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"ACE-Step Studio deployed: {app.url}")
