"""Gradio studio for the music-generation demo: a thin Flyte *app* that launches runs.

The architectural point, same as the image and video studios next door: this app is a
LAUNCHER, not a worker. No torch, no diffusers, holds no GPU, loads no model. Every
button submits a `flyte.run(...)` against the already-registered tasks and hands back a
link to that run's Report tab, where the tracks play.

Why it is built this way: a Gradio app pod stays alive for as long as the app is up. If
it generated in-process it would hold the Spark's only GPU forever and every pipeline
task would sit Unschedulable behind it. Launching runs means the GPU is held only while
a track is actually rendering.

── The tabs, and why they are split this way ────────────────────────────────────
Not one tab per model. A comparison is only meaningful if the shared inputs are
identical and each model contributes its OWN recipe, so:

  Generate : one checkpoint, every knob it has. For working on a single track.
  Takes    : one song, several ways at once, each take with its own knobs. The tab for
             actually MAKING something, as opposed to running a designed experiment.
  Compare  : N checkpoints, SHARED knobs only. Each contributes its own steps /
             guidance / shift because those describe how it was trained, not a
             preference. This is the tab that still works when the registry grows
             past ACE-Step: the common denominator (prompt, lyrics, duration, seed) is
             genuinely common, and everything else comes from the spec.
  Sweep    : one checkpoint, one knob, a row you play left to right.
  Reuse    : paste the JSON from any report card to reload its exact settings. The
             bridge that makes a report a working surface instead of a leaderboard.
  Knobs    : what every parameter does, why you would tweak it, and what it costs.

Generate and Takes look adjacent but answer different questions. Generate is one track
with full control. Takes is built on the observation that you almost never want one
track: every useful session on this project has been a comparison, and the difference
between a usable studio and a toy is whether the comparison is the default or something
you assemble by hand out of six separate runs. So Takes always renders at least two,
and the second defaults to a different seed.

All takes go out as ONE run, not one per take. Beyond the obvious saving (a checkpoint
loads once however many takes use it), this repo learned the hard way that N concurrent
runs means N orchestrator pods each holding 8Gi while awaiting a 96Gi child; past three,
they starve the very children they are waiting on.

Deploy (from the devbox):
    python app.py

Env:
    RUN_MODE=local     call the tasks directly instead of via remote refs
    GRADIO_SHARE=1     expose a public gradio.live URL
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import flyte
import flyte.app
import flyte.remote as remote

from config import APP_NAME, APP_PORT, studio_app_image
from models import DEFAULT_MODELS, MODELS, SWEEPS, Variant
from prompts import (BY_KEY, DEFAULT_BRIEF, SECONDS_PER_LINE, SUITES,
                     suggest_durations, sung_lines)

PROJECT = os.environ.get("FLYTE_PROJECT", "music-generation")
DOMAIN = os.environ.get("FLYTE_DOMAIN", "development")
FLYTE_UI_URL = os.environ.get("FLYTE_UI_URL", "http://localhost:30080")
RUN_MODE = os.environ.get("RUN_MODE", "remote")

MAX_TAKES = 6            # more than this and the report stops being scannable
DEFAULT_TAKES = 2        # never one: see the module docstring

_here = Path(__file__).parent

# Bake ONLY the import-light registries into the slim app image, so the pickers can be
# built without dragging torch or diffusers into the app pod. Both modules import
# nothing heavier than `dataclasses`, which is what keeps this honest. `Variant` lives
# in models.py for exactly this reason: it is built here and consumed by the pipeline,
# so it cannot live next to the render code.
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
# One place to extend when the registry grows past ACE-Step. `diffrhythm` genuinely
# honours far fewer of them, and saying so here is what stops the UI promising control
# it does not have.

_ACESTEP_KNOBS = {"steps", "guidance", "shift", "bpm", "keyscale", "timesignature",
                  "language", "lyrics", "cfg_interval"}
_DIFFRHYTHM_KNOBS = {"steps", "guidance", "lyrics"}
_MINIMAX_KNOBS = {"lyrics"}          # its call takes prompt, lyrics, duration, seed

_ADAPTER_KNOBS = {"diffrhythm": _DIFFRHYTHM_KNOBS, "minimax": _MINIMAX_KNOBS}


def knobs_for(model_key: str) -> set[str]:
    """The knobs this checkpoint understands. Unknown keys degrade to 'shared only'."""
    spec = MODELS.get(model_key)
    if spec is None:
        return set()
    return _ADAPTER_KNOBS.get(spec.adapter, _ACESTEP_KNOBS)


# Which models ignore most of the per-take knobs, phrased for the UI. Derived rather
# than hand-listed so it cannot fall out of step with `knobs_for`: the moment a model
# is added to the registry with a thin adapter, this sentence includes it.
_THIN_MODELS = sorted(k for k in MODELS
                      if len(knobs_for(k)) < len(_ACESTEP_KNOBS))


# ── The knob reference ───────────────────────────────────────────────────────────
#
# Every entry says what the knob does, why you would reach for it, and WHAT IT COSTS,
# because the cost answers turned out to be the surprising part and they change how you
# use the studio: only one of these is expensive, so there is no reason to be timid
# with the rest.
#
# Measurements are from this project's own runs on the Spark (a 240s track on xl-sft
# unless stated), not from the model card. Where something is unmeasured or provisional
# it says so; a reference that quietly presents a guess as a fact is worse than none,
# because it will be believed.

KNOB_DOCS = {
    "model_key": (
        "### Checkpoint — the biggest lever, and the only one confirmed by ear\n"
        "**What it does.** Which trained model renders. `xl-turbo` is *guidance "
        "distilled*: 8 denoising steps with CFG folded into the weights. `xl-sft` is "
        "instruction-tuned: 50 steps and a real guidance pass. `xl-base` is the "
        "pretrained model underneath both. `diffrhythm` and `minimax-music3` are "
        "different models entirely, and the only other two here that **sing**.\n\n"
        "**`minimax-music3`** (MiniMax, August 2026) is the newest and takes ACE-Step's "
        "`[verse]`/`[chorus]` tags verbatim, so unlike DiffRhythm it needs no lyric "
        "conversion and the head-to-head against ACE-Step is genuinely fair. 32kHz "
        "against ACE-Step's 48kHz, ~0.3x realtime (roughly 275s for a minute of "
        "audio), and it treats length as approximate: asked for 60s it returned 82.8s. "
        "Note its licence is **not** open source, unlike ACE-Step (MIT) and DiffRhythm "
        "(Apache-2.0): attribution and AI-disclosure are required.\n\n"
        "**Why tweak it.** This is the one change that clearly improved things, most "
        "audibly **on the voice**. Low-step flow matching does not sound broken, it "
        "sounds *smeared*, and smeared is most of what people mean by 'it sounds AI'.\n\n"
        "**Cost.** 240s track: turbo **23s**, sft **211s**. About 9x.\n\n"
        "**The trap.** Several knobs below do nothing on turbo. Guidance is silently "
        "coerced to 1.0. Tune on the checkpoint you intend to ship.\n\n"
        "**Try:** `xl-sft` for anything with vocals; `xl-turbo` to iterate on wording "
        "fast; `xl-base` to hear what the fine-tune bought."
    ),
    "seed": (
        "### Seed — the free lottery, and a diagnostic\n"
        "**What it does.** The noise the sampler starts from. Everything else fixed, a "
        "new seed is a new take.\n\n"
        "**Why tweak it.** Some takes are just better. But the more useful reason: "
        "*how much* the arrangement moves between seeds tells you how underspecified "
        "your caption is. Four wildly different songs means the model is filling in "
        "what you did not say, and tightening the caption should visibly shrink that "
        "spread. It is the fastest prompt feedback loop you get.\n\n"
        "**Cost.** None.\n\n"
        "**Try:** this is why take 2 defaults to a different seed. Leave it."
    ),
    "duration": (
        "### Length — not a crop, and the one every listening test has wanted MORE of\n"
        "**What it does.** Fed to the model up front, so it changes the *composition*. "
        "A 20s render must state the idea immediately; a 240s render has room for an "
        "intro, a build and a turnaround. These are different arrangements, not longer "
        "and shorter cuts of one take.\n\n"
        "**Why tweak it.** ACE-Step paces the **whole** lyric to fit the duration "
        "rather than truncating it, so a long lyric in a short render is not cut off, "
        "it is compressed: syllables shorten and breaths vanish. That is a leading "
        "cause of the synthetic feeling.\n\n"
        "**Cost.** Roughly linear in length.\n\n"
        "**Status: provisional.** In Takes, leaving this at 0 derives a length from "
        "your line count. Every listening test so far has preferred *more* room per "
        "line than the current estimate, and the knee has not been bracketed from "
        "above.\n\n"
        "**Try:** 0 first, then deliberately go longer than it suggests."
    ),
    "steps": (
        "### Steps — the only knob you actually pay for\n"
        "**What it does.** Denoising iterations. Flow matching degrades gracefully, so "
        "low-step output sounds *smeared* rather than broken: transients (kick attack, "
        "hi-hat, consonants) blur first and the stereo image narrows.\n\n"
        "**Why tweak it.** It is the straightforward quality-for-time trade, and 50 is "
        "sft's shipped recipe rather than a measured ceiling.\n\n"
        "**Cost.** Near-linear and it dominates everything else. 240s on sft: 50 steps "
        "**210s**, 100 steps **412s**, 200 steps **818s**.\n\n"
        "**Try:** 0 (the checkpoint's own recipe) unless you have a reason. If 200 does "
        "not clearly beat 50 to your ear, that is ten minutes a track you can stop "
        "spending."
    ),
    "guidance": (
        "### Guidance (CFG) — free, and the classic cause of 'harsh'\n"
        "**What it does.** How hard the model is pushed toward your caption. Low drifts "
        "and sounds generic. High obeys, then over-obeys, and turns brittle and glassy "
        "with the instruments fighting — the audio equivalent of an over-saturated "
        "photo.\n\n"
        "**Why tweak it.** It is aimed at both failure modes and costs nothing.\n\n"
        "**Cost.** **None.** cfg 7 and cfg 20 both render in ~210s, because the "
        "guidance pass runs per step regardless of its magnitude.\n\n"
        "**Inert on `xl-turbo`**, which coerces it to 1.0 and only warns.\n\n"
        "**Try:** 7 is sft's recipe. Push to 12-20 for adherence, and if it goes harsh, "
        "reach for CFG cutoff rather than backing the guidance off."
    ),
    "cfg_interval_end": (
        "### CFG cutoff — the one knob with *negative* cost\n"
        "**What it does.** *Where* on the schedule guidance applies. 1.0 is the whole "
        "run; 0.5 means guidance is on for the first half and off after.\n\n"
        "**Why tweak it.** Guidance does two things at two different times. Early it "
        "decides form, groove and arrangement, where pushing toward the caption helps. "
        "Late it renders texture and detail, and pushing hard there is what makes a "
        "high CFG harsh. Restricting it to the early part asks for adherence "
        "**without** the artifacts. Without this knob a guidance sweep only offers "
        "'drifts' or 'harsh' with nothing in between.\n\n"
        "**Cost: negative.** Cutting off at 0.25 rendered in **164s** against 209s for "
        "the full range, a 22% saving, because the later steps stop doing the second "
        "forward pass at all.\n\n"
        "**Try:** high guidance (12-20) with a cutoff of 0.4-0.6. If it works you get "
        "adherence, lose the sheen, and render faster."
    ),
    "cfg_interval_start": (
        "### CFG onset — the control, and it should sound *worse*\n"
        "**What it does.** The mirror of CFG cutoff. Raising it switches guidance OFF "
        "early and on only later.\n\n"
        "**Why tweak it.** Mostly to check the reasoning behind CFG cutoff. If guidance "
        "helps early and hurts late, doing the opposite should lose prompt adherence "
        "*and* keep the harshness. If raising this sounds fine, the early/late story is "
        "wrong and the cutoff result means something else.\n\n"
        "**Cost.** Same saving as the cutoff: fewer steps run the second forward pass.\n\n"
        "**Try:** leave at 0. Set 0.5 only when you want the control."
    ),
    "shift": (
        "### Shift — where the step budget gets spent (NOT a musical control)\n"
        "**First, what it is not.** Despite sitting near Length, BPM and Time sig, "
        "`shift` is short for *timestep* shift and has nothing to do with musical time. "
        "It never changes how long the track is or how fast it goes.\n\n"
        "**What it does.** Warps the flow-matching schedule. High front-loads the noisy "
        "end, where global structure (form, arrangement, groove) is decided. Low spends "
        "more budget at the clean end, which is detail and texture.\n\n"
        "**Why tweak it.** It is the specific knob for a problem that is *textural* "
        "rather than structural. Same number of steps either way, redistributed.\n\n"
        "**Cost.** None: 206s / 205s / 205s at shift 1, 2, 3.\n\n"
        "**Try:** 3 is the shipped recommendation. Drop to 1-2 if the arrangement is "
        "right but the surface is not."
    ),
    "bpm": (
        "### BPM — structured metadata, not prompt wording\n"
        "**What it does.** Goes into a `# Metas` block the text encoder was trained on, "
        "**not** into your caption. That makes it a real control surface rather than a "
        "wording trick.\n\n"
        "**Why tweak it.** Obedience is testable: tap along. The interesting case is a "
        "mismatch between the caption's genre and the requested tempo, like a ballad at "
        "160, where you find out which one the model believes.\n\n"
        "**Cost.** None.\n\n"
        "**Try:** 0 (let it choose) unless you are writing to a target."
    ),
    "keyscale": (
        "### Key / scale — the same metadata channel, aimed at harmony\n"
        "**What it does.** `\"C major\"`, `\"A minor\"`, `\"D dorian\"`.\n\n"
        "**Why tweak it.** Major versus minor should be unmissable. The real test is a "
        "modal scale: dorian is a minor scale that does *not* sound sad, so it is where "
        "a model that merely memorised 'minor = sad' falls apart.\n\n"
        "**Cost.** None.\n\n"
        "**Try:** empty, or `D dorian` to hear whether it genuinely knows modes."
    ),
    "timesignature": (
        "### Time signature — confirmed audible, and the last one anyone tested\n"
        "**What it does.** `4` common time, `3` waltz, `6` compound. Same `# Metas` "
        "channel as BPM and key.\n\n"
        "**Why tweak it.** It was the one metadata field nobody had exercised, and it "
        "**works**: asking a 1980s arena power ballad for `3` audibly changes it. So "
        "all three metadata fields are real controls, not decoration.\n\n"
        "**Cost.** None: 212s / 209s / 209s for 4, 3 and 6. Values well past the "
        "documented 3 and 4 (5, 7, 9, 12, 15) are accepted and cost the same; whether "
        "they do anything is unexplored.\n\n"
        "**Try:** `3` against a genre that strongly implies 4/4."
    ),
    "language": (
        "### Vocal language — match it to your lyric\n"
        "**What it does.** Sets the language header on the lyric. ACE-Step claims 50+.\n\n"
        "**Why tweak it.** If you write in a language other than English and leave this "
        "at `en`, the phrasing drifts toward an English reading of the spelling. It is "
        "the difference between singing Portuguese and reading it aloud in an English "
        "accent.\n\n"
        "**Cost.** None.\n\n"
        "**Try:** `en` unless your lyric is not English, then match it (`pt`, `zh`, "
        "`ja`, …)."
    ),
}

_DOC_LABELS = {
    "model_key": "Checkpoint (biggest lever)", "seed": "Seed",
    "duration": "Length", "steps": "Steps (the only expensive one)",
    "guidance": "Guidance / CFG", "cfg_interval_end": "CFG cutoff",
    "cfg_interval_start": "CFG onset (the control)",
    "shift": "Shift (sampler, not tempo)", "bpm": "BPM",
    "keyscale": "Key / scale", "timesignature": "Time signature",
    "language": "Vocal language",
}

# The per-take knob registry. Data rather than inline widgets because it is the same
# list three times over (build the row, read the row, document the row), and three
# hand-maintained copies is how a studio silently drops a knob.
TAKE_KNOBS = [
    ("seed",       "Seed",         "number", 42,   None),
    ("duration",   "Length (s)",   "number", 0,    "0 = derive from the lyric"),
    ("steps",      "Steps",        "number", 0,    "0 = checkpoint default (turbo 8, sft 50)"),
    ("guidance",   "Guidance",     "number", -1,   "-1 = checkpoint default. Inert on turbo"),
    ("shift",      "Shift (sampler, not tempo)", "number", -1,
     "-1 = default (3). Where denoising spends its budget: low = texture, high = structure"),
    ("cfg_interval_end", "CFG cutoff", "number", 1.0, "1 = full range. 0.5 = guidance early only"),
    ("cfg_interval_start", "CFG onset", "number", 0.0, "0 = from the start. Raising it is the control"),
    ("bpm",        "BPM",          "number", 0,    "0 = let the model choose"),
    ("keyscale",   "Key / scale",  "text",   "",   "e.g. A minor, D dorian"),
    ("timesignature", "Time sig",  "text",   "",   "4, 3, 6 … confirmed audible"),
    ("language",   "Vocal language", "text", "en", "en, pt, zh, ja … match your lyric"),
]

# Drift guards. A knob can exist in the pipeline, be reachable through Variant, and
# still have no widget or no doc, and every one of those failures is INVISIBLE: nothing
# errors, the control simply is not there. That already happened once with
# cfg_interval_start and language, so it is asserted rather than reviewed.
_VARIANT_FIELDS = {f for f in Variant().__dataclass_fields__} - {"label", "model_key"}
assert {n for n, *_ in TAKE_KNOBS} == _VARIANT_FIELDS, (
    f"take knobs and Variant have drifted: "
    f"{_VARIANT_FIELDS ^ {n for n, *_ in TAKE_KNOBS}}")
assert {n for n, *_ in TAKE_KNOBS} | {"model_key"} == set(KNOB_DOCS) == set(_DOC_LABELS), (
    "take knobs, their docs and their labels have drifted apart")


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


# How long the page follows a run before letting go. Knative's queue-proxy enforces a
# request timeout (300s by default) on the streaming response Gradio uses for generator
# functions, so a blocking `run.wait()` outlives its own connection and the browser is
# left with a click that appears to have done nothing.
#
# That is exactly what happened here, and it is a regression caused by success
# elsewhere: this app was written when a render was "5 to 13 seconds" and waiting was
# free. A 240s track on xl-sft now takes 3.5 to 11 minutes, which is past the timeout,
# so runs completed perfectly while the UI showed nothing.
_FOLLOW_BUDGET_S = 240
_POLL_EVERY_S = 5


def _submit(task_name: str, blurb: str, **kwargs):
    """Submit a run and follow it for a while. Yields (status, link) as it goes.

    Polls rather than blocking on `run.wait()`, for two reasons. It emits something on
    every tick, which keeps the streaming response alive instead of letting an idle
    connection be reaped; and it GIVES UP on purpose before the request timeout, so a
    long render ends with an honest "still going, here is the link" rather than a dead
    page. The run is never affected by the page letting go of it.
    """
    try:
        run = flyte.run(_task(task_name), **kwargs)
    except Exception as e:                      # a bad ref, a dead cluster, bad inputs
        yield f"❌ Could not submit: `{type(e).__name__}: {e}`", ""
        return

    link = (f'<a href="{_external_url(run.url)}" target="_blank">Open the report for '
            f'<code>{run.name}</code></a>')
    yield f"🎵 Submitted **{run.name}**. {blurb}", link

    t0 = time.monotonic()
    while True:
        time.sleep(_POLL_EVERY_S)
        elapsed = int(time.monotonic() - t0)
        try:
            run.sync()
            if run.done():
                phase = str(getattr(run, "phase", "") or "")
                if "SUCCEEDED" in phase.upper():
                    yield (f"✅ **{run.name}** finished in {elapsed}s. The tracks play "
                           f"in the report.", link)
                else:
                    yield (f"⚠️ **{run.name}** ended as `{phase}` after {elapsed}s. "
                           f"The report has whatever it managed to render.", link)
                return
        except Exception as e:
            # A transient control-plane hiccup must not look like a failed run. Say so
            # and keep polling; the run is not ours to lose.
            yield (f"⏳ **{run.name}** running ({elapsed}s). Status check failed "
                   f"(`{type(e).__name__}`), retrying.", link)
        else:
            yield f"⏳ **{run.name}** running… {elapsed}s elapsed.", link

        if time.monotonic() - t0 > _FOLLOW_BUDGET_S:
            yield (f"⏳ **{run.name}** is still going after {elapsed}s, which is normal "
                   f"for `xl-sft` or a long track. This page stops following it here so "
                   f"the connection is not cut mid-stream; **the run is unaffected**. "
                   f"Open the link to watch it finish.", link)
            return


def _weights_note(keys) -> str:
    gb = sum(MODELS[k].download_gb for k in keys if k in MODELS)
    return (f"First run for a checkpoint downloads its weights "
            f"(~{gb:.0f}GB across this set), cached forever after.")


def _lyric_hint(lyrics: str, duration: float) -> str:
    """Warn when a lyric cannot comfortably fit the requested length.

    ACE-Step paces the WHOLE lyric to fit `audio_duration` rather than truncating the
    end, so an over-long lyric in a short render gets compressed and dropped
    throughout, which sounds like the model skipping words at random. Cheap to catch
    here, confusing to diagnose by ear.

    The threshold now comes from `prompts.SECONDS_PER_LINE` rather than a hardcoded 4,
    so the one place that estimate lives is the one place it changes.
    """
    n = sung_lines(lyrics)
    if not n:
        return ""
    need = n * SECONDS_PER_LINE
    if duration >= need:
        return ""
    return (f"\n\n⚠️ {n} sung lines wants roughly **{need:.0f}s** but you asked for "
            f"{duration:g}s. ACE-Step paces the whole lyric to fit, so expect words to "
            f"be compressed throughout rather than the end being cut.")


def length_hint(lyrics: str) -> str:
    """Live feedback under the Takes lyric box: what length this many words wants."""
    n = sung_lines(lyrics)
    if not n:
        return ("No lyrics: instrumental. Length is a compositional choice here, not a "
                "constraint, so leaving Length at 0 gives 30 / 60 / 120s.")
    ladder = suggest_durations(lyrics)
    mid = ladder[len(ladder) // 2]
    return (f"**{n} sung lines.** Leaving Length at 0 gives **{mid:g}s** "
            f"(~{SECONDS_PER_LINE:g}s per line). Worth hearing: "
            f"{', '.join(f'{d:g}s' for d in ladder)}. That per-line figure is a working "
            f"estimate, not a measured rule, and every listening test so far has wanted "
            f"MORE room rather than less.")


def build_takes(n_visible, model_vals, knob_vals) -> list[Variant]:
    """Turn the widget values into Variants. Pure, so it is testable without Gradio."""
    out = []
    per = len(TAKE_KNOBS)
    for i in range(int(n_visible)):
        kv = knob_vals[i * per:(i + 1) * per]
        kw = {}
        for (name, _lbl, kind, default, _hint), raw in zip(TAKE_KNOBS, kv):
            if kind == "number":
                v = default if raw in (None, "") else raw
                kw[name] = int(v) if isinstance(default, int) else float(v)
            else:
                kw[name] = (raw or "").strip()
        out.append(Variant(model_key=model_vals[i], **kw))
    return out


def create_demo():
    import gradio as gr

    model_keys = list(MODELS)
    brief_keys = list(BY_KEY)
    sweep_keys = list(SWEEPS)

    def _brief_md(key: str) -> str:
        b = BY_KEY[key]
        lyr = ("instrumental" if not b.lyrics.strip()
               else f"{sung_lines(b.lyrics)} sung lines")
        return f"**{b.axis}** · {lyr}\n\n> {b.prompt}"

    with gr.Blocks(title="ACE-Step Studio") as demo:
        gr.Markdown(
            "# 🎵 ACE-Step Studio\n"
            "Launch music-generation runs on the Spark. Every button submits a Flyte "
            "run; the tracks play inline in that run's **Report** tab, alongside a "
            "waveform, a spectrogram, objective measurements, and the command to make "
            "each track again."
        )

        # ── Generate ─────────────────────────────────────────────────────────────
        with gr.Tab("Generate"):
            gr.Markdown("One checkpoint, one track, every knob it has.")
            with gr.Row():
                # Defaults to turbo for a fast first render, but says what that costs
                # you. This picker used to be the only control in the studio with no
                # info line, which made the biggest lever in the whole system look like
                # an incidental setting sitting next to the brief.
                g_model = gr.Dropdown(
                    model_keys, value="xl-turbo", label="Checkpoint",
                    info="THE biggest quality lever. xl-turbo is fast (~23s for a 240s "
                         "track) but 8-step distilled and audibly smeared; xl-sft is "
                         "clearly better, especially on vocals, at ~9x the cost "
                         "(~211s). Guidance below is INERT on turbo. See the Knobs tab.")
                g_brief = gr.Dropdown(
                    brief_keys, value=DEFAULT_BRIEF, label="Brief",
                    info="A named caption + lyrics + metadata from prompts.py. Use "
                         "'Write your own instead' below to override it.")
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
                g_duration = gr.Slider(10, 600, 60, step=5, label="Duration (s)")
                g_seed = gr.Slider(0, 99999, 42, step=1, label="Seed")
            with gr.Accordion("Advanced", open=False):
                gr.Markdown(
                    "`0` / `-1` means *use this checkpoint's own default*, which is what "
                    "keeps a comparison fair. Note **guidance is inert on `xl-turbo`**: "
                    "it is guidance-distilled and the pipeline coerces it to 1.0. The "
                    "**Knobs** tab explains every one of these, including what it costs."
                )
                with gr.Row():
                    g_steps = gr.Slider(0, 200, 0, step=1, label="Steps (0 = default)")
                    g_guidance = gr.Slider(-1, 25, -1, step=0.5, label="Guidance (-1 = default)")
                    g_shift = gr.Slider(-1, 5, -1, step=0.5, label="Shift (-1 = default)")
                with gr.Row():
                    g_bpm = gr.Slider(0, 200, 0, step=1, label="BPM (0 = model decides)")
                    g_keyscale = gr.Textbox("", label="Key / scale", placeholder="A minor")
                    g_timesig = gr.Textbox("", label="Time signature", placeholder="4")
                    g_language = gr.Textbox("en", label="Vocal language", max_lines=1)
            g_go = gr.Button("Generate", variant="primary")
            g_status = gr.Markdown()
            g_link = gr.HTML()

            g_brief.change(lambda k: _brief_md(k), g_brief, g_info)

            def _gen(model, brief, prompt, lyrics, duration, seed, steps, guidance,
                     shift, bpm, keyscale, timesig, language):
                yield from _submit(
                    "generate_one",
                    _weights_note([model]) + _lyric_hint(
                        lyrics if prompt.strip() else BY_KEY[brief].lyrics, duration),
                    model_key=model, brief=brief, prompt=prompt.strip(),
                    lyrics=lyrics if prompt.strip() else "",
                    duration=float(duration), seed=int(seed), steps=int(steps),
                    guidance=float(guidance), shift=float(shift), bpm=int(bpm),
                    keyscale=keyscale.strip(), timesignature=timesig.strip(),
                    language=language.strip() or "en")

            g_go.click(_gen,
                       [g_model, g_brief, g_prompt, g_lyrics, g_duration, g_seed,
                        g_steps, g_guidance, g_shift, g_bpm, g_keyscale, g_timesig,
                        g_language],
                       [g_status, g_link])

        # ── Takes ────────────────────────────────────────────────────────────────
        with gr.Tab("Takes"):
            gr.Markdown(
                "One song, rendered **several ways at once**, each take with its own "
                "knobs. This is the tab for making something, as opposed to running a "
                "designed experiment: you almost never want one track, and assembling "
                "a comparison by hand out of six separate runs is what makes a studio "
                "a toy.\n\n"
                "All takes go out as a **single** Flyte run, so a checkpoint loads once "
                "however many takes use it."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    t_prompt = gr.Textbox(
                        label="Style caption", lines=3,
                        placeholder="1980s arena power ballad, clean chorused guitar in "
                                    "the verses, huge distorted chorus, gated snare, "
                                    "soaring male lead, wide reverb",
                        info="Describing a RECORDING (room, mic, tape, natural "
                             "dynamics) tends to sound less synthetic than piling on "
                             "adjectives like huge and wide.")
                    t_lyrics = gr.Textbox(
                        label="Lyrics", lines=12,
                        placeholder="[verse]\nyour words here\n\n[chorus]\n…",
                        info="EMPTY = instrumental. Do not type the word "
                             "'instrumental'; the model will sing it.")
                    t_hint = gr.Markdown(length_hint(""))
                    t_title = gr.Textbox(label="Report title (optional)", lines=1)
                with gr.Column(scale=2):
                    gr.Markdown(
                        "### Takes\nAlways at least two. The second defaults to a "
                        "different seed, because that is the cheapest question worth "
                        "asking and it shows how much the caption leaves to chance.\n\n"
                        f"⚠️ **{', '.join(_THIN_MODELS)}** ignore most of the knobs "
                        f"below — they take the lyric, the length and the seed and "
                        f"little else. Their report cards print only what they "
                        f"actually used, so a knob set here simply will not appear.")
                    t_n = gr.State(DEFAULT_TAKES)
                    t_rows, t_models, t_knobs = [], [], []
                    for i in range(MAX_TAKES):
                        with gr.Accordion(f"Take {i + 1}", open=(i < DEFAULT_TAKES),
                                          visible=(i < DEFAULT_TAKES)) as acc:
                            m = gr.Dropdown(
                                model_keys, value="xl-sft", label="Checkpoint",
                                info="sft is clearly better on vocals, ~9x turbo's cost")
                            t_models.append(m)
                            for name, lbl, kind, default, khint in TAKE_KNOBS:
                                val = 7 if (name == "seed" and i == 1) else default
                                w = (gr.Number(value=val, label=lbl, info=khint)
                                     if kind == "number"
                                     else gr.Textbox(value=val, label=lbl, info=khint))
                                t_knobs.append(w)
                        t_rows.append(acc)
                    with gr.Row():
                        t_add = gr.Button("➕ Add take", size="sm")
                        t_rm = gr.Button("➖ Remove take", size="sm")

                    def _resize(n, delta):
                        n = max(1, min(MAX_TAKES, int(n) + delta))
                        return [n] + [gr.update(visible=(i < n), open=(i < n))
                                      for i in range(MAX_TAKES)]

                    t_add.click(lambda n: _resize(n, +1), t_n, [t_n, *t_rows])
                    t_rm.click(lambda n: _resize(n, -1), t_n, [t_n, *t_rows])

            t_go = gr.Button("🎧 Render takes", variant="primary")
            t_status = gr.Markdown()
            t_link = gr.HTML()

            t_lyrics.change(length_hint, t_lyrics, t_hint)

            def _takes(prompt, lyrics, title, n, *flat):
                if not (prompt or "").strip():
                    yield "⚠️ A style caption is required.", ""
                    return
                models = list(flat[:MAX_TAKES])
                try:
                    takes = build_takes(n, models, list(flat[MAX_TAKES:]))
                except (TypeError, ValueError) as e:
                    yield f"⚠️ Could not read the knobs: {e}", ""
                    return
                used = sorted({t.model_key for t in takes})
                yield from _submit(
                    "variants",
                    f"{len(takes)} take(s) on {', '.join(used)}. " + _weights_note(used),
                    prompt=prompt.strip(), lyrics=lyrics or "", takes=takes,
                    title=(title or "").strip())

            t_go.click(_takes, [t_prompt, t_lyrics, t_title, t_n, *t_models, *t_knobs],
                       [t_status, t_link])

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
                    info="xl-sft and xl-base run 50 steps with real CFG, so roughly 9x "
                         "the compute of turbo per track.")
                c_suite = gr.Dropdown(list(SUITES), value="quick", label="Suite")
            c_briefs = gr.CheckboxGroup(
                brief_keys, value=[], label="Or pick briefs (overrides the suite)")
            with gr.Row():
                c_duration = gr.Slider(10, 600, 60, step=5, label="Duration (s)")
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
                "another column costs one render, not another 11GB model load.\n\n"
                "The pins below fix the knobs the axis is *not* moving. Some axes are "
                "meaningless without them: a `cfg_end` sweep at the default CFG has "
                "little harshness to remove, so run it at `--guidance 20`."
            )
            with gr.Row():
                s_axis = gr.Dropdown(sweep_keys, value="seed", label="Axis")
                s_model = gr.Dropdown(model_keys, value="xl-sft", label="Checkpoint")
                s_brief = gr.Dropdown(brief_keys, value=DEFAULT_BRIEF, label="Brief")
            s_note = gr.Markdown()
            s_values = gr.Textbox("", label="Values (comma separated, blank = the axis defaults)",
                                  placeholder="4, 8, 16, 32")
            with gr.Row():
                s_duration = gr.Slider(10, 600, 60, step=5, label="Duration (s)")
                s_seed = gr.Slider(0, 99999, 42, step=1, label="Seed")
            with gr.Accordion("Pin the other knobs", open=False):
                with gr.Row():
                    s_steps = gr.Slider(0, 200, 0, step=1, label="Steps (0 = default)")
                    s_guidance = gr.Slider(-1, 25, -1, step=0.5, label="Guidance (-1 = default)")
                    s_shift = gr.Slider(-1, 5, -1, step=0.5, label="Shift (-1 = default)")
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

            def _sweep(axis, model, brief, values, duration, seed, steps, guidance, shift):
                vals = [v.strip() for v in (values or "").split(",") if v.strip()]
                yield from _submit(
                    "sweep",
                    f"axis **{axis}**, {len(vals) or len(SWEEPS[axis].values)} values. "
                    + _weights_note([model]),
                    axis=axis, model_key=model, brief=brief, values=vals or None,
                    duration=float(duration), seed=int(seed), steps=int(steps),
                    guidance=float(guidance), shift=float(shift))

            s_go.click(_sweep, [s_axis, s_model, s_brief, s_values, s_duration, s_seed,
                                s_steps, s_guidance, s_shift],
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
                r_duration = gr.Slider(10, 600, 60, step=5, label="Duration (s)")
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

        # ── Knobs ────────────────────────────────────────────────────────────────
        with gr.Tab("Knobs"):
            gr.Markdown(
                "## What each parameter does, and what it costs\n"
                "Everything here was **measured on this machine**, not copied off a "
                "model card. The headline, which is not obvious and changes how you "
                "use the rest of the studio:\n\n"
                "> **Steps are the only knob you actually pay for.** Guidance, shift, "
                "BPM, key and time signature are all free, and CFG cutoff is *cheaper* "
                "than leaving it alone. So there is no reason to be timid with any of "
                "them.\n\n"
                "Where a recommendation is provisional or a knob is inert on a given "
                "checkpoint, the entry says so."
            )
            k_pick = gr.Dropdown([(_DOC_LABELS[k], k) for k in KNOB_DOCS],
                                 value="model_key", label="Which parameter?")
            k_doc = gr.Markdown(KNOB_DOCS["model_key"])
            k_pick.change(lambda k: KNOB_DOCS.get(k, ""), k_pick, k_doc)

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
