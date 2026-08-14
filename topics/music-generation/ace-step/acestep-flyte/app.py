"""ACE-Step Studio: paste a song, render it several ways, pick the one that works.

A thin CPU launcher, exactly like the image and video studios next door. It loads no
model and touches no GPU: pressing Render submits the `variants` pipeline as a Flyte
run and hands back a link to the report. An app pod lives for as long as the app is up,
so a studio that generated in-process would pin the Spark's only GPU forever and every
pipeline task would queue behind it.

WHAT MAKES THIS ONE DIFFERENT from the other studios: they render one thing at a time.
This one is built around the observation that you never actually want one take. Every
useful session in this project has been a comparison, and the difference between a
usable studio and a toy is whether the comparison is the default or something you
assemble by hand from six separate runs.

So the unit here is a TAKE, and there are always at least two. The second one defaults
to a different seed, because that is the cheapest question worth asking and because a
one-card report teaches you nothing about how underspecified your prompt is. Every take
carries the full knob set, folded away until you want it.

All takes go out as ONE run, not one run per take. Beyond the obvious saving (a
checkpoint loads once however many takes use it), this repo learned the hard way that N
concurrent runs means N orchestrator pods each holding 8Gi while awaiting a 96Gi child;
past three, they starve the very children they are waiting on.

Development progression:
  1. Local app + local pipeline:   RUN_MODE=local python app.py
  2. Local app + remote pipeline:  python app.py
  3. Deploy the pipeline, then the app:
       flyte deploy compare_pipeline.py
       flyte deploy app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import flyte
import flyte.app
import flyte.remote as remote
import gradio as gr

from config import APP_NAME, APP_PORT, studio_app_image

RUN_MODE = os.environ.get("RUN_MODE", "remote")
PROJECT = os.environ.get("FLYTE_PROJECT", "music-generation")
DOMAIN = os.environ.get("FLYTE_DOMAIN", "development")
FLYTE_UI_URL = os.environ.get("FLYTE_UI_URL", "http://localhost:30080")

MAX_TAKES = 6            # more than this and the report stops being scannable
DEFAULT_TAKES = 2        # never one: see the module docstring

# The knob registry. Kept as data rather than inline widgets because it is the same
# list three times over (build the row, read the row, reset the row), and three
# hand-maintained copies of a knob list is how a studio ends up silently ignoring one.
#
# `sentinel` is the value meaning "inherit from the checkpoint", which is what every
# knob defaults to. A studio that shipped its OWN defaults would quietly override the
# per-checkpoint sampling recipes that make turbo-vs-sft a fair comparison at all.
KNOBS = [
    ("seed",       "Seed",         "number", 42,   None),
    ("duration",   "Length (s)",   "number", 0,    "0 = derive from the lyric"),
    ("steps",      "Steps",        "number", 0,    "0 = checkpoint default (turbo 8, sft 50)"),
    ("guidance",   "Guidance",     "number", -1,   "-1 = checkpoint default. Inert on turbo"),
    ("shift",      "Shift",        "number", -1,   "-1 = default (3). Low = texture, high = structure"),
    ("cfg_interval_end", "CFG cutoff", "number", 1.0, "1 = full range. 0.5 = guidance early only"),
    ("cfg_interval_start", "CFG onset", "number", 0.0, "0 = from the start. Raising it is the control, expect worse"),
    ("bpm",        "BPM",          "number", 0,    "0 = let the model choose"),
    ("keyscale",   "Key / scale",  "text",   "",   "e.g. A minor, D dorian"),
    ("timesignature", "Time sig",  "text",   "",   "4, 3, 6 … confirmed audible"),
    ("language",   "Vocal language", "text", "en", "en, pt, zh, ja … match your lyric"),
]

# diffrhythm is here because it is the only OTHER model that sings, which makes it the
# one genuine alternative rather than a baseline. It honours far fewer knobs (steps,
# guidance and the lyric, nothing else), and it wants timed lyrics rather than
# structural ones, so this repo converts and the report card says it did. The card
# prints only the knobs its adapter actually has, so a take on diffrhythm will not
# claim to have used a bpm it ignored.
MODEL_CHOICES = ["xl-sft", "xl-turbo", "xl-base", "diffrhythm"]

# The reference panel. Every entry says what the knob does, why you would reach for it,
# and WHAT IT COSTS, because the cost answers turned out to be the surprising part and
# they change how you use the studio: only one of these knobs is expensive, so there is
# no reason to be timid with the rest.
#
# Measurements are from this project's own runs on the Spark (a 240s track on xl-sft
# unless stated), not from the model card. Where something is unmeasured or provisional
# it says so; a reference panel that quietly presents a guess as a fact is worse than
# no panel, because it will be believed.
KNOB_DOCS = {
    "model_key": (
        "### Checkpoint — the biggest lever, and the only one confirmed by ear\n"
        "**What it does.** Which trained model renders. `xl-turbo` is *guidance "
        "distilled*: 8 denoising steps with CFG folded into the weights. `xl-sft` is "
        "instruction-tuned: 50 steps and a real guidance pass. `xl-base` is the "
        "pretrained model underneath both.\n\n"
        "**Why tweak it.** This is the one change that clearly improved things, most "
        "audibly **on the voice**. Low-step flow matching does not sound broken, it "
        "sounds *smeared*, and smeared is most of what people mean by 'it sounds AI'.\n\n"
        "**Cost.** 240s track: turbo **23s**, sft **211s**. About 9x.\n\n"
        "**The trap.** Several knobs below do nothing at all on turbo. Guidance is "
        "silently coerced to 1.0. Tune on the checkpoint you intend to ship.\n\n"
        "**Try:** `xl-sft` for anything with vocals; `xl-turbo` to iterate on wording "
        "fast; `xl-base` to hear what the fine-tune bought."
    ),
    "seed": (
        "### Seed — the free lottery, and a diagnostic\n"
        "**What it does.** The noise the sampler starts from. Everything else fixed, a "
        "new seed is a new take.\n\n"
        "**Why tweak it.** Two reasons. Some takes are just better. But the more useful "
        "one: *how much* the arrangement moves between seeds tells you how "
        "underspecified your caption is. Four wildly different songs means the model is "
        "filling in what you did not say, and tightening the caption should visibly "
        "shrink that spread. It is the fastest prompt feedback loop you get.\n\n"
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
        "**Status: provisional.** Leaving this at 0 derives a length from your line "
        "count. Every listening test so far has preferred *more* room per line than "
        "the current estimate, and the knee has not been bracketed from above.\n\n"
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
        "**Why tweak it.** It is the knob aimed most directly at both failure modes, "
        "and it costs nothing to explore.\n\n"
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
        "**Why tweak it.** Guidance does two different things at two different times. "
        "Early it decides form, groove and arrangement, where pushing toward the "
        "caption genuinely helps. Late it renders texture and detail, and pushing hard "
        "there is what makes a high CFG harsh. Restricting it to the early part asks "
        "for adherence **without** the artifacts. Without this knob, a guidance sweep "
        "only offers 'drifts' or 'harsh' with nothing in between.\n\n"
        "**Cost: negative.** Cutting off at 0.25 rendered in **164s** against 209s for "
        "the full range, a 22% saving, because the later steps stop doing the second "
        "forward pass at all.\n\n"
        "**Try:** high guidance (12-20) paired with a cutoff of 0.4-0.6. If it works "
        "you get adherence, lose the sheen, and render faster."
    ),
    "cfg_interval_start": (
        "### CFG onset — the control, and it should sound *worse*\n"
        "**What it does.** The mirror of CFG cutoff. Raising it switches guidance OFF "
        "for the early part of the schedule and on only later.\n\n"
        "**Why tweak it.** Mostly to check that the reasoning behind CFG cutoff is "
        "right. If guidance helps early (deciding form) and hurts late (rendering "
        "texture), then doing the opposite should lose prompt adherence *and* keep the "
        "harshness. If raising this sounds fine, the early/late story is wrong and the "
        "cutoff result means something else.\n\n"
        "**Cost.** Same saving as the cutoff: fewer steps run the second forward pass.\n\n"
        "**Try:** leave at 0. Set it to 0.5 only when you want the control."
    ),
    "language": (
        "### Vocal language — match it to your lyric\n"
        "**What it does.** Sets the language header on the lyric. ACE-Step claims 50+.\n\n"
        "**Why tweak it.** If you write in a language other than English and leave this "
        "at `en`, the phrasing drifts toward English pronunciation of the spelling "
        "rather than the language. It is the difference between singing Portuguese and "
        "reading Portuguese aloud in an English accent.\n\n"
        "**Cost.** None.\n\n"
        "**Try:** `en` unless your lyric is not English, then match it (`pt`, `zh`, "
        "`ja`, …). Worth one take in another language just to hear how far the claim "
        "goes."
    ),
    "shift": (
        "### Shift — where the step budget gets spent\n"
        "**What it does.** Warps the flow-matching schedule. High front-loads the noisy "
        "end, where global structure (form, arrangement, groove) is decided. Low spends "
        "more budget at the clean end, which is detail and texture.\n\n"
        "**Why tweak it.** It is the specific knob for a problem that is *textural* "
        "rather than structural. Same number of steps either way, just redistributed.\n\n"
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
        "**Cost.** None: 212s / 209s / 209s for 4, 3 and 6.\n\n"
        "**Try:** `3` against a genre that strongly implies 4/4. Higher and odd values "
        "are unexplored territory."
    ),
}
_DOC_LABELS = {
    "model_key": "Checkpoint (biggest lever)", "seed": "Seed",
    "duration": "Length", "steps": "Steps (the only expensive one)",
    "guidance": "Guidance / CFG", "cfg_interval_end": "CFG cutoff",
    "cfg_interval_start": "CFG onset (the control)",
    "shift": "Shift", "bpm": "BPM", "keyscale": "Key / scale",
    "timesignature": "Time signature", "language": "Vocal language",
}

# Same drift guard as Variant's, one layer up: a knob can exist in the pipeline, be
# reachable through Variant, and still have no widget here. Both directions are
# checked, because an orphan doc entry means a widget was renamed and the panel is now
# describing something that is not on screen.
assert {n for n, *_ in KNOBS} | {"model_key"} == set(KNOB_DOCS) == set(_DOC_LABELS), (
    "studio knobs, their docs and their labels have drifted apart")

_here = Path(__file__).parent
_bundled = studio_app_image.with_source_file(
    [_here / "models.py", _here / "prompts.py", _here / "config.py"])

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

_variants_ref = remote.Task.get(
    "acestep-orch.variants", project=PROJECT, domain=DOMAIN, auto_version="latest",
)


def _variants_task():
    if RUN_MODE == "local":
        from compare_pipeline import variants
        return variants
    return _variants_ref


def _external_url(url) -> str:
    """Rewrite an in-cluster run URL to the browser-reachable console URL."""
    if not url:
        return ""
    s = str(url)
    if s.startswith("http") and "flyte-binary" not in s and "flyte:" not in s:
        return s
    return f"{FLYTE_UI_URL}{urlparse(s).path}"


def _sung_lines(lyrics: str) -> int:
    """Sung lines: structure tags and blanks are not lyrics.

    Duplicated from prompts.sung_lines rather than imported at module scope, because
    the app image bundles prompts.py but this must not fail if that ever changes; it is
    four lines and it drives only a hint.
    """
    return len([l for l in (lyrics or "").splitlines()
                if l.strip() and not l.strip().startswith("[")])


def length_hint(lyrics: str) -> str:
    """Live feedback under the lyric box: what length this many words wants.

    The single most useful thing this studio knows that a first-time user does not.
    ACE-Step paces the WHOLE lyric to fit the duration rather than truncating it, so a
    long lyric in a short render is not cut off, it is compressed: syllables shorten,
    breaths vanish, and it is the main cause of the "sounds AI" reaction. The number
    below is provisional and the honest thing is to say so rather than present it as a
    rule.
    """
    n = _sung_lines(lyrics)
    if not n:
        return ("No lyrics: instrumental. Length is a compositional choice here, not a "
                "constraint, so the default ladder is 30 / 60 / 120s.")
    try:
        from prompts import SECONDS_PER_LINE, suggest_durations
        ladder = suggest_durations(lyrics)
        mid = ladder[len(ladder) // 2]
        return (f"**{n} sung lines.** Leaving Length at 0 gives **{mid:g}s** "
                f"(~{SECONDS_PER_LINE:g}s per line). Worth hearing: "
                f"{', '.join(f'{d:g}s' for d in ladder)}. That per-line figure is a "
                f"working estimate, not a measured rule, and every listening test so "
                f"far has wanted MORE room rather than less.")
    except Exception:
        return f"**{n} sung lines.**"


def build_takes(n_visible, model_vals, knob_vals):
    """Turn the widget values into Variant objects. Pure, so it is testable."""
    from compare_pipeline import Variant
    out = []
    per = len(KNOBS)
    for i in range(int(n_visible)):
        kv = knob_vals[i * per:(i + 1) * per]
        kw = {}
        for (name, _, kind, default, _hint), raw in zip(KNOBS, kv):
            if kind == "number":
                v = default if raw in (None, "") else raw
                kw[name] = int(v) if isinstance(default, int) else float(v)
            else:
                kw[name] = (raw or "").strip()
        out.append(Variant(model_key=model_vals[i], **kw))
    return out


def launch(prompt, lyrics, title, n_visible, *flat):
    """Submit ONE run holding every take, and hand back the report link."""
    if not (prompt or "").strip():
        yield "⚠️ A style caption is required: genre, instruments, production.", ""
        return

    model_vals = list(flat[:MAX_TAKES])
    knob_vals = list(flat[MAX_TAKES:])
    try:
        takes = build_takes(n_visible, model_vals, knob_vals)
    except (TypeError, ValueError) as e:
        yield f"⚠️ Could not read the knobs: {e}", ""
        return

    yield f"🚀 Submitting {len(takes)} take(s)…", ""
    try:
        run = flyte.run(_variants_task(), prompt=prompt.strip(),
                        lyrics=lyrics or "", takes=takes,
                        title=(title or "").strip())
    except Exception as e:
        yield f"❌ Could not launch: {type(e).__name__}: {e}", ""
        return

    url = _external_url(getattr(run, "url", None))
    link = (f'<a href="{url}" target="_blank" rel="noopener">🔗 Open '
            f'<code>{run.name}</code></a> — the <b>Report</b> tab has the players.'
            if url else f"Running as <code>{run.name}</code>…")
    models = sorted({t.model_key for t in takes})
    note = ("First run on a checkpoint downloads ~11GB, so it is slow once and cached "
            "after." if models else "")
    yield (f"🚀 Launched {len(takes)} take(s) on {', '.join(models)}. {note}", link)


def ui():
    with gr.Blocks(title="ACE-Step Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# ACE-Step Studio\n"
            "Paste a caption and a lyric, then render it **several ways at once** and "
            "pick the one that works. Everything runs as a single Flyte run on the "
            "Spark; this page holds no GPU and loads no model."
        )
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Style caption", lines=3,
                    placeholder="1980s arena power ballad, clean chorused guitar in "
                                "the verses, huge distorted chorus, gated snare, "
                                "soaring male lead, wide reverb",
                    info="Genre, instruments, production. Describing a RECORDING "
                         "(room, mic, tape, natural dynamics) tends to sound less "
                         "synthetic than piling on adjectives like huge and wide.")
                lyrics = gr.Textbox(
                    label="Lyrics", lines=12,
                    placeholder="[verse]\nyour words here\n\n[chorus]\n…",
                    info="[intro] [verse] [chorus] [bridge] [outro] are real "
                         "conditioning, not decoration. Leave empty for an "
                         "instrumental; do NOT type the word 'instrumental', the "
                         "model will sing it.")
                hint = gr.Markdown(length_hint(""))
                title = gr.Textbox(label="Report title (optional)", lines=1)
            with gr.Column(scale=2):
                gr.Markdown("### Takes\nAlways at least two. The second defaults to a "
                            "different seed, because that is the cheapest question "
                            "worth asking and it shows you how much the prompt is "
                            "leaving to chance.")
                n_visible = gr.State(DEFAULT_TAKES)
                rows, model_ws, knob_ws = [], [], []
                for i in range(MAX_TAKES):
                    with gr.Accordion(f"Take {i + 1}", open=(i < DEFAULT_TAKES),
                                      visible=(i < DEFAULT_TAKES)) as acc:
                        m = gr.Dropdown(MODEL_CHOICES, value="xl-sft",
                                        label="Checkpoint",
                                        info="sft is clearly better on vocals and ~9x "
                                             "the cost of turbo")
                        model_ws.append(m)
                        for name, lbl, kind, default, khint in KNOBS:
                            # Second take differs by seed: the built-in comparison.
                            val = 7 if (name == "seed" and i == 1) else default
                            w = (gr.Number(value=val, label=lbl, info=khint,
                                           precision=None)
                                 if kind == "number"
                                 else gr.Textbox(value=val, label=lbl, info=khint))
                            knob_ws.append(w)
                    rows.append(acc)

                with gr.Row():
                    add = gr.Button("➕ Add take", size="sm")
                    rm = gr.Button("➖ Remove take", size="sm")

                def _resize(n, delta):
                    n = max(1, min(MAX_TAKES, int(n) + delta))
                    return [n] + [gr.update(visible=(i < n), open=(i < n))
                                  for i in range(MAX_TAKES)]

                add.click(lambda n: _resize(n, +1), n_visible, [n_visible, *rows])
                rm.click(lambda n: _resize(n, -1), n_visible, [n_visible, *rows])

        with gr.Accordion("📖 What does each knob do, and why tweak it?", open=False):
            gr.Markdown(
                "Measured on this box, not copied off a model card. The short version: "
                "**steps are the only knob you pay for.** Guidance, shift and the "
                "metadata fields are free, and CFG cutoff is *cheaper* than leaving it "
                "alone, so there is no reason to be timid with any of them."
            )
            picker = gr.Dropdown(
                [(_DOC_LABELS[k], k) for k in KNOB_DOCS],
                value="model_key", label="Parameter", container=True)
            doc = gr.Markdown(KNOB_DOCS["model_key"])
            picker.change(lambda k: KNOB_DOCS.get(k, ""), picker, doc)

        go = gr.Button("🎧 Render takes", variant="primary")
        status = gr.Markdown()
        link = gr.HTML()

        lyrics.change(length_hint, lyrics, hint)
        go.click(launch, [prompt, lyrics, title, n_visible, *model_ws, *knob_ws],
                 [status, link])

        gr.Markdown(
            "---\n"
            "**Reading the report.** Each card is labelled with what makes it "
            "*different* from the first take. The small numbers under each are "
            "measurements (dynamic range, brightness, stereo width, transient "
            "sharpness), not quality scores: they are meaningful as differences "
            "across a row, and say nothing on their own about whether a take is good."
        )
    return demo


def main():
    if os.environ.get("FLYTE_INTERNAL_EXECUTION_ID") or RUN_MODE == "cluster":
        flyte.init_in_cluster(project=PROJECT, domain=DOMAIN)
    else:
        flyte.init_from_config(root_dir=Path(__file__).parent)
    ui().launch(server_name="0.0.0.0", server_port=APP_PORT,
                share=bool(os.environ.get("GRADIO_SHARE")))


if __name__ == "__main__":
    main()
