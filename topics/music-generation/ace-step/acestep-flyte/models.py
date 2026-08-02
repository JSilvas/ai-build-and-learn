"""Registry of ACE-Step 1.5 checkpoints, plus the parameter axes we sweep.

Two registries live here because the demo has two questions:

  MODELS : which *checkpoint*? The three XL variants differ in how they were trained
           (base -> sft -> turbo), and that difference shows up as a completely
           different sampling recipe, not just a quality delta.
  SWEEPS : which *knob*? Once you've picked a checkpoint, the report's second job is
           showing what each generation parameter actually does to the audio. A seed
           change and a step-count change are both "one number", and they do wildly
           different things; the sweep makes that audible side by side.

── Only the XL (5B DiT) line is here, and only the -diffusers repos ─────────────
ACE-Step publishes each checkpoint twice: a native repo for the upstream `acestep`
package, and a `-diffusers` repo converted into the standard pipeline layout. Only
the latter loads with `AceStepPipeline.from_pretrained`, and as of this writing only
the XL line has been converted, so the 2B checkpoints are not reachable from here.
That's fine on a 119.7GiB unified pool: XL is ~11GB of bf16 weights and the "needs
>=20GB VRAM" note on the model card is a consumer-GPU concern, not ours.

── The turbo / sft / base split, and why it is the interesting comparison ────────
`turbo` is *guidance-distilled*: CFG is baked into the weights, so it runs 8 steps
with no negative-prompt pass and the pipeline actively ignores `guidance_scale > 1.0`
(it warns and coerces to 1.0). One denoising pass per step instead of two, and 8 steps
instead of 50, is roughly a 12x compute cut. Whether you can hear the 12x is exactly
what `compare` is for.

`sft` is the instruction-tuned checkpoint: it wants real CFG and 30-60 steps, and it
is the one to reach for when prompt adherence matters more than latency. `base` is the
pretrained model underneath both, kept here because "what did fine-tuning buy?" is a
fair question and nobody ever gets to ask it with commercial models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MusicModelSpec:
    key: str                  # short handle used on the CLI and in reports
    repo: str                 # HuggingFace repo id (diffusers layout)
    family: str               # for the report card
    license: str
    params: str               # human string
    download_gb: float        # measured size of the repo, from the HF API
    sample_rate: int = 48000  # ACE-Step's Oobleck VAE is 48kHz stereo

    # The checkpoint's own sampling recipe. These are the defaults a job inherits when
    # it does not override them, and they differ per checkpoint by design: handing sft
    # the turbo recipe (8 steps, no CFG) produces mush, and handing turbo the sft
    # recipe wastes 6x the compute for output the pipeline partly discards.
    steps: int = 8
    guidance: float = 1.0
    shift: float = 3.0

    distilled: bool = False   # guidance-distilled: guidance_scale > 1 is ignored
    dtype: str = "bfloat16"   # GB10 (Blackwell) is happiest in bf16
    notes: str = ""


MODELS: dict[str, MusicModelSpec] = {
    "xl-turbo": MusicModelSpec(
        key="xl-turbo",
        repo="ACE-Step/acestep-v15-xl-turbo-diffusers",
        family="ACE-Step 1.5 XL · flow-matching DiT",
        license="MIT (weights) · Apache-2.0 (Qwen3 text encoder)",
        params="5B DiT + 0.6B text encoder",
        download_gb=11.1,
        steps=8, guidance=1.0, shift=3.0,
        distilled=True,
        notes="Guidance-distilled. 8 steps, no CFG pass. The one you reach for when "
              "you want to iterate on a prompt at conversational speed.",
    ),
    "xl-sft": MusicModelSpec(
        key="xl-sft",
        repo="ACE-Step/acestep-v15-xl-sft-diffusers",
        family="ACE-Step 1.5 XL · flow-matching DiT",
        license="MIT (weights) · Apache-2.0 (Qwen3 text encoder)",
        params="5B DiT + 0.6B text encoder",
        download_gb=11.5,
        steps=50, guidance=7.0, shift=3.0,
        notes="Instruction-tuned, real CFG. Slower per track but the better listener "
              "when the prompt is specific. Also ships the audio tokenizer pair the "
              "turbo repo omits, so it is the one that can do code-conditioned cover.",
    ),
    "xl-base": MusicModelSpec(
        key="xl-base",
        repo="ACE-Step/acestep-v15-xl-base-diffusers",
        family="ACE-Step 1.5 XL · flow-matching DiT",
        license="MIT (weights) · Apache-2.0 (Qwen3 text encoder)",
        params="5B DiT + 0.6B text encoder",
        download_gb=11.5,
        steps=50, guidance=7.0, shift=3.0,
        notes="The pretrained checkpoint under the other two. Here so 'what did the "
              "fine-tune actually buy?' is a question you can answer by ear.",
    ),
}

# The default `compare` set: the speed/quality pair. base is opt-in because a third
# 11GB pull and a third 50-step pass is a lot to spend on a question most runs are
# not asking.
DEFAULT_MODELS = ["xl-turbo", "xl-sft"]


def get_spec(key: str) -> MusicModelSpec:
    if key not in MODELS:
        raise ValueError(f"unknown model {key!r}; known: {', '.join(MODELS)}")
    return MODELS[key]


def resolve_models(keys: list[str] | None) -> list[MusicModelSpec]:
    return [get_spec(k) for k in (keys or DEFAULT_MODELS)]


# ── The parameter sweep ──────────────────────────────────────────────────────────
#
# One checkpoint, one prompt, one knob moved across N values, rendered as a row you
# play left to right. This is the part of the demo that is genuinely hard to do
# anywhere else: a hosted music API gives you a seed field and a vibe, not a
# controlled experiment.
#
# `field` names an attribute on music_core.GenSettings. `listen_for` is printed above
# the row in the report, because a grid of clips with no stated hypothesis is just
# noise: the point is to say what should change BEFORE you press play.


@dataclass(frozen=True)
class SweepAxis:
    field: str                 # GenSettings attribute to vary
    values: tuple              # the values to render, in report order
    label: str                 # short axis name for the report heading
    listen_for: str            # the hypothesis, stated up front
    fmt: str = "{}"            # how a value is rendered in a column label
    turbo_ok: bool = True      # False = meaningless on a distilled checkpoint


SWEEPS: dict[str, SweepAxis] = {
    "seed": SweepAxis(
        field="seed",
        values=(11, 42, 1234, 90210),
        label="seed",
        listen_for="Everything else is identical, so every difference you hear is the "
                   "noise the sampler started from. Listen for how much the *arrangement* "
                   "moves, not just the timbre: if four seeds give four genuinely "
                   "different songs, the prompt is underspecified and the model is "
                   "filling in the rest. Tightening the prompt should visibly shrink "
                   "this spread, which is the fastest prompt-engineering feedback loop "
                   "you will get out of a music model.",
    ),
    "steps": SweepAxis(
        field="steps",
        values=(4, 8, 16, 32),
        label="denoising steps",
        listen_for="The quality/latency dial. Flow matching degrades gracefully, so the "
                   "low-step clips will not sound broken, they will sound *smeared*: "
                   "transients (kick attack, hi-hat, consonants in the vocal) blur first, "
                   "and the stereo image narrows. Find the knee. On the turbo checkpoint "
                   "it is usually at or below its default 8, which is the whole point of "
                   "distillation, and paying for 32 steps buys you nothing.",
    ),
    "guidance": SweepAxis(
        field="guidance",
        values=(1.0, 3.0, 7.0, 12.0),
        label="guidance scale (CFG)",
        listen_for="How hard the model is pushed toward the prompt. Low: it drifts, "
                   "ignores half your tags, and sounds relaxed and slightly generic. "
                   "High: it obeys, then over-obeys, and you get the audio version of "
                   "an over-saturated image, harsh, brittle, compressed, with the "
                   "instruments fighting. Note this row is INERT on xl-turbo: guidance "
                   "is distilled into those weights and the pipeline coerces it to 1.0 "
                   "with a warning, which is itself worth seeing once.",
        fmt="{:g}",
        turbo_ok=False,
    ),
    "shift": SweepAxis(
        field="shift",
        values=(1.0, 2.0, 3.0),
        label="timestep shift",
        listen_for="The least-known knob and the most ACE-Step-specific one. `shift` "
                   "warps where the flow-matching schedule spends its steps: high shift "
                   "front-loads the noisy end, which is where global structure (form, "
                   "arrangement, groove) gets decided; low shift spends more budget on "
                   "the clean end, which is detail and texture. 3.0 is the shipped "
                   "recommendation. At a fixed low step count this trade is audible.",
        fmt="{:g}",
    ),
    "bpm": SweepAxis(
        field="bpm",
        values=(75, 100, 128, 160),
        label="BPM metadata",
        listen_for="ACE-Step takes bpm/key/time-signature as *structured metadata*, not "
                   "as words in the prompt: they go into a `# Metas` block the text "
                   "encoder was trained on. So this is a real control surface, and the "
                   "question is how obedient it is. Tap along. Does the track actually "
                   "land on the requested tempo, or does it pick a groove it likes and "
                   "ignore you? Mismatches between prompt genre and requested BPM (a "
                   "ballad at 160) are where it gets interesting.",
    ),
    "keyscale": SweepAxis(
        field="keyscale",
        values=("C major", "A minor", "F# minor", "D dorian"),
        label="key / scale metadata",
        listen_for="Same structured-metadata channel as BPM, aimed at harmony. Major vs "
                   "minor should be unmissable. The real test is the last two: an "
                   "unusual key and a modal scale are where a model that memorized "
                   "'minor = sad' falls apart, because dorian is a minor scale that "
                   "does not sound sad.",
    ),
    "duration": SweepAxis(
        field="duration",
        values=(20.0, 40.0, 80.0),
        label="audio duration",
        listen_for="Duration is not a crop: it is fed to the model up front and changes "
                   "the composition. A 20s render has to state the idea immediately; an "
                   "80s render has room for an intro, a build and a turnaround. Listen "
                   "for whether the long one uses that room or just loops. Cost scales "
                   "with length, so the timings under each card are the honest answer to "
                   "'what does a full-length track actually cost?'",
        fmt="{:g}s",
    ),
}


def get_sweep(name: str) -> SweepAxis:
    if name not in SWEEPS:
        raise ValueError(f"unknown sweep {name!r}; known: {', '.join(SWEEPS)}")
    return SWEEPS[name]
