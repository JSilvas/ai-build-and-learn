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
    repo: str                 # HuggingFace repo id
    family: str               # for the report card
    license: str
    params: str               # human string
    download_gb: float        # measured size of the repo, from the HF API
    sample_rate: int = 48000  # ACE-Step's Oobleck VAE is 48kHz stereo

    # Which loader/generator in music_core drives this model. The TTS demo next door
    # learned this the hard way: as soon as a registry spans model FAMILIES, "which
    # pipeline class" stops being a property you can infer and becomes one you declare.
    # Unlike that demo, an adapter here does NOT imply a separate image: ACE-Step needs
    # diffusers and MusicGen needs transformers, and one image already has both.
    adapter: str = "acestep"

    # What the model is FOR, printed on the card. Load-bearing for honesty once the
    # registry spans families: ACE-Step writes songs with vocals, MusicGen writes
    # ~30s instrumental beds, Stable Audio makes textures and SFX. Running one prompt
    # across all three is informative, but only if the report says what each was built
    # to do. Without this line the grid reads as a beauty contest that the model with
    # the widest remit always wins.
    intended_for: str = ""

    # Hard ceiling on a single render, where the architecture has one. MusicGen was
    # trained on 30s windows; ask for more and it degrades rather than refusing, so we
    # clamp and say so on the card instead of quietly shipping mush.
    max_duration: float = 0.0   # 0 = no practical limit

    # The checkpoint's own sampling recipe. These are the defaults a job inherits when
    # it does not override them, and they differ per checkpoint by design: handing sft
    # the turbo recipe (8 steps, no CFG) produces mush, and handing turbo the sft
    # recipe wastes 6x the compute for output the pipeline partly discards.
    steps: int = 8
    guidance: float = 1.0
    shift: float = 3.0

    distilled: bool = False   # guidance-distilled: guidance_scale > 1 is ignored
    gated: bool = False       # HF repo needs terms accepted on the HF_TOKEN account

    # True when the model's own code fetches its checkpoints at runtime rather than
    # loading from a path we hand it. DiffRhythm calls hf_hub_download internally, so
    # our fetch_weights step has nothing useful to pre-stage and the orchestrator skips
    # it; HF_HOME still points somewhere sane so the download is cached per pod.
    self_downloads: bool = False
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
        intended_for="Full songs with sung vocals, 10s to 10min, 48kHz stereo.",
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
        intended_for="Full songs with sung vocals, 10s to 10min, 48kHz stereo.",
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
        intended_for="Full songs with sung vocals, 10s to 10min, 48kHz stereo.",
        steps=50, guidance=7.0, shift=3.0,
        notes="The pretrained checkpoint under the other two. Here so 'what did the "
              "fine-tune actually buy?' is a question you can answer by ear.",
    ),

    # ── MusicGen (Meta) ──────────────────────────────────────────────────────────
    #
    # A different FAMILY, not another ACE-Step checkpoint, and the comparison is more
    # interesting for it. MusicGen is an autoregressive transformer over EnCodec tokens
    # (50 tokens/sec, 32kHz) rather than a diffusion model, it has no concept of lyrics
    # or vocals, and it was trained on 30s windows. So it is not "ACE-Step but worse":
    # it is a different tool, and the report says so via `intended_for`.
    #
    # Two things make it worth the registry entry:
    #   1. MELODY CONDITIONING (the -melody checkpoint), which ACE-Step has no
    #      equivalent of: hum or play a tune and get it arranged in a described style.
    #   2. LICENSING. The weights are CC-BY-NC: non-commercial, and that extends to
    #      what you generate. Next to ACE-Step's MIT that is the single most practical
    #      difference between them, and it costs nothing to demonstrate.
    #
    # No new image: transformers ships MusicGen and the ACE-Step image already has it.
    "musicgen-large": MusicModelSpec(
        key="musicgen-large",
        repo="facebook/musicgen-stereo-large",
        family="MusicGen · autoregressive transformer over EnCodec",
        license="CC-BY-NC-4.0 (weights AND outputs are non-commercial)",
        params="3.3B",
        # 6.9GB fetched, not the repo's 20.4GB: it ships .safetensors AND duplicate
        # .bin/state_dict copies of the same weights, and fetch_weights ignores those.
        download_gb=6.9,
        sample_rate=32000,
        adapter="musicgen",
        intended_for="~30s instrumental beds from a text prompt. No lyrics, no vocals.",
        max_duration=30.0,
        guidance=3.0,          # the value the model card recommends
        notes="The stereo 3.3B. Note it is a SMALLER download than any ACE-Step "
              "checkpoint yet caps out at 30s of instrumental, which is a fair "
              "one-line summary of how much moved in a year.",
    ),
    # ── AudioLDM 2 (Liu et al. / CVSSP) ──────────────────────────────────────────
    #
    # A THIRD architecture: latent diffusion over a mel VAE, conditioned by a GPT-2
    # that bridges CLAP audio-text embeddings and T5 text embeddings, decoded by a
    # HiFi-GAN vocoder. Not a flow-matching DiT, not an autoregressive token model.
    #
    # Its real contribution to the report is the FIDELITY axis. It renders 16kHz MONO,
    # so on the card's full-Nyquist spectrogram it shows a hard ceiling at 8kHz next to
    # ACE-Step's content out to 24kHz. That single image explains "audio quality" better
    # than any adjective, and it is why the spectrogram is not cropped to the voice band.
    #
    # `-music` is the music-finetuned variant; plain `cvssp/audioldm2` is general audio
    # and `-large` is bigger but not music-specific. Ungated, and diffusers-native, so
    # once again no new image.
    "audioldm2-music": MusicModelSpec(
        key="audioldm2-music",
        repo="cvssp/audioldm2-music",
        family="AudioLDM 2 · latent diffusion + CLAP/GPT-2/T5 conditioning",
        license="CC-BY-NC-SA-4.0 (non-commercial)",
        params="~1.1B across the stack",
        download_gb=4.5,
        sample_rate=16000,
        adapter="audioldm2",
        intended_for="Short instrumental clips and textures at 16kHz MONO. No lyrics, "
                     "no vocals. The fidelity floor in this comparison, by design.",
        max_duration=30.0,
        steps=200,          # the model card's own recommendation, and it is slow
        guidance=3.5,
        notes="200 denoising steps is its documented default, not a handicap we chose. "
              "Cheapest download in the registry at 4.5GB. Sweet spot is ~10s; it was "
              "trained on 10s segments and drifts on longer asks.",
    ),

    # ── Stable Audio Open (Stability AI) ─────────────────────────────────────────
    #
    # The most interesting comparison in the registry after ACE-Step, for three reasons.
    #
    # 1. ARCHITECTURE. It is the only other DiT here: autoencoder + T5 text encoder +
    #    transformer diffusion in latent space. MusicGen is autoregressive and AudioLDM2
    #    is a UNet, so this is the closest like-for-like against ACE-Step.
    # 2. DATES. June 2024, sitting between MusicGen/AudioLDM2 (2023) and ACE-Step 1.5 XL
    #    (April 2026). Listening to the three in order is a timeline, not a leaderboard.
    # 3. TRAINING DATA. ~48k recordings from Freesound and the Free Music Archive, all
    #    CC0 / CC BY / CC Sampling+. It is the only model here trained exclusively on
    #    licensed audio, which for a stream about *open* music generation is a sharper
    #    talking point than any quality delta.
    #
    # GATED (auto-approve): accept the terms at
    # huggingface.co/stabilityai/stable-audio-open-1.0 on the account behind HF_TOKEN,
    # or fetch_weights fails with a 401. Nothing in this repo can do that for you.
    #
    # Sound-design first, songs second: it is very good at textures, SFX and short
    # instrumental beds, and it does not sing.
    "stable-audio": MusicModelSpec(
        key="stable-audio",
        repo="stabilityai/stable-audio-open-1.0",
        family="Stable Audio Open · latent DiT + T5",
        license="Stability AI Community License (the model card is gated; read it "
                "before shipping anything commercial)",
        params="~1.3B DiT + T5",
        download_gb=15.7,
        sample_rate=44100,
        adapter="stableaudio",
        intended_for="Up to 47s of 44.1kHz stereo. Sound design, textures and short "
                     "instrumental beds. No lyrics, no vocals. Trained only on "
                     "CC-licensed audio.",
        max_duration=47.0,
        steps=100, guidance=7.0,     # the pipeline's own defaults
        gated=True,
        # float16, NOT the bf16 every other model here uses, and this is load-bearing.
        # The scheduler's noise sampler transforms sigmas with `sigma.log().neg()`, so a
        # sigma that underflows to exactly 0 becomes +inf and torchsde's interval search
        # never terminates: RecursionError at any limit, including 50,000. bf16 has ~3
        # decimal digits of mantissa against fp16's ~3.3 and a much coarser step near
        # zero, and the diffusers docs for this model use fp16 throughout. Every other
        # model in the registry is genuinely happier in bf16 on Blackwell; this one is
        # not, which is exactly why dtype lives on the spec rather than in the loader.
        dtype="float16",
        notes="Prompt it like a sound designer, not a songwriter: the docs are explicit "
              "that descriptive, context-specific prompts ('melodic techno with a fast "
              "beat and synths') beat bare genre tags ('techno').",
    ),

    # ── DiffRhythm (ASLP-lab / NWPU) ─────────────────────────────────────────────
    #
    # The only genuine ACE-Step RIVAL in this registry rather than a baseline: it is the
    # one other model here that writes full-length songs WITH SUNG VOCALS from lyrics,
    # and it is also flow-matching. Everything else added alongside it (MusicGen,
    # AudioLDM 2, Stable Audio) is instrumental-only, which is why they read as era
    # context and this one reads as a comparison.
    #
    # March 2025, Apache-2.0, and the busiest repo of the lot (2.3k stars, still being
    # updated). Two-part: a DiT/CFM song model plus MuQ-MuLan as the style encoder.
    #
    # THE ONE MODEL WITH ITS OWN IMAGE. No PyPI package, no packaging metadata at all,
    # so it is a git clone on PYTHONPATH (see config.diffrhythm_image). Its pinned
    # torchaudio==2.6.0 / transformers==4.49.0 turned out NOT to bind: verified in a GPU
    # pod importing fine on torch 2.13+cu130 and transformers 5.14.1.
    #
    # Lyrics are TIMED (.lrc, `[mm:ss.xx]line`), not structural like ACE-Step's
    # [verse]/[chorus]. music_core converts, and the card says so, because a lossy
    # format conversion that goes unmentioned would quietly disadvantage this model on a
    # shared brief.
    "diffrhythm": MusicModelSpec(
        key="diffrhythm",
        repo="ASLP-lab/DiffRhythm2",
        family="DiffRhythm 2 · block flow matching DiT + BigVGAN",
        license="Apache-2.0",
        params="~1.1B + MuQ-MuLan",
        download_gb=5.1,
        sample_rate=44100,
        adapter="diffrhythm",
        intended_for="Full-length songs with SUNG VOCALS from timed lyrics, 44.1kHz "
                     "stereo. The only model here besides ACE-Step that sings.",
        max_duration=210.0,   # inference.py --max-secs default
        steps=16, guidance=2.0,   # v2 CLI defaults
        self_downloads=True,
        notes="Style comes from a text prompt through MuQ-MuLan rather than a caption "
              "encoder, so it reads prompts differently from ACE-Step. `--max-secs` is "
              "a CEILING, not a target: asked for 95 it produced 96s of song, because "
              "it lays out the whole lyric and stops. Short asks are therefore not a "
              "reliable way to get short songs from it.",
    ),

    "musicgen-melody": MusicModelSpec(
        key="musicgen-melody",
        repo="facebook/musicgen-melody",
        family="MusicGen · autoregressive transformer over EnCodec",
        license="CC-BY-NC-4.0 (weights AND outputs are non-commercial)",
        params="1.5B",
        download_gb=6.2,   # 9.2GB repo, minus the duplicate .bin copies
        sample_rate=32000,
        adapter="musicgen",
        intended_for="~30s instrumental beds, optionally conditioned on a hummed or "
                     "played MELODY. The capability ACE-Step has no answer to.",
        max_duration=30.0,
        guidance=3.0,
        notes="Melody conditioning is not wired to an entry point yet; today this runs "
              "as plain text-to-music. Half the size of the large checkpoint.",
    ),
}

# The default `compare` set: the speed/quality pair. base is opt-in because a third
# 11GB pull and a third 50-step pass is a lot to spend on a question most runs are
# not asking.
DEFAULT_MODELS = ["xl-turbo", "xl-sft"]

# The models that can actually SING. Handing a vocal brief to the instrumental-only
# models is safe (they drop the lyric and their card says so), but a vocal comparison
# reads better without three cards explaining why they came back instrumental:
#   flyte run compare_pipeline.py compare --briefs '["indie-vocal"]' \
#       --models '["xl-turbo","diffrhythm"]'
VOCAL_MODELS = ["xl-turbo", "diffrhythm"]


@dataclass
class Variant:
    """One take of the same song, with its own knobs. The studio's unit of work.

    Lives HERE rather than next to the task that consumes it because the Gradio app
    builds these, and the app image is deliberately slim: it has no torch, no
    diffusers, no matplotlib. `compare_pipeline` imports `music_core`, so importing
    Variant from there would drag the whole render stack into a launcher pod whose
    entire job is to submit a run. This module imports `dataclasses` and nothing else,
    which is what keeps that honest.

    Every other entry point fixes WHICH knob varies: `sweep` moves one, `grid` crosses
    two, `takes` moves length. That is right for an experiment, where the value of the
    design is that nothing else can explain the difference. It is wrong for actually
    making music, where you want two seeds, and one of them longer, and one of them on
    the other checkpoint, in the same report, and you do not care that the design is
    unbalanced.

    Sentinels mean "inherit", matching GenSettings, with one addition: `duration = 0`
    means DERIVE it from the lyric, because the studio's point is that you paste words
    and get something sensible without having done the density homework first.
    """
    label: str = ""              # "" -> auto-labelled from what differs
    model_key: str = "xl-sft"
    seed: int = 42
    duration: float = 0.0        # 0 -> derived from the lyric
    steps: int = 0               # 0 -> checkpoint default
    guidance: float = -1.0       # <0 -> checkpoint default
    shift: float = -1.0
    bpm: int = 0
    keyscale: str = ""
    timesignature: str = ""
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    language: str = "en"


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
    "cfg_end": SweepAxis(
        field="cfg_interval_end",
        values=(0.25, 0.5, 0.75, 1.0),
        label="CFG cutoff (guidance applied from the start until X)",
        listen_for="Guidance does two different things at two different times, and this "
                   "separates them. EARLY on the schedule it is deciding form, groove "
                   "and arrangement, and pushing toward the prompt genuinely helps. "
                   "LATE it is rendering texture and detail, and pushing hard there is "
                   "what makes a high CFG sound harsh, brittle and over-saturated. So "
                   "1.0 is the usual full-range behaviour and the smaller values switch "
                   "guidance off partway through. Run this WITH a high --guidance: the "
                   "question is whether 0.5 gives you the adherence of a strong CFG "
                   "without its glassy sheen, which is the thing a plain guidance sweep "
                   "cannot offer you because there every option is all-or-nothing.",
        fmt="{:g}",
        turbo_ok=False,
    ),
    "cfg_start": SweepAxis(
        field="cfg_interval_start",
        values=(0.0, 0.25, 0.5),
        label="CFG onset (guidance applied from X onward)",
        listen_for="The mirror of `cfg_end`, and the one that should be WORSE if the "
                   "reasoning behind it is right. Here guidance is off while the "
                   "arrangement is being decided and only switches on later, for the "
                   "detail pass. Expect prompt adherence to fall away (the model "
                   "chose the shape unguided) while the harshness stays, since that is "
                   "the half of the schedule where it comes from. A control, in other "
                   "words: if these sound fine, the early/late story is wrong.",
        fmt="{:g}",
        turbo_ok=False,
    ),
    "timesignature": SweepAxis(
        field="timesignature",
        values=("4", "3", "6"),
        label="time signature metadata",
        listen_for="The last unexercised item in the structured-metadata block, "
                   "alongside bpm and keyscale. 4 is common time, 3 is a waltz, 6 is "
                   "compound. Count along. The interesting case is a caption whose "
                   "genre strongly implies 4/4 (a rock ballad, say) asked for 3: does "
                   "the metadata win, does the genre prior win, or do you get something "
                   "that lurches between them?",
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
