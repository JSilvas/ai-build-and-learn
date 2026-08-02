# ACE-Step 1.5 on Flyte: music you can compare in a report

Render music with [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) on the DGX
Spark and get **one Flyte report with tracks that play in the browser**, plus a
waveform and spectrogram for each one.

Two questions, two entry points:

| | question | command |
|---|---|---|
| `compare` | which **checkpoint**? | `flyte run compare_pipeline.py compare --suite quick` |
| `sweep` | what does this **knob** do? | `flyte run compare_pipeline.py sweep --axis seed` |

`compare` puts turbo, sft and base side by side on the same brief. `sweep` holds
everything fixed and moves exactly one parameter across a row you play left to right.
The second one is the reason this demo exists: a hosted music API gives you a seed
field and a vibe, not a controlled experiment.

> **Status.** Written and unit-tested against the real `AceStepPipeline` API (diffusers
> 0.39.0), renderer verified end to end with synthetic audio. **Not yet run on the
> Spark**, so there are no measured timings in this document. Where a number would
> normally go you will find an expectation, labelled as one. First run on the box is
> the next step; see [First run](#first-run).

---

## What ACE-Step 1.5 is

A text-to-music foundation model from the ACE Studio / StepFun team, released January
2026, with the XL line (4B-class DiT) following in April. It takes a **style caption**
("driving synthwave, analog bass, gated reverb drums") and **optional structured
lyrics**, and returns 48kHz stereo audio from 10 seconds to 10 minutes.

Three components, all loaded by one `from_pretrained`:

```
Qwen3-Embedding-0.6B  ->  encodes the caption + lyrics + metadata
AceStepTransformer1D  ->  flow-matching DiT, denoises in latent space (the 5B one here)
AutoencoderOobleck    ->  25Hz stereo latents  ->  48kHz stereo waveform
```

The thing that makes it interesting for a demo: it is **fast**. Flow matching plus a
guidance-distilled turbo checkpoint means a full song in single-digit seconds on a
datacenter GPU, which is what turns "generate a track" into "iterate on a track".

### The checkpoint family

Only the XL line is reachable from here, and only its `-diffusers` repos. ACE-Step
publishes each checkpoint twice: a native repo for the upstream `acestep` package, and
a converted one in the standard diffusers pipeline layout. Only the latter loads with
`AceStepPipeline.from_pretrained`, and so far only XL has been converted.

| key | repo | recipe | why it is here |
|---|---|---|---|
| `xl-turbo` | `ACE-Step/acestep-v15-xl-turbo-diffusers` | 8 steps, no CFG | Guidance-distilled. The iteration checkpoint. |
| `xl-sft` | `ACE-Step/acestep-v15-xl-sft-diffusers` | 50 steps, CFG 7.0 | Instruction-tuned. The better listener. |
| `xl-base` | `ACE-Step/acestep-v15-xl-base-diffusers` | 50 steps, CFG 7.0 | The pretrained model under both. |

Each is ~11GB of bf16 weights. `xl-turbo` and `xl-sft` are the default `compare` pair;
`xl-base` is opt-in because a third 11GB pull and a third 50-step pass is a lot to
spend on a question most runs are not asking.

**Guidance distillation is the whole story of turbo.** CFG normally costs you two
forward passes per step (conditional and unconditional). Turbo has that baked into the
weights, so it runs one pass per step, and 8 steps instead of 50. Call it a ~12x
compute cut. Whether you can hear the 12x is exactly what `compare` is for, and the
pipeline enforces it: pass `guidance_scale > 1.0` to a turbo checkpoint and it warns
and coerces to 1.0. We do the same coercion in `GenSettings.resolve` so the report
card states what actually ran, not what was asked for.

---

## First run

Runs happen **on the devbox**, not in local Python. From the Spark:

```bash
cd topics/music-generation/ace-step/acestep-flyte

# smoke test: one checkpoint, one track, ~11GB download the first time
flyte run compare_pipeline.py generate_one --model_key xl-turbo --duration 20

# the headline: turbo vs sft, 3 briefs each
flyte run compare_pipeline.py compare --suite quick

# the sweep
flyte run compare_pipeline.py sweep --axis seed
```

Weights are cached by `fetch_weights` with `flyte.Cache`, so the 11GB pull happens once
per checkpoint, ever. Bump `_WEIGHTS_CACHE_VERSION` in `compare_pipeline.py` when you
change *what* gets downloaded.

Want to skip the cluster entirely while you are debugging? `run_local.py` drives the
host GPU directly and writes a standalone HTML file using the identical renderer:

```bash
python run_local.py --model xl-turbo --brief synthwave
python run_local.py --sweep steps --values 4,8,16,32
```

If the Flyte report comes up **blank**, that is not a code bug: forward port 30002
(rustfs) so the presigned URLs resolve.

---

## The report

Every card is one rendered track:

```
┌──────────────────────────────────┐
│  [ waveform envelope           ] │  <- the arrangement as a shape
│  [ spectrogram to 24kHz        ] │  <- where a lowpass brickwall shows up
├──────────────────────────────────┤
│  ▶ ━━━━━━━━━━━━━━━━━━  0:30      │  <- plays inline, no JS needed
├──────────────────────────────────┤
│  seed 42 · 8 steps · cfg 1 ·     │  <- RESOLVED settings, what actually ran
│  shift 3 · 30s · 118 bpm         │
│  4.1s to render · 30.0s stereo   │
│  @ 48kHz · peak 12.3GB           │
└──────────────────────────────────┘
```

Three deliberate choices in there:

**Audio plays with no JavaScript.** Flyte reports render under a CSP that drops
external assets and `<script>` tags. HTML5 `<audio controls>` with a base64 data URI
needs neither, so the tracks just work. Same trick the video demo used for its clips.

**Vorbis, not PCM.** A 60s stereo 48kHz track is ~1MB as OGG and ~11.5MB as PCM16 wav,
and base64 adds a third on top. Embedding a grid of full-length tracks is only possible
because of that 10x. `audio_data_uri` walks a ladder (stereo OGG, mono OGG, stereo wav)
and each rung says in the card what it gave up, so "why is this one mono?" is never a
mystery.

**The spectrogram runs to the full 24kHz Nyquist**, not the 8kHz voice band the TTS
demo used. The top of that range is exactly where a lowpass brickwall shows up, and a
hard horizontal edge at 16kHz is the tell that a model is rendering through a lossy
bottleneck. Cropping would hide the most diagnostic thing on the plot. The waveform
above it is drawn as a filled envelope so an intro/build/drop arc reads as shape rather
than a block of ink, with the clipping line marked in red.

Every block also prints **what to listen for** before the audio. A grid of clips with
no stated hypothesis is just noise; the point is to say what should change *before* you
press play.

---

## The sweep axes

`flyte run compare_pipeline.py sweep --axis <name>`. All of them run in a **single GPU
task against one loaded pipeline**, so the marginal cost of another column is one
render, not another 11GB load.

| axis | default values | what it moves |
|---|---|---|
| `seed` | 11, 42, 1234, 90210 | Nothing but the starting noise. How much the arrangement moves tells you how underspecified the prompt is. |
| `steps` | 4, 8, 16, 32 | Quality vs latency. Flow matching degrades gracefully, so low-step clips sound *smeared*, not broken: transients blur first, stereo narrows. Find the knee. |
| `guidance` | 1, 3, 7, 12 | Prompt adherence. Low drifts, high over-obeys and gets brittle. **Inert on `xl-turbo`** by design. |
| `shift` | 1, 2, 3 | The ACE-Step-specific one. Warps where the flow schedule spends its steps: high front-loads global structure, low spends more on detail. |
| `bpm` | 75, 100, 128, 160 | Structured metadata, not prompt words. Tap along and find out how obedient it actually is. |
| `keyscale` | C major, A minor, F# minor, D dorian | Same channel, aimed at harmony. Dorian is the one that breaks a model that memorized "minor = sad". |
| `duration` | 20s, 40s, 80s | Not a crop: length is fed in up front and changes the composition. Does the long one use the room, or loop? |

Override the values for any axis:

```bash
flyte run compare_pipeline.py sweep --axis steps --values '["2","4","8","24","50"]'
```

(Values come in as strings because a CLI list cannot be heterogeneously typed, and are
cast back to `int`/`float`/`str` from the matching `GenSettings` field. There is a test
asserting every preset round-trips its own type.)

### bpm and keyscale are a real control surface

Worth calling out because it is easy to miss. ACE-Step does not want tempo and key as
words in your caption. It takes them as **structured metadata** that the pipeline
formats into a `# Metas` block the text encoder was trained on:

```
# Instruction
Fill the audio semantic mask based on the given conditions:

# Caption
driving synthwave instrumental, analog saturated bass, ...

# Metas
- bpm: 118
- timesignature: N/A
- keyscale: N/A
- duration: 30 seconds
```

Leave any of them unset and the model estimates it. That makes `--axis bpm` a genuine
controllability test rather than a prompt-wording test.

---

## The briefs

Six, in `prompts.py`, each aimed at a different failure mode. Named suites: `full`,
`quick` (the first three), `instrumental`, `vocal`.

| key | what it tests |
|---|---|
| `synthwave` | The control. Dense material is forgiving, so passing means little and failing means everything. |
| `acoustic-duo` | Exposure. Two instruments and silence between the notes: every artifact is naked. Smeared plucks and a sine-wave "upright bass" show up here first. |
| `indie-vocal` | Intelligibility and structure. Can you make out the words without reading along, and does `[chorus]` actually arrive as a chorus? |
| `odd-instruments` | Prompt adherence. 7/8, three unusual acoustic instruments by name, one explicit exclusion. Pair with `--axis guidance`. |
| `bossa-pt` | The 50+ languages claim, in Portuguese. Do the nasals survive, and does the phrasing sit behind the beat? |
| `arc` | Structure over time. At 20s it is a loop; at 80s it has to make an argument. Pair with `--axis duration`. |

Lyrics use ACE-Step's structure tags (`[verse]`, `[chorus]`, `[bridge]`, `[intro]`,
`[outro]`). They are conditioning, not decoration: drop them and you tend to get one
undifferentiated verse. An **empty** lyrics string is the right way to ask for an
instrumental. Do not write "instrumental" in the lyrics field; the model will sing it.

---

## How it is wired

```
compare ─┬─ fetch_weights(xl-turbo)  ·· CPU pod, cached forever
         ├─ fetch_weights(xl-sft)    ·· CPU pod, cached forever
         └─ render(model, weights, [job, job, job])  ·· GPU pod, one load, N renders
```

| file | what it holds |
|---|---|
| `config.py` | Images and `TaskEnvironment`s. Spark-pinned (arm64, cu130, local registry). |
| `models.py` | The checkpoint registry and the sweep-axis registry. |
| `prompts.py` | The six briefs. |
| `music_core.py` | The engine room: load, generate, embed, render. **Flyte-free on purpose**, so the same code runs in the GPU task and in `run_local.py`. If a track plays in the local HTML it plays in the Flyte report, because it is the identical renderer. |
| `compare_pipeline.py` | The Flyte tasks and the three entry points. |
| `run_local.py` | Host-GPU smoke test, no cluster. |

### One image, and why that is worth noticing

The text-to-speech demo next door needs **seven** images, one per model, because every
open TTS package ships its own mutually hostile pins (`transformers==4.57.3` vs
`transformers==5.2.0` vs `transformers==4.46.1`, and a `torch==2.6.0` pin with no cu130
arm64 wheel). That is a real tax and it shows up as a `config.py` three times this size.

Music generation, at least for ACE-Step, is back to the video demo's happy case:
ACE-Step 1.5 was merged into `diffusers` ([PR #13095](https://github.com/huggingface/diffusers/pull/13095),
shipped in 0.39.0), so every checkpoint loads through one `from_pretrained` on one
stack. **One image, one GPU env**, and adding the next checkpoint is a registry entry
rather than a new Dockerfile. That is the difference between an ecosystem that
converged on a runtime and one where every model is its own science project.

### Jobs are batched per checkpoint on purpose

Loading 11GB off disk takes far longer than rendering a 30s track at 8 steps. So
`render` takes a **list** of jobs and runs them all against one loaded pipeline. A
four-value sweep is one model load and four renders, not four of each. The report is
replaced after every track so a long run is watchable: play track one while track four
is still denoising.

The orchestrator is **CPU-only on purpose**. An orchestrator pod stays alive holding its
resources while awaiting children, so if it held the box's one GPU its own GPU children
would deadlock forever waiting for it.

---

## DGX Spark notes

GB10 Blackwell, arm64, cu130, 119.7GiB unified memory.

**The unified pool is the thing to respect.** "GPU memory" here is the same memory the
OS is using, and running it to the wall does not OOM, it **hangs the box**. ACE-Step XL
is ~11GB in bf16 and nowhere near that limit, but `from_pretrained(...).to("cuda")`
briefly holds two copies and the VAE decode of a long track allocates one very large
contiguous activation on top. `prepare_gpu()` caps the process at 0.90 of the pool
before any load, which costs nothing and removes the failure mode.

**VAE tiling is on.** A 4-minute stereo 48kHz track is a ~46M-sample activation plus
intermediates if you decode it in one shot. `load_pipeline` enables tiling, using
`enable_tiling()` when the installed diffusers has it and setting `vae.use_tiling`
directly when it does not (the attribute is stable across releases, the method is not).

**Expected timings, not measured ones.** The model card quotes under 2s per full song
on an A100 and under 10s on a 3090 for the 2B line. The Spark is memory-bandwidth-bound
and this is the 5B XL DiT, so expect slower than a 3090 on the same work, and expect
`xl-sft` at 50 steps with CFG to cost roughly 12x an 8-step turbo render of the same
length. The report prints real wall-clock and a real-time factor under every card, so
the first run on the box replaces this paragraph with numbers.

---

## Gotchas worth knowing before you hit them

**`diffusers>=0.39.0` is a hard floor.** `AceStepPipeline` did not exist before it, and
the failure mode is a bare `ImportError` deep inside a GPU pod several minutes into a
run. `config.py` floors it explicitly.

**`transformers>=4.51` is a softer but nastier floor.** The pipeline's text encoder is a
`Qwen3Model`; on older transformers you get a `KeyError` inside `from_pretrained`, not
an import error, so the image builds fine and the task dies in the pod.

**Turbo silently ignores your guidance scale.** The pipeline warns and coerces to 1.0.
That is correct behavior for a distilled checkpoint, but it means a naive report would
print "cfg 7.0" under a track that ran at 1.0. `GenSettings.resolve` mirrors the
coercion so the card is honest, and `sweep --axis guidance --model_key xl-turbo` says
up front that the row will be four identical clips and points you at `xl-sft`.

**The report holds base64 audio, the task boundary does not.** Task results carry
metadata plus a `flyte.io.Dir` of wavs; the parent re-reads the wavs and rebuilds the
embedded players. Passing data URIs across the boundary would bloat every result for no
reason. Same pattern as the TTS and video demos.

**A checkpoint that fails every job still returns a non-empty-looking `Dir`.**
Downloading a genuinely empty one raises `DownloadQueueEmpty` in the parent, which would
take the whole comparison down with it. `_to_results` checks there is something to
download first, so one dead checkpoint becomes an error column instead of a dead run.

---

## Licensing

ACE-Step weights are **MIT**. The bundled Qwen3-Embedding-0.6B text encoder is
**Apache-2.0**, redistributed per Qwen's license. Both are commercial-safe, which is not
something you can say about every model in this space: MusicGen ships under CC BY-NC and
its *outputs* are non-commercial.

That matters more for music than for most modalities, and it is worth a minute on the
stream. Check the license before you put a generated track behind anything.

---

## Not here yet

**A Gradio studio.** The image, video and TTS demos each have one; `config.py` already
reserves the name, port 7865, and the GPU pod template. When it lands it will be a thin
CPU launcher that submits runs and links the report, holding no GPU and loading no
model, like the imagegen and videogen studios rather than the resident-model voice app.

**Everything the diffusers path does not carry.** Taking the diffusers route buys one
image and a shared mental model with the image and video demos. What it costs is the
features the upstream `acestep` package has that diffusers has not absorbed: the 5Hz LM
stage, Flow-Edit prompt-guided editing, Retake variations, LoRA training from a handful
of songs, and the 2B checkpoints. `AceStepPipeline` *does* expose `repaint`, `cover`,
`extract`, `lego` and `complete` task types plus `src_audio` / `reference_audio`
conditioning, so audio-to-audio work is reachable from here; it is just not wired into
an entry point yet. Repaint (regenerate seconds 10 through 20 of a track, keep the rest)
is the obvious next one.

**Other models.** ACE-Step first because it is the best open option right now, but the
topic is wider: YuE for long-form lyrics-to-song, MusicGen for melody conditioning,
Stable Audio Open for SFX and texture, and Magenta RT2 for live steerable streaming,
which already has [its own demo](../../magenta/magenta-rt-flyte/) in this topic. Adding
one here is a `models.py` entry if it loads through diffusers, and a second image if it
does not.
