# ACE-Step 1.5 on Flyte: music you can compare in a report

Render music with [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) on the DGX
Spark and get **one Flyte report with tracks that play in the browser**, plus a
waveform, a spectrogram, and a copy-paste command to make any track again.

Two questions, two entry points:

| | question | command |
|---|---|---|
| `compare` | which **checkpoint**? | `flyte run compare_pipeline.py compare --suite quick` |
| `sweep` | what does this **knob** do? | `flyte run compare_pipeline.py sweep --axis seed` |

`compare` puts turbo, sft and base side by side on the same brief. `sweep` holds
everything fixed and moves exactly one parameter across a row you play left to right.
The second one is the reason this demo exists: a hosted music API gives you a seed
field and a vibe, not a controlled experiment.

**Measured on the Spark** (GB10, xl-turbo, 8 steps, `duration` sweep in one load):

| track length | render time | real-time factor |
|---|---|---|
| 20s | 3.9s | 5.2x |
| 60s | 4.9s | 12.2x |
| 120s | 7.7s | 15.7x |
| 180s | **12.8s** | 14.1x |

Three minutes of finished stereo music in under thirteen seconds, weights already
warm. Cost is roughly linear in length plus a fixed overhead, so longer tracks are
*more* efficient per second of audio. Peak GPU is 11.1GB regardless of length, which
is essentially just the weights.

### The number that makes the point

| | track | render | real-time factor | peak | per second of audio |
|---|---|---|---|---|---|
| ACE-Step xl-turbo | 180s | 12.8s | **14.1x faster than real time** | 11.1GB | 0.07s |
| MusicGen large | 30s | 56.6s | **0.5x** (slower than real time) | 6.9GB | 1.89s |

**~27x cheaper per second of audio**, and ACE-Step is the bigger model doing the harder
job, because it is also singing. That is guidance-distilled flow matching against
autoregressive token generation, and it is a better argument for why the field moved to
diffusion than any benchmark chart. MusicGen is from June 2023; ACE-Step 1.5 XL from
April 2026. Under three years.

---

## What ACE-Step 1.5 is

A text-to-music foundation model from the ACE Studio / StepFun team, released January
2026, with the XL line following in April. It takes a **style caption** ("driving
synthwave, analog bass, gated reverb drums") and **optional structured lyrics**, and
returns 48kHz stereo audio from 10 seconds to 10 minutes.

Three components, all loaded by one `from_pretrained`:

```
Qwen3-Embedding-0.6B  ->  encodes the caption + lyrics + metadata
AceStepTransformer1D  ->  flow-matching DiT, denoises in latent space
AutoencoderOobleck    ->  25Hz stereo latents  ->  48kHz stereo waveform
```

## The models

Seven models, four architectures, **one image**. Every one is a text-conditioned
generative audio model, and that is where the similarity ends: they disagree about what
"generating music" means, and the report is built to keep those differences visible
rather than flatten them into a ranking.

### Four architectures, four ideas about how to make audio

**Flow-matching DiT (ACE-Step 1.5).** A diffusion transformer that learns a
straight-line *velocity field* from noise to data rather than a noise-prediction
schedule. Straighter paths mean fewer steps, and distillation folds the guidance pass
into the weights on top. Audio lives as 25Hz stereo latents from an Oobleck VAE, and a
Qwen3 encoder conditions on caption, lyrics and structured metadata together. This is
the only model here that **sings**.

**Latent DiT (Stable Audio Open).** Also a transformer diffusion in latent space, but
noise-prediction rather than flow-matching, and conditioned by T5 text embeddings only.
No lyric path, no metadata channel. A stochastic (SDE) sampler draws its noise from a
Brownian tree, which is the source of its one integration headache below.

**Autoregressive over audio tokens (MusicGen).** Not diffusion at all. EnCodec
compresses audio into discrete tokens at 50/sec and a transformer predicts them one
frame at a time, like a language model over sound. That is why it has no "steps" knob:
its cost is *tokens*, so a 30s clip is 1500 sequential forward passes and cannot be
parallelised across time. It is also why it is by far the slowest thing here.

**Latent diffusion with a UNet (AudioLDM 2).** The oldest design in the set. Diffusion
over a mel-spectrogram latent, with a GPT-2 bridging CLAP audio-text embeddings and T5
text embeddings, decoded by a HiFi-GAN vocoder. Mel plus vocoder is what caps it at
16kHz mono.

### The registry

| key | family | adapter | output | fetched | licence |
|---|---|---|---|---|---|
| `xl-turbo` | ACE-Step 1.5 XL | `acestep` | 48kHz stereo, **sings**, to 10min | 11.1GB | MIT |
| `xl-sft` | ACE-Step 1.5 XL | `acestep` | as above | 11.5GB | MIT |
| `xl-base` | ACE-Step 1.5 XL | `acestep` | as above | 11.5GB | MIT |
| `stable-audio` | Stable Audio Open | `stableaudio` | 44.1kHz stereo, to 47s | 15.7GB | Stability Community (**gated**) |
| `musicgen-large` | MusicGen | `musicgen` | 32kHz stereo, to 30s | 6.9GB | CC-BY-NC |
| `musicgen-melody` | MusicGen | `musicgen` | 32kHz stereo, melody-conditionable | 6.2GB | CC-BY-NC |
| `audioldm2-music` | AudioLDM 2 | `audioldm2` | 16kHz **mono**, ~10s sweet spot | 4.5GB | CC-BY-NC-SA |

`adapter` selects the loader and generator in `music_core`. It does **not** select an
image: diffusers and transformers coexist happily, so all seven share one. Compare that
to the TTS demo next door, which needs seven images for seven models because every open
TTS package ships mutually hostile pins.

### Measured on the Spark

| model | track | render | real-time factor | peak |
|---|---|---|---|---|
| ACE-Step xl-turbo | 180s | 12.8s | **14.1x** | 11.1GB |
| ACE-Step xl-turbo | 30s | 1.5-4.2s | 7-20x | 11.1GB |
| Stable Audio Open | 30s | 19.5s | 1.5x | 2.7GB |
| AudioLDM 2 music | 10s | 6.4s | 1.6x | 2.2GB |
| MusicGen large | 30s | 55.3s | **0.5x** | 6.9GB |

ACE-Step is ~27x cheaper per second of audio than MusicGen while being the larger model
doing the harder job. That is autoregressive token generation versus guidance-distilled
flow matching, and it is a better argument for why the field moved than any chart.

Read the table as a **timeline**, not a leaderboard: MusicGen and AudioLDM 2 are 2023,
Stable Audio Open is mid-2024, ACE-Step 1.5 XL is April 2026. The older models are
context for how fast this moved, and comparing them on a vocal brief is unfair by
construction, which is what `intended_for` on every card is there to say.

### Which knobs each model actually has

`ADAPTER_KNOBS` in `music_core.py` is the single source of truth, used by the report
card, the reproduce command, and the studio's Advanced panel. Printing a dial a model
does not have is the same class of dishonesty as printing a value it silently ignored.

| adapter | steps | guidance | shift | bpm / key / lyrics |
|---|---|---|---|---|
| `acestep` | yes | yes | yes | yes |
| `stableaudio` | yes | yes | no | no |
| `audioldm2` | yes | yes | no | no |
| `musicgen` | no | yes | no | no |

So a MusicGen card reads `seed 42 · cfg 3 · 30s` and its reproduce command omits
`--steps` and `--shift`, because it is autoregressive and has neither. A binary
"is it diffusion?" test was not enough to get this right: AudioLDM 2 and Stable Audio
have steps but no flow-matching `shift`.

`max_duration` works the same way. MusicGen was trained on 30s windows and Stable Audio
tops out at 47s; neither refuses a longer request, they degrade. So the spec clamps and
the card shows the **clamped** number, making the limit legible as a constraint instead
of looking like a quality failure.

### What each one cost to integrate

Three of the four families needed a workaround. All are documented at their call sites
with the evidence that motivated them.

- **ACE-Step**: nothing. `diffusers>=0.39`, one `from_pretrained`, done.
- **MusicGen**: nothing. transformers already ships it.
- **AudioLDM 2**: a one-method shim. diffusers 0.39 drives its GPT-2 by hand and calls
  `_update_model_kwargs_for_generation`, which transformers dropped from
  `PreTrainedModel` in 4.53. Pinning the shared image back fifteen months to suit the
  oldest model in the registry was the wrong trade, so `_shim_audioldm2_generation`
  rebinds that one method onto that one instance.
- **Stable Audio Open**: two. `torchsde` is a required backend for its scheduler and is
  not mentioned on the model card; and its Brownian noise tree is built over
  `[sigma_min, sigma_max]` = `[0.3, 500]` while the schedule it generates runs
  `500.00006 … 0.3, 0.0`, so **both endpoints fall outside the tree's own domain** and
  torchsde recurses forever looking for them. `_widen_brownian_interval` widens the
  domain to `[0, sigma_max × 1.001]`. It does not touch the schedule, so sampling is
  unchanged; a wider domain does change the Brownian path for a given seed, so seeds
  will not match another implementation bit-for-bit.

That last one took five failed hypotheses (torchsde version, recursion limit, bf16,
fp16, fp32) before one instrumented run logging the sigma schedule against the interval
bounds gave the answer immediately. The lesson is in the Gotchas section and it is the
same one `faulthandler` taught: **instrument before theorising.**

### Adding another

Write a `_load_<adapter>` and `_generate_<adapter>` in `music_core`, add an
`ADAPTER_KNOBS` entry, and add a `MusicModelSpec` with `adapter`, `intended_for` and
`max_duration`. No image change if it loads through diffusers or transformers.

**DiffRhythm** (`ASLP-lab/DiffRhythm2`, 5.07GB, Apache-2.0) is the obvious next one and
the only true ACE-Step rival found so far: full-length **lyrics-to-song with vocals**,
also flow-matching, from November 2025. It would be the first model here to need its own
image, since it ships a custom package rather than loading through either library.

---

## Parameters

Everything below is a real knob on `AceStepPipeline`. The **CLI** column is the flag on
`generate_one`; a dash means the pipeline supports it but this demo has not wired it to
an entry point yet.

### Content

| parameter | CLI | default | what it does |
|---|---|---|---|
| `prompt` | `--prompt` | from brief | The **style caption**: genre, mood, instrumentation, production. Not a description of the song's story. Naming specific instruments and a production style ("tape-saturated", "close-mic'd") works far better than adjectives alone. |
| `lyrics` | `--lyrics` | from brief | Sung text with structure tags. **Empty string means instrumental.** Do not write "instrumental" here; the model will sing the word. |
| `vocal_language` | `--language` | `en` | Language code for the lyric header (`en`, `pt`, `zh`, `ja`, ...). 50+ supported. Set it to match your lyric or the phrasing drifts toward English. |

**Lyric structure tags are conditioning, not decoration.** `[intro]`, `[verse]`,
`[chorus]`, `[bridge]`, `[outro]`. Drop them and you tend to get one undifferentiated
verse.

**Budget roughly 4 seconds of track per sung line, plus intro and outro.** The model
paces the whole lyric to fit `audio_duration` rather than truncating the end, so an
over-long lyric in a short render gets compressed and dropped throughout. The 22-line
`synthwave-vocal` brief needs ~120-180s; at 60s it audibly skips.

### Sampling

| parameter | CLI | default | what it does |
|---|---|---|---|
| `audio_duration` | `--duration` | 30.0 | Track length in seconds (10 to 600). **Not a crop**: it is fed to the model up front and changes the composition. A 20s render states the idea immediately; a 180s render has room for an intro, a build and a turnaround. |
| `seed` | `--seed` | 42 | The sampler's starting noise. Everything else fixed, a new seed is a new take. How *much* the arrangement moves tells you how underspecified your prompt is. |
| `num_inference_steps` | `--steps` | per checkpoint | Denoising steps. Quality vs latency. Flow matching degrades gracefully: low-step output sounds *smeared* rather than broken, with transients blurring first and the stereo image narrowing. |
| `guidance_scale` | `--guidance` | per checkpoint | CFG. How hard the model is pushed toward the prompt. Low drifts and sounds generic; high over-obeys and gets harsh and brittle. **Inert on turbo** (coerced to 1.0). |
| `shift` | `--shift` | 3.0 | Warps where the flow-matching schedule spends its steps. High front-loads the noisy end, where global structure (form, arrangement, groove) is decided; low spends more budget on the clean end, which is detail and texture. 3.0 is the shipped recommendation. |
| `cfg_interval_start/end` | - | 0.0 / 1.0 | Restrict CFG to a slice of the schedule. Guidance early only, or late only. |
| `timesteps` | - | none | A fully custom schedule, overriding steps and shift. |

### Musical metadata

These go into a `# Metas` block the text encoder was trained on, **not** into your
caption. That makes them a real control surface rather than a prompt-wording trick.
Leave any of them unset and the model estimates it.

| parameter | CLI | what it does |
|---|---|---|
| `bpm` | `--bpm` | Target tempo. Tap along to check how obedient it actually is; a ballad caption at 160 is where it gets interesting. |
| `keyscale` | `--keyscale` | `"C major"`, `"A minor"`, `"D dorian"`. Major/minor is unmissable; a modal scale is where a model that memorized "minor = sad" falls apart. |
| `timesignature` | - | `"4"` for 4/4, `"3"` for 3/4. |

The assembled prompt looks like this:

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

### Audio-to-audio (supported by the pipeline, not yet wired here)

This is the "that take was almost right" surface, and it is the obvious next thing to
build on top of the reports.

| parameter | what it does |
|---|---|
| `task_type` | `text2music` (default), `repaint`, `cover`, `extract`, `lego`, `complete`. |
| `src_audio` | Source audio as a `[channels, samples]` 48kHz tensor, for every task above except plain generation. |
| `repainting_start` / `_end` | Regenerate only seconds X to Y and keep the rest. |
| `reference_audio` | Timbre conditioning: keep your arrangement, borrow another track's voice or tone. |
| `audio_cover_strength` | 0.0-1.0. How far to move from the source. Lower blends more of the original. |
| `track_name` | For `extract` / `lego`: which stem (`vocals`, `drums`, ...). |
| `audio_codes` | 5Hz semantic codes; switches to `cover` automatically. Needs the tokenizer pair, which the **sft** and **base** repos ship and turbo does not. |

---

## The sweep axes

`flyte run compare_pipeline.py sweep --axis <name>`. All values run in a **single GPU
task against one loaded pipeline**, so another column costs one render, not another
11GB load.

| axis | default values | what it moves |
|---|---|---|
| `seed` | 11, 42, 1234, 90210 | Nothing but the starting noise. |
| `steps` | 4, 8, 16, 32 | Quality vs latency. Find the knee; on turbo it is usually at or below 8. |
| `guidance` | 1, 3, 7, 12 | Prompt adherence. **Inert on `xl-turbo`** by design. |
| `shift` | 1, 2, 3 | Where the schedule spends its budget: structure vs detail. |
| `bpm` | 75, 100, 128, 160 | Tempo obedience. |
| `keyscale` | C major, A minor, F# minor, D dorian | Harmony, including a modal curveball. |
| `duration` | 20s, 40s, 80s | Composition, not crop. |

Override the values for any axis:

```bash
flyte run compare_pipeline.py sweep --axis steps --values '["2","4","8","24","50"]'
```

Values arrive as strings because a CLI list cannot be heterogeneously typed, and are
cast back to `int`/`float`/`str` from the matching `GenSettings` field. There is a test
asserting every preset round-trips its own type.

---

## The briefs

Seven, in `prompts.py`, each aimed at a different failure mode. Named suites: `full`,
`quick`, `instrumental`, `vocal`, `vocal-ab`.

| key | what it tests |
|---|---|
| `synthwave` | The control. Dense material is forgiving, so passing means little and failing means everything. |
| `synthwave-vocal` | The same caption plus a female lead and an emotional robot lyric. Direct A/B for what adding a singer does to the arrangement. |
| `acoustic-duo` | Exposure. Two instruments and silence between the notes: every artifact is naked. |
| `indie-vocal` | Intelligibility and structure. Can you make out the words, and does `[chorus]` arrive as a chorus? |
| `odd-instruments` | Prompt adherence. 7/8, three unusual acoustic instruments by name, one explicit exclusion. Pair with `--axis guidance`. |
| `bossa-pt` | The 50+ languages claim, in Portuguese. |
| `arc` | Structure over time. At 20s a loop; at 180s it has to make an argument. Pair with `--axis duration`. |

---

## The report

Every card is one rendered track: waveform envelope, spectrogram to the full 24kHz
Nyquist, an inline player, the resolved settings, and a **reproduce** fold.

**Audio plays with no JavaScript.** Flyte reports render under a CSP that drops
external assets and `<script>` tags. HTML5 `<audio controls>` with a base64 data URI
needs neither.

**Vorbis, not PCM.** A 60s stereo 48kHz track is ~0.23MB as OGG and 11.5MB as PCM16
wav, and base64 adds a third on top. Embedding a grid of full-length tracks is only
possible because of that.

**The spectrogram runs to the full 24kHz Nyquist**, not the 8kHz voice band the TTS
demo used, because a hard horizontal edge at 16kHz is the tell that a model renders
through a lossy bottleneck. Cropping would hide the most diagnostic thing on the plot.

**Every card carries the command that made it:**

```
flyte run compare_pipeline.py \
    generate_one \
    --model_key xl-turbo \
    --brief synthwave-vocal \
    --duration 180 --seed 42 --steps 8 --guidance 1 --shift 3
```

The settings shown are the **resolved** ones, so the command reproduces what actually
ran rather than what was requested. Ask turbo for `--guidance 7` and the handle says
`--guidance 1`, because that is what the model did. A handle that echoed the request
would quietly lie on exactly the checkpoint you most want to iterate on. Named briefs
stay named rather than inlining a 22-line lyric; the full prompt and lyrics are in the
JSON block underneath. There is no copy button because the CSP would kill it.

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
| `prompts.py` | The seven briefs. |
| `music_core.py` | The engine room: load, generate, embed, render. **Flyte-free on purpose**, so the same code runs in the GPU task and in `run_local.py`. |
| `compare_pipeline.py` | The Flyte tasks and the three entry points. |
| `run_local.py` | Host-GPU smoke test, no cluster. |

**One image, unlike the TTS demo next door**, which needs seven because every open TTS
package ships mutually hostile pins. ACE-Step 1.5 was merged into `diffusers`
([PR #13095](https://github.com/huggingface/diffusers/pull/13095), shipped in 0.39.0),
so every checkpoint loads through one `from_pretrained` on one stack. Adding the next
checkpoint is a registry entry, not a Dockerfile.

**Jobs are batched per checkpoint.** Loading 11GB off disk takes far longer than
rendering a 30s track at 8 steps, so `render` takes a list of jobs and runs them all
against one loaded pipeline. The report is replaced after every track so a long run is
watchable.

**The orchestrator is CPU-only on purpose.** It stays alive holding its resources while
awaiting children, so if it held the box's one GPU its own GPU children would deadlock.

**Multi-GPU needs no code change.** `compare` already fans out with `asyncio.gather`;
on one GPU the scheduler serializes, on four it does not.

---

## Running it

```bash
cd topics/music-generation/ace-step/acestep-flyte

flyte run compare_pipeline.py generate_one --model_key xl-turbo --duration 20
flyte run compare_pipeline.py compare --suite quick
flyte run compare_pipeline.py sweep --axis seed
flyte run compare_pipeline.py compare --suite vocal-ab --models '["xl-turbo"]' --duration 120
```

Weights are cached by `fetch_weights` with `flyte.Cache`, so the 11GB pull happens once
per checkpoint. Bump `_WEIGHTS_CACHE_VERSION` when you change *what* gets downloaded.

`run_local.py` drives the host GPU directly and writes standalone HTML with the
identical renderer:

```bash
python run_local.py --model xl-turbo --brief synthwave
python run_local.py --sweep steps --values 4,8,16,32
```

If the Flyte report comes up **blank**, forward port 30002 (rustfs) so the presigned
URLs resolve. That is not a code bug.

---

## Gotchas, all of them found the hard way on this box

**`sf.write()` of long audio to OGG/Vorbis SEGFAULTS the process.** libsndfile 1.2.2:
20s and 30s encode fine, 60s and 90s crash. Not an exception, a SIGSEGV with exit 139
and no Python traceback. It is a function of the size of the single write, not the
destination: it crashes to a `BytesIO` and to a real file alike, mono and stereo alike.
Writing the same audio through `sf.SoundFile` in ~5 second slices produces the same
file and never crashes, which is what `music_core.encode_audio` does.

This one cost four cluster runs and two wrong theories, because the crash lands *after*
a successful render while building the report, so the last log line is always
`loaded; peak 11.1GB` and it looks like the model died. The failures tracked audio
LENGTH, which sent us hunting length-dependent behaviour in the model. There is a
length-dependent thing in there: `AutoencoderOobleck` switches to tiled decode above
512 latent frames, which at 25 frames/sec is 20.5 seconds, almost exactly where our
successes stopped. That coincidence was convincing and completely wrong.

**Arm `faulthandler`.** `faulthandler.enable()` in `music_core` turns a silent exit 139
into a Python traceback naming the exact frame. Two lines of stack ended the hunt above
in minutes after three crashes had produced nothing at all. On a box where CUDA
allocations are host pages, native crashes are not exotic.

**`diffusers>=0.39.0` is a hard floor.** `AceStepPipeline` did not exist before it, and
the failure is a bare `ImportError` deep in a GPU pod minutes into a run.

**`transformers>=4.51` is a softer, nastier floor.** The text encoder is a `Qwen3Model`;
older transformers gives a `KeyError` inside `from_pretrained`, so the image builds fine
and the task dies in the pod. (5.14.1 works.)

**`hf_transfer` is dead lore.** Current huggingface_hub routes through **Xet** and
ignores it; setting `HF_HUB_ENABLE_HF_TRANSFER` only earns a FutureWarning, and
`HF_HUB_DOWNLOAD_TIMEOUT` does not bound the Xet path either. The real failure is hard
and retryable:
`CAS Client Error: Request middleware error: error sending request for url (...)`,
seen on this pipeline's very first run. Protection lives in `retries=3`, not env vars.

**Fetch with `local_dir=`, not `cache_dir=`.** The first version produced an HF hub
cache and pointed `HF_HUB_CACHE` at the downloaded Dir. Two things went wrong: the Dir
upload dereferenced the cache's symlinks so every shard was stored **twice** (11GB
became ~22GB in the blob store), and the GPU task re-downloaded all 11GB from
HuggingFace anyway, costing 2.5 minutes on every run. A plain repo layout handed to
`from_pretrained` as a path skips the cache machinery entirely.

**`torch.cuda.mem_get_info()` is unusable on GB10, in two different ways.** It
**raises** `AcceleratorError: CUDA error: out of memory` from `cudaMemGetInfo` on a
healthy box with 100GB+ free; a bare `except: pass` around it then skips the memory cap
entirely and logs a cheerful `pool 0/0GB, capped at 0GB` that reads like a formatting
quirk rather than "this process is now uncapped". And when it *does* return, its `free`
counts reclaimable page cache as used (3GB of 129GB, while MemAvailable said 108GB).
Use `MemAvailable` for free and `get_device_properties(0).total_memory` as the fallback
for total, and log loudly when the cap cannot be applied.

**Cap GPU memory against the cgroup, not the host.** On GB10 the GPU pool *is* host
memory, so CUDA allocations are charged to the pod's cgroup: the box can have 100GB
free while the pod is capped at 48Gi. Also do **not** use `torch.cuda.mem_get_info()`
for this. It reported "3GB free of 129GB" here because reclaimable page cache counts as
used, which capped a process at 2.55GB and OOMed the model load instantly. Read
`MemAvailable` from `/proc/meminfo` and floor the result.

**Reclaim leaked memory before a big run.** `kubectl rollout restart deploy/rustfs -n
flyte` took this box from 90GB to 110GB available; rustfs had 19GB of leaked heap. On
unified memory that is 19GB the renderer cannot have.

**VAE tiling stays off**, matching the diffusers default. Forcing it on was our change
and it bought nothing on a box with 90GB of headroom. Note the model card's
`pipe.vae.enable_tiling()` does not exist on `AutoencoderOobleck` in 0.39; only the
`use_tiling` attribute does.

**Turbo silently ignores guidance.** Correct behaviour for a distilled checkpoint, but
it means a naive report prints "cfg 7.0" under a track that ran at 1.0.
`GenSettings.resolve` mirrors the coercion so cards and repro handles stay honest.

---

## Licensing

ACE-Step weights are **MIT**. The bundled Qwen3-Embedding-0.6B text encoder is
**Apache-2.0**. Both commercial-safe, which is not something you can say about every
model here: MusicGen ships under CC BY-NC and its *outputs* are non-commercial. Worth a
minute on the stream.

---

## Next

**A `refine` entry point** wrapping `repaint` / `cover`. Since `ModelRun` already
returns a `flyte.io.Dir` of wavs, a follow-up run can take a previous run's Dir as
`src_audio` and Flyte records the lineage for free. That turns the report from a
leaderboard into a working surface: pick the take that was almost right, regenerate
seconds 40-55, keep the rest.

**A Gradio studio.** `config.py` reserves the name, port 7865, and a GPU pod template.
A thin CPU launcher that submits runs and links the report, holding no GPU and loading
no model, plus a picker over past runs' tracks to use as reference audio.

**Other models.** ACE-Step first because it is the best open option right now, but the
topic is wider: YuE for long-form lyrics-to-song, MusicGen for melody conditioning,
Stable Audio Open for SFX and texture, and Magenta RT2 for live steerable streaming,
which already has [its own demo](../../magenta/magenta-rt-flyte/). Adding one is a
`models.py` entry if it loads through diffusers, and a second image if it does not.
