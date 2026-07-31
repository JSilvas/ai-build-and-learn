# Open-source TTS, side by side (on Flyte + DGX Spark)

Read the same script through several state-of-the-art open TTS models and get back a
single Flyte report where every clip **plays inline**, next to a waveform and
spectrogram, with speed (real-time factor) on each card. The point is to *hear* them
against each other, on the same words, in one place.

Two pipelines share the machinery:

- **`compare_pipeline.py`** — seven models read the same scripts in their own built-in
  voices. Judged by ear.
- **`clone_pipeline.py`** — five models clone one reference recording, and every clip is
  **scored** for speaker similarity and word error rate. Judged by number *and* ear.

This is the audio sibling of the `video-generation/videogen-flyte` demo and shares its
shape: cached HF downloads on CPU pods, GPU synthesis tasks, one aggregated report.

## The models

Seven models, all commercial-safe (Apache-2.0 / MIT). Same text, each in its own
built-in voice(s).

| key | model | params | license | why it's here |
|---|---|---|---|---|
| `qwen3-1.7b` | Qwen3-TTS 1.7B | 1.7B | Apache-2.0 | Alibaba's Jan-2026 flagship: expressive, multilingual. The one to beat. |
| `qwen3-0.6b` | Qwen3-TTS 0.6B | 0.6B | Apache-2.0 | Same family, a third the size. The size/speed contrast. |
| `chatterbox` | Chatterbox | 0.5B | MIT | Beat ElevenLabs in a blind test. The natural-speech bar. |
| `kokoro-82m` | Kokoro-82M | 82M | Apache-2.0 | Tiny, CPU-capable, 54 fixed voices. The efficiency floor. |
| `dia-1.6b` | Dia 1.6B | 1.6B | Apache-2.0 | Best open **dialogue** voice; shines on the two-speaker script. |
| `parler-mini` | Parler-TTS mini v1 | 880M | Apache-2.0 | Voice controlled by a text **description**. A different UX entirely. |
| `csm-1b` | Sesame CSM-1B | 1B | Apache-2.0 | Very natural conversational voice. **Gated** (see below). |

`csm-1b` is a **gated** repo: accept the license once at
[hf.co/sesame/csm-1b](https://huggingface.co/sesame/csm-1b) with the account tied to
your `HF_TOKEN`, or its fetch 403s (the run degrades to an error column, it doesn't crash).

**`voxtral-4b` (Mistral) is built but deferred** — it's the one *served* model (a
vLLM-omni HTTP server the task starts, not a `from_pretrained` load; CC BY-NC 4.0,
non-commercial). The image builds and the adapter works, but its server cold-starts past
vLLM-omni's hardcoded 600s orchestrator timeout on an ephemeral GB10 pod, so per-task
boot fails. It's left fully wired (`--models '["voxtral-4b"]'` to iterate); the fix is a
persistent warm server or a launcher that raises the timeout. Not in the default set.

### Male / female voices

Models with named built-in voices render M/F as separate columns. `--voices` picks:
`all` (every variant), `female`, `male`, or `default` (one column per model).

| model | female | male |
|---|---|---|
| `kokoro-82m` | af_heart | am_michael |
| `qwen3-1.7b` | Vivian | Ryan |
| `qwen3-0.6b` | Serena | Eric |
| `parler-mini` | "Laura's voice…" | "Gary's voice…" |

Chatterbox, Dia and CSM are zero-shot / random-speaker models: they have no named
built-in voices, so controllable M/F needs a reference clip. That's the **voice cloning**
pipeline, which is the other half of this demo: see below.

## The one thing that's different from the video demo: per-model images

The video models all loaded through `diffusers`, so they shared one image. TTS models
do not: each ships its own package with mutually hostile pins.

```
qwen-tts    pins transformers==4.57.3, accelerate==1.12.0
chatterbox  pins transformers==5.2.0, torch==2.6.0, diffusers==0.29.0, numpy<2
kokoro      wants misaki + espeak-ng
dia + csm   via transformers (Dia/CsmForConditionalGeneration) — SHARE one image
parler-tts  pins transformers==4.46.1 + old protobuf; needs numba/llvmlite floors
```

So every adapter gets its **own image and its own GPU `TaskEnvironment`**, and the
orchestrator dispatches each model to the task whose image has its package
(`config.GPU_ENVS`, `compare_pipeline.GEN_TASKS`). The two Qwen models share the `qwen`
image; Dia and CSM share the `transformers` image (built once). Three images needed
hand-work, all isolated to that one image and documented in `config.py`:

- **Chatterbox** `torch==2.6.0` has no cu130 arm64 wheel, so it's installed `--no-deps`
  on the Spark's cu130 torch with its real deps by hand.
- **Parler** backtracks to `llvmlite==0.36.0` (no py3.12 wheel) without a `numba`/
  `llvmlite` floor; and its `descript-audiotools` caps `protobuf<5`, which the Flyte
  runtime (needs `>=6.30.1` to serialize task outputs) can't live with, so a final layer
  forces protobuf back up.

## Run it

On the DGX Spark devbox (not local Python; images build on the cluster):

```bash
# default: all 7 models, M+F voice columns, the 5-script suite
flyte run compare_pipeline.py compare

# one column per model (skip the M/F expansion)
flyte run compare_pipeline.py compare --voices default

# only the female voices, quick 3-script pass on a couple of models
flyte run compare_pipeline.py compare --suite quick --voices female \
    --models '["qwen3-1.7b","kokoro-82m"]'

# your own lines
flyte run compare_pipeline.py compare --texts '["The quick brown fox jumps over the lazy dog."]'

# single-model smoke test (does it even load?), optionally a specific voice
flyte run compare_pipeline.py generate_one --model_key kokoro-82m --voice am_michael
```

> Note: `flyte run` needs its own local `.venv` (flyte + a few light deps) because it
> imports the task module on the host to discover tasks. Drive runs with
> `.venv/bin/flyte run …`. The heavy model packages stay lazy inside the adapters, so
> they are only ever installed in the per-adapter images, never locally.

Open the run's **Report** tab. Scripts are rows, models are columns; play a row
top-to-bottom to compare the same words across models.

### Fast local iteration

`run_local.py` drives one model on the host GPU (no cluster) and writes a `.wav` plus a
standalone `.html` using the identical renderer, so if it plays there it plays in the
Flyte report. Note only one model's package fits a given venv (the reason the cluster
uses per-adapter images), so install per model:

```bash
pip install qwen-tts && python run_local.py --model qwen3-1.7b --text "Hello there."
pip install kokoro   && python run_local.py --model kokoro-82m --text "Hello there."
```

### The studio app (`app.py`)

The same comparison without the CLI: paste however many lines you want (one script per
line), tick the models, pick the voice expansion, and it launches the run and links the
report.

```bash
# fast loop: the UI runs on the devbox host, runs still execute in the cluster
RUN_MODE=local .venv/bin/python app.py        # http://localhost:7864

# deploy it as a Flyte app instead (register the pipeline first)
.venv/bin/flyte deploy compare_pipeline.py orch_env
.venv/bin/python app.py
# -> http://tts-studio-text-to-speech-development.localhost:30081
```

`flyte deploy` takes exactly one environment, and `orch_env` is the one to name:
`depends_on=[cpu_env, *GPU_ENVS.values()]` means deploying it registers `fetch_weights`
and all seven `generate_*` tasks alongside the orchestrator. The deployed app **needs**
that, because it launches through `remote.Task.get("tts-orch.compare")`, which resolves
a registered task; skip the deploy and the studio still comes up and then errors the
moment you press the button. The host-mode UI above is immune, since it imports
`compare_pipeline` directly.

If a redeploy reports `failed!`, `flyte delete app tts-studio` and deploy again: a stale
revision that never became ready will block the new one (there is no `flyte stop app`).

Like the image and video studios, this one is a thin **launcher**: 1 CPU, 1Gi, no torch
and no TTS package in its image, so it cannot pin a model or hold the GPU that its own
runs need. That is exactly the opposite of `voice_app.py` below, which holds the GPU on
purpose, and the reason both cannot be up at once on a one-GPU box.

The box under the Voices radio does the arithmetic before you commit to it: with
`Voices = all` the default 7 models expand to **11 voice columns**, and 5 scripts × 11
columns is 55 clips on one GPU, one column at a time. `default` (one column per model)
is the setting to demo with; `all` is the one to leave running while you go get coffee.

## Voice cloning (`clone_pipeline.py`)

Hand it **one recording of one person** and it says five new scripts in that voice
across every cloning-capable model, then **scores** each clip. Same report shape, but
this one has numbers, because cloning has a ground truth the compare run never did.

Every line is generated **twice**: once cloned, once in the model's own default voice.
The report stacks the pair in one cell, so the question stops being "does this clip
sound good" and becomes "did the reference clip change anything" (see
[the control that makes the number mean something](#the-control-that-makes-the-number-mean-something)).

```bash
# a public-domain reference ships with the repo, so this runs out of the box:
flyte run clone_pipeline.py clone \
    --ref_audio refs/librispeech.wav --ref_text "$(cat refs/librispeech.txt)"

# your own voice: record ~15s reading refs/sage.txt, save it as refs/sage.wav, then:
flyte run clone_pipeline.py clone \
    --ref_audio refs/sage.wav --ref_text "$(cat refs/sage.txt)"

# fast pass: two scripts, two models
flyte run clone_pipeline.py clone \
    --ref_audio refs/sage.wav --ref_text "$(cat refs/sage.txt)" \
    --suite clone-quick --models '["chatterbox","qwen3-1.7b-clone"]'

# does this model clone this voice at all? (no scoring, just audio)
flyte run clone_pipeline.py clone_one \
    --ref_audio refs/sage.wav --ref_text "$(cat refs/sage.txt)" --model_key chatterbox
```

`flyte run` uploads the local wav for you: `--ref_audio` is a `flyte.io.File`.

### The two numbers

| | what it is | why both |
|---|---|---|
| **SIM** | cosine between WavLM x-vector speaker embeddings, reference vs clone | "is this the same person" |
| **WER** | Whisper transcribes the clip, diffed against the text it was asked to say | "did it say the words" |

Reported together on purpose. A model can nail someone's timbre while slurring every
third word, and SIM alone cannot see that: **high similarity is the classic way to make
a cloning demo look better than it is.** The report also shows Whisper's actual
transcript per clip, which is the receipt for the WER number.

### The control that makes the number mean something

SIM is a ranking, not a grade: a raw cosine has no natural top **and no natural bottom**,
so the run brackets it from both ends.

- The **ceiling** is the reference split in half, scored A against B. Same person,
  different audio, identical embedding path, i.e. realistically the best any clone could
  score. It's the dashed line on the scoreboard.
- The **floor** is each model saying the same line in its own default voice, imitating
  nobody. It's the pale bar under every model's bar, and the gap between the two is the
  only honest measure of what the reference clip actually bought.

The floor is not decoration. The first green run of this pipeline had Chatterbox's
"default voice" scoring **0.944** against the reference while its clone scored **0.925**:
the control beat the clone. That was a bug, and only the pairing made it visible.
Chatterbox keeps its speaker conditionals as mutable state on the model, so
`generate(audio_prompt_path=…)` overwrites them and every later plain `generate()`
silently keeps cloning. The control take was a second clone. `load_model` now snapshots
the pristine conditionals and `synth_one` restores them; the same run then reads 0.904 /
0.965 cloned against 0.748 / 0.665 stock, with the ceiling at 0.975 and the model card's
same-speaker threshold at 0.86.

Worth sitting with: a single similarity number would have called that broken run a
**success**. Two clips that were both the reference voice scored 0.93-0.96 against it,
which is exactly what a working clone looks like.

Models with no default voice (the Qwen `-Base` checkpoints exist only to clone) say so
in the cell and on the chart, and their clones still rank normally.

### What the first full run actually showed

Five models x the five-script clone suite, against `refs/librispeech.wav` (ceiling
**0.975**, same-speaker threshold 0.86). Means across the five scripts, ranked by what
the reference actually bought:

| model | cloned SIM | its own voice | lift | mean WER |
|---|---|---|---|---|
| `csm-1b` | 0.972 | 0.718 | +0.254 | 4.5% |
| `chatterbox` | 0.938 | 0.642 | +0.296 | 11.2% |
| `dia-1.6b` | 0.727 | 0.569 | +0.158 | 24.3% |
| `qwen3-1.7b-clone` | 0.968 | — | — | **0.0%** |
| `qwen3-0.6b-clone` | 0.970 | — | — | 5.0% |

**Qwen3-TTS 1.7B-Base is the one to beat.** Its similarity is a three-way tie with CSM and
the 0.6B — 0.968 / 0.972 / 0.970 against a 0.975 ceiling is not a real ordering — and it
breaks the tie on the other axis by transcribing perfectly on all five scripts. That is
the whole argument for reporting both numbers.

Three things the pairing makes visible that a similarity column alone would not:

- **A high floor is not a good sign.** CSM tops the cloned column, but its stock voice
  averages 0.718 and reached **0.917** on one script, meaning it started most of the way
  to the reference before hearing it. Chatterbox scores lower cloned (0.938) off a much
  lower floor (0.642), so it gains more. Ranking by the cloned bar puts CSM first;
  ranking by lift puts Chatterbox first.
- **Dia is not really cloning.** 0.727 mean, ranging 0.576 to 0.910 across five scripts,
  off a floor of 0.569 — the smallest lift of the three, with the worst WER. Its "default
  voice" is a *random* speaker per call, and in one run it landed at 0.917 against a
  reference it had never heard. Without the control that would have read as a good clone.
  Dia is a dialogue model being asked to do a job it was not built for.
- **Five scripts cannot resolve neighbours.** Across three identical full runs the top
  three kept swapping, and the WER blowups moved with them: the 1.7B was flawless, then
  lost a line at 119%, then flawless again; the 0.6B went 150%, 6%, 25%. Dia's own voice
  produced a **1480% WER** runaway on one line, generating far more speech than asked.
  Believe the gaps (everything above Dia, every clone above its own floor), not the order
  within a tenth.

Rough cost on the GB10: the full five-model run takes ~35 minutes wall clock including
the two Qwen `-Base` downloads (5.2GB + 3.2GB), and the second take roughly doubles
generation time. `--suite clone-quick --models '["chatterbox"]'` is the ~4 minute
smoke test.

### The models that can clone

Five, and **all of them ride images the compare demo already built**, which is the nice
property of this half: the only new thing on disk is weights.

| key | how it clones | needs a transcript? | has a default voice? | image |
|---|---|---|---|---|
| `qwen3-1.7b-clone` | `generate_voice_clone()` | yes | no, `-Base` only clones | existing `qwen` |
| `qwen3-0.6b-clone` | same, a third the size | yes | no, same reason | existing `qwen` |
| `chatterbox` | `generate(audio_prompt_path=…)` | **no** | yes, but it is sticky (see above) | existing `chatterbox` |
| `dia-1.6b` | audio as decoder prefix, transcript prepended to the target | yes | yes, a random speaker per call | existing `transformers` |
| `csm-1b` | the reference becomes a prior conversation turn | yes | yes, speaker `[0]` with no context | existing `transformers` |

The "default voice" column is the control take: same model, same line, no reference. Two
of the five have nothing to put there, which is itself the answer to "what does this
checkpoint do" — the Qwen `-Base` weights have no built-in speakers, that is what
`-CustomVoice` is for.

Cloning Qwen means a **different checkpoint**, not a different call: `-CustomVoice` (the
compare run's) has no `generate_voice_clone`; only `-Base` does. Same `qwen-tts` package,
so no new image, just a second download.

Kokoro and Parler are absent because they genuinely cannot clone (fixed voicepacks; a
text description). Asking for one fails in the orchestrator before any weights are
fetched or any GPU pod is scheduled, because the alternative is worse than slow: Kokoro
would load fine, ignore the reference clip, and render its stock voice into the grid as
if it had cloned something.

The scoring task reuses the shared `transformers` image too. That's why the scorers are
WavLM + Whisper rather than speechbrain + jiwer: both are transformers-native, so the
whole cloning demo adds **zero images** to the build.

### The reference clip matters more than the model

Three of the five models condition on the reference **transcript**, so it has to be
exact; that's why the demo ships a fixed read-aloud script instead of Whisper-transcribing
the reference and inheriting ASR errors. 8-15s, mono, no reverb, don't normalize to full
scale. `RefVoice.warnings()` checks length, clipping and near-silence and surfaces them
on the report's reference card rather than failing the run.

**Reference wavs are gitignored.** A reference clip is a voiceprint: exactly the input
someone else needs to clone that voice with this same pipeline. Transcripts are
committed, audio isn't. See `refs/README.md`.

### Watermarking

Chatterbox embeds a [Perth](https://github.com/resemble-ai/perth) watermark in
everything it generates, so its clones stay detectable as synthetic afterwards. The
other four ship no watermark at all: nothing downstream can distinguish their output
from a recording. The report shows the verdict per clip.

Detection runs in the **generation** task, not the scoring one, because `perth` only
exists in Chatterbox's image (it comes in as one of that package's deps). Since
`metrics.py` is Flyte-free it rides into every image with the code bundle, so the same
call returns a real verdict where the detector exists and "not checked" everywhere else,
at zero build cost. Cells distinguish "no watermark" from "couldn't check" — they are
different claims.

## Voice chat (`voice_app.py`)

The payoff for both pipelines: speak to a local LLM and hear it answer, with **every
layer swappable mid-conversation**.

```
mic -> Whisper -> ollama (LLM) -> Kokoro / your cloned voice -> speakers
        ^^^^^^^   ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
         dropdown    dropdown              dropdown
```

STT is kept as its **own layer** rather than leaning on a multimodal LLM's native audio.
That costs ~1.6GB for Whisper and buys a free choice of LLM: Qwen, a 1B, anything the
backend can serve, not just models with an audio encoder.

### Two deployments, one seam

The box is one node with `nvidia.com/gpu: 1` and no time-slicing, so **only one pod can
hold the GPU**. That single fact produces two valid shapes, and `voice_core.stream_llm()`
is the only code that differs between them:

```bash
# A. vLLM holds the GPU in its own app; the voice app is CPU-only.
pip install --pre flyteplugins-vllm      # deploy-time only
python voice_vllm.py                     # prints the app URL
LLM_BACKEND=vllm VLLM_URL=<url> python voice_app.py

# B. ollama runs inside the voice app, which therefore holds the GPU.
LLM_BACKEND=ollama python voice_app.py

# either one, on the devbox host, for a fast iteration loop:
RUN_MODE=host LLM_BACKEND=ollama python voice_app.py
```

| | **A · vLLM** (default) | **B · ollama** |
|---|---|---|
| LLM | own GPU pod, `VLLMAppEnvironment` | in-pod subprocess |
| Speed | NVFP4 ~16.5GB on vLLM kernels | Q4 18GB, **measured 14.1 tok/s** |
| Weights | `flyte.prefetch.hf_model` + `RunOutput`, fetched once | pulled on cold start |
| Model swap | one model per app: switching is a redeploy | seconds, from the dropdown |
| Voice app pod | **CPU** (Kokoro ~5x real-time, still streams) | **GPU** (Kokoro 78x) |
| Cloned voice | no (Chatterbox is 2.2x on a *GPU*) | yes |
| Compare/clone pipelines | keep running | queue behind the app |

**A is the default** because it reuses the serving pattern that already works on this box
(`gemma4-dgx-devbox/vllm_server.py`) and leaves the GPU free. **B exists** because fast
model swapping and the cloned voice are worth a deployment of their own.

The quantized checkpoint matters in both: the Spark is memory-bandwidth-bound, so
tokens/sec tracks bytes-read-per-token, and bf16 at ~52GB would fall to roughly speech
rate. `VOICE_LLM_QUANT=nvfp4|fp8|bf16` picks it; NVFP4 is smallest and should be fastest.

> **Untested:** NVFP4 kernels target Blackwell, and the GB10 is Blackwell but **sm_121**,
> not the B200's sm_100. This repo has hit sm_121 gaps repeatedly (CUDA-graph capture
> hangs, `torch.compile` failures, the voxtral deferral). If it won't load, fall back
> through `fp8` then `bf16` — the voice app doesn't care what's behind the endpoint.

### It streams sentence by sentence

The naive loop (what `topics/gemma4/voice/app.py` does today) waits for the whole reply,
then synthesizes, then plays: time-to-first-audio is the entire generation plus the
entire synthesis. Instead we chunk the token stream at sentence boundaries and
synthesize each chunk while the next generates.

Measured on this box, `gemma4:26b` decodes **14.1 tok/s**. Speech is roughly 3 tok/s of
actual talking, so the LLM runs **~4x ahead of playback** and never lets the audio catch
up. First audio should land ~1.5s after you stop talking: Whisper (~0.3s) + the first
sentence (~1s) + Kokoro synthesizing it (~0.04s). The UI prints the real number.

Two rules shape `SentenceChunker`: chunks break on sentence boundaries (TTS models need
a whole clause for prosody) and must be **over ~1 second of audio**, because Gradio's
streaming player stutters on shorter ones. So `"Sure."` gets merged forward instead of
emitted alone, and `Dr.` / `p.m.` don't split a sentence.

### Everything is resident, and it all fits (deployment B)

| | resident |
|---|---|
| Gemma 4 26B-A4B (Q4, MoE, ~4B active) | ~18 GB |
| Whisper large-v3-turbo | ~1.6 GB |
| Kokoro 82M | ~0.4 GB |
| Chatterbox (cloned voice) | ~2 GB |
| **total** | **~26 GB of ~98 GB usable** |

No external LLM API is needed either way: in B ollama runs *inside* the app pod, in A
vLLM runs in a pod next door. Both are reached over local HTTP.

### The GPU-holding trade

The image and video studios are deliberately thin **launchers** that submit runs and
hold no model, so the GPU stays free (see their `app.py` docstrings). A voice app can't
be a launcher: keeping models resident *is* the latency feature.

Deployment **A** dodges this by putting only the LLM on the GPU and leaving the voice app
on CPU. Deployment **B** accepts it: while the app is up it owns the GPU and the
compare/clone pipelines queue behind it. In both, `scaledown_after` is the release valve
rather than an architectural fix — idle and the pod exits, handing the GPU back.

### Which voices actually stream

Only Kokoro is comfortably faster than speech (78x real-time here). Chatterbox is 2.2x
and Qwen3-TTS 1.4x, so those lag; they're in the dropdown so you can *hear* the
difference. The engine list is built at runtime from what actually imports in the image,
since the TTS packages can't all share one (see above), so an unavailable engine never
appears and then fails on click.

## The scripts

Five scripts, each stressing one axis you can only judge by ear (see `prompts.py`):
naturalness/prosody, **text normalization** (numbers, dates, currency, a URL, an
acronym: the clearest quality separator), questions + emphasis, hard phonemes, and a
two-speaker dialogue with a nonverbal (Dia's home turf). For single-voice models the
`[S1]`/`[S2]` tags and `(laughs)` are stripped so they read one clean narrator.

The **clone** suite (`--suite clone`) is a different five, because the question is
different: not "is this good speech" but "is this still *them*, and did it say the
words". They target how a clone specifically comes apart: identity under **expression**
(the big one, where a clone holds up on flat narration and snaps back to the model's
default voice the moment the line gets excited), drift over a long utterance, phonemes
the reference never contained, and normalization, now objectively scored instead of
judged by ear. No dialogue script: two speakers is incoherent when the whole point is
that every word is one specific person's voice.

## Files

- `config.py` — Flyte images/envs, per-adapter, Spark-pinned (arm64, cu130, local registry).
- `models.py` — the model registry (`TTSModelSpec`, one `adapter` per model; `clone_capable`).
- `prompts.py` — the read-aloud scripts and what each one tests, for both suites.
- `tts_core.py` — Flyte-free: the per-adapter loaders, `synth_one`/`synth_clone`,
  `RefVoice`, audio + spectrogram embedding, and both HTML report renderers. Shared by
  the tasks, run_local, and (later) the app.
- `metrics.py` — Flyte-free: speaker similarity (WavLM x-vectors), WER (Whisper +
  a hand-rolled word-level Levenshtein), Perth watermark detection.
- `compare_pipeline.py` — the compare tasks: `fetch_weights`, seven adapter `generate_*`
  tasks, and the `compare` orchestrator.
- `clone_pipeline.py` — the cloning tasks: four adapter `clone_*` tasks, `score_clones`,
  and the `clone` orchestrator. Reuses `fetch_weights`, so weights the compare run
  already pulled are cache hits. Every line is generated in two `mode`s, `"clone"` and
  `"stock"`, and results/scores are keyed `(text, model_key, mode)` all the way to the
  renderer.
- `app.py` — the compare studio: a Gradio launcher (checkbox model picker, one script
  per line) that submits `compare` runs and links the report. Holds no GPU, loads no model.
- `voice_core.py` — Flyte-free: the `SentenceChunker`, the Whisper `Transcriber`, the
  ollama streaming client, the `Speaker` (TTS engine cache), and the `converse` loop.
- `voice_app.py` — the Gradio voice-chat app + its Flyte `AppEnvironment`. The pod is
  GPU or CPU depending on `LLM_BACKEND`.
- `voice_vllm.py` — the vLLM serving app for deployment A, mirroring
  `gemma4-dgx-devbox/vllm_server.py` with a quantized checkpoint.
- `refs/` — reference voices: committed transcripts, gitignored audio.
- `run_local.py` — one model on the host GPU, for quick checks. `--ref-audio` switches
  it to cloning and it scores the clip too when the scorers are importable.

## Adding a model later

One entry in `models.py` (`key`, `repo`, `adapter`, defaults; add a `voices` tuple for
M/F variants). If it needs a new package, add an adapter branch in
`tts_core.load_model`/`synth_one`, an image + GPU env in `config.py`, and a `generate_*`
task in `compare_pipeline.py`. If it reuses an existing package (e.g. another
transformers-native model), point its adapter at the shared image and you skip the new
image. Candidates parked for later: Voxtral (Mistral, needs a vLLM server), CosyVoice2
(painful install: Matcha-TTS + ttsfrd submodules), IndexTTS-2, VibeVoice.
