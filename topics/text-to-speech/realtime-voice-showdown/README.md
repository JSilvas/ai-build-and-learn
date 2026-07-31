# Realtime Voice Showdown

A head-to-head comparison of realtime/conversational voice stacks, run on a
MacBook Pro M4 Max (64 GB unified memory). This sits alongside the batch TTS
demo one level up (`../`, Voxtral) — that one answers "how do I turn text into
speech," this one answers "how do I hold a live spoken conversation with
low enough latency that it feels natural."

> **Built in a headless Linux sandbox, meant to run on Apple Silicon.**
> Everything here was written and reviewed for correctness against each
> project's documented API, but none of it has been executed — this dev
> environment has no microphone/speaker and no MLX (Apple-silicon-only)
> runtime to test against. Treat the scripts as a working first draft: pull
> this branch onto the Mac, `pip install` each track's requirements, and
> expect to patch a few API details against upstream's current examples
> before it runs clean. Flagged the specific spots I'm least sure of inline.

## The three tracks

| Track | What it is | Where it runs | Why it's in the comparison |
|---|---|---|---|
| [A — OpenAI Realtime](track-a-openai-realtime/) | `gpt-realtime` over the Realtime API (speech-in → speech-out, one hosted model) | Cloud (needs `OPENAI_API_KEY`) | The cloud SOTA baseline — ~300ms average latency, no local compute at all |
| [B — Kyutai Moshi (MLX)](track-b-moshi-mlx/) | Full-duplex speech-to-speech foundation model, no intermediate text | Local, on-device via MLX | The open-weights answer to "true" full-duplex — claims ~200ms end-to-end |
| [C — Local chained pipeline](track-c-local-pipeline/) | STT (mlx-whisper) → LLM (mlx-lm) → TTS (Sesame CSM, MLX) | Local, on-device via MLX | The classic 3-stage architecture (same shape as `topics/gemma4/voice`), to see what full-duplex buys you over it |

## Methodology

For each track, capture:

1. **Time-to-first-audio** — from the moment you stop talking to the moment
   sound starts coming back. `common/latency.py` gives every track the same
   timer/logging shape so the numbers are comparable.
2. **Turn-taking / barge-in** — can you interrupt it mid-sentence? Does it
   handle overlapping speech, or does it wait for silence?
3. **Setup friction** — cold-start time, model download size, how many
   moving pieces.
4. **Subjective quality** — naturalness of the voice, whether prosody holds
   up, whether it feels "realtime" or like a fast walkie-talkie.

Log results as you go in [`results.md`](results.md) (one row per run) so the
comparison has actual numbers behind it instead of vibes.

## Setup (shared)

```bash
cd topics/voxtral/realtime-voice-showdown
uv venv .venv --python 3.11
source .venv/bin/activate
```

Each track then has its own `requirements.txt` — install per-track, since
they don't all need to coexist in the same env (MLX packages in particular
can be picky about torch/transformers versions).

## Files

```
common/latency.py         # shared timing/logging helper used by all three tracks
track-a-openai-realtime/  # cloud baseline
track-b-moshi-mlx/        # local full-duplex
track-c-local-pipeline/   # local chained STT->LLM->TTS
results.md                # fill in as you run each track
```
