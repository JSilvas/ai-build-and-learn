# Track C — Local chained pipeline (STT → LLM → TTS)

The classic architecture — same shape as `topics/gemma4/voice`, just every
stage swapped for an MLX-native model so it runs well on Apple Silicon
instead of leaning on Ollama/CTranslate2/edge-tts:

1. **STT**: [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Whisper ported to MLX
2. **LLM**: [`mlx-lm`](https://github.com/ml-explore/mlx-lm) — a small instruct model (default below: `mlx-community/Llama-3.2-3B-Instruct-4bit`)
3. **TTS**: Sesame CSM via the community [`csm-mlx`](https://github.com/senstella/csm-mlx) port — expressive speech conditioned on audio context (not full-duplex, but higher quality than most drop-in TTS)

This is the one to compare against Track B to see what full-duplex actually
buys you over three well-optimized stages bolted together — expect higher
latency (three model loads/inferences in series vs. one), but potentially
comparable or better voice quality since CSM is purpose-built for
expressiveness.

## Setup

```bash
cd topics/voxtral/realtime-voice-showdown/track-c-local-pipeline
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

`csm-mlx` isn't reliably on PyPI under that exact name at any given moment —
if `uv pip install csm-mlx` fails, install straight from the repo:

```bash
uv pip install "git+https://github.com/senstella/csm-mlx"
```

## Run

```bash
python pipeline.py
```

Records a fixed-length chunk from the mic (see `RECORD_SECONDS` in
`pipeline.py` — push-to-talk by wall-clock timer rather than VAD, since
that's simpler to get right than reimplementing voice-activity detection),
runs it through all three stages, and plays back the reply. Logs
per-stage timing plus overall time-to-first-audio to `../results.jsonl`
via `common/latency.py`.

## What to double check

- **`csm-mlx`'s actual API** is the shakiest part of this script — I wrote
  `generate(...)` based on the project's example usage, but small MLX ports
  like this one change their function signatures across versions faster
  than their README updates. Check `csm-mlx`'s own example script if
  `pipeline.py` throws a `TypeError`/`AttributeError` on the CSM call.
- **Model IDs** — swap `LLM_MODEL` for whatever `mlx-community` build you
  want to try; 3B is a reasonable first pick for turnaround speed but a
  bigger model may be worth the latency trade given 64 GB of headroom.
- Whisper model size (`WHISPER_MODEL`) — `base` for fast iteration,
  step up to `small`/`medium` once the pipeline is wired up if transcription
  accuracy is the bottleneck rather than latency.
