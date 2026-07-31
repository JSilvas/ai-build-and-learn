# Open-Source TTS Shootout

Comparing the open-source text-to-speech models called out in this week's
[AI Build & Learn brief](https://luma.com/oxhti59k) — "Open-Source
Text-to-Speech: Natural Voices." Same spirit as the Realtime Voice Showdown
one level up, but batch TTS instead of live conversation: given text (and
sometimes a reference voice clip), how fast, how natural, and how flexible
is each model?

> **Built in a headless Linux sandbox, meant to run on Apple Silicon.**
> Written and reviewed against each project's documented API but not
> executed here — no MPS/MLX runtime, no audio playback in this
> environment. Pull onto the M4 Max, `pip install` per-model, expect to
> patch a few things against current upstream examples. Flagged the
> spots I'm least sure of inline.

## The lineup

| Model | License | Size | Cloning | Multilingual | Notes |
|---|---|---|---|---|---|
| [Kokoro-82M](kokoro/) | Apache-2.0 | 82M | No (54 preset voices, blendable) | en-us/en-gb + ja/zh/es/fr/hi/it/pt-br packs | Tiny, CPU-fast — the "lightweight narration" end of the spectrum |
| [Chatterbox (Resemble AI)](chatterbox/) | MIT | ~0.5B | Yes, zero-shot from a short clip | English-only in the base model; multilingual variant adds ~23 languages | Turbo variant is the one Resemble claims beat ElevenLabs in blind listening tests |
| [XTTS v2 (Coqui)](xtts-v2/) | **Non-commercial** (Coqui Public Model License) | ~460M | Yes, from ~6s reference | ~17 languages | Coqui the company is defunct; using the community-maintained fork |
| [F5-TTS](f5-tts/) | **Non-commercial** (CC-BY-NC) | ~330M (flow-matching DiT) | Yes, zero-shot, needs ref audio + its transcript | English/Chinese primary, community finetunes for more | Research-grade quality, diffusion/flow-matching architecture — different lineage than the others |
| [Orpheus 3B](orpheus/) | Apache-2.0 / MIT (community finetunes vary) | 3B | Partial/zero-shot, less reliable than XTTS/Chatterbox | English-focused; community finetunes for FR/DE/KR/ZH | Llama-backbone TTS — the heavyweight of the group, supports inline emotion tags |
| [Piper](piper/) | MIT | Tens of MB per voice | No | 30+ languages, huge pretrained voice catalog | The "will it run on a Raspberry Pi" end — ONNX, CPU-only, near-instant |

Optional reference points called out in the brief:
- **ElevenLabs** (commercial) — worth a quick side-by-side if you have API
  credits, to see how close the best open models actually get.
- **[HF TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena)** —
  crowd-ranked leaderboard, useful sanity check against your own ears.

## Methodology

Every model gets run against the same fixed set of prompts
(`common/test_sentences.py`):

1. **Narration** — a plain, neutral sentence (tests baseline quality/speed)
2. **Expressive** — a sentence with clear emotional content (tests prosody —
   this is where the "fast vs natural" tradeoff shows up most)
3. **Multilingual** — a non-English sentence (skip for English-only models)
4. **Cloning target** — a sentence to render in a cloned voice, for the
   models that support it (drop your own ~10s reference clip at
   `common/reference_voice.wav` — gitignored, bring your own)

`common/benchmark.py` gives every model folder the same timing helper:
wall-clock synth time, output audio duration, and **real-time factor**
(audio seconds produced per second of compute — >1.0 means faster than
real-time). Log results in [`results.md`](results.md).

## Setup (shared)

```bash
cd topics/voxtral/open-source-tts-shootout
uv venv .venv --python 3.11
source .venv/bin/activate
```

Each model has its own `requirements.txt` — install per-model rather than
all at once. These libraries pin very different torch/transformers
versions and will fight each other in one shared environment.

## Files

```
common/
  test_sentences.py    # shared prompts (narration / expressive / multilingual / cloning)
  benchmark.py          # timing helper: wall-clock, audio duration, real-time factor
  reference_voice.wav    # gitignored — drop your own short clip here for cloning tests
kokoro/       chatterbox/       xtts-v2/       f5-tts/       orpheus/       piper/
  README.md   requirements.txt  synth.py        (per model)
results.md    # fill in as you run each model
```
