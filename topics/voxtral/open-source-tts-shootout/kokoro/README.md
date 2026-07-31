# Kokoro-82M

The lightweight end of the lineup: 82M params, Apache-2.0, runs comfortably
on CPU. No voice cloning — you pick from 54 built-in voices (and can blend
between them), so this is the one to benchmark for "fast, good-enough
narration" rather than "sounds like a specific person."

- License: **Apache-2.0** — no commercial-use caveats, unlike half this lineup
- Model: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- Languages: English (US `a` / UK `b`) built-in; extra packs for
  ja/zh/es/fr/hi/it/pt-br exist but need their own phonemizer setup
- Needs `espeak-ng` as a system dependency for phonemization

## Setup

```bash
cd topics/voxtral/open-source-tts-shootout/kokoro
brew install espeak-ng   # macOS system dependency, not a pip package
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
python synth.py --category narration --voice af_heart
python synth.py --category expressive --voice af_heart
```

Writes `output_<category>.wav` and logs timing/RTF via
`common/benchmark.py`. Try a couple of different voice codes (see the
[voice list on the model card](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md))
— `af_*`/`am_*` are American female/male, `bf_*`/`bm_*` are British.

## What to double check

- Exact `KPipeline` call shape and `lang_code` values — written from the
  model card's documented usage, but this is a fast-moving repo.
- Whether `espeak-ng` is actually on your `PATH` after `brew install` —
  Kokoro's phonemizer shells out to it and fails silently-ish if it's
  missing.
