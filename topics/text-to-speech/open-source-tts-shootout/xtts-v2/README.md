# XTTS v2 (Coqui)

Broad multilingual voice cloning (~17 languages) from a ~6-second reference
clip. Coqui the company shut down in early 2024, but the model and the
`TTS` toolkit are kept alive by a community fork — check licensing
carefully before using this one for anything beyond this comparison.

- License: **Coqui Public Model License — non-commercial.** Read it before
  using XTTS v2 output for anything beyond learning/comparison here.
- Model: `tts_models/multilingual/multi-dataset/xtts_v2`
- Languages: ~17, including en/es/fr/de/it/pt/pl/tr/ru/nl/cs/ar/zh-cn/ja/hu/ko/hi
- Cloning: yes, ~6s reference clip, same clip works across languages

## Setup

```bash
cd topics/voxtral/open-source-tts-shootout/xtts-v2
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

The original `TTS` PyPI package is Coqui's, now unmaintained upstream —
`requirements.txt` below points at the community-maintained fork instead
(`coqui-tts`), which keeps the same `TTS.api.TTS` import path. If that
fork has also moved on by the time you run this, `pip install TTS` (the
original) still works for XTTS v2 specifically, just without recent fixes.

## Run

```bash
python synth.py --category narration --language en
python synth.py --category cloning --language en --reference ../common/reference_voice.wav
python synth.py --category multilingual --language es --reference ../common/reference_voice.wav
```

XTTS v2 always wants a `speaker_wav` — for non-cloning categories this
script falls back to a built-in example speaker shipped with the model.

## What to double check

- Whether MPS (Apple GPU) actually helps here — Coqui's `TTS` library has
  had spotty MPS support historically; falling back to `device="cpu"` may
  end up faster/more stable than fighting MPS compatibility issues.
- The exact license terms for your use case — CPML is stricter than most
  of the other models in this lineup, which is worth calling out during
  the stream's licensing discussion.
