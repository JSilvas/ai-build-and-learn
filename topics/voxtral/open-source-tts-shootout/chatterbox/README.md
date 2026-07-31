# Chatterbox (Resemble AI)

Zero-shot voice cloning from a short reference clip, MIT license, and the
one Resemble claims beat ElevenLabs in blind listening tests (the "Turbo"
variant). This is the natural-quality + cloning end of the lineup, next to
Kokoro's fast-but-fixed-voice approach.

- License: **MIT**
- Model: [ResembleAI/chatterbox](https://github.com/resemble-ai/chatterbox)
- Base model is **English-only**; a separate multilingual release adds
  ~23 languages — see `synth.py` for the import path if you want to try it
- Runs on CUDA, MPS (Apple Silicon), or CPU — pass `device="mps"` on the
  M4 Max for the GPU path
- Outputs carry an inaudible Resemble watermark by default (responsible-AI
  provenance tagging, not a quality issue)

## Setup

```bash
cd topics/voxtral/open-source-tts-shootout/chatterbox
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Drop a ~10s clip of your own voice at
`../common/reference_voice.wav` before running the cloning category.

## Run

```bash
python synth.py --category narration
python synth.py --category expressive
python synth.py --category cloning --reference ../common/reference_voice.wav
```

## What to double check

- `ChatterboxTTS.from_pretrained(device=...)` — confirm `"mps"` is accepted
  directly, or whether you need `torch.device("mps")` instead of the string.
- The multilingual class name/import (`chatterbox.mtl_tts` or similar) if
  you want to test the multilingual category — I wasn't able to confirm
  the exact module path against current docs.
- Watermarking: if you're doing subjective A/B listening tests, know that
  the watermark is inaudible by design, so it shouldn't affect scoring —
  but worth confirming it doesn't trip anything if you pipe outputs through
  a detector as part of the comparison.
