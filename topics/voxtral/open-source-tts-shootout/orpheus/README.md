# Orpheus 3B

The heavyweight of the lineup — a Llama-backbone TTS model (3B params)
from Canopy Labs, with inline emotion tags (`<laugh>`, `<sigh>`, etc.) and
partial zero-shot cloning. Worth including specifically as the
counterpoint to Kokoro: does 3B of weight actually buy better expressiveness
than the 82M model, or is the gap smaller than the size difference suggests?

- License: Apache-2.0 / MIT depending on checkpoint — **verify per specific
  finetune you use**, community variants don't all inherit the same terms
- Model/repo: [canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- Cloning: partial — less reliable than Chatterbox or XTTS v2 in practice
- Languages: English-focused; community finetunes exist for French, German,
  Korean, Mandarin

## Apple Silicon caveat — read before installing

The reference implementation (`orpheus-speech`) runs inference through
**vLLM, which is CUDA-only** — it will not run natively on the M4 Max.
Two ways around that for a Mac:

1. **llama.cpp / GGUF route**: Orpheus is Llama-architecture, so
   community GGUF quantizations exist and run fine on Apple Silicon via
   `llama.cpp`'s Metal backend — but Orpheus's output is audio *token
   codes*, not raw audio, so you additionally need the **SNAC** vocoder
   to decode those tokens into a waveform. This is meaningfully more setup
   than every other model in this lineup.
2. **Skip it / note it as a limitation**: if the GGUF+SNAC path eats more
   time than it's worth, it's a legitimate finding for the comparison —
   "heaviest model, worst Mac-native support" is a real data point.

`requirements.txt` and `synth.py` below assume you've gone the GGUF+SNAC
route. I was not able to verify current package names/APIs for either
piece with confidence — this is the roughest sketch in the whole shootout.

## Setup (GGUF + SNAC route, unverified)

```bash
cd topics/voxtral/open-source-tts-shootout/orpheus
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## What to double check (a lot, here)

- Whether `llama-cpp-python` is built with Metal support (`CMAKE_ARGS`
  during install) — without it, this falls back to slow CPU inference.
- The actual GGUF model repo/filename to download (search Hugging Face for
  community Orpheus GGUF quantizations — the canonical one may change).
- SNAC's current package name and decode API — check
  [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac) directly,
  `synth.py`'s import is a best guess.
- Given the setup cost here, it may be more efficient to try Orpheus
  through a hosted/CPU-inference demo space on Hugging Face first, purely
  to judge voice quality, before investing in a local Mac setup.
