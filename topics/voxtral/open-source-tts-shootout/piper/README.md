# Piper

The "will it run on a Raspberry Pi" end of the lineup. ONNX-based, CPU-only,
tens of megabytes per voice, huge catalog of pretrained voices across 30+
languages. No cloning — like Kokoro, you pick a voice rather than clone
one — but even lighter-weight and explicitly built for offline/edge use.

- License: **MIT**
- Project: [rhasspy/piper](https://github.com/rhasspy/piper)
- Voices: browse/download from the
  [voices index](https://github.com/rhasspy/piper/blob/master/VOICES.md) —
  each is a small `.onnx` + `.onnx.json` pair

## Setup

```bash
cd topics/voxtral/open-source-tts-shootout/piper
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# Download a voice (English example — swap for whatever you want to test,
# including a non-English one for the multilingual category)
python -m piper.download_voices en_US-lessac-medium
```

## Run

```bash
python synth.py --category narration --voice en_US-lessac-medium
python synth.py --category multilingual --voice <some-non-English-voice>
```

## What to double check

- The voice-download command/module path — Piper's packaging has shifted
  between a standalone CLI binary and a pip-installable `piper-tts` package
  with its own downloader; if `python -m piper.download_voices` doesn't
  exist, grab the `.onnx`/`.onnx.json` pair directly from the voices index
  instead and point `synth.py` at the local files.
- This should be the fastest RTF in the whole lineup by a wide margin —
  if it isn't clearly beating Kokoro on speed, something's likely
  misconfigured (e.g. accidentally not using the ONNX runtime's CPU
  optimizations).
