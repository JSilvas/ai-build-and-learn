# F5-TTS

Diffusion/flow-matching TTS — a different architectural lineage than the
autoregressive models in this lineup (Kokoro, Chatterbox, Orpheus).
Research-grade zero-shot cloning quality, but the weights are
non-commercial and it needs a transcript of the reference clip, not just
the clip itself.

- License: **CC-BY-NC — non-commercial**
- Model/repo: [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
- Cloning: yes, zero-shot, but requires **both** a reference audio clip
  *and* its exact transcript (unlike Chatterbox/XTTS which only need the
  audio)
- Languages: English/Chinese in the base checkpoint; community finetunes
  exist for others

## Setup

```bash
cd topics/voxtral/open-source-tts-shootout/f5-tts
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

This wraps the project's own CLI (`f5-tts_infer-cli`) via subprocess rather
than a direct Python import — F5-TTS's Python API has changed shape across
releases, while the CLI flags have stayed comparatively stable.

## Run

You need the reference clip's transcript up front — type or paste it in:

```bash
python synth.py --category cloning \
  --reference ../common/reference_voice.wav \
  --reference-text "the exact words spoken in reference_voice.wav"
```

## What to double check

- Confirm the CLI entrypoint name — I've seen this project ship it as
  `f5-tts_infer-cli` and also as a plain `f5_tts.infer.infer_cli` module
  invocation across versions. Run `f5-tts_infer-cli --help` after install
  to confirm before trusting `synth.py`'s subprocess call.
- Model checkpoint selection — `--model F5TTS_v1_Base` is the current
  default in the repo as of when this was written; verify against the
  repo's README for the latest recommended checkpoint name.
- This is CPU/MPS-unfriendly relative to the others — flow-matching models
  are heavier per-inference-step than autoregressive ones, so expect the
  lowest real-time factor in the lineup even on the M4 Max.
