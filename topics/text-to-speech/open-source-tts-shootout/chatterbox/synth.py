"""Synthesize one of the shared test prompts with Chatterbox.

Untested (headless sandbox, no MPS/audio here) — see README for what to
double check if the ChatterboxTTS call shape has drifted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from benchmark import time_synthesis  # noqa: E402
from test_sentences import PROMPTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=list(PROMPTS), default="narration")
    parser.add_argument("--device", default="mps", help="mps | cuda | cpu")
    parser.add_argument("--reference", default=None, help="reference wav for cloning")
    args = parser.parse_args()

    if args.category == "cloning" and not args.reference:
        parser.error("--category cloning requires --reference <wav path>")

    text = PROMPTS[args.category]
    model = ChatterboxTTS.from_pretrained(device=args.device)

    with time_synthesis("chatterbox", args.category) as bench:
        if args.reference:
            wav = model.generate(text, audio_prompt_path=args.reference)
        else:
            wav = model.generate(text)
        bench.audio_seconds = wav.shape[-1] / model.sr

    out_path = f"output_{args.category}.wav"
    ta.save(out_path, wav, model.sr)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
