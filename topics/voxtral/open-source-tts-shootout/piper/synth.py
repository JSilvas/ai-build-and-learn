"""Synthesize one of the shared test prompts with Piper.

Untested (headless sandbox, no audio here) — see README for the
voice-download caveat.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

from piper import PiperVoice

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from benchmark import time_synthesis  # noqa: E402
from test_sentences import PROMPTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=list(PROMPTS), default="narration")
    parser.add_argument("--voice", default="en_US-lessac-medium", help="voice name (must be downloaded already)")
    args = parser.parse_args()

    text = PROMPTS[args.category]
    voice = PiperVoice.load(f"{args.voice}.onnx")

    out_path = f"output_{args.category}.wav"
    with time_synthesis("piper", args.category) as bench:
        with wave.open(out_path, "wb") as wav_file:
            voice.synthesize(text, wav_file)
        with wave.open(out_path, "rb") as wav_file:
            bench.audio_seconds = wav_file.getnframes() / wav_file.getframerate()

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
