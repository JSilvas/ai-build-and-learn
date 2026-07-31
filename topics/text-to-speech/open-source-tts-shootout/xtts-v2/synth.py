"""Synthesize one of the shared test prompts with XTTS v2.

Untested (headless sandbox, no audio here) — see README for what to double
check, particularly around MPS support and licensing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
from TTS.api import TTS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from benchmark import time_synthesis  # noqa: E402
from test_sentences import PROMPTS  # noqa: E402

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=list(PROMPTS), default="narration")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    parser.add_argument(
        "--reference",
        default=None,
        help="reference wav for cloning; falls back to a built-in example speaker if omitted",
    )
    args = parser.parse_args()

    text = PROMPTS[args.category]
    tts = TTS(MODEL_NAME).to(args.device)

    out_path = f"output_{args.category}_{args.language}.wav"
    kwargs = {"text": text, "language": args.language, "file_path": out_path}
    if args.reference:
        kwargs["speaker_wav"] = args.reference
    else:
        kwargs["speaker"] = tts.speakers[0]  # built-in example speaker

    with time_synthesis("xtts-v2", args.category) as bench:
        tts.tts_to_file(**kwargs)
        audio, sr = sf.read(out_path)
        bench.audio_seconds = len(audio) / sr

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
