"""Synthesize one of the shared test prompts with Kokoro-82M.

Untested (headless sandbox, no espeak-ng/audio here) — see README for what
to double check if the KPipeline call shape has drifted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from benchmark import time_synthesis  # noqa: E402
from test_sentences import PROMPTS  # noqa: E402

SAMPLE_RATE = 24_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=list(PROMPTS), default="narration")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--lang-code", default="a", help="'a' = American English, 'b' = British")
    args = parser.parse_args()

    text = PROMPTS[args.category]
    pipeline = KPipeline(lang_code=args.lang_code)

    with time_synthesis("kokoro", args.category) as bench:
        chunks = [audio for _, _, audio in pipeline(text, voice=args.voice)]
        full_audio = np.concatenate(chunks)
        bench.audio_seconds = len(full_audio) / SAMPLE_RATE

    out_path = f"output_{args.category}.wav"
    sf.write(out_path, full_audio, SAMPLE_RATE)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
