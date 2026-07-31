"""Synthesize the cloning prompt with F5-TTS, via its CLI (subprocess),
since the Python API has been less stable across releases than the CLI.

Untested (headless sandbox, no audio here) — confirm the CLI entrypoint
name per the README before trusting this.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from benchmark import time_synthesis  # noqa: E402
from test_sentences import PROMPTS  # noqa: E402

CLI_ENTRYPOINT = "f5-tts_infer-cli"  # verify with `f5-tts_infer-cli --help`


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=list(PROMPTS), default="cloning")
    parser.add_argument("--reference", required=True, help="reference wav for cloning")
    parser.add_argument("--reference-text", required=True, help="exact transcript of the reference wav")
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    text = PROMPTS[args.category]
    out_path = Path(args.out_dir) / f"output_{args.category}.wav"

    cmd = [
        CLI_ENTRYPOINT,
        "--model", args.model,
        "--ref_audio", args.reference,
        "--ref_text", args.reference_text,
        "--gen_text", text,
        "--output_dir", args.out_dir,
    ]

    with time_synthesis("f5-tts", args.category) as bench:
        subprocess.run(cmd, check=True)
        if out_path.exists():
            audio, sr = sf.read(out_path)
            bench.audio_seconds = len(audio) / sr

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
