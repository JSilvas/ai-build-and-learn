"""Best-effort GGUF+SNAC route for Orpheus 3B on Apple Silicon.

This is the roughest script in the shootout — see the README's "Apple
Silicon caveat" and "what to double check" sections. Written from general
knowledge of llama.cpp + SNAC-style codec-token TTS pipelines, not a
verified Orpheus example. Expect to fix the GGUF repo path, the prompt
formatting, and the SNAC decode call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
from llama_cpp import Llama
from snac import SNAC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from benchmark import time_synthesis  # noqa: E402
from test_sentences import PROMPTS  # noqa: E402

# Placeholder — replace with whatever community GGUF quantization you find
# for Orpheus 3B (search Hugging Face; the canonical repo moves around).
GGUF_PATH = "orpheus-3b-0.1-ft-Q4_K_M.gguf"
SAMPLE_RATE = 24_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=list(PROMPTS), default="narration")
    parser.add_argument("--voice", default="tara")
    args = parser.parse_args()

    text = PROMPTS[args.category]

    llm = Llama(model_path=GGUF_PATH, n_gpu_layers=-1)  # -1 -> offload all layers to Metal
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

    with time_synthesis("orpheus", args.category) as bench:
        prompt = f"{args.voice}: {text}"
        output = llm(prompt, max_tokens=2048)
        # Orpheus emits SNAC codec tokens as text, not raw audio — parsing
        # those tokens out of `output` and feeding them to `snac.decode(...)`
        # is the step I could not verify against current docs. This is a
        # stub to fill in once you've confirmed the token format.
        raise NotImplementedError(
            "Parse SNAC codec tokens from `output` and call snac.decode(...) "
            "here — see README for what to check first."
        )

    out_path = f"output_{args.category}.wav"
    sf.write(out_path, audio, SAMPLE_RATE)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
