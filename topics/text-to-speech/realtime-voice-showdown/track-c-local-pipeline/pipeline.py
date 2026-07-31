"""Fixed-length record -> mlx-whisper STT -> mlx-lm LLM -> csm-mlx TTS -> playback.

Untested (written in a headless Linux sandbox — no MLX runtime, no mic).
The csm-mlx call in particular is the least certain part of this file; see
the README's "what to double check" section before assuming a bug here
over an upstream API drift.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx_whisper
import numpy as np
import sounddevice as sd
import soundfile as sf
from mlx_lm import generate as llm_generate
from mlx_lm import load as llm_load

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from latency import LatencyLog  # noqa: E402

SAMPLE_RATE = 16_000
RECORD_SECONDS = 5
WHISPER_MODEL = "mlx-community/whisper-base-mlx"
LLM_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
SYSTEM_PROMPT = "You are a concise voice assistant. Reply in one or two short sentences."


def record_audio(seconds: float, sample_rate: int) -> np.ndarray:
    print(f"Recording {seconds}s... speak now.")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def transcribe(audio: np.ndarray) -> str:
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL)
    return result["text"].strip()


def reply_text(llm_model, tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return llm_generate(llm_model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)


def synthesize(text: str) -> np.ndarray:
    """Speak `text` with CSM. This is the shakiest part of the pipeline —
    csm-mlx's API has been a moving target. Check the repo's own example
    script if this call shape is wrong for the version you installed.
    """
    from csm_mlx import CSM, csm_1b, generate as csm_generate  # local import: optional/heavy

    model = CSM(csm_1b())
    model.load_weights("csm-1b-mlx")  # placeholder ref — swap for the actual weights path/repo id
    audio = csm_generate(model, text=text, speaker=0, context=[], max_audio_length_ms=10_000)
    return np.asarray(audio)


def main() -> None:
    log = LatencyLog("track-c-local-pipeline")

    print("Loading Whisper + LLM (first run downloads weights)...")
    llm_model, tokenizer = llm_load(LLM_MODEL)

    audio_in = record_audio(RECORD_SECONDS, SAMPLE_RATE)
    log.mark("user_stopped_speaking")

    t0 = time.perf_counter()
    text_in = transcribe(audio_in)
    print(f"[stt] {time.perf_counter() - t0:.2f}s -> {text_in!r}")

    t0 = time.perf_counter()
    text_out = reply_text(llm_model, tokenizer, text_in)
    print(f"[llm] {time.perf_counter() - t0:.2f}s -> {text_out!r}")

    t0 = time.perf_counter()
    audio_out = synthesize(text_out)
    print(f"[tts] {time.perf_counter() - t0:.2f}s")

    log.mark("first_audio_out")
    log.finish(extra={"transcript_in": text_in, "reply_text": text_out})

    sf.write("last_reply.wav", audio_out, SAMPLE_RATE)
    sd.play(audio_out, samplerate=SAMPLE_RATE)
    sd.wait()


if __name__ == "__main__":
    main()
