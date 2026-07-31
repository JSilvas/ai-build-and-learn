"""Shared timing helper: wall-clock synth time, output audio duration, and
real-time factor (RTF = audio seconds / synth seconds; >1.0 is faster than
real-time).

Usage:

    from benchmark import time_synthesis

    with time_synthesis("kokoro", "narration") as bench:
        audio = my_model.synth(text)
        bench.audio_seconds = len(audio) / sample_rate
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results.jsonl"


@dataclass
class Bench:
    model: str
    prompt_key: str
    audio_seconds: float | None = None
    _start: float = 0.0
    _elapsed: float | None = None

    def rtf(self) -> float | None:
        if self.audio_seconds is None or self._elapsed is None or self._elapsed == 0:
            return None
        return self.audio_seconds / self._elapsed

    def finish(self, extra: dict | None = None) -> None:
        rtf = self.rtf()
        if rtf is not None:
            print(
                f"[{self.model}/{self.prompt_key}] synth={self._elapsed:.2f}s "
                f"audio={self.audio_seconds:.2f}s RTF={rtf:.2f}x"
            )
        else:
            print(
                f"[{self.model}/{self.prompt_key}] synth={self._elapsed:.2f}s "
                f"(set bench.audio_seconds for an RTF reading)"
            )

        record = {
            "model": self.model,
            "prompt": self.prompt_key,
            "synth_seconds": self._elapsed,
            "audio_seconds": self.audio_seconds,
            "rtf": rtf,
            "timestamp": time.time(),
            **(extra or {}),
        }
        with RESULTS_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")


@contextmanager
def time_synthesis(model: str, prompt_key: str):
    bench = Bench(model=model, prompt_key=prompt_key)
    bench._start = time.perf_counter()
    try:
        yield bench
    finally:
        bench._elapsed = time.perf_counter() - bench._start
        bench.finish()
