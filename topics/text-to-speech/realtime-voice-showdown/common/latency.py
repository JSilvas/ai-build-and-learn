"""Shared timing/logging helper so all three tracks report comparable numbers.

Usage:

    from latency import LatencyLog

    log = LatencyLog("track-a-openai-realtime")
    log.mark("user_stopped_speaking")
    ...
    log.mark("first_audio_out")
    log.finish()  # prints TTFA and appends a JSON line to results.jsonl
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results.jsonl"

# The two marks every track should call at minimum. Extra marks are fine —
# they just get included in the log line for reference.
START_MARK = "user_stopped_speaking"
END_MARK = "first_audio_out"


@dataclass
class LatencyLog:
    track: str
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str) -> float:
        t = time.perf_counter()
        self.marks[name] = t
        return t

    def ttfa_ms(self) -> float | None:
        if START_MARK not in self.marks or END_MARK not in self.marks:
            return None
        return (self.marks[END_MARK] - self.marks[START_MARK]) * 1000

    def finish(self, extra: dict | None = None) -> None:
        ttfa = self.ttfa_ms()
        if ttfa is not None:
            print(f"[{self.track}] time-to-first-audio: {ttfa:.0f} ms")
        else:
            print(
                f"[{self.track}] incomplete run — need both "
                f"'{START_MARK}' and '{END_MARK}' marks for a TTFA reading"
            )

        record = {
            "track": self.track,
            "ttfa_ms": ttfa,
            "marks": sorted(self.marks.keys()),
            "timestamp": time.time(),
            **(extra or {}),
        }
        with RESULTS_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
