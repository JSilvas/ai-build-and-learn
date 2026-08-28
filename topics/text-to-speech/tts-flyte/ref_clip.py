"""Turn a long recording into a reference clip. Flyte-free, numpy + soundfile only.

A stream recording is the opposite of what `refs/README.md` asks for: it is an hour
long, it has music and other people in it, and nobody read a fixed script. This module
is the bridge. It finds the best few seconds of continuous speech in a long file and
trims them out, so the thing handed to the cloners is still a clean 8-15s of one voice.

Deliberately light on dependencies: the studio app imports it to offer drag-in cloning,
and that app must stay a launcher (no torch, no TTS package). The scan is plain RMS
framing, which is enough to find continuous speech and reject dead air and clipping. It
is NOT diarization: it cannot tell your voice from a guest's, so preview the clip.

Quality thresholds live in `tts_core.RefVoice.warnings()`, not here, so there is one
source of truth for what makes a bad reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

HOP = 0.02          # 20ms analysis frames
SCAN_STRIDE = 0.5   # candidate window starts every 0.5s


@dataclass
class ClipInfo:
    seconds: float
    sample_rate: int
    channels: int
    format: str

    def describe(self) -> str:
        m, s = divmod(self.seconds, 60)
        return (f"{int(m)}m {s:04.1f}s · {self.sample_rate}Hz · "
                f"{self.channels}ch · {self.format}")


def probe(path: str | Path) -> ClipInfo:
    i = sf.info(str(path))
    return ClipInfo(float(i.duration), int(i.samplerate), int(i.channels), i.format)


def frame_rms(path: str | Path, start: float = 0.0, end: float | None = None
              ) -> tuple[np.ndarray, int]:
    """Per-frame RMS over [start, end), read in blocks.

    Blocked rather than one read because an hour of 48kHz stereo float32 is ~1.4GB
    decoded, and the app pod has 1Gi. The RMS track for the same hour is 180k floats.
    """
    info = sf.info(str(path))
    sr = info.samplerate
    hop = int(sr * HOP)
    end = min(end if end is not None else info.duration, info.duration)
    out: list[np.ndarray] = []
    with sf.SoundFile(str(path)) as f:
        f.seek(int(start * sr))
        want = max(int((end - start) * sr), 0)
        read = 0
        while read < want:
            block = f.read(min(hop * 2048, want - read), dtype="float32", always_2d=True)
            if not len(block):
                break
            read += len(block)
            mono = block.mean(axis=1)
            n = (len(mono) // hop) * hop
            if n:
                out.append(np.sqrt((mono[:n].reshape(-1, hop) ** 2).mean(axis=1)))
    rms = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    return rms.astype(np.float32), sr


def _longest_gap(unvoiced: np.ndarray) -> int:
    """Longest run of unvoiced frames, in frames. Vectorized: the run boundaries are
    where the boolean track changes, so the runs are the diffs of those indices."""
    if not unvoiced.any():
        return 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], unvoiced.view(np.int8), [0]))))
    return int(np.max(idx[1::2] - idx[0::2])) if len(idx) >= 2 else 0


def best_window(path: str | Path, secs: float = 12.0, skip_start: float = 0.0,
                skip_end: float = 0.0, top: int = 1) -> list[tuple[float, float]]:
    """The best (start_seconds, score) windows of length `secs`, best first.

    Score rewards continuous speech and punishes the two things that ruin a reference:
    dead air inside the window (a pause between sentences is reference you paid for and
    cannot use) and a peak that is either clipped or near-silent. Candidate starts are
    strided by 0.5s, and winners are kept non-overlapping so `top > 1` returns genuinely
    different passages instead of shifted copies of one sentence.
    """
    info = sf.info(str(path))
    end = max(info.duration - skip_end, 0.0)
    rms, _ = frame_rms(path, skip_start, end)
    w = int(secs / HOP)
    if len(rms) <= w:
        return [(skip_start, 0.0)]

    # Noise floor from the quiet quartile: robust on a recording that is mostly speech.
    floor = max(float(np.percentile(rms, 25)) * 2.5, 0.005)
    stride = max(int(SCAN_STRIDE / HOP), 1)
    starts = np.arange(0, len(rms) - w, stride)

    scored: list[tuple[float, float]] = []
    for i in starts:
        win = rms[i:i + w]
        voiced = win > floor
        peak = float(win.max())
        gap_s = _longest_gap(~voiced) * HOP
        # A hot RMS frame is not proof of clipping (RMS < peak), so this term is a
        # soft preference for a healthy level, not a clipping detector. The real
        # clipping check runs on the trimmed samples, in RefVoice.warnings().
        level_pen = 0.0 if 0.02 <= peak <= 0.35 else (0.4 if peak <= 0.5 else 0.8)
        scored.append((skip_start + i * HOP,
                       float(voiced.mean()) - 0.35 * min(gap_s, 1.0) - level_pen))

    scored.sort(key=lambda t: -t[1])
    picked: list[tuple[float, float]] = []
    for t0, sc in scored:
        if all(abs(t0 - p) >= secs for p, _ in picked):
            picked.append((t0, sc))
        if len(picked) == top:
            break
    return picked


def trim(path: str | Path, start: float, secs: float, out_path: str | Path
         ) -> tuple[str, float, int]:
    """Write [start, start+secs) as a mono 16-bit wav at the SOURCE rate.

    No resampling on purpose: `RefVoice.at()` resamples per model (Dia wants 44.1k, CSM
    24k), so resampling here would only throw away detail twice.
    """
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        f.seek(min(int(start * sr), len(f)))
        block = f.read(int(secs * sr), dtype="float32", always_2d=True)
    mono = block.mean(axis=1) if block.size else np.zeros(1, dtype=np.float32)
    sf.write(str(out_path), mono, sr, subtype="PCM_16")
    return str(out_path), float(np.abs(mono).max()) if mono.size else 0.0, int(sr)
