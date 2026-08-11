"""Getting video into V-JEPA 2, and getting back the exact pixels it saw.

That second half is not a detail. `AutoVideoProcessor` resizes the shortest edge to
256*256/224 = 292 and then centre-crops to 256, so a 480x270 Kinetics clip loses most
of its width. Every per-patch overlay in this demo is a 16x16 grid painted onto the
frame, and if you paint it onto a naive resize of the ORIGINAL clip instead of onto the
crop, the overlay is silently misaligned by tens of pixels. The first version of this
demo did exactly that, and the resulting heatmaps looked plausible and meant nothing.

So `shown_pixels()` inverts the normalisation and hands back precisely the tensor the
patch embedding consumed. Overlays are composited on that, never on the source clip.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def list_clips(repo: str) -> list[tuple[str, str, str]]:
    """Every clip in the dataset as (split, label, path), sorted for determinism."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    out = []
    for f in sorted(files):
        if not f.endswith(".mp4"):
            continue
        parts = f.split("/")
        if len(parts) >= 3:
            out.append((parts[0], parts[1], f))
    return out


def labels_of(clips: list[tuple[str, str, str]]) -> list[str]:
    return sorted({label for _, label, _ in clips})


def fetch(repo: str, name: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, name, repo_type="dataset")


def decode(path: str, num_frames: int) -> np.ndarray:
    """Decode an mp4 and sample `num_frames` evenly across its whole duration.

    Even sampling rather than a contiguous window on purpose: these clips are 10s and
    the model takes 32-64 frames, so a contiguous window would see well under a second
    and miss the action the label refers to.
    """
    import av

    with av.open(path) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if not frames:
        raise ValueError(f"{path} decoded to zero frames")
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    return np.stack([frames[i] for i in idx])


def preprocess(processor, frames: np.ndarray):
    """Frames -> the [1, T, 3, 256, 256] normalised tensor the model wants."""
    return processor(list(frames), return_tensors="pt")["pixel_values_videos"]


def shown_pixels(processor, pixel_values) -> np.ndarray:
    """Invert the normalisation: [1, T, 3, H, W] -> [T, H, W, 3] uint8.

    This is the ground truth for every overlay in the demo. Patch (h, w) of the model's
    grid is exactly pixels [16h:16h+16, 16w:16w+16] of these frames, and of no others.
    """
    import torch

    mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
    px = (pixel_values[0].float().cpu() * std + mean).clamp(0, 1)
    return (px.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


def load_clip(processor, repo: str, name: str, num_frames: int):
    """Fetch, decode and preprocess one clip. Returns (pixel_values, shown_frames)."""
    frames = decode(fetch(repo, name), num_frames)
    pv = preprocess(processor, frames)
    return pv, shown_pixels(processor, pv)


def pick(clips: list[tuple[str, str, str]], label: str, split: str = "val") -> str:
    """First clip of a class, deterministically. `label` may also be a full path."""
    if label.endswith(".mp4"):
        return label
    for sp, lb, path in clips:
        if lb == label and sp == split:
            return path
    for sp, lb, path in clips:
        if lb == label:
            return path
    raise ValueError(f"no clip for {label!r}; have {sorted({c[1] for c in clips})}")


def pixel_baseline(frames: np.ndarray) -> np.ndarray:
    """A deliberately dumb clip descriptor: a 4x12x12 greyscale space-time thumbnail.

    The point of this is to make the probe number in `probe` mean something. "78% on
    5-way action recognition" is unreadable on its own; "78% where a 576-dimensional
    pile of downsampled pixels gets 40% and chance is 20%" is a claim about the
    representation rather than about the difficulty of the dataset.
    """
    import torch
    import torch.nn.functional as F

    grey = torch.tensor(frames.mean(-1), dtype=torch.float32)[None, None]
    return F.adaptive_avg_pool3d(grey, (4, 12, 12)).flatten().numpy()
