"""Turning V-JEPA 2's inputs and measurements into something a Flyte report can show.

── Why there is no "generated video" here ──────────────────────────────────────
Cosmos and the video-generation demos base64 a clip the model produced. V-JEPA 2 has
no decoder: the predictor emits 1024-dimensional vectors and there is no head anywhere
in the released checkpoints that turns one back into pixels. Anything claiming to show
"what V-JEPA 2 predicted" as an image is showing you a projection of a vector, not a
prediction.

So the video in these reports is honest about what it is:

  masked_video()   The model's ACTUAL input with the masked tokens blacked out. Not a
                   visualisation, the literal thing the encoder was given. This is what
                   makes the inpainting task legible: you can see the hole.
  heat_video()     A per-patch MEASUREMENT painted onto those same pixels. Always a
                   number we computed (prediction quality), never a claim about
                   semantics.
  clip_video()     The source clip, for the retrieval results, where the point is
                   whether two clips are the same kind of event.

Everything composites onto `clips.shown_pixels()` output, i.e. the post-crop tensor the
patch embedding actually consumed, so patch (h, w) is pixels [16h:16h+16, 16w:16w+16]
and the overlay cannot drift out of alignment with the grid.

Same three-step encode/probe/embed contract as topics/cosmos/media.py, for the same
reasons: PyAV because aarch64 wheels exist, a luminance probe because a black clip is a
perfectly valid mp4 that a report will happily embed as a black rectangle, and base64
into a <video> tag so the report is one self-contained document with no object-store
round trip that can point at the wrong rustfs.
"""

from __future__ import annotations

import base64
import io
import logging

import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# base64 inflates by 4/3 and a Flyte report is HTML held in memory to render.
_EMBED_LIMIT_MB = 24


# ── mp4 ─────────────────────────────────────────────────────────────────────────


def encode_mp4(frames, fps: int = 12, crf: int = 24) -> bytes:
    """[T, H, W, 3] uint8 -> H.264 mp4 bytes."""
    frames = np.asarray(frames)
    if frames.size == 0:
        return b""
    import av

    buf = io.BytesIO()
    h, w = frames.shape[1:3]
    with av.open(buf, "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        # libx264 needs even dimensions for yuv420p; 256 already is, but overlays get
        # built at other sizes and a stray odd height is an unhelpful way to fail.
        stream.width, stream.height = w - (w % 2), h - (h % 2)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        for frame in frames:
            cropped = np.ascontiguousarray(frame[: stream.height, : stream.width])
            for pkt in stream.encode(av.VideoFrame.from_ndarray(cropped, format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return buf.getvalue()


def probe(mp4: bytes) -> str:
    """One line saying whether the clip shows anything, and whether it moves.

    Two failure modes that both decode without error and have both bitten this repo:
    every frame black (a renderer that produced nothing), and every frame identical (a
    "video" that is one still repeated).
    """
    if not mp4:
        return "no clip"
    try:
        import av

        with av.open(io.BytesIO(mp4)) as c:
            grays = [f.to_ndarray(format="gray").astype("float32") for f in c.decode(video=0)]
        if not grays:
            return "clip decodes to ZERO frames"
        means = [float(g.mean()) for g in grays]
        motion = (
            float(np.mean([np.abs(b - a).mean() for a, b in zip(grays, grays[1:])]))
            if len(grays) > 1
            else 0.0
        )
        return (
            f"{len(means)} frames, {grays[0].shape[1]}x{grays[0].shape[0]}, "
            f"luminance min {min(means):.1f} / mean {sum(means) / len(means):.1f} / "
            f"max {max(means):.1f}, {sum(1 for m in means if m < 1.0)} black, "
            f"inter-frame motion {motion:.2f}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"probe failed: {exc}"


def video_html(mp4: bytes, caption: str = "", max_width: int = 420) -> str:
    """base64 an mp4 into a self-contained <video> tag. No JS, no external assets."""
    if not mp4:
        return '<p style="color:#888;font-family:monospace;">no clip</p>'
    mb = len(mp4) / 2**20
    if mb > _EMBED_LIMIT_MB:
        return (
            f'<p style="color:#888;font-family:monospace;">clip is {mb:.1f} MB, over the '
            f"{_EMBED_LIMIT_MB} MB embed limit</p>"
        )
    b64 = base64.b64encode(mp4).decode()
    cap = (
        f'<p style="color:#888;font-family:monospace;font-size:12px;margin:6px 0 0;">{caption}</p>'
        if caption
        else ""
    )
    return (
        f'<div style="background:#0f0f23;padding:12px;border-radius:8px;">'
        f'<video src="data:video/mp4;base64,{b64}" controls autoplay loop muted playsinline '
        f'style="max-width:{max_width}px;width:100%;border:2px solid #333;border-radius:4px;'
        f'display:block;"></video>{cap}</div>'
    )


def strip(frames, count: int = 6, width: int = 130) -> str:
    """A row of evenly spaced stills as inline PNGs.

    Always alongside the clip, never instead of it: a strip survives a browser that
    will not autoplay a data: URI video, and laying frames out in time is how you see
    whether something is changing, which a short loop hides.
    """
    frames = np.asarray(frames)
    if frames.size == 0:
        return ""
    from PIL import Image

    n = len(frames)
    picks = [round(i * (n - 1) / max(count - 1, 1)) for i in range(min(count, n))]
    cells = ""
    for idx in picks:
        img = Image.fromarray(frames[idx])
        img.thumbnail((width, width * 2))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        cells += (
            f'<figure style="margin:0;"><img src="data:image/png;base64,{b64}" '
            f'style="width:{width}px;border-radius:3px;display:block;"/>'
            f'<figcaption style="color:#888;font-family:monospace;font-size:10px;'
            f'text-align:center;padding-top:3px;">frame {idx}</figcaption></figure>'
        )
    return (
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;background:#0f0f23;padding:12px;'
        f'border-radius:8px;">{cells}</div>'
    )


# ── overlays ────────────────────────────────────────────────────────────────────


def _patch_size(shown: np.ndarray, grid: int) -> int:
    return shown.shape[1] // grid


def masked_video(shown: np.ndarray, mask3d, tubelet: int = 2) -> np.ndarray:
    """The model's input with the masked tokens blacked out.

    Frame f belongs to tubelet f // tubelet_size, and patch (h, w) of that tubelet is
    the 16x16 pixel block at [16h:16h+16, 16w:16w+16]. Blanking exactly those blocks is
    what the encoder's context actually was.
    """
    mask = np.asarray(mask3d)
    out = shown.copy()
    grid = mask.shape[1]
    ps = _patch_size(shown, grid)
    for f in range(len(out)):
        t = min(f // tubelet, mask.shape[0] - 1)
        hs, ws = np.nonzero(mask[t])
        for h, w in zip(hs, ws):
            out[f, h * ps : (h + 1) * ps, w * ps : (w + 1) * ps] = 20
    return out


def _hot(x: np.ndarray) -> np.ndarray:
    """Blue (low) -> orange (high). NaN renders as flat grey, i.e. 'not measured'."""
    nan = np.isnan(x)
    s = np.nan_to_num(x, nan=0.0)
    rgb = np.stack(
        [
            np.clip(s * 1.7, 0, 1),
            np.clip(s * 1.2 - 0.25, 0, 1) * 0.8,
            np.clip(1 - s * 1.9, 0, 1) * 0.95,
        ],
        -1,
    )
    rgb[nan] = 0.28
    return rgb


def heat_video(
    shown: np.ndarray, field, tubelet: int = 2, alpha: float = 0.62,
    lo: float | None = None, hi: float | None = None,
) -> np.ndarray:
    """Paint a per-patch [T, G, G] measurement onto the pixels the model saw.

    Scaled between the 5th and 95th percentile of the FINITE values so the colours use
    their range; NaN (an unmasked token, nothing to score) stays grey.
    """
    from PIL import Image

    field = np.asarray(field, dtype=np.float32)
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return shown.copy()
    lo = float(np.percentile(finite, 5)) if lo is None else lo
    hi = float(np.percentile(finite, 95)) if hi is None else hi
    norm = (field - lo) / (hi - lo + 1e-6)
    norm = np.where(np.isfinite(field), np.clip(norm, 0, 1), np.nan)

    h, w = shown.shape[1:3]
    out = np.empty_like(shown)
    for f in range(len(shown)):
        t = min(f // tubelet, field.shape[0] - 1)
        heat = (_hot(norm[t]) * 255).astype(np.uint8)
        heat = np.asarray(Image.fromarray(heat).resize((w, h), Image.NEAREST))
        out[f] = ((1 - alpha) * shown[f] + alpha * heat).astype(np.uint8)
    return out


def side_by_side_video(left: np.ndarray, right: np.ndarray, gap: int = 6) -> np.ndarray:
    """Two aligned clips in one mp4, so a viewer cannot compare the wrong frames."""
    n = min(len(left), len(right))
    sep = np.full((n, left.shape[1], gap, 3), 15, dtype=np.uint8)
    return np.concatenate([left[:n], sep, right[:n]], axis=2)


# ── charts ──────────────────────────────────────────────────────────────────────


def _fig_html(fig, width: int = 620) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#0f0f23")
    import matplotlib.pyplot as plt

    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<img src="data:image/png;base64,{b64}" style="max-width:{width}px;width:100%;'
        f'border-radius:6px;display:block;margin:8px 0;"/>'
    )


def _axes(title: str, xlabel: str, ylabel: str, size=(6.4, 3.4)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=size, facecolor="#0f0f23")
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="#fdcb6e", fontsize=11)
    ax.set_xlabel(xlabel, color="#ccc", fontsize=9)
    ax.set_ylabel(ylabel, color="#ccc", fontsize=9)
    ax.tick_params(colors="#888", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#333")
    ax.grid(alpha=0.15, color="#888")
    return fig, ax


def horizon_chart(
    series: dict[str, list[tuple[int, float]]],
    hlines: dict[str, float] | None = None,
) -> str:
    """Prediction quality against how far ahead it was asked to predict.

    `hlines` carries the two references that make the curve readable: the in-
    distribution tube-mask score (the ceiling this predictor reaches when asked the
    question it was trained on) and the shuffled chance floor.
    """
    fig, ax = _axes(
        "Latent prediction quality vs horizon",
        "tubelets ahead of the last visible frame (1 tubelet = 2 frames)",
        "mean centered cosine to truth",
    )
    colors = ["#00b894", "#fdcb6e", "#74b9ff", "#e17055"]
    for i, (label, pts) in enumerate(series.items()):
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.8, color=colors[i % len(colors)], label=label)
    for i, (label, y) in enumerate((hlines or {}).items()):
        ax.axhline(y, ls="--", lw=1.4, color=["#888", "#74b9ff", "#e17055"][i % 3], label=label)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


def bar_chart(title: str, groups: list[str], series: dict[str, list[float]],
              ylabel: str, floor: float | None = None, floor_label: str = "chance") -> str:
    """Grouped bars: the metric for each model, with the floor drawn across it."""
    fig, ax = _axes(title, "", ylabel, size=(6.4, 3.2))
    colors = ["#00b894", "#fdcb6e", "#74b9ff", "#e17055"]
    n = max(len(series), 1)
    width = 0.8 / n
    xs = np.arange(len(groups))
    for i, (label, vals) in enumerate(series.items()):
        pos = xs - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width * 0.9, label=label, color=colors[i % len(colors)])
        for x, v in zip(pos, vals):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", color="#ccc", fontsize=7)
    if floor is not None:
        ax.axhline(floor, ls="--", lw=1.4, color="#888", label=floor_label)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=8)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


def confusion_chart(cm: np.ndarray, labels: list[str], title: str) -> str:
    """Confusion matrix. Which classes it confuses is more informative than the total."""
    fig, ax = _axes(title, "predicted", "true", size=(4.6, 4.0))
    ax.grid(False)
    ax.imshow(cm, cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    hi = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="#fff" if cm[i, j] < hi * 0.6 else "#000")
    return _fig_html(fig, width=380)
