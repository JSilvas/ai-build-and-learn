"""Turn Cosmos 3 output into something a Flyte report can show.

Same three-step contract as topics/dreamerv3/replay.py, for the same reasons:

  encode()      frames -> H.264 mp4 bytes, via PyAV. aarch64 wheels exist for PyAV
                and do not reliably for imageio-ffmpeg, which is the conclusion the
                video-generation and Isaac Sim demos both reached independently.

  probe()       Say whether the clip actually shows anything. A black clip is a
                perfectly valid mp4 that a report will happily embed as a black
                rectangle, which is exactly how the Isaac Sim demo lost an
                afternoon. Mean luminance separates a real render from a decoder
                that silently produced nothing.

  video_html()  base64 the mp4 into a <video> tag. No JavaScript and no object-store
                round trip: the report is one self-contained HTML document, so it
                cannot break because a presigned URL points at the wrong rustfs.
                See topics/video-generation for the full diagnosis of that one.

The size ceiling is the reason `strip()` exists. base64 inflates by 4/3 and a Flyte
report is HTML held in memory, so a 720p clip at 189 frames is not something to
embed. Generate small (256p/480p), or fall back to the frame strip.
"""

from __future__ import annotations

import base64
import io
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Above this, embedding is a bad idea: the base64 payload is ~1.33x the raw bytes and
# the whole report has to be held in memory to render.
_EMBED_LIMIT_MB = 24


def _to_uint8_frames(video) -> list:
    """Normalize whatever the pipeline returned into a list of HxWx3 uint8 arrays.

    `Cosmos3OmniPipeline` returns PIL frames for output_type="pil" (the default),
    a [T, H, W, C] float array for "np", and a [T, C, H, W] tensor for "pt". The
    report path only ever wants uint8 RGB, so the conversion lives here rather than
    being repeated at every call site.
    """
    import numpy as np

    if video is None:
        return []
    # PIL frames.
    if isinstance(video, list):
        return [np.asarray(f.convert("RGB"), dtype=np.uint8) for f in video]

    arr = video
    if hasattr(arr, "detach"):  # torch tensor, [T, C, H, W]
        arr = arr.detach().float().cpu().numpy()
        if arr.ndim == 4 and arr.shape[1] in (1, 3):
            arr = arr.transpose(0, 2, 3, 1)
    arr = np.asarray(arr)
    # Float outputs are in [0, 1]; integer outputs are already [0, 255].
    if arr.dtype.kind == "f":
        arr = (arr.clip(0, 1) * 255).round()
    return [np.ascontiguousarray(f, dtype=np.uint8) for f in arr.astype("uint8")]


def encode(video, fps: int = 24, crf: int = 26) -> bytes:
    """Frames to H.264 mp4 bytes, small enough to base64 into a report."""
    frames = _to_uint8_frames(video)
    if not frames:
        return b""

    import av

    buf = io.BytesIO()
    h, w = frames[0].shape[:2]
    with av.open(buf, "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        # libx264 requires even dimensions for yuv420p. Cosmos 3 output sizes track
        # the input content and are not always divisible by 2, let alone 16, which is
        # also why NVIDIA's own example passes macro_block_size=1 to export_to_video.
        stream.width, stream.height = w - (w % 2), h - (h % 2)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        for frame in frames:
            cropped = frame[: stream.height, : stream.width]
            for pkt in stream.encode(av.VideoFrame.from_ndarray(cropped, format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return buf.getvalue()


def probe(mp4: bytes) -> str:
    """One line saying whether the clip shows anything, and whether it moves.

    Two failure modes, both of which decode without error and both of which have
    bitten this repo before:
      * every frame black   -> a renderer that produced nothing
      * every frame identical -> a "video" that is one still frame repeated, which is
        what a world model looks like when conditioning has silently swallowed the
        generation
    """
    if not mp4:
        return "no clip"
    try:
        import av
        import numpy as np

        with av.open(io.BytesIO(mp4)) as c:
            grays = [f.to_ndarray(format="gray").astype("float32") for f in c.decode(video=0)]
        if not grays:
            return "clip decodes to ZERO frames"
        means = [float(g.mean()) for g in grays]
        dark = sum(1 for m in means if m < 1.0)
        # Mean absolute difference between consecutive frames: ~0 means nothing moved.
        motion = (
            float(np.mean([np.abs(b - a).mean() for a, b in zip(grays, grays[1:])]))
            if len(grays) > 1
            else 0.0
        )
        return (
            f"{len(means)} frames, {grays[0].shape[1]}x{grays[0].shape[0]}, "
            f"luminance min {min(means):.1f} / mean {sum(means) / len(means):.1f} / "
            f"max {max(means):.1f}, {dark} black, inter-frame motion {motion:.2f}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"probe failed: {exc}"


def video_html(mp4: bytes, caption: str = "", max_width: int = 560) -> str:
    """base64 an mp4 into a self-contained <video> tag."""
    if not mp4:
        return '<p style="color:#888;font-family:monospace;">no clip</p>'
    mb = len(mp4) / 2**20
    if mb > _EMBED_LIMIT_MB:
        return (
            f'<p style="color:#888;font-family:monospace;">clip is {mb:.1f} MB, over '
            f"the {_EMBED_LIMIT_MB} MB embed limit; generate fewer frames or a lower "
            f"resolution</p>"
        )
    b64 = base64.b64encode(mp4).decode()
    cap = (
        f'<p style="color:#888;font-family:monospace;font-size:12px;margin:6px 0 0;">'
        f"{caption}</p>"
        if caption
        else ""
    )
    return (
        f'<div style="background:#0f0f23;padding:12px;border-radius:8px;">'
        f'<video src="data:video/mp4;base64,{b64}" controls autoplay loop muted '
        f'playsinline style="max-width:{max_width}px;width:100%;border:2px solid #333;'
        f'border-radius:4px;display:block;"></video>{cap}</div>'
    )


def strip(video, count: int = 6, width: int = 160) -> str:
    """A row of evenly-spaced still frames as inline PNGs.

    Always rendered alongside the clip, never instead of it. Two reasons: a strip
    survives a browser that will not autoplay an inline data: URI video, and laying
    the frames out in time is how you see whether the world model is predicting a
    sequence or repeating one frame, which a looping video hides.
    """
    frames = _to_uint8_frames(video)
    if not frames:
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
            f'<figure style="margin:0;">'
            f'<img src="data:image/png;base64,{b64}" style="width:{width}px;'
            f'border-radius:3px;display:block;"/>'
            f'<figcaption style="color:#888;font-family:monospace;font-size:10px;'
            f'text-align:center;padding-top:3px;">frame {idx}</figcaption></figure>'
        )
    return (
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;background:#0f0f23;'
        f'padding:12px;border-radius:8px;">{cells}</div>'
    )


def image_html(img, caption: str = "", width: int = 320) -> str:
    """A single still (the conditioning frame) as an inline PNG."""
    if img is None:
        return ""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    cap = (
        f'<p style="color:#888;font-family:monospace;font-size:12px;margin:6px 0 0;">'
        f"{caption}</p>"
        if caption
        else ""
    )
    return (
        f'<div style="background:#0f0f23;padding:12px;border-radius:8px;">'
        f'<img src="data:image/png;base64,{b64}" style="max-width:{width}px;width:100%;'
        f'border:2px solid #333;border-radius:4px;display:block;"/>{cap}</div>'
    )
