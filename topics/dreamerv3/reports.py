"""HTML for the Flyte report: run summary, the learning curve, and world-model losses.

Same palette and helpers as topics/rl-mujoco/reports.py and topics/isaac-sim/reports.py,
so a viewer moving between the demos is not re-learning the colours. Charts are
hand-rolled SVG rather than matplotlib: the image is already large, and one polyline
needs no dependency.
"""

from __future__ import annotations

_BG = "#0f0f23"
_PANEL = "#1a1a2e"
_TEXT = "#ccc"
_ACCENT = "#00b894"
_HILITE = "#fdcb6e"
_MUTED = "#888"

_TITLE = "DreamerV3 - learning a world model of MuJoCo"


def _table(rows: list[tuple[str, str]]) -> str:
    body = ""
    for i, (k, v) in enumerate(rows):
        border = "border-bottom:1px solid #333;" if i < len(rows) - 1 else ""
        body += (
            f'<tr><td style="padding:6px;{border}">{k}</td>'
            f'<td style="padding:6px;{border}color:{_ACCENT};">{v}</td></tr>'
        )
    return f'<table style="border-collapse:collapse;width:100%;">{body}</table>'


def _panel(title: str, inner: str) -> str:
    return (
        f'<div style="font-family:monospace;background:{_BG};color:{_TEXT};'
        f'padding:20px;border-radius:8px;">'
        f'<h3 style="color:{_ACCENT};margin-top:0;">{title}</h3>{inner}</div>'
    )


def _heading(text: str) -> str:
    return f'<h3 style="color:{_HILITE};font-family:monospace;">{text}</h3>'


def curve(
    points: list[tuple[float, float]],
    label: str,
    colour: str = _ACCENT,
    w: int = 760,
    h: int = 240,
) -> str:
    """(x, y) as an SVG polyline. x is environment steps, y whatever is being plotted."""
    if len(points) < 2:
        return f'<p style="color:{_MUTED};font-family:monospace;">{label}: not enough points yet</p>'

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    xspan = (x1 - x0) or 1.0
    yspan = (y1 - y0) or 1.0
    pad = 40
    pts = " ".join(
        f"{pad + (x - x0) / xspan * (w - 2 * pad):.1f},"
        f"{h - pad - (y - y0) / yspan * (h - 2 * pad):.1f}"
        for x, y in points
    )
    return (
        f'<div style="background:{_BG};padding:12px 16px;border-radius:8px;">'
        f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;background:{_PANEL};'
        f'border-radius:4px;" font-family="monospace">'
        f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>'
        f'<text x="{pad}" y="16" fill="{_HILITE}" font-size="12">{label}</text>'
        f'<text x="{pad}" y="{h - 12}" fill="{_MUTED}" font-size="11">'
        f'{y0:.3g} to {y1:.3g}</text>'
        f'<text x="{w - pad}" y="{h - 12}" fill="{_MUTED}" font-size="11" '
        f'text-anchor="end">{x1:,.0f} steps</text>'
        f"</svg></div>"
    )


def _losses_block(losses: dict[str, list[tuple[float, float]]]) -> str:
    """The world model's own losses, which is where you see it learning the dynamics.

    Named so the report explains itself: `dyn` and `rep` are the two sides of the KL
    between what the model predicted the next latent would be and what the encoder
    actually saw, `rew` and `con` are the reward and continuation heads, and
    `policy`/`value` are the actor and critic trained inside imagination.
    """
    meaning = {
        "dyn": "dynamics KL: prediction vs encoder",
        "rep": "representation KL: encoder vs prediction",
        "rew": "reward head",
        "con": "continuation head",
        "policy": "actor, trained in imagination",
        "value": "critic, trained in imagination",
    }
    blocks = ""
    for key, pts in losses.items():
        if len(pts) < 2:
            continue
        blocks += curve(pts, f"loss/{key} ({meaning.get(key, '')})", _HILITE) + "<br/>"
    return blocks or f'<p style="color:{_MUTED};">no loss points logged yet</p>'


def progress_html(task: str, step: int, total: int, score: list, losses: dict) -> str:
    pct = (100.0 * step / total) if total else 0.0
    rows = [
        ("Task", task),
        ("Progress", f"{step:,} / {total:,} env steps ({pct:.0f}%)"),
        ("Episodes scored", f"{len(score)}"),
        ("Latest score", f"{score[-1][1]:.1f}" if score else "no episode finished yet"),
    ]
    return (
        f"<h2>{_TITLE}</h2>"
        + _panel("Training", _table(rows))
        + "<br/>"
        + _heading("Episode return")
        + curve(score, "episode score")
    )


def final_html(
    task: str,
    steps: int,
    secs: float,
    score: list,
    losses: dict,
    params: str,
    tail: list[str],
    video: str = "",
    clip_probe: str = "",
) -> str:
    best = max((y for _, y in score), default=0.0)
    rows = [
        ("Task", task),
        ("Environment", "DeepMind Control Suite (dm_control on MuJoCo)"),
        ("Agent", "DreamerV3, model-based: policy trained inside imagined rollouts"),
        ("Env steps", f"{steps:,}"),
        ("Wall clock", f"{secs / 60:.1f} min"),
        ("Agent size", params or "n/a"),
        ("Episodes scored", f"{len(score)}"),
        ("Score, first", f"{score[0][1]:.1f}" if score else "n/a"),
        ("Score, final", f"{score[-1][1]:.1f}" if score else "n/a"),
        ("Score, best", f"{best:.1f}" if score else "n/a"),
    ]
    logs = "\n".join(tail[-25:]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    replay = ""
    if video:
        replay = (
            _heading("The trained policy")
            + video
            + (f'<p style="color:{_MUTED};font-family:monospace;font-size:12px;">'
               f"{clip_probe}</p>" if clip_probe else "")
            + "<br/>"
        )

    return (
        f"<h2>{_TITLE}</h2>"
        + _panel("Run summary", _table(rows))
        + "<br/>"
        + replay
        + _heading("Episode return")
        + curve(score, "episode score")
        + "<br/>"
        + _heading("World model")
        + _panel("Losses", _losses_block(losses))
        + "<br/>"
        + _panel(
            "Logs",
            f'<details><summary style="cursor:pointer;color:{_MUTED};">training tail</summary>'
            f'<pre style="font-size:11px;color:{_TEXT};background:{_PANEL};padding:12px;'
            f'border-radius:4px;overflow-x:auto;">{logs}</pre></details>',
        )
    )
