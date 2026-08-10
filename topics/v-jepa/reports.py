"""HTML for the Flyte reports.

Same palette and helpers as topics/cosmos/reports.py, topics/rl-mujoco/reports.py and
topics/isaac-sim/reports.py, so a viewer moving between the world-model demos is not
re-learning the colours. Everything is inline HTML and inline data: URIs; no
JavaScript, no external assets, no object-store round trip.
"""

from __future__ import annotations

import html

_BG = "#0f0f23"
_PANEL = "#1a1a2e"
_TEXT = "#ccc"
_ACCENT = "#00b894"
_HILITE = "#fdcb6e"
_MUTED = "#888"
_WARN = "#e17055"

_TITLE = "V-JEPA 2 - a world model that predicts representations, not pixels"


def _table(rows: list[tuple[str, str]]) -> str:
    body = ""
    for i, (k, v) in enumerate(rows):
        border = "border-bottom:1px solid #333;" if i < len(rows) - 1 else ""
        body += (
            f'<tr><td style="padding:6px;{border}white-space:nowrap;">{k}</td>'
            f'<td style="padding:6px;{border}color:{_ACCENT};">{v}</td></tr>'
        )
    return f'<table style="border-collapse:collapse;width:100%;">{body}</table>'


def _panel(title: str, inner: str) -> str:
    return (
        f'<div style="font-family:monospace;background:{_BG};color:{_TEXT};padding:20px;'
        f'border-radius:8px;"><h3 style="color:{_ACCENT};margin-top:0;">{title}</h3>{inner}</div>'
    )


def heading(text: str) -> str:
    return f'<h3 style="color:{_HILITE};font-family:monospace;">{text}</h3>'


def note(text: str) -> str:
    return (
        f'<p style="color:{_MUTED};font-family:monospace;font-size:12px;line-height:1.55;">'
        f"{text}</p>"
    )


def details(summary: str, body: str) -> str:
    return (
        f'<details><summary style="cursor:pointer;color:{_MUTED};font-family:monospace;">'
        f"{summary}</summary>"
        f'<pre style="font-size:11px;color:{_TEXT};background:{_PANEL};padding:12px;'
        f'border-radius:4px;overflow-x:auto;white-space:pre-wrap;">{html.escape(body)}</pre>'
        f"</details>"
    )


def progress_html(stage: str, detail: str, rows: list[tuple[str, str]]) -> str:
    """Painted while the pod is still working, so the report is never blank.

    A pod that spends its first minutes pulling weights and clips while its report
    stays empty is indistinguishable from a hung pod. Same lesson as the Cosmos and
    DreamerV3 tasks next door.
    """
    return f"<h2>{_TITLE}</h2>" + _panel(stage, _table(rows) + note(detail))


def side_by_side(cells: list[tuple[str, str]], basis: int = 380) -> str:
    """Lay blocks out in a responsive row.

    flex-wrap rather than a grid: Flyte reports render at a width you do not control,
    and two side-by-side clips have to be allowed to become two stacked clips.
    """
    inner = ""
    for label, body in cells:
        inner += (
            f'<div style="flex:1 1 {basis}px;min-width:300px;">'
            f'<h4 style="color:{_HILITE};font-family:monospace;margin:0 0 8px;">{label}</h4>'
            f"{body}</div>"
        )
    return f'<div style="display:flex;gap:16px;flex-wrap:wrap;">{inner}</div>'


def score_table(rows: list[tuple[str, dict]], highlight: str | None = None) -> str:
    """The prediction scorecard: one row per method, chance floors included.

    Columns are in the order you should read them, worst metric first. `cos` is the
    intuitive one and the least trustworthy: it is not comparable between masks, since
    it depends on how many tokens were hidden and which. `top-1` and the median
    displacements are measured against candidates drawn from the masked tokens only.
    The last column, `time localised`, is the one that settles the argument: the
    fraction of the chance-level temporal error the prediction removed, so 1.00 means
    it found the right moment and 0.00 means it did no better than shuffling.
    """
    head = (
        f'<tr style="color:{_HILITE};">'
        + "".join(
            f'<th style="padding:6px 10px;text-align:left;border-bottom:1px solid #444;">{c}</th>'
            for c in ("method", "cosine", "top-1", "median dt", "median dh", "median dw",
                      "time localised")
        )
        + "</tr>"
    )
    body = ""
    for name, s in rows:
        strong = highlight is not None and name == highlight
        colour = _ACCENT if strong else _TEXT
        weight = "bold" if strong else "normal"
        cells = "".join(
            f'<td style="padding:6px 10px;border-bottom:1px solid #2a2a3e;color:{colour};'
            f'font-weight:{weight};">{v}</td>'
            for v in (
                name,
                f"{s['cos']:.3f}",
                f"{s['top1']:.1%}",
                f"{s['dt']:.1f}",
                f"{s['dh']:.1f}",
                f"{s['dw']:.1f}",
                f"{s['loc']:.2f}" if "loc" in s else "-",
            )
        )
        body += f"<tr>{cells}</tr>"
    return (
        f'<table style="border-collapse:collapse;font-family:monospace;font-size:12px;'
        f'width:100%;color:{_TEXT};">{head}{body}</table>'
    )


def verdict(text: str, good: bool = True) -> str:
    colour = _ACCENT if good else _WARN
    return (
        f'<div style="font-family:monospace;border-left:3px solid {colour};background:{_PANEL};'
        f'padding:12px 16px;margin:12px 0;color:{_TEXT};font-size:13px;line-height:1.6;">'
        f"{text}</div>"
    )


def final_html(subtitle: str, rows: list[tuple[str, str]], body: str, explainer: str = "") -> str:
    out = f"<h2>{_TITLE}</h2>" + _panel(subtitle, _table(rows))
    if explainer:
        out += _panel("What to look for", note(explainer))
    return out + "<br/>" + body


# ── Explainers ──────────────────────────────────────────────────────────────────
#
# Kept in the report rather than only in the README, because the report is the artifact
# people scroll past on the stream and a heatmap with no claim attached to it is just a
# picture.

INPAINT_EXPLAINER = (
    "V-JEPA 2 has no decoder. The predictor emits 1024-dimensional vectors, and there "
    "is no head in any released checkpoint that turns one back into a pixel, so "
    "'show me what it predicted' is not a screenshot anyone can take. What the video "
    "shows instead is the model's ACTUAL input with the masked patches blacked out, "
    "and next to it a per-patch measurement of how close the predicted vector landed "
    "to the true one. "
    "The argument is the two masks. A TUBE mask removes a spatial block across the "
    "whole clip, which is the shape V-JEPA 2 was pretrained on. A FUTURE mask removes "
    "everything after a moment in time, which it never saw: pretraining masks always "
    "span all of time, so no token was ever predicted from a strictly earlier one. "
    "Read the median dt column. Under the trained mask the predictor's nearest "
    "retrieved token is at the RIGHT MOMENT; asked to extrapolate forward it lands "
    "several tubelets away, which is a representation of roughly the right scene at "
    "the wrong time. That gap is the honest boundary of what this checkpoint is, and "
    "it is what the action-conditioned V-JEPA 2-AC post-train exists to close."
)

PROBE_EXPLAINER = (
    "Nothing here is fine-tuned. The encoder is frozen, every clip becomes one mean-"
    "pooled vector, and the only trained parameters are a single linear layer. So the "
    "accuracy is a statement about the representation, not about the classifier. "
    "Read it against the two floors rather than on its own: chance is 20% for five "
    "balanced classes, and the pixel baseline is the same probe trained on a "
    "downsampled space-time thumbnail of the same clips, which is how much of the "
    "score you get for free because the classes look different. The retrieval row is "
    "the version with no training at all: are clips of the same action each other's "
    "nearest neighbours in a space nobody supervised?"
)

SCALE_EXPLAINER = (
    "Same clips, same masks, same probe, three encoder sizes. Worth separating the two "
    "questions this answers. Probe accuracy asks whether a bigger self-supervised "
    "encoder carries more linearly accessible semantics. The inpainting scores ask "
    "whether its predictor is better at the job it was trained on. These do not have "
    "to move together, and the interesting outcome is if they don't."
)

GEOMETRY_NOTE = (
    "Every similarity in this report is computed on CENTERED features (the per-clip "
    "mean token subtracted before normalising). This is not a stylistic choice. "
    "V-JEPA 2's token cloud sits in a narrow cone: the mean cosine between two random "
    "patches of the same clip is measured below, and raw it is dominated by a "
    "component every token shares. Centered, random pairs sit at ~0.00 while adjacent "
    "patches sit around 0.33, which is a geometry you can actually measure against."
)
