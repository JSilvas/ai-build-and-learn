"""
Generate an interactive Plotly progress chart from results.tsv.

Mirrors karpathy's progress.png but interactive — hover for details,
zoom into regions, dark theme, memory usage on secondary axis.

Usage:
    python plot_progress.py                          # reads upstream/results.tsv
    python plot_progress.py --file path/to/results.tsv
    python plot_progress.py --title "Gemma 4 Clean Run"
    python plot_progress.py --out progress.html      # default
    python plot_progress.py --png progress.png       # also save static PNG
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import plotly.graph_objects as go

DEFAULT_TSV = Path(__file__).parent.parent / "upstream" / "results.tsv"

STATUS_COLORS = {
    "keep": "#1a7f37",
    "baseline": "#6c757d",
    "discard": "#d4d4d4",
    "crash": "#d97706",
}

STATUS_SYMBOLS = {
    "keep": "circle",
    "baseline": "diamond",
    "discard": "circle-open",
    "crash": "x",
}


def load_results(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            try:
                val_bpb = float(row["val_bpb"]) if row["val_bpb"] != "0.000000" else None
            except (ValueError, KeyError):
                val_bpb = None
            try:
                memory_gb = float(row["memory_gb"]) if row["memory_gb"] != "0.0" else None
            except (ValueError, KeyError):
                memory_gb = None
            rows.append({
                "experiment": i,
                "commit": row.get("commit", ""),
                "val_bpb": val_bpb,
                "memory_gb": memory_gb,
                "status": row.get("status", "").strip(),
                "description": row.get("description", "").strip(),
            })
    return rows


def build_chart(rows: list[dict], title: str = "", show_vram: bool = True) -> go.Figure:
    keeps = [r for r in rows if r["status"] in ("keep", "baseline") and r["val_bpb"] is not None]
    discards = [r for r in rows if r["status"] == "discard" and r["val_bpb"] is not None]
    crashes = [r for r in rows if r["status"] == "crash" or r["val_bpb"] is None]

    num_experiments = len(rows)
    num_keeps = len(keeps)

    if not title:
        title = f"Autoresearch Progress: {num_experiments} Experiments, {num_keeps} Kept Improvements"

    fig = go.Figure()

    # --- Running best line ---
    best_so_far = float("inf")
    best_x, best_y = [], []
    for r in rows:
        if r["val_bpb"] is not None and r["status"] in ("keep", "baseline"):
            if r["val_bpb"] < best_so_far:
                best_so_far = r["val_bpb"]
            best_x.append(r["experiment"])
            best_y.append(best_so_far)

    fig.add_trace(go.Scatter(
        x=best_x, y=best_y,
        mode="lines",
        name="Running best",
        line=dict(color="#1a7f37", width=2.5, shape="hv"),
        hoverinfo="skip",
    ))

    # --- Fill under running best ---
    if best_x:
        fig.add_trace(go.Scatter(
            x=best_x + best_x[::-1],
            y=best_y + [max(best_y) * 1.01] * len(best_y),
            fill="toself",
            fillcolor="rgba(26, 127, 55, 0.07)",
            line=dict(width=0),
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
        ))

    # --- Discards (gray, background) ---
    if discards:
        fig.add_trace(go.Scatter(
            x=[r["experiment"] for r in discards],
            y=[r["val_bpb"] for r in discards],
            mode="markers",
            name="Discarded",
            marker=dict(color="#8b949e", size=10, symbol="circle-open", line=dict(width=2, color="#8b949e")),
            text=[r["description"][:80] for r in discards],
            hovertemplate="<b>Experiment %{x}</b><br>val_bpb: %{y:.6f}<br>%{text}<extra>discarded</extra>",
        ))

    # --- Crashes (red x) ---
    if crashes:
        fig.add_trace(go.Scatter(
            x=[r["experiment"] for r in crashes],
            y=[0] * len(crashes),
            mode="markers",
            name="Crash / Timeout",
            marker=dict(color="#d97706", size=10, symbol="x", line=dict(width=2)),
            text=[r["description"][:80] for r in crashes],
            hovertemplate="<b>Experiment %{x}</b><br>%{text}<extra>crash</extra>",
            yaxis="y",
        ))

    # --- Keeps (green, foreground, with labels) ---
    if keeps:
        fig.add_trace(go.Scatter(
            x=[r["experiment"] for r in keeps],
            y=[r["val_bpb"] for r in keeps],
            mode="markers+text",
            name="Kept",
            marker=dict(color="#1a7f37", size=12, symbol="circle",
                        line=dict(width=2, color="white")),
            text=[r["description"][:40] for r in keeps],
            textposition="top right",
            textfont=dict(size=10, color="#555"),
            hovertemplate=(
                "<b>Experiment %{x}</b><br>"
                "val_bpb: %{y:.6f}<br>"
                "commit: %{customdata[0]}<br>"
                "memory: %{customdata[1]}<br>"
                "%{customdata[2]}"
                "<extra>kept</extra>"
            ),
            customdata=[[r["commit"], f"{r['memory_gb']:.1f} GB" if r["memory_gb"] else "—",
                         r["description"]] for r in keeps],
        ))

    # --- Memory usage on secondary axis (auto-hide if VRAM barely varies) ---
    mem_rows = [r for r in rows if r["memory_gb"] is not None and r["val_bpb"] is not None]
    mem_values = [r["memory_gb"] for r in mem_rows] if mem_rows else []
    vram_varies = (max(mem_values) - min(mem_values)) > 2.0 if mem_values else False

    if show_vram and mem_rows and vram_varies:
        fig.add_trace(go.Bar(
            x=[r["experiment"] for r in mem_rows],
            y=[r["memory_gb"] for r in mem_rows],
            name="Peak VRAM (GB)",
            marker=dict(color="rgba(100, 100, 255, 0.15)"),
            yaxis="y2",
            hovertemplate="VRAM: %{y:.1f} GB<extra></extra>",
        ))

    # --- Layout ---
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        xaxis=dict(
            title="Experiment #",
            gridcolor="#21262d",
            zeroline=False,
        ),
        yaxis=dict(
            title="Validation BPB (lower is better)",
            gridcolor="#21262d",
            zeroline=False,
        ),
        yaxis2=dict(
            title="Peak VRAM (GB)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            range=[0, max((r["memory_gb"] or 0) for r in rows) * 3],
        ),
        legend=dict(
            x=1.0, y=1.0, xanchor="right",
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#30363d",
            borderwidth=1,
        ),
        hovermode="x unified",
        margin=dict(l=60, r=80, t=80, b=60),
    )

    return fig


def main():
    p = argparse.ArgumentParser(description="Plot autoresearch progress from results.tsv")
    p.add_argument("--file", type=Path, default=DEFAULT_TSV, help="Path to results.tsv")
    p.add_argument("--title", default="", help="Chart title (auto-generated if omitted)")
    p.add_argument("--out", type=Path, default=Path("progress.html"), help="Output HTML path")
    p.add_argument("--png", type=Path, default=None, help="Also save a static PNG")
    p.add_argument("--no-vram", action="store_true", help="Hide VRAM bars")
    args = p.parse_args()

    rows = load_results(args.file)
    fig = build_chart(rows, title=args.title, show_vram=not args.no_vram)

    fig.write_html(str(args.out), include_plotlyjs="cdn")
    print(f"Written: {args.out}")

    if args.png:
        fig.write_image(str(args.png), width=1400, height=600, scale=2)
        print(f"Written: {args.png}")


if __name__ == "__main__":
    main()
