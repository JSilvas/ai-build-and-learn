"""HTML for the Flyte report: training curves, a summary table, and the replay clip.

Kept out of train.py/render.py so the GPU task's file stays about training and the
render task's stays about MuJoCo. Nothing here imports jax or mujoco.

── Why the replay is an mp4 and not a frame scrubber ───────────────────────────
The workshop tutorials base64'd a list of JPEGs into the page and drove them with a
range input. That works, but a 1000-step episode is a lot of JPEGs, and the report
gets huge and scrubs badly. A single base64 `<video>` is smaller, plays at the right
frame rate on its own, and needs no JavaScript at all (the same conclusion the
video-generation demo reached: see reference_videogen_report_playback).
"""

from __future__ import annotations

import base64
import io
import json

# The palette the other build-learn demo reports use, so a viewer moving between
# them isn't re-learning the colours.
_BG = "#0f0f23"
_PANEL = "#1a1a2e"
_TEXT = "#ccc"
_ACCENT = "#00b894"
_HILITE = "#fdcb6e"
_MUTED = "#888"


def _fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor=_BG)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;" />'


def reward_chart(history: list[dict], title: str = "G1 walking - training progress") -> str:
    """Reward vs environment steps, with the +/- std band the eval reports."""
    if not history:
        return ""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history]
    rewards = [h["reward"] for h in history]
    stds = [h.get("reward_std", 0.0) for h in history]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_BG)
    ax.set_facecolor(_PANEL)
    lo = [r - s for r, s in zip(rewards, stds)]
    hi = [r + s for r, s in zip(rewards, stds)]
    ax.fill_between(steps, lo, hi, color=_ACCENT, alpha=0.18, linewidth=0)
    ax.plot(steps, rewards, color=_ACCENT, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Environment steps", color=_TEXT)
    ax.set_ylabel("Mean episode reward", color=_TEXT)
    ax.set_title(title, color=_HILITE, fontsize=14)
    ax.tick_params(colors=_MUTED)
    ax.grid(alpha=0.15)
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()
    html = _fig_to_img(fig)
    plt.close(fig)
    return html


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


def progress_html(preset, params, history: list[dict], extra: str = "") -> str:
    """The live view, redrawn on every eval callback while training runs."""
    latest = history[-1] if history else {}
    done = latest.get("step", 0)
    total = int(params.num_timesteps)
    pct = (100.0 * done / total) if total else 0.0
    rows = [
        ("Preset", f"{preset.key} - {preset.notes}"),
        ("Progress", f"{done:,} / {total:,} steps ({pct:.0f}%)"),
        ("Parallel envs", f"{int(params.num_envs):,}"),
        ("Latest reward", f"{latest.get('reward', 0.0):.1f}"),
        ("Elapsed", f"{latest.get('elapsed_s', 0.0) / 60:.1f} min"),
    ]
    return (
        "<h2>Unitree G1 - learning to walk (MJX + Brax PPO)</h2>"
        + _panel("Training", _table(rows))
        + "<br/>"
        + (extra + "<br/>" if extra else "")
        + reward_chart(history)
    )


def training_snapshot_html(latest: dict, timeline: list[dict]) -> str:
    """Mid-training video: the current policy, plus how it got there.

    `latest` is overwritten each snapshot, so the report always answers "what does it
    look like RIGHT NOW". `timeline` keeps one still per snapshot, which answers the
    different and more useful question of whether it is getting better: a reward curve
    that climbs while the robot is visibly hopping on one leg is a reward-hacking
    story the number alone will not tell you.

    Only the latest clip carries video bytes. The timeline is stills, so this block
    grows by a few KB per snapshot rather than by a whole mp4.
    """
    if not latest:
        return ""
    block = video_html(
        latest["mp4"],
        f"step {latest['step']:,} | eval reward {latest['reward']:.1f} | "
        f"survived {latest['steps']}/{latest['max_steps']} steps",
    )
    strip = ""
    if len(timeline) > 1:
        cells = "".join(
            f'<div style="text-align:center;margin-right:8px;">'
            f'<img src="data:image/jpeg;base64,{s["frame"]}" '
            f'style="height:120px;border:1px solid #333;border-radius:3px;display:block;" />'
            f'<span style="color:{_MUTED};font-size:11px;">'
            f'{s["step"] / 1e6:.0f}M · {s["reward"]:.0f}</span></div>'
            for s in timeline
        )
        strip = (
            f'<p style="color:{_TEXT};margin:12px 0 4px;">Progress so far '
            f'<span style="color:{_MUTED};">(steps · reward)</span></p>'
            f'<div style="display:flex;overflow-x:auto;padding-bottom:6px;">{cells}</div>'
        )
    return _panel("Latest policy", block + strip)


def video_html(mp4_bytes: bytes, caption: str = "") -> str:
    """A self-contained, autoplaying, looping clip. No JS, no external requests."""
    b64 = base64.b64encode(mp4_bytes).decode()
    cap = f'<p style="color:{_MUTED};margin:6px 0 0;">{caption}</p>' if caption else ""
    return (
        f'<div style="font-family:monospace;background:{_BG};padding:16px;border-radius:8px;">'
        f'<video src="data:video/mp4;base64,{b64}" controls autoplay loop muted playsinline '
        f'style="max-width:640px;width:100%;border:2px solid #333;border-radius:4px;display:block;">'
        f"</video>{cap}</div>"
    )


def frame_strip_html(frames_b64: list[str], caption: str = "") -> str:
    """Fallback when the mp4 encode fails: a filmstrip of stills.

    Worth having because a report that shows *something* beats one that shows an
    error, and encoder problems are the most likely way this pipeline breaks on a
    box where the GL stack is doing headless EGL.
    """
    if not frames_b64:
        return ""
    imgs = "".join(
        f'<img src="data:image/jpeg;base64,{f}" style="height:150px;border:1px solid #333;'
        f'border-radius:3px;margin-right:4px;" />'
        for f in frames_b64
    )
    cap = f'<p style="color:{_MUTED};margin:6px 0 0;">{caption}</p>' if caption else ""
    return (
        f'<div style="font-family:monospace;background:{_BG};padding:16px;border-radius:8px;'
        f'overflow-x:auto;white-space:nowrap;">{imgs}{cap}</div>'
    )


def before_after_html(before: dict, after: dict) -> str:
    """The two replays side by side, with the numbers that make them comparable.

    Both clips run the same command, camera and episode cap, so "steps survived" is a
    fair like-for-like: a policy that walks stays up for the full cap, a policy that
    doesn't hits the floor and the episode ends early.
    """
    cap = max(before.get("steps", 0), after.get("steps", 0))
    rows = [
        (
            "Steps survived",
            f'{after.get("steps", 0)} <span style="color:{_MUTED};">'
            f'(random: {before.get("steps", 0)})</span>',
        ),
        (
            "Episode reward",
            f'{after.get("reward", 0.0):.1f} <span style="color:{_MUTED};">'
            f'(random: {before.get("reward", 0.0):.1f})</span>',
        ),
        (
            "Command",
            ", ".join(f"{v:.1f}" for v in after.get("command", [])) + "  (vx, vy, yaw)",
        ),
    ]
    return (
        _panel("Trained vs random", _table(rows))
        + "<br/>"
        + '<div style="display:flex;flex-wrap:wrap;gap:16px;">'
        + f'<div style="flex:1 1 320px;min-width:280px;"><h4 style="color:{_HILITE};'
        f'font-family:monospace;">After training</h4>{after.get("html", "")}</div>'
        + f'<div style="flex:1 1 320px;min-width:280px;"><h4 style="color:{_MUTED};'
        f'font-family:monospace;">Before (random policy)</h4>{before.get("html", "")}</div>'
        + "</div>"
    )


def preset_index_html(runs: list[dict], replays: list[dict]) -> str:
    """A scoreboard pointing at the per-preset tabs, with no video bytes of its own.

    The clips live in the tabs. This is the table that tells you which tab to open.
    """
    header = (
        f'<tr style="color:{_HILITE};text-align:left;">'
        f'<th style="padding:6px;">Preset</th>'
        f'<th style="padding:6px;">Best reward</th>'
        f'<th style="padding:6px;">Steps survived</th>'
        f'<th style="padding:6px;">Replay</th></tr>'
    )
    body = ""
    for run, replay in zip(runs, replays):
        best = max((h["reward"] for h in run.get("history", [])), default=0.0)
        body += (
            f'<tr><td style="padding:6px;border-top:1px solid #333;">{run.get("preset", "?")}</td>'
            f'<td style="padding:6px;border-top:1px solid #333;color:{_ACCENT};">{best:.1f}</td>'
            f'<td style="padding:6px;border-top:1px solid #333;">{replay.get("steps", 0)}</td>'
            f'<td style="padding:6px;border-top:1px solid #333;color:{_MUTED};">'
            f'see the "{run.get("preset", "?")}" tab</td></tr>'
        )
    return _panel(
        "Results",
        f'<table style="border-collapse:collapse;width:100%;">{header}{body}</table>',
    )


def summary_html(ckpt: dict, replay: dict | None = None) -> str:
    history = ckpt.get("history", [])
    best = max((h["reward"] for h in history), default=0.0)
    final = history[-1]["reward"] if history else 0.0
    rows = [
        ("Environment", ckpt.get("env_name", "?")),
        ("Reward preset", ckpt.get("preset", "?")),
        ("Steps trained", f"{ckpt.get('num_timesteps', 0):,}"),
        ("Parallel envs", f"{ckpt.get('num_envs', 0):,}"),
        ("Domain randomization", "on" if ckpt.get("domain_randomization") else "off"),
        ("Wall clock", f"{ckpt.get('elapsed_s', 0) / 60:.1f} min"),
        ("Best eval reward", f"{best:.1f}"),
        ("Final eval reward", f"{final:.1f}"),
    ]
    if replay:
        rows.append(("Replay episode reward", f"{replay.get('reward', 0.0):.1f}"))
        rows.append(("Replay length", f"{replay.get('steps', 0)} steps"))
    return _panel("Run summary", _table(rows))


def final_html(ckpt: dict, replay_html_block: str) -> str:
    return (
        "<h2>Unitree G1 - learning to walk (MJX + Brax PPO)</h2>"
        + summary_html(ckpt, None)
        + "<br/>"
        + f'<h3 style="color:{_HILITE};font-family:monospace;">Training progress</h3>'
        + reward_chart(ckpt.get("history", []))
        + "<br/>"
        + (
            f'<h3 style="color:{_HILITE};font-family:monospace;">Replay</h3>'
            + replay_html_block
            if replay_html_block
            else ""
        )
    )


def compare_html(runs: list[dict]) -> str:
    """One chart with every preset's reward curve, for the preset sweep."""
    if not runs:
        return ""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = [_ACCENT, _HILITE, "#0984e3", "#e17055", "#a29bfe", "#fd79a8"]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_BG)
    ax.set_facecolor(_PANEL)
    for i, run in enumerate(runs):
        history = run.get("history", [])
        if not history:
            continue
        ax.plot(
            [h["step"] for h in history],
            [h["reward"] for h in history],
            color=palette[i % len(palette)],
            linewidth=2,
            marker="o",
            markersize=3,
            label=run.get("preset", f"run {i}"),
        )
    ax.set_xlabel("Environment steps", color=_TEXT)
    ax.set_ylabel("Mean episode reward", color=_TEXT)
    ax.set_title("Reward preset comparison", color=_HILITE, fontsize=14)
    ax.tick_params(colors=_MUTED)
    ax.grid(alpha=0.15)
    legend = ax.legend(facecolor=_PANEL, edgecolor="#333", labelcolor=_TEXT)
    legend.get_frame().set_alpha(0.9)
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()
    html = _fig_to_img(fig)
    plt.close(fig)
    return html
