"""Render ACE-Step directly on the host GPU, no Flyte, no cluster.

The fastest way to answer "does this checkpoint even load and render on this box?"
before a cluster round-trip. Writes .wav files plus a standalone .html report using
the SAME renderer the Flyte report uses, so if a track plays here it plays there.

    # one track from a named brief
    python run_local.py --model xl-turbo --brief synthwave

    # your own brief
    python run_local.py --prompt "lo-fi hip hop, dusty rhodes, vinyl crackle, boom bap"

    # a local parameter sweep: same load, N renders, one HTML row
    python run_local.py --sweep seed
    python run_local.py --sweep steps --model xl-turbo --values 4,8,16,32

Unlike the TTS demo's run_local, there is no per-model venv caveat: every checkpoint
here loads through the same `diffusers` install, so one venv drives all of them.

    uv venv && uv pip install "diffusers>=0.39.0" transformers accelerate \
        soundfile matplotlib click torch --torch-backend=auto

Weights land in the normal HF cache (~/.cache/huggingface), shared with anything else
on the host but NOT with the Flyte tasks, which cache their own copy in the blob store.
"""

from __future__ import annotations

import time
from pathlib import Path

import click

import music_core
from models import MODELS, SWEEPS, get_spec, get_sweep
from music_core import Block, GenSettings
from prompts import BY_KEY, DEFAULT_BRIEF, get_brief


def _coerce(field_name: str, raw: str):
    """Cast a CLI string to the type that GenSettings field holds (see compare_pipeline)."""
    default = getattr(GenSettings(), field_name)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


@click.command()
@click.option("--model", "model_key", default="xl-turbo", type=click.Choice(list(MODELS)),
              help="Which checkpoint to load.")
@click.option("--brief", default=DEFAULT_BRIEF, type=click.Choice(list(BY_KEY)),
              help="A named brief from prompts.py.")
@click.option("--prompt", default="", help="Your own style caption. Overrides --brief.")
@click.option("--lyrics", default="", help="Lyrics with [verse]/[chorus] tags. Empty = "
              "instrumental. Only used with --prompt.")
@click.option("--sweep", "sweep_axis", default="", type=click.Choice(["", *SWEEPS]),
              help="Vary ONE parameter across its preset values and render the row.")
@click.option("--values", default="", help="Comma-separated override for the sweep's "
              "preset values, e.g. '4,8,16,32'.")
@click.option("--duration", default=30.0, help="Track length in seconds.")
@click.option("--seed", default=42, help="Sampler seed.")
@click.option("--steps", default=0, help="Denoising steps. 0 = the checkpoint's default.")
@click.option("--guidance", default=-1.0, help="CFG scale. <0 = the checkpoint's default.")
@click.option("--out", default="./out", help="Where to write the .wav files and report.")
def main(model_key, brief, prompt, lyrics, sweep_axis, values, duration, seed, steps,
         guidance, out):
    """Render on the host GPU and write .wav files + an .html report."""
    spec = get_spec(model_key)
    b = get_brief(brief)
    the_prompt = prompt or b.prompt
    the_lyrics = lyrics if prompt else b.lyrics
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = GenSettings(seed=seed, steps=steps, guidance=guidance, duration=duration,
                       bpm=0 if prompt else b.bpm,
                       keyscale="" if prompt else b.keyscale,
                       language="en" if prompt else b.language)

    # Build the job list: one render, or a row across a swept axis.
    ax = get_sweep(sweep_axis) if sweep_axis else None
    if ax:
        vals = ([_coerce(ax.field, v) for v in values.split(",")] if values
                else list(ax.values))
        jobs = []
        for v in vals:
            st = GenSettings(**vars(base))
            setattr(st, ax.field, v)
            jobs.append((f"{ax.label} = {ax.fmt.format(v)}", st))
        heading = f"{ax.label}: {brief} on {model_key}"
        note = ax.listen_for
        title = f"ACE-Step 1.5: what does {ax.label} do? (host GPU)"
    else:
        jobs = [(model_key, base)]
        heading = f"{model_key}: single track"
        note = spec.notes
        title = f"ACE-Step 1.5 · {model_key} (host GPU)"

    click.echo(f"[{model_key}] {spec.repo}  ({spec.params}, {spec.license})")
    click.echo(f"  ~{spec.download_gb:.1f}GB download (cached after the first run)")
    click.echo(f"  {len(jobs)} render(s) against one loaded pipeline")

    music_core.prepare_gpu()
    t0 = time.time()
    pipe = music_core.load_pipeline(spec)
    click.echo(f"  loaded in {time.time() - t0:.0f}s")

    results = []
    try:
        for i, (label, st) in enumerate(jobs):
            music_core.reset_peak_memory()
            audio, sr, secs, resolved = music_core.generate(
                pipe, spec, the_prompt, the_lyrics, st)
            peak = music_core.peak_memory_gb()

            wav_path = out_dir / f"{model_key}__{i:02d}.wav"
            music_core.write_wav(audio, sr, wav_path)
            r = music_core.build_track_result(
                label, audio, sr, secs, sublabel=spec.family,
                settings=resolved.summary(), peak_gb=peak)
            results.append(r)
            click.echo(f"  {i+1}/{len(jobs)} {label}: {secs:.1f}s render -> "
                       f"{r.audio_seconds:.1f}s audio ({r.speedup:.1f}x RT, "
                       f"peak {peak:.1f}GB) -> {wav_path.name}")
    finally:
        pipe = None
        music_core.free_gpu_memory()

    stem = f"{model_key}-{sweep_axis}" if sweep_axis else model_key
    html_path = out_dir / f"{stem}.html"
    html_path.write_text(music_core.render_report(
        [Block(heading=heading, note=note, prompt=the_prompt, lyrics=the_lyrics,
               results=results)],
        title=title, meta=f"{spec.repo} · {spec.license} · host GPU, no cluster"))

    click.echo(f"\n  report: {html_path}  (open it in a browser; the tracks play inline)")


if __name__ == "__main__":
    main()
