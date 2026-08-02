"""The Flyte pipeline: render music with ACE-Step 1.5, one report with playable tracks.

Usage (runs on the DGX Spark devbox, not local Python):

    # the headline run: turbo vs sft on the 3-brief quick suite
    flyte run compare_pipeline.py compare --suite quick

    # one brief across all three checkpoints
    flyte run compare_pipeline.py compare \
        --briefs '["acoustic-duo"]' --models '["xl-turbo","xl-sft","xl-base"]'

    # THE PARAMETER SWEEP: one checkpoint, one brief, one knob moved
    flyte run compare_pipeline.py sweep --axis seed
    flyte run compare_pipeline.py sweep --axis steps --model_key xl-turbo
    flyte run compare_pipeline.py sweep --axis guidance --model_key xl-sft
    flyte run compare_pipeline.py sweep --axis bpm --brief acoustic-duo

    # a custom value list for any axis
    flyte run compare_pipeline.py sweep --axis steps --values '["2","4","8","24","50"]'

    # your own brief, one track
    flyte run compare_pipeline.py generate_one \
        --prompt "lo-fi hip hop, dusty rhodes, vinyl crackle, boom bap drums" \
        --duration 45

Shape (identical to the TTS and video demos next door):

    compare ─┬─ fetch_weights(model)  ·· CPU, cached: one HF download per checkpoint
             └─ render(model, weights, jobs)  ·· GPU: load once, render every job

fetch is serial (parallel HF pulls just congest the uplink); the GPU tasks are
gathered, and since the box has one GPU the scheduler runs them one at a time.

Unlike the TTS demo there is ONE GPU task, not one per adapter, because every
checkpoint here loads through the same `AceStepPipeline`. That is the entire reason
this file is half the length of its TTS counterpart.

── Why the jobs are batched into one task per checkpoint ────────────────────────
Loading ACE-Step XL is ~11GB off disk and takes far longer than rendering a 30s track
at 8 steps. So a task takes a LIST of jobs and renders them all against one loaded
pipeline. For `sweep` that is the whole run in a single load: four seeds cost one
model load and four renders, not four of each.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import flyte
import flyte.io
import flyte.report
import soundfile as sf

import music_core
from config import cpu_env, gpu_env, orch_env
from models import DEFAULT_MODELS, get_sweep, get_spec, resolve_models
from music_core import Block, GenSettings, TrackResult
from prompts import Brief, DEFAULT_BRIEF, get_brief, get_suite

log = logging.getLogger("acestep")
logging.basicConfig(level=logging.INFO)


# ── Data crossing the task boundary ──────────────────────────────────────────────
#
# NOTE these are the LIGHTWEIGHT carriers: metadata plus a Dir of wavs, never the
# multi-hundred-KB base64 data URIs. The parent re-derives the report objects (with
# the embedded audio and spectrogram) from the wav files. A dataclass full of data
# URIs would bloat every task result needlessly.

@dataclass
class GenJob:
    """One track to render: what to ask for, and how to label it in the report."""
    label: str                       # the card title ("xl-turbo", "seed 42", ...)
    prompt: str
    lyrics: str = ""
    badges: list[str] = field(default_factory=list)
    settings: GenSettings = field(default_factory=GenSettings)


@dataclass
class TrackItem:
    label: str
    filename: str = ""               # wav in the run's Dir; "" if this job failed
    seconds: float = 0.0             # render wall-clock
    audio_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    peak_gb: float = 0.0
    settings: str = ""               # RESOLVED settings summary, what actually ran
    badges: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class ModelRun:
    model_key: str
    items: list[TrackItem] = field(default_factory=list)
    tracks: flyte.io.Dir | None = None   # the wavs


# ── Fetch: cache the HF download, keyed per checkpoint ───────────────────────────
#
# BUMP this when you change WHAT gets downloaded (a repo id, an allow/ignore pattern).
# Then a re-download is a deliberate act, not a side effect of editing a neighbouring
# function. Keyed on model_key, so each checkpoint keeps its own entry.
_WEIGHTS_CACHE_VERSION = "v1"
_WEIGHTS_CACHE = flyte.Cache(behavior="override", version_override=_WEIGHTS_CACHE_VERSION)


@cpu_env.task(cache=_WEIGHTS_CACHE, retries=2)
async def fetch_weights(model_key: str) -> flyte.io.Dir:
    """Snapshot a checkpoint's HF repo into an HF-cache-layout Dir. Cached forever.

    We download into `<dir>/` with snapshot_download(cache_dir=...), so the Dir is a
    ready-made HF hub cache: the GPU task points HF_HUB_CACHE at it and
    `from_pretrained` reads from disk. Runs on a CPU pod so no GPU sits idle during an
    11GB pull.
    """
    from huggingface_hub import snapshot_download

    spec = get_spec(model_key)
    dest = Path(tempfile.mkdtemp(prefix=f"weights_{model_key}_"))
    log.info(f"[{model_key}] downloading {spec.repo} (~{spec.download_gb:.1f}GB) -> {dest}")
    await asyncio.to_thread(
        snapshot_download, repo_id=spec.repo, cache_dir=str(dest),
        token=os.environ.get("HF_TOKEN"),
        # The .pt is a debug artifact the pipeline never reads at runtime, and the
        # duplicate .bin shards would double the pull on any repo that ships both.
        ignore_patterns=["*.pt", "*.bin", "*.msgpack", "*.onnx"],
    )
    return await flyte.io.Dir.from_local(str(dest))


# ── The GPU work ─────────────────────────────────────────────────────────────────

@gpu_env.task(report=True, retries=1)
async def render(model_key: str, weights: flyte.io.Dir, jobs: list[GenJob]) -> ModelRun:
    """Load one checkpoint once, render every job, live-update the report.

    The report is replaced after each track rather than at the end, so a long run is
    watchable: you can play track one while track four is still denoising. That
    matters more here than in the TTS demo because a 50-step sft render of a 60s track
    is minutes, not seconds.
    """
    spec = get_spec(model_key)
    local = await weights.download()

    out_dir = Path(tempfile.mkdtemp(prefix=f"acestep_{model_key}_"))
    items: list[TrackItem] = []
    results: list[TrackResult] = []
    meta = f"{spec.repo} · {spec.params} · {spec.license}"

    pipe = None
    try:
        music_core.prepare_gpu()
        log.info(f"[{model_key}] loading {spec.repo}")
        pipe = music_core.load_pipeline(spec, local_dir=local)
        log.info(f"[{model_key}] loaded; peak {music_core.peak_memory_gb():.1f}GB")

        for i, job in enumerate(jobs):
            try:
                music_core.reset_peak_memory()
                audio, sr, secs, resolved = music_core.generate(
                    pipe, spec, job.prompt, job.lyrics, job.settings)
                peak = music_core.peak_memory_gb()

                fn = f"{model_key}__{i:02d}.wav"
                music_core.write_wav(audio, sr, out_dir / fn)
                r = music_core.build_track_result(
                    job.label, audio, sr, secs,
                    sublabel=spec.family, settings=resolved.summary(),
                    peak_gb=peak, badges=job.badges)
                results.append(r)
                items.append(TrackItem(
                    label=job.label, filename=fn, seconds=secs,
                    audio_seconds=r.audio_seconds, sample_rate=sr,
                    channels=r.channels, peak_gb=peak,
                    settings=resolved.summary(), badges=job.badges))
                log.info(f"[{model_key}] {i+1}/{len(jobs)} {job.label}: {secs:.1f}s -> "
                         f"{r.audio_seconds:.1f}s audio ({r.speedup:.1f}x RT)")
            except Exception as e:  # one bad job must not kill the rest of the batch
                log.exception(f"[{model_key}] job {i} ({job.label}) failed")
                results.append(TrackResult(label=job.label, error=repr(e)))
                items.append(TrackItem(label=job.label, error=repr(e)))

            # One block per finished job, each carrying ITS OWN prompt. A single block
            # with jobs[0].prompt would be wrong for `compare`, where every job in the
            # batch is a different brief, and quietly mislabelling a track in the live
            # view is worse than having no live view at all.
            await flyte.report.replace.aio(music_core.render_report(
                [Block(heading=f"{j.label} ({n+1}/{len(jobs)})", note=", ".join(j.badges),
                       prompt=j.prompt, lyrics=j.lyrics, results=[r])
                 for n, (j, r) in enumerate(zip(jobs, results))],
                title=f"{model_key} · rendering ({i+1}/{len(jobs)})", meta=meta))
            await flyte.report.flush.aio()
    finally:
        pipe = None
        music_core.free_gpu_memory()

    tracks = await flyte.io.Dir.from_local(str(out_dir))
    return ModelRun(model_key=model_key, items=items, tracks=tracks)


# ── Re-derive report objects from a run's wavs, in the parent ────────────────────

async def _to_results(run: ModelRun, *, sublabel: str = "") -> list[TrackResult]:
    """Read a run's wavs back out and rebuild TrackResults (with the embedded audio and
    spectrogram) for the aggregate report.

    Returned IN JOB ORDER, not keyed by label, and that is deliberate: in a `compare`
    run every job from one checkpoint carries the same label (the checkpoint's name,
    which is what the card should say), so a label-keyed dict would silently collapse N
    briefs into one. Position is the only identity that survives.
    """
    # A checkpoint that failed EVERY job still returns a non-None but EMPTY Dir, and
    # downloading an empty Dir raises DownloadQueueEmpty in the parent, which would
    # take the whole comparison down. So check there is something to download first.
    if not run.tracks or not any(it.filename and not it.error for it in run.items):
        return [TrackResult(label=it.label, error=it.error or "no output")
                for it in run.items]

    local = Path(await run.tracks.download())
    out: list[TrackResult] = []
    for it in run.items:
        if it.error or not it.filename:
            out.append(TrackResult(label=it.label, error=it.error or "no output"))
            continue
        try:
            audio, sr = sf.read(str(local / it.filename), dtype="float32", always_2d=True)
            out.append(music_core.build_track_result(
                it.label, audio.T, sr, it.seconds, sublabel=sublabel,
                settings=it.settings, peak_gb=it.peak_gb, badges=it.badges))
        except Exception as e:
            out.append(TrackResult(label=it.label,
                                   error=f"could not read {it.filename}: {e}"))
    return out


def _at(results: list[TrackResult], i: int, label: str) -> TrackResult:
    """results[i], or an error card carrying the right label if the run came up short."""
    if i < len(results):
        return results[i]
    return TrackResult(label=label, error="no result")


def _settings_for(brief: Brief, *, seed: int, duration: float) -> GenSettings:
    """A brief's musical metadata plus the run-level seed/length knobs.

    bpm/key/language come from the brief because they are part of the *musical* ask;
    seed and duration come from the CLI because they are part of the *run*. Steps,
    guidance and shift stay at their sentinels so each checkpoint contributes its own
    recipe: that is what makes turbo-vs-sft a fair comparison rather than a rigged one.
    """
    return GenSettings(seed=seed, duration=duration, bpm=brief.bpm,
                       keyscale=brief.keyscale, language=brief.language)


# ── compare: checkpoints side by side ────────────────────────────────────────────

@orch_env.task(report=True)
async def compare(
    briefs: list[str] | None = None,
    suite: str = "quick",
    models: list[str] | None = None,
    duration: float = 30.0,
    seed: int = 42,
) -> list[ModelRun]:
    """Render the same briefs through several checkpoints, one report.

    `briefs` (keys from prompts.SUITE) wins over `suite`. Every checkpoint gets the
    same prompt, the same seed and the same length, and its OWN steps/guidance/shift,
    because those are properties of how it was trained, not knobs to hold constant.
    """
    specs = resolve_models(models)
    the_briefs = [get_brief(k) for k in briefs] if briefs else get_suite(suite)

    await flyte.report.replace.aio(music_core.render_status(
        "ACE-Step 1.5: checkpoints side by side",
        f"{len(specs)} checkpoints x {len(the_briefs)} briefs at {duration:g}s, seed "
        f"{seed}. Fetching weights (~{sum(s.download_gb for s in specs):.0f}GB total, "
        f"cached after the first run), then rendering. "
        f"Checkpoints: {', '.join(s.key for s in specs)}."))
    await flyte.report.flush.aio()

    # Fetch serially so parallel HF pulls do not fight for the uplink. Tolerate a
    # per-checkpoint failure: it becomes an error COLUMN, it does not kill the run.
    weights: dict[str, flyte.io.Dir] = {}
    fetch_errors: dict[str, str] = {}
    for s in specs:
        try:
            weights[s.key] = await fetch_weights.override(short_name=f"fetch {s.key}")(s.key)
        except Exception as e:
            log.exception(f"[{s.key}] fetch failed")
            fetch_errors[s.key] = f"weights fetch failed: {e}"

    # One task per checkpoint, each rendering every brief against one loaded pipeline.
    launched = [s for s in specs if s.key in weights]
    jobs_by_model = {
        s.key: [GenJob(label=s.key, prompt=b.prompt, lyrics=b.lyrics,
                       badges=[b.axis] if b.axis else [],
                       settings=_settings_for(b, seed=seed, duration=duration))
                for b in the_briefs]
        for s in launched
    }
    raw = await asyncio.gather(*[
        render.override(short_name=f"render {s.key}")(s.key, weights[s.key], jobs_by_model[s.key])
        for s in launched
    ], return_exceptions=True)

    # Reassemble: rows are briefs, cards within a row are checkpoints. Each GPU task
    # returned ONE run per checkpoint holding every brief in job order, so a card is
    # (checkpoint, position in the brief list).
    runs: list[ModelRun] = []
    per_model: dict[str, list[TrackResult]] = {}
    for s, res in zip(launched, raw):
        if isinstance(res, Exception):
            log.exception(f"[{s.key}] render task failed: {res}")
            per_model[s.key] = [TrackResult(label=s.key, error=f"render failed: {res}")
                                for _ in the_briefs]
            continue
        runs.append(res)
        per_model[s.key] = await _to_results(res, sublabel=s.family)

    blocks = []
    for i, b in enumerate(the_briefs):
        cards = []
        for s in specs:
            if s.key in fetch_errors:
                cards.append(TrackResult(label=s.key, error=fetch_errors[s.key]))
            else:
                cards.append(_at(per_model.get(s.key, []), i, s.key))
        blocks.append(Block(heading=f"{b.key}: {b.axis}", note=b.listen_for,
                            prompt=b.prompt, lyrics=b.lyrics, results=cards))

    meta = (f"{len(specs)} checkpoints · {len(the_briefs)} briefs · {duration:g}s each · "
            f"seed {seed} · same prompt, each checkpoint's own sampling recipe · "
            f"play a row left to right to compare")
    await flyte.report.replace.aio(music_core.render_report(
        blocks, title="ACE-Step 1.5: same brief, every checkpoint", meta=meta))
    await flyte.report.flush.aio()
    return runs


# ── sweep: one knob, moved ───────────────────────────────────────────────────────

def _coerce(field_name: str, raw: str):
    """Cast a CLI string to whatever type that GenSettings field holds.

    The CLI has to take `--values` as strings (a list can't be heterogeneously typed
    through the arg parser), but `seed` must arrive as an int and `shift` as a float or
    the pipeline's own validation rejects it. GenSettings' defaults are the schema.
    """
    default = getattr(GenSettings(), field_name)
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


@orch_env.task(report=True)
async def sweep(
    axis: str = "seed",
    model_key: str = "xl-turbo",
    brief: str = DEFAULT_BRIEF,
    values: list[str] | None = None,
    duration: float = 30.0,
    seed: int = 42,
) -> ModelRun:
    """Hold everything fixed, move ONE parameter, render the row side by side.

    `axis` is a key from models.SWEEPS (seed, steps, guidance, shift, bpm, keyscale,
    duration). `values` overrides the axis's preset list. Everything runs in a single
    GPU task against one loaded pipeline, so the marginal cost of another column is
    one render, not another 11GB load.
    """
    ax = get_sweep(axis)
    spec = get_spec(model_key)
    b = get_brief(brief)
    vals = [_coerce(ax.field, v) for v in values] if values else list(ax.values)

    # A distilled checkpoint ignores guidance entirely. We still run the sweep when
    # asked, because a row of four identical clips IS the demonstration; we just say so
    # up front instead of letting someone conclude the knob does nothing in general.
    inert = spec.distilled and not ax.turbo_ok
    warn = (f" NOTE: {model_key} is guidance-distilled, so this axis is inert on it "
            f"(the pipeline coerces guidance to 1.0) and every card should sound "
            f"identical. Re-run with --model_key xl-sft to hear the real thing."
            if inert else "")

    await flyte.report.replace.aio(music_core.render_status(
        f"ACE-Step sweep: {ax.label}",
        f"{model_key} · brief '{brief}' · {len(vals)} values: "
        f"{', '.join(ax.fmt.format(v) for v in vals)}. Fetching weights, then rendering "
        f"all {len(vals)} against one loaded pipeline.{warn}"))
    await flyte.report.flush.aio()

    w = await fetch_weights.override(short_name=f"fetch {model_key}")(model_key)

    jobs = []
    for v in vals:
        st = _settings_for(b, seed=seed, duration=duration)
        setattr(st, ax.field, v)
        badges = [f"{ax.label}"] + (["inert on this checkpoint"] if inert else [])
        jobs.append(GenJob(label=f"{ax.label} = {ax.fmt.format(v)}",
                           prompt=b.prompt, lyrics=b.lyrics, badges=badges, settings=st))

    run = await render.override(short_name=f"sweep {axis}")(model_key, w, jobs)
    cards = await _to_results(run, sublabel=f"{model_key} · {spec.family}")

    block = Block(heading=f"{ax.label}: {brief} on {model_key}",
                  note=ax.listen_for + warn,
                  prompt=b.prompt, lyrics=b.lyrics,
                  results=[_at(cards, i, j.label) for i, j in enumerate(jobs)])
    meta = (f"{model_key} · everything fixed except {ax.label} · seed "
            f"{seed if ax.field != 'seed' else 'varied'} · {duration:g}s each")
    await flyte.report.replace.aio(music_core.render_report(
        [block], title=f"ACE-Step 1.5: what does {ax.label} do?", meta=meta))
    await flyte.report.flush.aio()
    return run


# ── generate_one: the smoke test ─────────────────────────────────────────────────

@orch_env.task(report=True)
async def generate_one(
    model_key: str = "xl-turbo",
    prompt: str = "",
    brief: str = DEFAULT_BRIEF,
    lyrics: str = "",
    duration: float = 30.0,
    seed: int = 42,
    steps: int = 0,
    guidance: float = -1.0,
    shift: float = -1.0,
    bpm: int = 0,
    keyscale: str = "",
    language: str = "en",
) -> ModelRun:
    """Render a single track. The cheapest 'does this even load on the box' check.

    With no `--prompt` it uses the named brief from prompts.py; with one, your text
    wins and `--lyrics` is yours to fill (leave it empty for an instrumental).
    """
    spec = get_spec(model_key)
    b = get_brief(brief)
    the_prompt = prompt or b.prompt
    the_lyrics = lyrics if prompt else b.lyrics

    st = GenSettings(seed=seed, steps=steps, guidance=guidance, shift=shift,
                     duration=duration, bpm=bpm or (0 if prompt else b.bpm),
                     keyscale=keyscale or ("" if prompt else b.keyscale),
                     language=language)

    w = await fetch_weights.override(short_name=f"fetch {model_key}")(model_key)
    run = await render.override(short_name=f"render {model_key}")(
        model_key, w, [GenJob(label=model_key, prompt=the_prompt, lyrics=the_lyrics,
                              badges=[spec.params], settings=st)])

    cards = await _to_results(run, sublabel=spec.family)
    block = Block(heading=f"{model_key}: single track", note=spec.notes,
                  prompt=the_prompt, lyrics=the_lyrics,
                  results=[_at(cards, 0, model_key)])
    await flyte.report.replace.aio(music_core.render_report(
        [block], title=f"ACE-Step 1.5 · {model_key}",
        meta=f"{spec.repo} · {spec.license}"))
    await flyte.report.flush.aio()
    return run


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(compare, suite="quick", models=DEFAULT_MODELS)
    print(f"Compare run: {run.name}")
    print(f"  {run.url}")
