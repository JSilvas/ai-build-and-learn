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
import prompts
from config import cpu_env, diffrhythm_env, gpu_env, orch_env
from models import DEFAULT_MODELS, Variant, get_sweep, get_spec, resolve_models
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
# v2: switched from cache_dir= to local_dir=. See the note below.
_WEIGHTS_CACHE_VERSION = "v2"
_WEIGHTS_CACHE = flyte.Cache(behavior="override", version_override=_WEIGHTS_CACHE_VERSION)


@cpu_env.task(cache=_WEIGHTS_CACHE, retries=3)
async def fetch_weights(model_key: str) -> flyte.io.Dir:
    """Snapshot a checkpoint into a plain diffusers-layout Dir. Cached forever.

    `local_dir=`, NOT `cache_dir=`, and that was a bug fix. The first version used
    `cache_dir=`, producing an HF hub cache (`blobs/` holding the real files,
    `snapshots/<rev>/` holding symlinks to them) and pointed HF_HUB_CACHE at the
    downloaded Dir in the GPU task. Two things went wrong on the box:

      1. The Dir upload DEREFERENCED the symlinks, so every shard was stored twice and
         an 11GB checkpoint became ~22GB in the blob store.
      2. The GPU task re-downloaded all 11GB from HuggingFace anyway, taking 2.5
         minutes on every single run, cache or no cache.

    `local_dir=` writes a plain repo layout (`model_index.json` next to `transformer/`,
    `vae/`, ...), which `from_pretrained` accepts as a path. No symlinks to dereference,
    no duplication, and no cache-resolution machinery to miss.

    retries=3, not 2, because the failure here is not a timeout. huggingface_hub now
    routes downloads through Xet content-addressed storage, and a transient CDN blip
    surfaces as `CAS Client Error: Request middleware error` and kills the whole
    download. Observed on the very first run of this pipeline. It is retryable and the
    retry resumes, so the cheap fix is another attempt.

    Runs on a CPU pod so no GPU sits idle during an 11GB pull.
    """
    from huggingface_hub import snapshot_download

    spec = get_spec(model_key)
    dest = Path(tempfile.mkdtemp(prefix=f"weights_{model_key}_")) / model_key
    log.info(f"[{model_key}] downloading {spec.repo} (~{spec.download_gb:.1f}GB) -> {dest}")
    await asyncio.to_thread(
        snapshot_download, repo_id=spec.repo, local_dir=str(dest),
        token=os.environ.get("HF_TOKEN"),
        # The .pt is a debug artifact the pipeline never reads at runtime, and the
        # duplicate .bin shards would double the pull on any repo that ships both.
        ignore_patterns=["*.pt", "*.bin", "*.msgpack", "*.onnx"],
    )
    # A diffusers pipeline announces itself with model_index.json; a plain transformers
    # checkpoint (MusicGen) with config.json. Check the right one, and fail here with a
    # directory listing rather than letting from_pretrained fail in a GPU pod later.
    marker = "config.json" if spec.adapter == "musicgen" else "model_index.json"
    if not (dest / marker).exists():
        raise RuntimeError(
            f"{spec.repo} downloaded without a {marker}; not a {spec.adapter}-layout "
            f"repo. Got: {sorted(p.name for p in dest.iterdir())[:12]}")
    return await flyte.io.Dir.from_local(str(dest))


# ── The GPU work ─────────────────────────────────────────────────────────────────

@gpu_env.task(report=True, retries=1)
async def render(model_key: str, weights: flyte.io.Dir, jobs: list[GenJob]) -> ModelRun:
    """The default renderer: every model whose image is the shared one."""
    return await _render_jobs(model_key, weights, jobs)


async def _render_jobs(model_key: str, weights: flyte.io.Dir,
                       jobs: list[GenJob]) -> ModelRun:
    """Load one checkpoint once, render every job, live-update the report.

    The report is replaced after each track rather than at the end, so a long run is
    watchable: you can play track one while track four is still denoising. That
    matters more here than in the TTS demo because a 50-step sft render of a 60s track
    is minutes, not seconds.
    """
    spec = get_spec(model_key)
    # Do NOT download for models that fetch their own checkpoints: `_weights_for` hands
    # those an EMPTY Dir, and `Dir.download()` on an empty Dir raises
    # DownloadQueueEmpty. The same trap is already documented in `_to_results`; it bit
    # again here because this is a second, independent call site.
    local = "" if getattr(spec, "self_downloads", False) else await weights.download()

    out_dir = Path(tempfile.mkdtemp(prefix=f"acestep_{model_key}_"))
    items: list[TrackItem] = []
    results: list[TrackResult] = []
    meta = f"{spec.repo} · {spec.params} · {spec.license}"

    pipe = None
    try:
        # Adapters that shell out (DiffRhythm) render in a CHILD process, so the parent
        # must NOT touch CUDA: `prepare_gpu` would create a context here and cap this
        # process at ~90% of a unified pool the child then has to allocate from. The
        # child died with `CUDA error: out of memory` moving a 2.2GB model onto a GPU
        # with 90GB+ free, which is what that contention looks like from the inside.
        subprocess_adapter = spec.adapter in music_core.SUBPROCESS_ADAPTERS
        cap = 0.0 if subprocess_adapter else music_core.prepare_gpu()
        avail_gb, total_gb = (0.0, 0.0) if subprocess_adapter else music_core.gpu_pool_gb()
        # Log the ceiling BEFORE loading. On this box a long render can die as a bare
        # SIGSEGV with no Python traceback, and when that happens the only useful
        # forensic question is "how much memory did it actually have?". This line is
        # also what caught the cap regression: it read "capped at 3GB" out loud.
        if subprocess_adapter:
            log.info(f"[{model_key}] subprocess adapter: leaving CUDA untouched in the "
                     f"parent so the child owns the GPU; loading {spec.repo}")
        else:
            log.info(f"[{model_key}] pool {avail_gb:.0f}/{total_gb:.0f}GB available, "
                     f"capped at {cap:.0f}GB; loading {spec.repo}")
        pipe = music_core.load_pipeline(spec, local_dir=local)
        log.info(f"[{model_key}] loaded; peak {music_core.peak_memory_gb():.1f}GB")

        for i, job in enumerate(jobs):
            try:
                if not subprocess_adapter:
                    music_core.reset_peak_memory()
                audio, sr, secs, resolved = music_core.generate(
                    pipe, spec, job.prompt, job.lyrics, job.settings)
                peak = 0.0 if subprocess_adapter else music_core.peak_memory_gb()

                fn = f"{model_key}__{i:02d}.wav"
                music_core.write_wav(audio, sr, out_dir / fn)
                r = music_core.build_track_result(
                    job.label, audio, sr, secs,
                    sublabel=spec.family, settings=resolved.summary(spec),
                    peak_gb=peak, badges=job.badges,
                    intended_for=spec.intended_for)
                results.append(r)
                items.append(TrackItem(
                    label=job.label, filename=fn, seconds=secs,
                    audio_seconds=r.audio_seconds, sample_rate=sr,
                    channels=r.channels, peak_gb=peak,
                    settings=resolved.summary(spec), badges=job.badges))
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


@diffrhythm_env.task(report=True, retries=1)
async def render_diffrhythm(model_key: str, weights: flyte.io.Dir,
                            jobs: list[GenJob]) -> ModelRun:
    """Identical body to `render`, different ENV, and that is the only difference.

    DiffRhythm is the one model with its own image (no PyPI package, no packaging
    metadata, a git clone on PYTHONPATH), so it needs its own TaskEnvironment. The work
    is shared through `_render_jobs`; the TTS demo next door does the same thing seven
    times over for seven mutually hostile packages.
    """
    return await _render_jobs(model_key, weights, jobs)


# adapter -> which task (hence which image) renders it.
RENDER_TASKS = {"diffrhythm": render_diffrhythm}


def render_task_for(spec):
    """The render task whose image can actually load this model."""
    return RENDER_TASKS.get(spec.adapter, render)


# ── Re-derive report objects from a run's wavs, in the parent ────────────────────

async def _to_results(run: ModelRun, *, sublabel: str = "", spec=None,
                      jobs: list[GenJob] | None = None,
                      brief: str = "") -> list[TrackResult]:
    """Read a run's wavs back out and rebuild TrackResults (with the embedded audio and
    spectrogram) for the aggregate report.

    Returned IN JOB ORDER, not keyed by label, and that is deliberate: in a `compare`
    run every job from one checkpoint carries the same label (the checkpoint's name,
    which is what the card should say), so a label-keyed dict would silently collapse N
    briefs into one. Position is the only identity that survives.

    `spec` + `jobs` attach a Repro to each card. They are rebuilt HERE rather than
    shipped back from the GPU task, because `GenSettings.resolve(spec)` is
    deterministic: the parent already holds the job it sent and the spec it sent it to,
    so it can derive exactly what ran without widening the task boundary.
    """
    # A checkpoint that failed EVERY job still returns a non-None but EMPTY Dir, and
    # downloading an empty Dir raises DownloadQueueEmpty in the parent, which would
    # take the whole comparison down. So check there is something to download first.
    def _repro(i: int):
        if spec is None or not jobs or i >= len(jobs):
            return None
        j = jobs[i]
        return music_core.build_repro(spec, j.prompt, j.lyrics, j.settings, brief=brief)

    if not run.tracks or not any(it.filename and not it.error for it in run.items):
        return [TrackResult(label=it.label, error=it.error or "no output")
                for it in run.items]

    local = Path(await run.tracks.download())
    out: list[TrackResult] = []
    for i, it in enumerate(run.items):
        if it.error or not it.filename:
            out.append(TrackResult(label=it.label, error=it.error or "no output"))
            continue
        try:
            audio, sr = sf.read(str(local / it.filename), dtype="float32", always_2d=True)
            tr = music_core.build_track_result(
                it.label, audio.T, sr, it.seconds, sublabel=sublabel,
                settings=it.settings, peak_gb=it.peak_gb, badges=it.badges,
                repro=_repro(i),
                intended_for=getattr(spec, "intended_for", "") if spec else "")
            # Log them as well as rendering them. The numbers are on the card, but a
            # card is a thing you read one at a time with your eyes; a log line is
            # greppable, diffable across runs, and survives the report being too heavy
            # to open. Comparing a sweep is exactly the case where you want the whole
            # row as text.
            if tr.metrics:
                log.info(f"[metrics] {it.label}: " + " ".join(
                    f"{k}={v:g}" for k, v in tr.metrics.items()))
            out.append(tr)
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


@cpu_env.task
async def empty_dir() -> flyte.io.Dir:
    """An empty Dir, for models that fetch their own weights at render time."""
    return await flyte.io.Dir.from_local(tempfile.mkdtemp(prefix="noweights_"))


async def _weights_for(spec) -> flyte.io.Dir:
    """Pre-staged weights, or an empty Dir for models that download their own."""
    if getattr(spec, "self_downloads", False):
        return await empty_dir.override(short_name=f"skip fetch {spec.key}")()
    return await fetch_weights.override(short_name=f"fetch {spec.key}")(spec.key)


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
        if getattr(s, "self_downloads", False):
            # Its own code calls hf_hub_download at render time, so there is nothing
            # useful for us to pre-stage. An empty Dir keeps the task signature uniform.
            weights[s.key] = await empty_dir.override(short_name=f"skip fetch {s.key}")()
            continue
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
        render_task_for(s).override(short_name=f"render {s.key}")(
            s.key, weights[s.key], jobs_by_model[s.key])
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
        per_model[s.key] = await _to_results(res, sublabel=s.family, spec=s,
                                             jobs=jobs_by_model[s.key])

    # A run whose briefs all share one lyric is a GENRE SWAP: the words are the control
    # and the style caption is the variable. Say so, because the alternative is a report
    # that prints the same lyric fold under all ten rows and looks broken. This is
    # detected rather than passed in so it stays true of any brief set someone builds,
    # including one assembled ad hoc with --briefs.
    lyric_set = {b.lyrics for b in the_briefs if b.lyrics.strip()}
    shared_lyric = len(the_briefs) > 1 and len(lyric_set) == 1 and not [
        b for b in the_briefs if not b.lyrics.strip()]
    lyr_label = ("lyrics (the control: identical in every row below)" if shared_lyric
                 else "lyrics")

    blocks = []
    for i, b in enumerate(the_briefs):
        cards = []
        for s in specs:
            if s.key in fetch_errors:
                cards.append(TrackResult(label=s.key, error=fetch_errors[s.key]))
            else:
                card = _at(per_model.get(s.key, []), i, s.key)
                # Each card in this row came from brief `b`, so name it in the repro:
                # `--brief acoustic-duo` beats inlining a paragraph of prompt text.
                if card.repro is not None:
                    card.repro.brief = b.key
                cards.append(card)
        blocks.append(Block(heading=f"{b.key}: {b.axis}", note=b.listen_for,
                            prompt=b.prompt, lyrics=b.lyrics, results=cards,
                            lyrics_label=lyr_label))

    if shared_lyric:
        title = "One lyric, every genre"
        meta = (f"{len(the_briefs)} genres · {len(specs)} model(s) · {duration:g}s each · "
                f"seed {seed} · the LYRIC is held fixed and only the style caption "
                f"changes, so the same words appear under every row on purpose · the "
                f"question is whether the melody and phrasing change with the genre or "
                f"whether one tune is wearing {len(the_briefs)} costumes")
    else:
        title = "ACE-Step 1.5: same brief, every checkpoint"
        meta = (f"{len(specs)} checkpoints · {len(the_briefs)} briefs · {duration:g}s "
                f"each · seed {seed} · same prompt, each checkpoint's own sampling "
                f"recipe · play a row left to right to compare")
    await flyte.report.replace.aio(music_core.render_report(
        blocks, title=title, meta=meta))
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
    steps: int = 0,
    guidance: float = -1.0,
    shift: float = -1.0,
) -> ModelRun:
    """Hold everything fixed, move ONE parameter, render the row side by side.

    `axis` is a key from models.SWEEPS. `values` overrides the axis's preset list.
    Everything runs in a single GPU task against one loaded pipeline, so the marginal
    cost of another column is one render, not another 11GB load.

    `steps` / `guidance` / `shift` pin the knobs the axis is NOT moving, instead of
    leaving them at the checkpoint's defaults. Some axes are meaningless without this:
    a `cfg_end` sweep asks whether restricting guidance to part of the schedule buys
    you adherence without harshness, and at the default CFG of 7 there is not much
    harshness to remove, so the row shows nothing. Run it at `--guidance 20` and the
    question has teeth. The axis always wins over these if they name the same field.
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

    w = await _weights_for(spec)

    jobs = []
    for v in vals:
        st = _settings_for(b, seed=seed, duration=duration)
        # Pins first, axis second: if someone sweeps steps AND passes --steps, the axis
        # is the thing they asked to vary and it has to win.
        st.steps, st.guidance, st.shift = steps, guidance, shift
        setattr(st, ax.field, v)
        badges = [f"{ax.label}"] + (["inert on this checkpoint"] if inert else [])
        jobs.append(GenJob(label=f"{ax.label} = {ax.fmt.format(v)}",
                           prompt=b.prompt, lyrics=b.lyrics, badges=badges, settings=st))

    run = await render_task_for(spec).override(short_name=f"sweep {axis}")(
        model_key, w, jobs)
    cards = await _to_results(run, sublabel=f"{model_key} · {spec.family}",
                              spec=spec, jobs=jobs, brief=brief)

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


# ── density: how many seconds does a line of lyric need? ─────────────────────────

# ACE-Step's floor. Asking for less is rejected by the pipeline, so the shortest cells
# in the grid get clamped up to it and the card says the density it ACTUALLY ran at.
_MIN_DURATION = 10.0


@orch_env.task(report=True)
async def density(
    model_key: str = "xl-turbo",
    lines: list[str] | None = None,
    per_line: list[str] | None = None,
    durations: list[str] | None = None,
    seed: int = 42,
    instrumental_control: bool = True,
) -> ModelRun:
    """Cross lyric LENGTH against room-per-line, to find where singing gets crammed.

    Two modes, because the two things you want to hold constant are different
    experiments and the first one cannot answer the second's question.

    DENSITY MODE (default, or `--per_line`). Columns are seconds per sung line and
      duration is DERIVED (`lines x per_line`). Driving the grid by duration instead
      would sample every row at a different density, leaving no column to compare down.
        ACROSS a row: the same words with more and more room.
        DOWN a column: constant density at very different absolute durations. If short
          tracks sound worse here, duration is doing something on its own.

    DURATION MODE (`--durations`). Columns are absolute track lengths and DENSITY is
      derived, so a column holds the track length fixed while the lyric gets longer and
      the words get more crammed. This is the mode that breaks the confound at the
      roomy end of the grid, where the best-sounding cell is also the longest track:
      render that same length with a quarter of the words and you find out which of the
      two you were hearing.

    `instrumental_control` adds a wordless row at the same durations as the 8-line row.
    A wordless clip has nothing to cram, so if those still sound synthetic when short,
    the penalty was never about lyric density.
    """
    spec = get_spec(model_key)
    lens = [int(x) for x in lines] if lines else sorted(prompts.DENSITY_LYRICS)
    by_duration = bool(durations)
    fixed = [float(x) for x in durations] if durations else []
    dens = [float(x) for x in per_line] if per_line else [1.5, 2.5, 4.0, 6.0]

    unknown = [n for n in lens if n not in prompts.DENSITY_LYRICS]
    if unknown:
        raise ValueError(f"no lyric with {unknown} sung lines; have "
                         f"{sorted(prompts.DENSITY_LYRICS)}")

    def _cell(n_lines: int, col: float) -> tuple[float, float, bool]:
        """(duration, seconds per line, was it clamped up to the model's floor).

        `col` is a density in density mode and an absolute duration in duration mode;
        one of the two is always derived from the other and the card shows BOTH, so a
        cell means the same thing however the grid was driven.
        """
        want = col if by_duration else n_lines * col
        got = max(want, _MIN_DURATION)
        return got, got / n_lines, got > want

    cols = fixed if by_duration else dens
    await flyte.report.replace.aio(music_core.render_status(
        "ACE-Step: how much room does a lyric line need?",
        f"{model_key} · {len(lens)} lyric lengths x {len(cols)} "
        f"{'durations' if by_duration else 'densities'}"
        f"{' + an instrumental control row' if instrumental_control else ''} · seed "
        f"{seed} · "
        + (f"the TRACK LENGTH is held fixed down each column and the lyric gets "
           f"longer, so density is derived and the words get more crammed as you read "
           f"down. This is the mode that separates 'roomy' from 'long'."
           if by_duration else
           f"duration is DERIVED as lines x seconds-per-line, so a column is a "
           f"constant density and a row is the same words with more and more room.")
        + " Rendering all of it against one loaded pipeline."))
    await flyte.report.flush.aio()

    # One job list, one GPU task: the whole grid runs against a single loaded pipeline,
    # so cell number twenty costs a render rather than another 11GB load.
    jobs: list[GenJob] = []
    layout: list[tuple[str, str, str, list[GenJob]]] = []   # heading, note, lyrics, jobs
    for n in lens:
        lyr = prompts.DENSITY_LYRICS[n]
        row: list[GenJob] = []
        seen_durations: set[float] = set()
        for d in cols:
            dur, real_d, clamped = _cell(n, d)
            # Two requested densities can clamp to the same duration on a short lyric
            # (4 lines at 1.5s and at 2.5s are both a 10s track). Rendering both would
            # put two identical cards in a row, which is the same "why is this repeated"
            # confusion the shared-lyric fold caused. Render it once.
            if dur in seen_durations:
                continue
            seen_durations.add(dur)
            badges = [f"{real_d:.1f}s per line", f"{dur:g}s track"]
            if clamped:
                badges.append(f"clamped up from {n * d:g}s (model floor is "
                              f"{_MIN_DURATION:g}s)")
            j = GenJob(label=f"{real_d:.1f}s/line · {dur:g}s",
                       prompt=prompts.DENSITY_PROMPT, lyrics=lyr, badges=badges,
                       settings=GenSettings(seed=seed, duration=dur,
                                            bpm=prompts.DENSITY_BPM))
            row.append(j)
            jobs.append(j)
        layout.append((
            f"{n} sung lines",
            (f"The same {n}-line lyric at {len(row)} fixed track lengths. Density is "
             f"derived, so compare this row against the others AT THE SAME LENGTH: the "
             f"track is equally long in every case and only the number of words "
             f"changes."
             if by_duration else
             f"The same {n}-line lyric at {len(row)} densities. Left is crammed, right "
             f"is roomy.")
            + " Listen for syllables clipped short, breaths disappearing between "
              "lines, and consonants softening into the backing: those go first, "
              "before anything sounds obviously broken.",
            lyr, row))

    if instrumental_control:
        # Same DURATIONS as the 8-line row, no words. Density is undefined without a
        # lyric, which is the point: this row isolates whatever short renders do on
        # their own. If these sound fine at 12s while the sung 12s cell does not, the
        # penalty was cramming. If they degrade too, it is duration.
        ref = lens[0] if 8 not in lens else 8
        row = []
        seen_durations = set()
        for d in cols:
            dur, _, _ = _cell(ref, d)
            if dur in seen_durations:
                continue
            seen_durations.add(dur)
            j = GenJob(label=f"instrumental · {dur:g}s",
                       prompt=prompts.DENSITY_PROMPT.replace(
                           "with a clear female lead vocal, ", "instrumental, "),
                       lyrics="", badges=[f"{dur:g}s track", "no lyric"],
                       settings=GenSettings(seed=seed, duration=dur,
                                            bpm=prompts.DENSITY_BPM))
            row.append(j)
            jobs.append(j)
        layout.append((
            "instrumental control (no lyric)",
            f"The same durations as the {ref}-line row with NO words, so there is "
            f"nothing to cram. This row is the control for the competing explanation: "
            f"if these short clips also sound synthetic, the problem is duration "
            f"itself and not lyric density.",
            "", row))

    w = await _weights_for(spec)
    run = await render_task_for(spec).override(short_name=f"density {model_key}")(
        model_key, w, jobs)
    cards = await _to_results(run, sublabel=f"{model_key} · {spec.family}", spec=spec,
                              jobs=jobs)

    by_job = {id(j): i for i, j in enumerate(jobs)}
    blocks = [Block(heading=heading, note=note, prompt=rjobs[0].prompt, lyrics=lyr,
                    lyrics_label=f"lyrics ({prompts.sung_lines(lyr)} sung lines)"
                                 if lyr else "lyrics",
                    results=[_at(cards, by_job[id(j)], j.label) for j in rjobs])
              for heading, note, lyr, rjobs in layout]

    if by_duration:
        meta = (f"{model_key} · {len(jobs)} tracks · seed {seed} · one caption, one "
                f"nested lyric · columns are FIXED track lengths and density is "
                f"derived · compare rows at the same length: same duration, more and "
                f"more words crammed into it")
        title = "Same length, more words: is it roomy or just long?"
    else:
        meta = (f"{model_key} · {len(jobs)} tracks · seed {seed} · one caption, one "
                f"nested lyric · columns are seconds per sung line, duration is "
                f"derived · read ACROSS for the effect, DOWN for whether short is bad "
                f"on its own")
        title = "How much room does a lyric line need?"
    await flyte.report.replace.aio(music_core.render_report(
        blocks, title=title, meta=meta))
    await flyte.report.flush.aio()
    return run


# ── variants: the studio's entry point ───────────────────────────────────────────

def _auto_label(v: Variant, base: Variant, i: int) -> str:
    """Name a card by what makes it DIFFERENT, falling back to its position.

    A row of cards all reading "xl-sft" is useless, and a row reading the full settings
    twice over is unreadable. The useful label is the delta against the first variant,
    which is exactly the thing the eye is looking for when scanning the row.
    """
    if v.label:
        return v.label
    bits = []
    if v.model_key != base.model_key:
        bits.append(v.model_key)
    if v.seed != base.seed:
        bits.append(f"seed {v.seed}")
    if v.duration != base.duration:
        bits.append(f"{v.duration:g}s" if v.duration else "auto length")
    if v.steps != base.steps:
        bits.append(f"{v.steps} steps")
    if v.guidance != base.guidance:
        bits.append(f"cfg {v.guidance:g}")
    if v.shift != base.shift:
        bits.append(f"shift {v.shift:g}")
    if v.bpm != base.bpm:
        bits.append(f"{v.bpm} bpm")
    if v.keyscale != base.keyscale:
        bits.append(v.keyscale)
    if v.timesignature != base.timesignature:
        bits.append(f"{v.timesignature}/4")
    if (v.cfg_interval_start, v.cfg_interval_end) != (base.cfg_interval_start,
                                                      base.cfg_interval_end):
        bits.append(f"cfg over {v.cfg_interval_start:g}-{v.cfg_interval_end:g}")
    if v.language != base.language:
        bits.append(f"lang {v.language}")
    return " · ".join(bits) if bits else (f"take {i + 1}" if i else "base")


def _variant_covers_gensettings() -> None:
    """Variant must expose every GenSettings knob, or the studio silently cannot reach one.

    This drifted once already: `cfg_interval_start` and `language` were wired into
    GenSettings and the pipeline but never added here, so no amount of clicking in the
    studio could set them and nothing anywhere said so. A missing knob is invisible by
    construction, which is exactly the kind of bug that needs an assertion rather than
    a reviewer.
    """
    from dataclasses import fields as _f
    gs = {f.name for f in _f(GenSettings())}
    mine = {f.name for f in _f(Variant())} - {"label", "model_key"}
    missing = gs - mine
    assert not missing, f"Variant is missing GenSettings knobs: {sorted(missing)}"


_variant_covers_gensettings()


@orch_env.task(report=True)
async def variants(
    prompt: str,
    lyrics: str = "",
    takes: list[Variant] | None = None,
    title: str = "",
) -> list[ModelRun]:
    """Render one song several ways and put the takes side by side.

    This is what the studio submits. Variants are grouped by checkpoint so each one
    loads once however many takes use it, which is the whole reason this is a single
    entry point rather than the app firing N separate runs: N runs would mean N pods,
    N 11GB loads, and (as this repo learned the hard way) N orchestrators competing for
    the memory their own children need.
    """
    vs = list(takes or [Variant(seed=42), Variant(seed=7)])
    base = vs[0]

    # Derive any auto durations up front so the labels and the status line agree with
    # what actually renders.
    auto = prompts.suggest_durations(lyrics)
    default_len = auto[len(auto) // 2] if auto else 30.0
    for v in vs:
        if v.duration <= 0:
            v.duration = default_len

    n_lines = prompts.sung_lines(lyrics)
    await flyte.report.replace.aio(music_core.render_status(
        title or "ACE-Step studio",
        f"{len(vs)} take(s) · "
        + (f"{n_lines} sung lines, default length {default_len:g}s "
           f"({default_len / n_lines:.1f}s per line)" if n_lines else "instrumental")
        + f" · checkpoints: {', '.join(sorted({v.model_key for v in vs}))}. "
          f"Rendering; the report is replaced as each checkpoint finishes."))
    await flyte.report.flush.aio()

    # Group by checkpoint, remembering each take's ORIGINAL position so the report can
    # put the cards back in the order the user built them, not in load order.
    by_model: dict[str, list[tuple[int, Variant]]] = {}
    for i, v in enumerate(vs):
        by_model.setdefault(v.model_key, []).append((i, v))

    jobs_by_model: dict[str, list[GenJob]] = {}
    for mk, items in by_model.items():
        jobs_by_model[mk] = [
            GenJob(label=_auto_label(v, base, i), prompt=prompt, lyrics=lyrics,
                   badges=[mk],
                   settings=GenSettings(
                       seed=v.seed, steps=v.steps, guidance=v.guidance, shift=v.shift,
                       duration=v.duration, bpm=v.bpm, keyscale=v.keyscale,
                       timesignature=v.timesignature, language=v.language,
                       cfg_interval_start=v.cfg_interval_start,
                       cfg_interval_end=v.cfg_interval_end))
            for i, v in items]

    specs = {mk: get_spec(mk) for mk in by_model}
    weights: dict[str, flyte.io.Dir] = {}
    for mk, s in specs.items():
        weights[mk] = await _weights_for(s)

    raw = await asyncio.gather(*[
        render_task_for(specs[mk]).override(short_name=f"take {mk}")(
            mk, weights[mk], jobs_by_model[mk])
        for mk in by_model
    ], return_exceptions=True)

    runs: list[ModelRun] = []
    cards: dict[int, TrackResult] = {}
    for mk, res in zip(by_model, raw):
        items = by_model[mk]
        if isinstance(res, Exception):
            log.exception(f"[{mk}] render failed: {res}")
            for pos, _ in items:
                cards[pos] = TrackResult(label=mk, error=f"render failed: {res}")
            continue
        runs.append(res)
        got = await _to_results(res, sublabel=specs[mk].family, spec=specs[mk],
                                jobs=jobs_by_model[mk])
        for k, (pos, _) in enumerate(items):
            cards[pos] = _at(got, k, jobs_by_model[mk][k].label)

    block = Block(
        heading=title or "Takes",
        note="Same words and same caption in every card; only the knobs differ, and "
             "each card is labelled with what makes it different. The numbers under "
             "each card are measurements, not scores.",
        prompt=prompt, lyrics=lyrics,
        lyrics_label=(f"lyrics ({n_lines} sung lines)" if n_lines else "lyrics"),
        results=[cards[i] for i in sorted(cards)])
    await flyte.report.replace.aio(music_core.render_report(
        [block], title=title or "ACE-Step studio",
        meta=f"{len(vs)} takes · " + ", ".join(
            f"{_auto_label(v, base, i)}" for i, v in enumerate(vs))))
    await flyte.report.flush.aio()
    return runs


# ── grid: two knobs at once, because they may interact ───────────────────────────

@orch_env.task(report=True)
async def grid(
    model_key: str = "xl-sft",
    brief: str = "ballad-flyte-callouts",
    steps: list[str] | None = None,
    guidance: list[str] | None = None,
    duration: float = 240.0,
    seed: int = 42,
) -> ModelRun:
    """Cross STEPS against GUIDANCE and render every cell against one loaded pipeline.

    `sweep` deliberately moves one knob so the cause of any difference is unambiguous,
    and that is the right default. But it cannot answer whether two knobs INTERACT, and
    these two plausibly do: guidance pushes each denoising step harder toward the
    prompt, so the damage a high CFG does may depend on how many steps it gets to do it
    over. More steps could refine away the harshness, or compound it. A pair of
    one-knob sweeps cannot distinguish those.

    Rows are step counts, columns are guidance values, so the diagonal is "both cranked"
    and the corners bracket it. Defaults to `xl-sft` because on a distilled checkpoint
    the guidance axis is inert and the whole grid collapses to one column.
    """
    spec = get_spec(model_key)
    b = get_brief(brief)
    step_vals = [int(float(x)) for x in steps] if steps else [50, 200]
    cfg_vals = [float(x) for x in guidance] if guidance else [7.0, 20.0]

    inert = spec.distilled
    warn = (f" NOTE: {model_key} is guidance-distilled, so every column will be "
            f"identical (the pipeline coerces guidance to 1.0). Use --model_key xl-sft."
            if inert else "")

    await flyte.report.replace.aio(music_core.render_status(
        f"ACE-Step: {len(step_vals)}x{len(cfg_vals)} steps x guidance",
        f"{model_key} · brief '{brief}' · steps {step_vals} x guidance {cfg_vals} · "
        f"{duration:g}s each · seed {seed}. Cost scales with steps, so the "
        f"bottom-right cell is the expensive one.{warn}"))
    await flyte.report.flush.aio()

    jobs: list[GenJob] = []
    rows: list[tuple[int, list[GenJob]]] = []
    for st_val in step_vals:
        row = []
        for cfg in cfg_vals:
            s = _settings_for(b, seed=seed, duration=duration)
            s.steps, s.guidance = st_val, cfg
            j = GenJob(label=f"{st_val} steps · cfg {cfg:g}",
                       prompt=b.prompt, lyrics=b.lyrics,
                       badges=[f"{st_val} steps", f"cfg {cfg:g}"], settings=s)
            row.append(j)
            jobs.append(j)
        rows.append((st_val, row))

    w = await _weights_for(spec)
    run = await render_task_for(spec).override(short_name=f"grid {model_key}")(
        model_key, w, jobs)
    cards = await _to_results(run, sublabel=f"{model_key} · {spec.family}", spec=spec,
                              jobs=jobs, brief=brief)

    idx = {id(j): i for i, j in enumerate(jobs)}
    blocks = [Block(
        heading=f"{st_val} denoising steps",
        note=("Across this row, guidance rises at a fixed step count. Low drifts and "
              "sounds generic; high obeys, then over-obeys, and turns harsh and "
              "brittle with the instruments fighting. Compare the SAME guidance value "
              "against the other rows: if a high CFG is harsh at a low step count but "
              "not at a high one, the extra steps are refining the damage away and the "
              "two knobs are not independent."),
        prompt=b.prompt, lyrics=b.lyrics,
        results=[_at(cards, idx[id(j)], j.label) for j in row])
        for st_val, row in rows]

    meta = (f"{model_key} · {len(jobs)} cells · {brief} · {duration:g}s · seed {seed} · "
            f"rows are steps, columns are guidance")
    await flyte.report.replace.aio(music_core.render_report(
        blocks, title="Steps x guidance: do they interact?", meta=meta))
    await flyte.report.flush.aio()
    return run


# ── takes: a menu of lengths to choose from ──────────────────────────────────────

@orch_env.task(report=True)
async def takes(
    model_key: str = "xl-turbo",
    brief: str = DEFAULT_BRIEF,
    prompt: str = "",
    lyrics: str = "",
    durations: list[str] | None = None,
    seeds: list[str] | None = None,
) -> ModelRun:
    """Render one song at several LENGTHS (and optionally several seeds), then pick.

    Not the same job as `sweep --axis duration`, though they render similar things.
    A sweep moves one knob to explain what it does; this is the working surface for
    actually making a song, where length is not a parameter you reason about but a
    judgement you make by ear once you have heard the options next to each other.

    That distinction is why the lengths are DERIVED from the lyric by default rather
    than fixed. `prompts.suggest_durations` brackets the estimated
    seconds-per-line target by roughly 3x, so the ladder contains the right answer
    even though the target itself is still provisional. A 4-line hook and a 24-line
    ballad get very different ladders, which is exactly the thing a fixed 20/40/80
    gets wrong.

    Every card carries its own reproduce command, so choosing is: listen, then copy the
    handle off the winner and render that one again at a higher step count or another
    seed.
    """
    spec = get_spec(model_key)
    b = get_brief(brief)
    the_prompt = prompt or b.prompt
    the_lyrics = lyrics if prompt else b.lyrics

    durs = ([float(x) for x in durations] if durations
            else prompts.suggest_durations(the_lyrics))
    the_seeds = [int(x) for x in seeds] if seeds else [42]
    n_lines = prompts.sung_lines(the_lyrics)

    suggested = "" if durations else (
        f" Lengths chosen for this lyric: {n_lines} sung lines x ~"
        f"{prompts.SECONDS_PER_LINE:g}s, bracketed. Override with --durations."
        if n_lines else " No lyric, so the ladder is the fixed instrumental one.")

    await flyte.report.replace.aio(music_core.render_status(
        f"ACE-Step: {len(durs) * len(the_seeds)} takes of '{brief}'",
        f"{model_key} · lengths {', '.join(f'{d:g}s' for d in durs)} · seeds "
        f"{', '.join(map(str, the_seeds))}.{suggested} Rendering all of them against "
        f"one loaded pipeline."))
    await flyte.report.flush.aio()

    jobs: list[GenJob] = []
    rows: list[tuple[int, list[GenJob]]] = []
    for s in the_seeds:
        row = []
        for d in durs:
            st = _settings_for(b, seed=s, duration=d)
            if prompt:                      # a hand-written prompt brings no metadata
                st.bpm, st.keyscale = 0, ""
            badges = [f"{d:g}s"] + ([f"{d / n_lines:.1f}s per line"] if n_lines else
                                    ["instrumental"])
            j = GenJob(label=f"{d:g}s", prompt=the_prompt, lyrics=the_lyrics,
                       badges=badges, settings=st)
            row.append(j)
            jobs.append(j)
        rows.append((s, row))

    w = await _weights_for(spec)
    run = await render_task_for(spec).override(short_name=f"takes {brief}")(
        model_key, w, jobs)
    cards = await _to_results(run, sublabel=f"{model_key} · {spec.family}", spec=spec,
                              jobs=jobs, brief=brief if not prompt else "")

    idx = {id(j): i for i, j in enumerate(jobs)}
    note = ("Same words, same seed, different amounts of room. Length is not a crop: "
            "it is fed to the model up front, so these are different arrangements of "
            "the same song rather than longer and shorter cuts of one take. Pick the "
            "one that breathes, then copy its reproduce command.")
    blocks = [Block(heading=f"seed {s}", note=note, prompt=the_prompt,
                    lyrics=the_lyrics,
                    lyrics_label=(f"lyrics ({n_lines} sung lines)" if n_lines
                                  else "lyrics"),
                    results=[_at(cards, idx[id(j)], j.label) for j in row])
              for s, row in rows]

    what = "custom prompt" if prompt else brief
    meta = (f"{model_key} · {len(jobs)} takes · {what} · lengths "
            f"{', '.join(f'{d:g}s' for d in durs)}"
            + (f" (derived from {n_lines} sung lines)" if not durations and n_lines
               else ""))
    await flyte.report.replace.aio(music_core.render_report(
        blocks, title=f"Takes: '{brief}' at {len(durs)} lengths", meta=meta))
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
    timesignature: str = "",
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
                     timesignature=timesignature,
                     language=language)

    jobs = [GenJob(label=model_key, prompt=the_prompt, lyrics=the_lyrics,
                   badges=[spec.params], settings=st)]
    w = await _weights_for(spec)
    run = await render_task_for(spec).override(short_name=f"render {model_key}")(
        model_key, w, jobs)

    # `brief` only names the card honestly when the brief actually supplied the text;
    # a caller-supplied --prompt overrides it, and claiming otherwise would emit a
    # command that reproduces something different from what you are looking at.
    cards = await _to_results(run, sublabel=spec.family, spec=spec, jobs=jobs,
                              brief="" if prompt else brief)
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


# ── DiffRhythm smoke test ────────────────────────────────────────────────────────
#
# Deliberately separate from `render`: DiffRhythm is the one model with its own image,
# so before writing an adapter against a repo that has no packaging and no Python API,
# prove the image builds and its modules import on aarch64. The build itself already
# runs an import check; this confirms the same thing inside a real GPU pod, with CUDA.
@diffrhythm_env.task
async def diffrhythm_smoke() -> str:
    import sys

    import torch

    from config import DIFFRHYTHM_DIR

    sys.path.insert(0, DIFFRHYTHM_DIR)
    from model import CFM, DiT           # noqa: F401  (the CFM + DiT stack)
    from muq import MuQMuLan            # noqa: F401  (the style encoder)

    import transformers
    from transformers.models.llama import LlamaConfig  # noqa: F401  dit.py needs this

    out = (f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
           f"transformers={transformers.__version__} imports=OK")
    log.info(f"[diffrhythm] {out}")
    return out
