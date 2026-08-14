"""The engine room: load ACE-Step, render a track, embed it in a Flyte report.

Deliberately Flyte-free, so the exact same code runs inside the GPU task and inside
run_local.py. If a track plays in the standalone HTML run_local writes, it plays in
the Flyte report, because it is the identical renderer.

Three things carry over from the TTS and video demos and matter here too:

  1. Flyte reports render under a CSP that drops external assets and <script> tags.
     Audio still plays, because HTML5 <audio> needs no JS: a base64 clip in a data URI
     on an <audio controls> element is enough. That's `audio_data_uri`.
  2. The report needs a *visual* comparison surface, not just players. For music that
     is a waveform + full-bandwidth spectrogram: you can see the arrangement (a drop
     is a visible step in the envelope), see clipping, and see the lowpass brickwall
     that gives away a model rendering through a lossy bottleneck. That's
     `waveform_spectrogram_png`.
  3. Every listening claim in the report needs the numbers next to it, so each card
     carries wall-clock, audio length, the realized settings, and peak GPU.

── Stereo is not an implementation detail here ──────────────────────────────────
The TTS core collapsed everything to mono, which was right: a single voice has no
stereo image to lose. ACE-Step's Oobleck VAE decodes 48kHz *stereo*, and stereo width
is one of the things you are grading (a narrow, mono-ish mix is a real and audible
failure). So audio stays stereo end to end: stereo through the wav, stereo through
the embedded OGG. Only the spectrogram sums to mono, because a two-channel
spectrogram is unreadable at report size.
"""

from __future__ import annotations

import base64
import faulthandler
import gc
import html
import io
import json
import logging
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless: no display, render straight to PNG bytes
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

log = logging.getLogger("acestep.core")

# Print a Python traceback to stderr if this process dies on SIGSEGV/SIGBUS/SIGFPE.
# Earned the hard way: three renders on this box died with exit 139 and left NOTHING
# in the logs after "loaded", which is what let a cgroup ceiling masquerade first as a
# track-length limit and then as a VAE tiling bug. A native crash in a CUDA workload
# is not exotic on unified memory, and one traceback would have saved two wrong turns.
faulthandler.enable()

# Past this the <audio> data URI is dropped and only the spectrogram is shown. A 60s
# stereo 48kHz Vorbis clip is ~1MB, so this is headroom for a full-length track, not a
# real constraint on the default runs.
MAX_EMBED_BYTES = 8_000_000

# The GB10's "GPU memory" is the same unified pool the OS is using. Letting an
# allocation run the pool to the wall does not OOM, it HANGS the whole box, so we cap
# the process before any large load.
#
# MEASURED AGAINST AVAILABLE, NOT TOTAL, and not against CUDA's idea of "free"
# either. Both halves of that were learned the expensive way:
#
#  1. `set_per_process_memory_fraction` is a fraction of the pool's TOTAL size. On a
#     unified-memory box every other process spends from the same pool, so a leaky
#     neighbour holding 19GB is 19GB the renderer cannot have. Cap against total and
#     torch believes it owns 107GB of a pool with 83GB left, then dies mid-render:
#     once as a catchable CUDA OOM, once as a bare SIGSEGV with no traceback.
#
#  2. The obvious fix, `torch.cuda.mem_get_info()`, is WRONG here and is worse than
#     doing nothing. On this box it reported "3GB free of 129GB" while the host had
#     108GB available, because the kernel's ~69GB of page cache counts as not-free
#     even though it is reclaimable on demand. That capped the process at 2.55GB and
#     the 11GB model load OOMed instantly.
#
# So: read MemAvailable from /proc/meminfo, which is the kernel's own estimate of what
# is obtainable *including* reclaimable cache, and floor the result so that a bad
# reading can never cap below what the model needs to load at all.
GPU_MEMORY_FRACTION = 0.90      # ceiling as a share of the whole pool
GPU_AVAIL_HEADROOM = 0.92       # and no more than this share of what is available now
GPU_MIN_FRACTION = 0.30         # ~36GB of a 120GB pool: never cap tighter than this

# ── Which knobs does each adapter actually honour? ───────────────────────────────
#
# The single source of truth for "does this model have this dial", used by the report
# card, by the reproduce command, and by the studio's Advanced panel. Printing a knob a
# model does not have is the same class of dishonesty as printing a value it silently
# ignored, and once a registry spans architectures the answer stops being guessable:
#
#   acestep   flow-matching DiT   : steps, guidance, shift, and musical metadata
#   musicgen  autoregressive      : guidance only. No denoising steps, no flow shift.
#   audioldm2 latent diffusion    : steps and guidance, but no shift and no metadata
#
# A binary "is it diffusion" test was not enough: AudioLDM2 has steps but no shift.
ADAPTER_KNOBS: dict[str, set[str]] = {
    "acestep": {"steps", "guidance", "shift", "bpm", "keyscale", "timesignature",
                "language", "lyrics", "cfg_interval"},
    "musicgen": {"guidance"},
    "audioldm2": {"steps", "guidance"},
    "stableaudio": {"steps", "guidance"},
    "diffrhythm": {"steps", "guidance", "lyrics"},
}


# Adapters that render by spawning a child process rather than in-process. The parent
# must leave CUDA alone for these: any context or memory cap it creates only competes
# with the child for the same unified pool.
SUBPROCESS_ADAPTERS = {"diffrhythm"}


def knobs_for(adapter: str) -> set[str]:
    """The knobs an adapter honours; an unknown adapter gets the shared minimum."""
    return ADAPTER_KNOBS.get(adapter, {"guidance"})


# ── GPU helpers (import-safe on a CPU-only host) ─────────────────────────────────

def free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def reset_peak_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def peak_memory_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0


def container_memory_limit_gb() -> float:
    """The cgroup memory ceiling this process actually lives under, in GB. 0 if none.

    A real ceiling that host MemAvailable does not capture: on GB10 the GPU pool IS
    host memory, so a CUDA allocation is charged to this container's cgroup like any
    other page, and the box can have 100GB free while the pod is capped at 48GB.
    Telling torch it may use 92GB inside a 48Gi container is asking for trouble, so
    always cap against the smaller of the two.

    (Honesty note: this was introduced while hunting a run of SIGSEGVs that turned out
    to be a libsndfile bug in `encode_audio`, NOT a cgroup overrun. Capping against the
    cgroup is still correct and still worth doing, it just was not the fix for that.)
    """
    for path, unlimited in (("/sys/fs/cgroup/memory.max", "max"),                    # v2
                            ("/sys/fs/cgroup/memory/memory.limit_in_bytes", None)):  # v1
        try:
            raw = Path(path).read_text().strip()
            if not raw or raw == unlimited:
                continue
            b = int(raw)
            # cgroup v1 reports a huge sentinel rather than "max" when unlimited.
            if 0 < b < (1 << 62):
                return b / 1e9
        except Exception:
            continue
    return 0.0


def host_available_gb() -> float:
    """The kernel's MemAvailable, in GB. 0.0 if it cannot be read.

    This, not `torch.cuda.mem_get_info()`, is the honest number on a unified-memory
    box: MemAvailable counts reclaimable page cache, and CUDA's `free` does not (see
    the constants above for what that cost).
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9   # kB -> bytes -> GB
    except Exception:
        pass
    return 0.0


def memory_budget_gb() -> tuple[float, float, float]:
    """(budget_gb, host_available_gb, cgroup_limit_gb) for this process.

    The budget is the smaller of what the host can spare and what the container is
    allowed, because on unified memory both are ceilings on the same pages and the
    tighter one wins.
    """
    host = host_available_gb()
    cg = container_memory_limit_gb()
    candidates = [x for x in (host, cg) if x > 0]
    return (min(candidates) if candidates else 0.0), host, cg


def gpu_pool_gb() -> tuple[float, float]:
    """(budget_gb, total_gb) for the pool, or (0, 0) off-GPU.

    `budget` is the min of host MemAvailable and the cgroup limit, NOT CUDA's `free`
    (which counts reclaimable page cache as used and once reported 3GB of 129GB here).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        # mem_get_info needs a live CUDA context. Without this it can raise on the
        # first call of a fresh process, and a silent `except: pass` then reports
        # "0/0GB, capped at 0GB" and skips the cap entirely: observed in a real run,
        # where the only symptom was a log line reading zero.
        torch.cuda.init()
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            total_gb = total_b / 1e9
        except Exception:
            log.warning("mem_get_info failed; falling back to device properties",
                        exc_info=True)
            free_b = 0
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        budget, _, _ = memory_budget_gb()
        return (budget or free_b / 1e9), total_gb
    except Exception:
        log.exception("could not size the GPU pool; the memory cap will be SKIPPED")
    return 0.0, 0.0


def prepare_gpu() -> float:
    """Clean the pool, cap the process against what is actually free, reset the peak.

    Returns the cap in GB (0.0 off-GPU) so the caller can log it; a run that later dies
    on OOM should be able to say what ceiling it was working under.

    The cap is the important line. ACE-Step XL is ~11GB in bf16 and nowhere near the
    unified pool's limit, but `from_pretrained(...).to("cuda")` briefly holds two
    copies and a long render allocates far more than the weights on top. Both are
    survivable; running a GB10's pool dry is not, because it hangs the box instead of
    raising.
    """
    free_gpu_memory()
    cap_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            _, total_gb = gpu_pool_gb()
            budget_gb, host_gb, cgroup_gb = memory_budget_gb()
            frac = GPU_MEMORY_FRACTION
            if budget_gb and total_gb:
                frac = min(frac, GPU_AVAIL_HEADROOM * budget_gb / total_gb)
            # The floor guards against the cure being worse than the disease: a misread
            # of AVAILABLE memory must degrade to "roughly unconstrained" rather than a
            # cap so tight the model cannot load. But a cgroup limit is not a misread,
            # it is a hard ceiling, so never floor above one we actually read.
            if frac < GPU_MIN_FRACTION:
                if cgroup_gb:
                    log.warning(f"cgroup limits this container to {cgroup_gb:.0f}GB of a "
                                f"{total_gb:.0f}GB pool; honoring it. Raise the task's "
                                f"`memory=` if renders need more.")
                else:
                    log.warning(f"computed cap {frac:.2f} of pool is implausibly tight "
                                f"(host available {host_gb:.0f}GB); flooring at "
                                f"{GPU_MIN_FRACTION:.2f}")
                    frac = GPU_MIN_FRACTION
            torch.cuda.set_per_process_memory_fraction(frac)
            cap_gb = frac * total_gb
    except Exception:
        log.exception("could not set the GPU memory cap; running UNCAPPED")
    if cap_gb <= 0:
        # Loud on purpose. An uncapped process on GB10 is the failure mode this whole
        # function exists to prevent, and it previously announced itself only as a
        # cheerful "capped at 0GB" that read like a formatting quirk.
        log.warning("GPU memory cap NOT applied (pool size unreadable); a large render "
                    "can exhaust the unified pool and take the box down with it")
    reset_peak_memory()
    return cap_gb


# ── Generation settings ──────────────────────────────────────────────────────────

@dataclass
class GenSettings:
    """One render's knobs. Zero / empty means 'inherit the checkpoint's default'.

    Sentinels rather than Optionals so a whole settings object survives a round trip
    through the Flyte CLI as plain JSON, and so `--steps 0` reads as "whatever this
    checkpoint wants" instead of an error. `resolve` folds a MusicModelSpec in and
    returns the concrete values actually handed to the pipeline; the report shows
    those, never the sentinels.
    """
    seed: int = 42
    steps: int = 0             # 0 -> spec.steps
    guidance: float = -1.0     # <0 -> spec.guidance
    shift: float = -1.0        # <0 -> spec.shift
    duration: float = 30.0
    bpm: int = 0               # 0 -> let the model estimate
    keyscale: str = ""         # "" -> let the model estimate
    timesignature: str = ""    # "" -> let the model estimate
    language: str = "en"

    # WHERE on the denoising schedule CFG is applied, as a fraction. The pipeline's own
    # defaults are the full range, and unlike the knobs above these are real values
    # rather than sentinels: 0.0 is a meaningful start, so there is no spare value to
    # mean "inherit".
    #
    # Worth having because guidance has two effects that arrive at different times. Its
    # useful effect is early, where the schedule is deciding form, groove and
    # arrangement and being pushed toward the prompt actually helps. Its damage is late,
    # where the remaining work is texture and detail, and pushing hard there is what
    # makes a high CFG sound harsh, brittle and over-saturated. Restricting CFG to the
    # first part of the schedule is therefore the standard way to ask for adherence
    # WITHOUT the artifacts, and it is the reason a plain guidance sweep can leave you
    # choosing between "drifts" and "harsh" with nothing in between.
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0

    def resolve(self, spec) -> "GenSettings":
        """Concrete settings for `spec`, with the checkpoint's defaults filled in."""
        out = GenSettings(**vars(self))
        if out.steps <= 0:
            out.steps = spec.steps
        if out.guidance < 0:
            out.guidance = spec.guidance
        if out.shift < 0:
            out.shift = spec.shift
        # A distilled checkpoint ignores CFG (the pipeline warns and coerces to 1.0).
        # Do the coercion here too, so the report card states what the model actually
        # ran instead of the number that was requested and silently discarded.
        if spec.distilled and out.guidance > 1.0:
            out.guidance = 1.0
        # Same principle for a hard architectural ceiling. MusicGen was trained on 30s
        # windows and does not refuse a longer request, it degrades into repetition, so
        # a `compare` run at 120s would quietly hand it an unwinnable task and let the
        # report imply it simply sounds worse. Clamp, and let the card show the clamped
        # number so the difference is legible as a CONSTRAINT rather than a failure.
        cap = getattr(spec, "max_duration", 0.0)
        if cap and out.duration > cap:
            out.duration = cap
        return out

    def as_dict(self) -> dict:
        """The knobs as plain JSON-able data. Resolve() first if you want what ran."""
        return dict(vars(self))

    def summary(self, spec=None) -> str:
        """The one-line settings string under a report card.

        Adapter-aware, because printing a knob a model does not have is the same class
        of lie as printing a value it ignored. MusicGen is autoregressive: it has no
        denoising steps and no flow-matching shift, so a card reading "8 steps · shift
        3" next to a MusicGen track would be inventing numbers. It does use CFG, so
        that stays.
        """
        adapter = getattr(spec, "adapter", "acestep") if spec is not None else "acestep"
        has = knobs_for(adapter)
        bits = [f"seed {self.seed}"]
        if "steps" in has:
            bits.append(f"{self.steps} steps")
        if "guidance" in has:
            bits.append(f"cfg {self.guidance:g}")
        if "shift" in has:
            bits.append(f"shift {self.shift:g}")
        # Only shown when it is not the full range, because "cfg 0-1" on every card is
        # noise; a restricted interval is the notable thing and deserves to stand out.
        if "cfg_interval" in has and (self.cfg_interval_start > 0.0
                                      or self.cfg_interval_end < 1.0):
            bits.append(f"cfg over {self.cfg_interval_start:g}-"
                        f"{self.cfg_interval_end:g}")
        bits.append(f"{self.duration:g}s")
        if self.bpm:
            bits.append(f"{self.bpm} bpm")
        if self.keyscale:
            bits.append(self.keyscale)
        if self.timesignature:
            bits.append(f"{self.timesignature}/4")
        return " · ".join(bits)


# ── Load + generate ──────────────────────────────────────────────────────────────

def resolve_weights(local_dir: str | None, marker: str = "model_index.json") -> str | None:
    """Find the model root inside a downloaded weights Dir.

    `marker` is the file that identifies the root: `model_index.json` for a diffusers
    pipeline, `config.json` for a plain transformers checkpoint.

    The fetch task snapshots with `local_dir=`, so the Dir contains a plain repo
    layout: `model_index.json` next to `transformer/`, `vae/`, and friends. But the
    Dir may arrive wrapped in the temp directory's own basename, so look one and two
    levels down before giving up.

    Returning a PATH rather than pointing `HF_HUB_CACHE` at a cache-layout directory is
    the whole point, and it was a bug fix, not a style choice. The first version
    downloaded with `cache_dir=` and set `HF_HUB_CACHE` at load time; the GPU task then
    re-downloaded all 11GB from HuggingFace anyway, and the uploaded Dir carried the
    weights twice (once under `blobs/`, once under `snapshots/`, because the upload
    dereferenced the HF cache's symlinks). Handing `from_pretrained` a local path skips
    the cache-resolution machinery entirely: no network, no symlinks, no duplication.
    """
    if not local_dir:
        return None
    p = Path(local_dir)
    if (p / marker).exists():
        return str(p)
    for cand in sorted(p.glob("*/")) + sorted(p.glob("*/*/")):
        if (cand / marker).exists():
            return str(cand)
    # Nothing recognizable: fall back to the repo id so the run still works (slowly,
    # over the network) rather than failing on a layout surprise.
    return None


def load_pipeline(spec, local_dir: str | None = None, tile: bool = False):
    """Load a model onto the GPU, dispatching on `spec.adapter`.

    One image serves every adapter here (diffusers and transformers coexist fine), so
    unlike the TTS demo next door an adapter selects a code path, not a container.
    Each branch imports its package LAZILY so a future image that lacks one does not
    fail at module import.
    """
    if spec.adapter == "musicgen":
        return _load_musicgen(spec, local_dir)
    if spec.adapter == "audioldm2":
        return _load_audioldm2(spec, local_dir)
    if spec.adapter == "stableaudio":
        return _load_stableaudio(spec, local_dir)
    if spec.adapter == "diffrhythm":
        return None      # a subprocess per render; nothing to hold resident
    return _load_acestep(spec, local_dir, tile)


def _load_stableaudio(spec, local_dir: str | None = None):
    """Stable Audio Open: autoencoder + T5 + latent DiT, 44.1kHz stereo.

    The closest architectural like-for-like against ACE-Step in this registry, and the
    only other transformer-diffusion model here.
    """
    import torch
    from diffusers import StableAudioPipeline

    source = resolve_weights(local_dir, marker="model_index.json") or spec.repo
    log.info(f"loading stable-audio from {source}")
    pipe = StableAudioPipeline.from_pretrained(source, torch_dtype=getattr(torch, spec.dtype))
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _load_audioldm2(spec, local_dir: str | None = None):
    """AudioLDM 2: latent diffusion with a GPT-2 bridging CLAP and T5 embeddings.

    A third architecture in the registry and the one that most changes what the report
    looks like, because it decodes through a HiFi-GAN vocoder at **16kHz mono** rather
    than ACE-Step's 48kHz stereo. That is not a small difference: on the card's
    full-Nyquist spectrogram it shows up as a hard ceiling at 8kHz, right next to a
    track that has content out to 24kHz. It is the clearest single image in the whole
    demo for what "audio quality" means before you have listened to anything.
    """
    import torch
    from diffusers import AudioLDM2Pipeline

    source = resolve_weights(local_dir, marker="model_index.json") or spec.repo
    log.info(f"loading audioldm2 from {source}")
    pipe = AudioLDM2Pipeline.from_pretrained(source, torch_dtype=getattr(torch, spec.dtype))
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    _shim_audioldm2_generation(pipe)
    return pipe


def _shim_audioldm2_generation(pipe) -> None:
    """Re-attach the generation helper transformers 5 removed from `PreTrainedModel`.

    diffusers 0.39's AudioLDM2 pipeline drives its GPT-2 "language model" by hand and
    calls `_update_model_kwargs_for_generation` on it. Until transformers 4.53,
    `PreTrainedModel` inherited `GenerationMixin` so every model had that method; from
    4.53 onward it does not, and on 5.x the pipeline dies with:

        AttributeError: 'GPT2Model' object has no attribute
                        '_update_model_kwargs_for_generation'

    The alternatives were worse. Pinning the shared image to transformers<4.53 would
    drag ACE-Step and MusicGen back fifteen months to accommodate the oldest and
    weakest model in the registry; a separate image for AudioLDM2 would mean pairing a
    2026 diffusers with a 2025 transformers and hoping. Binding the one missing method
    back onto the instance is smaller than either, and it is honest about being a
    workaround for an upstream lag rather than a fix.

    Deliberately narrow: it patches ONE method on ONE instance, only when missing, and
    it is loud if the shim itself stops applying.
    """
    import types

    target = getattr(pipe, "language_model", None)
    if target is None or hasattr(target, "_update_model_kwargs_for_generation"):
        return
    try:
        from transformers.generation import GenerationMixin

        target._update_model_kwargs_for_generation = types.MethodType(
            GenerationMixin._update_model_kwargs_for_generation, target)
        log.info("shimmed _update_model_kwargs_for_generation onto AudioLDM2's GPT-2 "
                 "(transformers >=4.53 dropped it from PreTrainedModel)")
    except Exception:
        log.exception("could not shim AudioLDM2's language model; generation will "
                      "likely fail on this transformers version")


def _load_musicgen(spec, local_dir: str | None = None):
    """MusicGen: an autoregressive transformer over EnCodec tokens.

    Returns (model, processor) rather than a pipeline object, which is why `generate`
    dispatches too. `device_map="auto"` rather than `.to("cuda")`: the GB10's unified
    pool makes a naive load hold two copies of the weights at once, and at 20.4GB for
    the stereo-large checkpoint that is worth avoiding.
    """
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    source = resolve_weights(local_dir, marker="config.json") or spec.repo
    log.info(f"loading musicgen from {source}")
    processor = AutoProcessor.from_pretrained(source)
    model = MusicgenForConditionalGeneration.from_pretrained(
        source, dtype=getattr(torch, spec.dtype), device_map="auto")
    model.eval()
    return model, processor


def _load_acestep(spec, local_dir: str | None = None, tile: bool = False):
    """Load one ACE-Step checkpoint onto the GPU.

    `local_dir` is the fetch task's downloaded Dir. When it holds a usable pipeline
    root the load is a pure disk read; otherwise we fall back to the repo id and pull
    from HuggingFace, which is slow but correct.

    `tile` turns on the VAE's tiled decode. Default False, deliberately: see the long
    note below, it crashed every render over ~20.5s on this box.
    """
    import torch
    from diffusers import AceStepPipeline

    root = resolve_weights(local_dir, marker="model_index.json")
    source = root or spec.repo
    if local_dir and not root:
        log.warning(f"no model_index.json under {local_dir}; falling back to "
                    f"{spec.repo} over the network (this costs a full re-download)")
    else:
        log.info(f"loading from {source}")

    dtype = getattr(torch, spec.dtype)
    pipe = AceStepPipeline.from_pretrained(source, torch_dtype=dtype)
    pipe = pipe.to("cuda")

    # ── VAE tiling is OFF by default, matching upstream ─────────────────────────────
    #
    # An earlier version forced `use_tiling = True` on the theory that a long track
    # decoded in one shot is a huge contiguous activation. Turning it on was wrong for
    # a boring reason: diffusers defaults it to False, and on a box with ~90GB of
    # headroom a whole-track decode fits fine. Tiling is what you reach for on a 12GB
    # consumer card.
    #
    # HISTORICAL NOTE, because the comment that used to live here was confidently
    # wrong: tiling was briefly blamed for a run of SIGSEGVs on long tracks. The
    # correlation looked airtight (`tile_latent_min_length` is 512 frames at 25
    # frames/sec, so tiling engages at ~20.5s, and 20s renders worked while 60s and 90s
    # died). It was a coincidence. The real cause was libsndfile's Vorbis encoder
    # crashing in `encode_audio` while building the REPORT, long after the decode had
    # succeeded, and it reproduced with tiling both on and off. Two unrelated things
    # happened to share a length threshold. See `encode_audio` for the actual bug.
    #
    # `tile=True` remains available for genuinely long renders where a single decode
    # will not fit. Note the model card's `pipe.vae.enable_tiling()` does not exist on
    # AutoencoderOobleck in diffusers 0.39 (ModelMixin has no such method), so setting
    # the attribute is the only way in.
    if tile:
        log.info("VAE tiling enabled (non-default): bounding decode memory")
        try:
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            else:
                pipe.vae.use_tiling = True
        except Exception:
            log.exception("could not enable VAE tiling; continuing without it")

    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, spec, prompt: str, lyrics: str = "",
             settings: GenSettings | None = None) -> tuple[np.ndarray, int, float, GenSettings]:
    """Render one track. Returns (audio [channels, samples], sample_rate, seconds, resolved).

    The resolved settings come back out because they are what the report must show: a
    turbo run asked for cfg 7.0 did not run cfg 7.0, and a card that claims otherwise
    is the kind of quiet lie that makes a comparison worthless.
    """
    if spec.adapter == "musicgen":
        return _generate_musicgen(pipe, spec, prompt, settings)
    if spec.adapter == "audioldm2":
        return _generate_audioldm2(pipe, spec, prompt, settings)
    if spec.adapter == "stableaudio":
        return _generate_stableaudio(pipe, spec, prompt, settings)
    if spec.adapter == "diffrhythm":
        return _generate_diffrhythm(spec, prompt, lyrics, settings)
    return _generate_acestep(pipe, spec, prompt, lyrics, settings)


# MusicGen's EnCodec runs at 50 frames/sec, so this converts seconds of audio into the
# `max_new_tokens` the generate call actually wants.
MUSICGEN_TOKENS_PER_SEC = 50


def _generate_musicgen(pipe, spec, prompt: str, settings: GenSettings | None):
    """Text-to-music through MusicGen.

    Ignores lyrics entirely, because the model has no vocal path at all: feeding it
    lyrics would silently treat them as style text and muddy the prompt. `compare`
    passes them anyway (every model gets the same brief), so dropping them here is the
    right place to do it, and `intended_for` on the card explains why the vocal briefs
    come back instrumental.
    """
    import torch

    model, processor = pipe
    st = (settings or GenSettings()).resolve(spec)

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(model.device)
    tokens = int(st.duration * MUSICGEN_TOKENS_PER_SEC)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=True, guidance_scale=float(st.guidance),
                             max_new_tokens=tokens)
    seconds = time.perf_counter() - t0

    audio = out[0].detach().to(torch.float32).cpu().numpy()   # (channels, samples)
    sr = int(model.config.audio_encoder.sampling_rate)
    return audio, sr, seconds, st


# AudioLDM2's own docs recommend a negative prompt; without one the output skews toward
# the low-fidelity, noisy end of what it was trained on. This is the string from the
# model card, applied by default so the comparison uses the model the way its authors
# intend rather than a deliberately hobbled version of it.
AUDIOLDM2_NEGATIVE = "Low quality, average quality, noisy"


def _generate_audioldm2(pipe, spec, prompt: str, settings: GenSettings | None):
    """Text-to-audio through AudioLDM 2.

    Like MusicGen it has no vocal path, so lyrics are dropped; `intended_for` on the
    card explains why a vocal brief comes back instrumental.
    """
    import torch

    st = (settings or GenSettings()).resolve(spec)
    gen = torch.Generator(device="cuda").manual_seed(int(st.seed))

    t0 = time.perf_counter()
    out = pipe(
        prompt=prompt,
        negative_prompt=AUDIOLDM2_NEGATIVE,
        audio_length_in_s=float(st.duration),
        num_inference_steps=int(st.steps),
        guidance_scale=float(st.guidance),
        generator=gen,
        output_type="np",
    )
    seconds = time.perf_counter() - t0

    audio = np.asarray(out.audios[0], dtype=np.float32)      # (samples,), mono
    sr = int(pipe.vocoder.config.sampling_rate)              # 16000
    return audio, sr, seconds, st


# The diffusers docs are explicit that a negative prompt "can significantly improve the
# quality of the generated audio" for this model, and name this exact string. Applying
# it by default means the comparison uses the model the way its authors intend.
STABLE_AUDIO_NEGATIVE = "Low quality, average quality"


def _widen_brownian_interval() -> None:
    """Make the SDE noise sampler's Brownian tree span the sigmas it will be asked for.

    THE root cause of Stable Audio's `RecursionError`, found by instrumenting rather
    than guessing. Measured on the box:

        interval = [0.3, 500]        (scheduler config sigma_min / sigma_max)
        sigmas   = 500.00006 ... 0.323, 0.3, 0.0      (101 values)
        outside  = index 0 (500.00006) and index 100 (0.0)

    `CosineDPMSolverMultistepScheduler` builds ONE `BrownianInterval` over
    [sigma_min, sigma_max], then queries it with consecutive pairs of `sigmas`. The
    schedule ends at exactly 0.0, which is below sigma_min, so on the final step
    torchsde bisects toward a point outside its own domain and never converges. That is
    why it survived a 50,000-frame recursion limit: the recursion is infinite, not deep.

    Ruled out first, so nobody repeats them: torchsde 0.2.5 vs 0.2.6 (identical), and
    bf16 / fp16 / fp32 (identical). None of them could matter: the domain is wrong in
    every dtype.

    The patch widens the domain to [0, sigma_max * 1.001] so both out-of-range endpoints
    fall inside. It does NOT touch the sigma schedule, so the sampling path is unchanged
    and only the noise tree's bounds move. Patched at the class, because the scheduler
    constructs the sampler lazily on its first step and there is no earlier hook.
    """
    try:
        import diffusers.schedulers.scheduling_dpmsolver_sde as sde_mod
    except Exception:
        log.exception("could not import the SDE scheduler module to widen its interval")
        return
    cls = sde_mod.BrownianTreeNoiseSampler
    if getattr(cls, "_interval_widened", False):
        return
    original = cls.__init__

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda v: v):
        hi = float(sigma_max) * 1.001      # covers the 500.00006 rounding overshoot
        original(self, x, 0.0, hi, seed, transform)

    cls.__init__ = __init__
    cls._interval_widened = True
    log.info("widened BrownianTreeNoiseSampler domain to [0, sigma_max*1.001]; the "
             "schedule's final sigma of 0.0 was outside it")


def _generate_stableaudio(pipe, spec, prompt: str, settings: GenSettings | None):
    """Text-to-audio through Stable Audio Open. No vocal path, so lyrics are dropped."""
    import sys
    import torch

    st = (settings or GenSettings()).resolve(spec)
    gen = torch.Generator(device="cuda").manual_seed(int(st.seed))

    _widen_brownian_interval()
    prev_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(prev_limit, 50_000))
    t0 = time.perf_counter()
    try:
        out = pipe(
            prompt=prompt,
            negative_prompt=STABLE_AUDIO_NEGATIVE,
            audio_end_in_s=float(st.duration),
            num_inference_steps=int(st.steps),
            guidance_scale=float(st.guidance),
            generator=gen,
        )
    finally:
        sys.setrecursionlimit(prev_limit)
    seconds = time.perf_counter() - t0

    # .audios is (batch, channels, samples) as a torch tensor; the docs' own example
    # transposes audios[0] before writing, which confirms the channels-first layout.
    audio = out.audios[0].detach().to(torch.float32).cpu().numpy()
    sr = int(pipe.vae.sampling_rate)
    return audio, sr, seconds, st


def _generate_acestep(pipe, spec, prompt: str, lyrics: str = "",
                      settings: GenSettings | None = None):
    """Render one track through ACE-Step."""
    import torch

    st = (settings or GenSettings()).resolve(spec)
    gen = torch.Generator(device="cuda").manual_seed(int(st.seed))

    t0 = time.perf_counter()
    out = pipe(
        prompt=prompt,
        lyrics=lyrics or "",
        audio_duration=float(st.duration),
        vocal_language=st.language or "en",
        num_inference_steps=int(st.steps),
        guidance_scale=float(st.guidance),
        shift=float(st.shift),
        generator=gen,
        # None is the pipeline's "estimate it yourself" for all three; the sentinels
        # only exist to survive the CLI, they must not reach the model.
        bpm=st.bpm or None,
        keyscale=st.keyscale or None,
        timesignature=st.timesignature or None,
        cfg_interval_start=float(st.cfg_interval_start),
        cfg_interval_end=float(st.cfg_interval_end),
    )
    seconds = time.perf_counter() - t0

    audio = out.audios[0]                       # (channels, samples), float32 on device
    audio = audio.detach().to(torch.float32).cpu().numpy()
    return audio, int(pipe.sample_rate), seconds, st


def lyrics_to_lrc(lyrics: str, seconds: float) -> str:
    """Convert ACE-Step's STRUCTURAL lyrics into DiffRhythm's TIMED .lrc format.

    The two models want different things and this conversion is lossy, which is why the
    report card says it happened rather than passing it off as equivalent:

      ACE-Step  : `[verse]` / `[chorus]` structure tags. WHAT part of the song a line
                  belongs to; the model decides when it lands.
      DiffRhythm: `[mm:ss.xx]line`. WHEN each line is sung; there is no notion of
                  section at all.

    There is no faithful mapping. Structure tags are dropped (they are not lyrics and
    would be sung), and the remaining lines are spread evenly across the track with a
    short instrumental head and tail. Even spacing is a real handicap for a model built
    to follow a supplied vocal rhythm, so a DiffRhythm card on a shared brief is being
    judged on a lyric it was handed less information about than ACE-Step got.
    """
    lines = [l.strip() for l in (lyrics or "").splitlines()
             if l.strip() and not l.strip().startswith("[")]
    if not lines:
        return ""
    head, tail = 6.0, 4.0                       # intro / outro breathing room
    span = max(seconds - head - tail, 1.0)
    step = span / len(lines)
    out = []
    for i, line in enumerate(lines):
        t = head + i * step
        out.append(f"[{int(t // 60):02d}:{t % 60:05.2f}]{line}")
    return "\n".join(out)


def _generate_diffrhythm(spec, prompt: str, lyrics: str, settings: GenSettings | None):
    """Render through DiffRhythm 2's CLI.

    A subprocess, not an import: the repo ships no packaging and no Python entry point,
    only `inference.py` behind argparse. Calling the script is the honest interface, it
    keeps their globals out of our process, and it means we are not coupled to a module
    layout with no stability promise.

    v2 rather than v1, and the reason is structural: v1's `dit.py` drives transformers'
    `LlamaDecoderLayer` directly and never sets `num_attention_heads`, so on any modern
    transformers its attention and rotary embedding disagree about head_dim and the
    first forward pass dies. v2 ships its own DiT and a BigVGAN vocoder and never
    imports Llama at all.

    Input is a JSONL, one song per line:
        {"song_name": ..., "style_prompt": "<text or a wav path>", "lyrics": "<.lrc>"}
    """
    import json as _json
    import subprocess
    import sys
    import tempfile

    st = (settings or GenSettings()).resolve(spec)

    work = Path(tempfile.mkdtemp(prefix="diffrhythm_"))
    out_dir = work / "out"
    out_dir.mkdir()
    lrc = work / "lyrics.lrc"
    lrc.write_text(lyrics_to_lrc(lyrics, st.duration))
    jsonl = work / "input.jsonl"
    jsonl.write_text(_json.dumps({
        "song_name": "track",
        "style_prompt": prompt,     # free text; a .wav path would also be accepted
        "lyrics": str(lrc),
    }) + "\n")

    root = os.environ.get("DIFFRHYTHM_DIR", "/opt/diffrhythm2")

    # Hand the child a CLEAN allocator config. Our pod sets
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for the in-process models and a
    # subprocess inherits it silently. DiffRhythm never asked for that allocator, and
    # leaving it in place produced a bare driver-level `CUDA error: out of memory` from
    # `.to(device)` with no "Tried to allocate" size, on a GPU where a child probe
    # allocated happily. Imposing our tuning on someone else's process is our bug.
    # Strip ALL of it, not just the PyTorch allocator knobs. `CUDA_MODULE_LOADING=EAGER`
    # is the dangerous one: it forces every kernel in every loaded CUDA library into
    # memory when the context is created, and this child loads torch + muq + BigVGAN. On
    # a unified-memory box that is a large upfront allocation, and a driver-level
    # `cudaMalloc` failure with no "Tried to allocate" size is what it looks like when it
    # does not fit. Stripping only the two PyTorch vars was not enough: both DiffRhythm
    # v1 and v2 still OOMed identically, which is what pointed at the env rather than at
    # either codebase.
    _OURS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
             "CUDA_MODULE_LOADING", "CUDA_CACHE_MAXSIZE")
    child_env = {k: v for k, v in os.environ.items() if k not in _OURS}
    child_env["PYTHONPATH"] = root
    # Measure the child's ACTUAL allocation ceiling instead of theorising about it.
    # Every env-based hypothesis has now failed, so the question is empirical: how much
    # CAN this child allocate, and what does the driver say when it stops?
    probe_src = (
        "import torch\n"
        "p=torch.cuda.get_device_properties(0)\n"
        "print('total_gb',round(p.total_memory/1e9,1))\n"
        "try:\n"
        "  f,t=torch.cuda.mem_get_info(); print('free_gb',round(f/1e9,1))\n"
        "except Exception as e: print('mem_get_info_RAISED',type(e).__name__)\n"
        "try: print('cgroup',open('/sys/fs/cgroup/memory.max').read().strip())\n"
        "except Exception: print('cgroup_unreadable')\n"
        "ok=0\n"
        "for gb in (0.25,1,2,4,8,16,32):\n"
        "  try:\n"
        "    x=torch.zeros(int(gb*1e9),dtype=torch.uint8,device='cuda'); ok=gb; del x\n"
        "  except Exception as e:\n"
        "    print('FAILED_at_gb',gb,type(e).__name__); break\n"
        "print('max_alloc_gb',ok)\n"
    )
    probe = subprocess.run([sys.executable, "-c", probe_src],
                           capture_output=True, text=True, env=child_env)
    log.info(f"[diffrhythm] child probe rc={probe.returncode} | "
             f"{' '.join((probe.stdout or '').split())} | "
             f"err={(probe.stderr or '').strip()[-160:]}")


    cmd = [sys.executable, f"{root}/inference.py",
           "--repo-id", spec.repo,
           "--input-jsonl", str(jsonl),
           "--output-dir", str(out_dir),
           "--steps", str(int(st.steps)),
           "--cfg-strength", str(float(st.guidance)),
           "--max-secs", str(float(st.duration))]
    log.info(f"[diffrhythm] inference.py steps={int(st.steps)} "
             f"cfg={st.guidance:g} max-secs={st.duration:g}")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                          env={**child_env, "CUDA_LOG_FILE": "stderr"})
    seconds = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1800:]
        raise RuntimeError(f"diffrhythm inference.py exited {proc.returncode}:\n{tail}")

    made = sorted(out_dir.rglob("*.wav")) + sorted(out_dir.rglob("*.mp3")) \
        + sorted(out_dir.rglob("*.flac"))
    if not made:
        raise RuntimeError(f"diffrhythm produced no audio in {out_dir}; stdout tail:\n"
                           f"{(proc.stdout or '')[-800:]}")
    audio, sr = sf.read(str(made[0]), dtype="float32", always_2d=True)
    return audio.T, int(sr), seconds, st


# ── Audio utilities ──────────────────────────────────────────────────────────────

def to_stereo_float32(audio) -> np.ndarray:
    """Coerce whatever came back to a (channels, samples) float32 array, channels first.

    ACE-Step returns (channels, samples), but run_local reads files back as
    (samples, channels) and a mono checkpoint would return 1-D, so normalize once.
    """
    try:
        import torch
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().to(torch.float32).cpu().numpy()
    except Exception:
        pass
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim > 2:
        a = a.reshape(a.shape[-2], a.shape[-1])
    # Channels-first: a music clip is always longer than it is wide.
    if a.shape[0] > a.shape[1]:
        a = a.T
    return np.ascontiguousarray(a)


def to_mono(audio) -> np.ndarray:
    return to_stereo_float32(audio).mean(axis=0)


def write_wav(audio, sr: int, path) -> None:
    """PCM16 wav, stereo preserved. soundfile wants (samples, channels)."""
    sf.write(str(path), to_stereo_float32(audio).T, sr, subtype="PCM_16")


def encode_audio(data: np.ndarray, sr: int, fmt: str, sub: str) -> bytes:
    """Encode (samples, channels) audio into an in-memory container.

    ── Written in CHUNKS, and that is not a style choice ────────────────────────────
    A single large `sf.write()` into libsndfile's Vorbis encoder SEGFAULTS the whole
    process. Not an exception, not an error return: SIGSEGV, exit 139, no Python
    traceback unless faulthandler is armed.

    Reproduced standalone on libsndfile 1.2.2 / soundfile 0.14.0, no GPU involved:
    20s and 30s of 48kHz audio encode fine, 60s and 90s crash. It is a function of the
    size of the one-shot write, not of the destination or the layout: it crashes to a
    BytesIO and to a real file alike, and in mono as well as stereo. Writing the same
    audio through `sf.SoundFile` in ~5 second slices produces a byte-identical-sized
    file and never crashes.

    This cost four failed cluster runs and two wrong theories (a memory cap, then VAE
    tiling), because the crash lands AFTER a successful render, while building the
    report. The give-away in hindsight: the failures tracked audio LENGTH, and the
    renderer is the only thing downstream of length that touches native code.
    """
    buf = io.BytesIO()
    channels = 1 if data.ndim == 1 else int(data.shape[1])
    step = max(1, sr * 5)
    with sf.SoundFile(buf, "w", samplerate=sr, channels=channels,
                      format=fmt, subtype=sub) as f:
        for i in range(0, len(data), step):
            f.write(data[i:i + step])
    return buf.getvalue()


def analyze_audio(audio, sr: int) -> dict:
    """Objective measurements of a rendered track. NOT a quality score.

    Nothing here says whether a track is good; every one of these is a named physical
    property, and they earn their place because several of the complaints this demo
    chases are physical rather than aesthetic:

      dynamics    Loud-to-quiet spread, p95 minus p10 of short-term RMS in dB. This is
                  the "everything is huge all the time" measurement. A maximalist,
                  brickwalled render leaves the ear no variation to read as human
                  performance, and it shows up here as a small number. It is the direct
                  test of the production-language A/B: `BALLAD_PROMPT_DRY` explicitly
                  asks for quiet verses and loud choruses, so if that caption works at
                  all this number must rise.
      crest       Peak-to-RMS in dB. The same idea instant by instant rather than
                  section by section, and the classic over-compression tell.
      clipped     Fraction of samples pinned at full scale. High guidance pushing the
                  output into the rails is a real failure mode and an invisible one.
      centroid    Spectral centre of mass in Hz, i.e. brightness. The harshness proxy:
                  a high CFG that has gone glassy moves energy upward.
      hf_ratio    Share of energy above 8kHz. Same question, aimed at the band where
                  "brittle" and "sizzly" actually live.
      width       1 minus the L/R correlation. 0 is mono, larger is wider. The README
                  claims low step counts narrow the stereo image; this is how you check
                  that by measurement instead of by assertion.
      onset       Mean onset strength. Transient sharpness, which is the first thing to
                  blur when steps drop, so it is the numeric form of "smeared".

    Read them as DIFFERENTIALS across a sweep, never as absolutes. Every experiment in
    this repo already varies one knob with everything else fixed, which is exactly the
    setup that makes a row of these numbers meaningful; a single number on a single
    track tells you almost nothing.
    """
    out: dict[str, float] = {}
    try:
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[0] > a.shape[1]:          # (samples, channels) -> (channels, samples)
            a = a.T
        mono = a.mean(axis=0)
        if mono.size < sr // 4:
            return out

        peak = float(np.abs(mono).max())
        rms = float(np.sqrt(np.mean(mono ** 2)))
        if rms > 0:
            out["crest"] = round(20 * np.log10(peak / rms), 1)

        # Short-term loudness, 100ms hops. The spread matters, not the level.
        hop = max(sr // 10, 1)
        n = (mono.size // hop) * hop
        if n >= hop * 8:
            frames = mono[:n].reshape(-1, hop)
            fr = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
            db = 20 * np.log10(fr)
            # Drop near-silence so an intro or a fade does not masquerade as dynamics.
            db = db[db > db.max() - 60]
            if db.size > 8:
                out["dynamics"] = round(float(np.percentile(db, 95)
                                              - np.percentile(db, 10)), 1)

        out["clipped"] = round(float(np.mean(np.abs(mono) >= 0.99)), 5)

        spec = np.abs(np.fft.rfft(mono[:sr * 30] * np.hanning(min(mono.size, sr * 30))))
        freqs = np.fft.rfftfreq(min(mono.size, sr * 30), 1 / sr)
        tot = spec.sum()
        if tot > 0:
            out["centroid"] = round(float((freqs * spec).sum() / tot))
            out["hf_ratio"] = round(float(spec[freqs > 8000].sum() / tot), 4)

        if a.shape[0] == 2:
            l, r = a[0], a[1]
            if l.std() > 0 and r.std() > 0:
                out["width"] = round(1.0 - float(np.corrcoef(l, r)[0, 1]), 3)

        # Onset strength without librosa: positive spectral flux over a coarse STFT.
        win = 2048
        m = mono[:sr * 30]
        n2 = (m.size // win) * win
        if n2 >= win * 4:
            mags = np.abs(np.fft.rfft(m[:n2].reshape(-1, win) * np.hanning(win), axis=1))
            flux = np.diff(mags, axis=0)
            out["onset"] = round(float(np.maximum(flux, 0).sum(axis=1).mean()), 1)
    except Exception:
        log.exception("analyze_audio failed; card will simply omit the metrics")
    return out


_METRIC_HELP = {
    "dynamics": "loud-to-quiet spread in dB; small = compressed",
    "crest": "peak-to-RMS in dB; small = compressed",
    "clipped": "fraction of samples at full scale",
    "centroid": "spectral centre of mass in Hz; brightness",
    "hf_ratio": "share of energy above 8kHz; harshness band",
    "width": "1 - L/R correlation; 0 = mono",
    "onset": "mean onset strength; transient sharpness",
}


def audio_data_uri(audio, sr: int, budget: int = MAX_EMBED_BYTES) -> tuple[str, str]:
    """(data_uri, note). An empty uri plus a note means 'too big, spectrogram only'.

    A ladder, most-faithful first. Vorbis is ~10x smaller than PCM and plays natively
    in <audio>, which is what makes a grid of full-length tracks embeddable at all: the
    same 60s stereo track is ~1MB as OGG and ~11.5MB as PCM16 wav. Each rung says in
    the note what it gave up, because "why is this one mono?" should never be a mystery.
    """
    stereo = to_stereo_float32(audio)
    rungs = (
        (stereo.T, "OGG", "VORBIS", "audio/ogg", ""),
        (to_mono(stereo), "OGG", "VORBIS", "audio/ogg",
         "stereo track was over the embed budget; player is a mono downmix"),
        (stereo.T, "WAV", "PCM_16", "audio/wav",
         "no Vorbis encoder in this image; embedded as uncompressed wav"),
    )
    last = ""
    for data, fmt, sub, mime, note in rungs:
        try:
            raw = encode_audio(data, sr, fmt, sub)
            if len(raw) > budget:
                last = f"clip is {len(raw)/1e6:.1f} MB, over the embed budget"
                continue
            return f"data:{mime};base64,{base64.b64encode(raw).decode()}", note
        except Exception:
            continue
    return "", (last or "could not encode audio for embedding") + "; spectrogram shown"


def waveform_spectrogram_png(audio, sr: int) -> str:
    """A stacked waveform + spectrogram as a base64 PNG data URI.

    The at-a-glance visual, and the fallback surface if a player fails. Two choices
    differ from the TTS version, both because this is music:

      - the waveform is the mono sum, drawn as an envelope, so an intro/build/drop
        arc reads as shape rather than as a solid block of ink;
      - the spectrogram runs to the FULL Nyquist (24kHz), not the 8kHz voice band.
        The top of that range is exactly where a lowpass brickwall shows up, and a
        hard horizontal edge at 16kHz is the tell that a model is rendering through a
        lossy bottleneck. Cropping to the voice band would hide the most diagnostic
        thing on the plot.
    """
    mono = to_mono(audio)
    if mono.size == 0:
        mono = np.zeros(int(sr * 0.1), dtype=np.float32)
    t = np.arange(mono.size) / float(sr)

    fig, (ax_w, ax_s) = plt.subplots(
        2, 1, figsize=(4.8, 2.6), dpi=110, gridspec_kw={"height_ratios": [1, 1.5]}
    )
    ax_w.fill_between(t, mono, -mono, linewidth=0, color="#7c3aed", alpha=0.85)
    ax_w.set_xlim(0, max(float(t[-1]), 0.1))
    ax_w.set_ylim(-1.05, 1.05)
    ax_w.axhline(1.0, color="#ef4444", linewidth=0.5, alpha=0.6)   # the clipping line
    ax_w.axhline(-1.0, color="#ef4444", linewidth=0.5, alpha=0.6)
    ax_w.set_yticks([])
    ax_w.set_xticks([])
    ax_w.margins(0)

    nfft = 1024 if mono.size >= 1024 else max(32, 1 << int(np.log2(max(mono.size, 32))))
    ax_s.specgram(mono, NFFT=nfft, Fs=sr, noverlap=nfft // 2, cmap="magma")
    ax_s.set_ylim(0, sr / 2)
    ax_s.set_yticks([0, 8000, 16000, sr / 2])
    ax_s.set_yticklabels(["0", "8k", "16k", f"{sr/2000:.0f}k"], fontsize=6)
    ax_s.set_xlabel("seconds", fontsize=7, color="#6b7280")
    ax_s.tick_params(axis="both", labelsize=6, colors="#6b7280")

    fig.subplots_adjust(left=0.06, right=0.99, top=0.98, bottom=0.16, hspace=0.08)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── Reproducing a card ───────────────────────────────────────────────────────────
#
# The report used to show settings as prose: "seed 42 · 8 steps · cfg 1 · shift 3".
# Fine for reading, useless for acting. The whole point of a comparison is that one
# cell is closer than the others, and the next thing you want is that cell again with
# one thing changed. Retyping six parameters off a screenshot is where that intent
# goes to die.
#
# So every card also carries a Repro: the exact command that made it, plus the same
# thing as JSON. Note the settings are the RESOLVED ones (turbo's coerced cfg, each
# checkpoint's own step count), so the command reproduces what actually ran rather
# than what was asked for.
#
# No copy button, deliberately: Flyte reports render under a CSP that drops <script>,
# so a clipboard widget would be dead HTML. A <details> block with selectable text
# works everywhere, including in the standalone HTML run_local writes.


@dataclass
class Repro:
    """Everything needed to make one card again."""
    model_key: str
    prompt: str
    lyrics: str = ""
    brief: str = ""              # the named brief this came from, if any
    settings: dict = field(default_factory=dict)
    entrypoint: str = "generate_one"
    adapter: str = "acestep"     # drives which flags the command emits

    def as_json(self) -> str:
        return json.dumps({
            "entrypoint": self.entrypoint,
            "model_key": self.model_key,
            "brief": self.brief,
            "prompt": self.prompt,
            "lyrics": self.lyrics,
            "settings": self.settings,
        }, indent=2)

    def as_cli(self) -> str:
        """A runnable `flyte run` line.

        Prefers `--brief` when the card came from one: it keeps the command to a
        readable length and it is exact, because the brief carries the prompt AND the
        lyrics AND the musical metadata. Falls back to an inline `--prompt`, which for
        a lyric of any size is genuinely unwieldy, which is what the JSON is for.
        """
        parts = ["flyte run compare_pipeline.py", self.entrypoint,
                 f"--model_key {shlex.quote(self.model_key)}"]
        if self.brief:
            parts.append(f"--brief {shlex.quote(self.brief)}")
        else:
            parts.append(f"--prompt {shlex.quote(self.prompt)}")
            if self.lyrics.strip():
                parts.append(f"--lyrics {shlex.quote(self.lyrics)}")
        s = self.settings
        # Only emit flags this model actually honours: `--steps 8 --shift 3` on a
        # MusicGen command would run without complaint and mean nothing.
        has = knobs_for(self.adapter)
        flags = [("--duration", "duration"), ("--seed", "seed")]
        if "guidance" in has:
            flags.append(("--guidance", "guidance"))
        if "steps" in has:
            flags.append(("--steps", "steps"))
        if "shift" in has:
            flags.append(("--shift", "shift"))
        for flag, key in flags:
            if key in s:
                v = s[key]
                parts.append(f"{flag} {v:g}" if isinstance(v, float) else f"{flag} {v}")
        # Only emit the musical metadata when it was actually set; a brief already
        # carries its own, and repeating it adds noise without changing the result.
        if not self.brief and "bpm" in has:
            if s.get("bpm"):
                parts.append(f"--bpm {s['bpm']}")
            if s.get("keyscale"):
                parts.append(f"--keyscale {shlex.quote(s['keyscale'])}")
            if s.get("language") and s["language"] != "en":
                parts.append(f"--language {shlex.quote(s['language'])}")
        return " \\\n    ".join(parts)


def build_repro(spec, job_prompt: str, job_lyrics: str, settings: "GenSettings",
                brief: str = "", entrypoint: str = "generate_one") -> Repro:
    """Repro for one card, with `settings` resolved against the checkpoint."""
    return Repro(model_key=spec.key, prompt=job_prompt, lyrics=job_lyrics, brief=brief,
                 settings=settings.resolve(spec).as_dict(), entrypoint=entrypoint,
                 adapter=getattr(spec, "adapter", "acestep"))


# ── Report data ──────────────────────────────────────────────────────────────────

@dataclass
class TrackResult:
    """One rendered card in the report."""
    label: str                  # the column identity: a model key, or "seed 42"
    sublabel: str = ""          # the family / axis line under the title
    settings: str = ""          # GenSettings.summary() of what actually ran
    seconds: float = 0.0        # render wall-clock
    audio_seconds: float = 0.0  # duration of the produced audio
    sample_rate: int = 0
    channels: int = 0
    audio_uri: str = ""
    spec_uri: str = ""
    peak_gb: float = 0.0
    badges: list[str] = field(default_factory=list)
    embed_note: str = ""
    error: str = ""
    intended_for: str = ""       # what this model is FOR (cross-family fairness)
    repro: Repro | None = None   # how to make this card again
    metrics: dict = field(default_factory=dict)   # analyze_audio(); measurements only

    @property
    def speedup(self) -> float:
        """Audio seconds produced per second of compute. >1 is faster than real time."""
        return (self.audio_seconds / self.seconds) if self.seconds else 0.0


def build_track_result(label: str, audio, sr: int, seconds: float, *,
                       sublabel: str = "", settings: str = "", peak_gb: float = 0.0,
                       badges: list[str] | None = None,
                       repro: Repro | None = None,
                       intended_for: str = "") -> TrackResult:
    a = to_stereo_float32(audio)
    audio_seconds = a.shape[1] / float(sr) if sr else 0.0
    uri, note = audio_data_uri(a, sr)
    return TrackResult(
        label=label, sublabel=sublabel, settings=settings,
        seconds=seconds, audio_seconds=audio_seconds, sample_rate=sr,
        channels=int(a.shape[0]), audio_uri=uri,
        spec_uri=waveform_spectrogram_png(a, sr),
        peak_gb=peak_gb, badges=list(badges or []), embed_note=note, repro=repro,
        intended_for=intended_for,
        # Measured here rather than at each call site so every report gets them: the
        # sweeps, the grids, `takes` and `run_local` all funnel through this function.
        metrics=analyze_audio(a, sr),
    )


@dataclass
class Block:
    """One row of the report: a heading, the hypothesis, and the cards to compare."""
    heading: str
    note: str = ""          # what to listen for
    prompt: str = ""        # the style caption used, shown verbatim
    lyrics: str = ""        # shown collapsed; "" renders as "instrumental"
    results: list[TrackResult] = field(default_factory=list)

    # What the collapsed lyric fold is called. It exists because a report whose rows
    # deliberately SHARE one lyric (the genre swap) renders the same ten lines under
    # every row, and a reader who does not already know that reads it as a bug: "it
    # printed the same lyrics for every genre". The fold looked identical whether the
    # sameness was the experiment or a broken loop. Naming it is the whole fix.
    lyrics_label: str = "lyrics"


# ── Rendering ────────────────────────────────────────────────────────────────────

REPORT_CSS = """
<style>
  .mg-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
             color: #0b0b0b; }
  .mg-wrap h2 { margin: 0 0 4px; }
  .mg-meta { color: #6b7280; font-size: 13px; margin-bottom: 20px; }
  .mg-head { background: #f9fafb; border-left: 3px solid #7c3aed; padding: 10px 14px;
             border-radius: 6px; margin: 22px 0 10px; }
  .mg-title { font-weight: 600; font-size: 15px; }
  .mg-note { color: #4b5563; font-size: 13px; line-height: 1.5; margin-top: 5px; }
  .mg-prompt { color: #374151; font-size: 12.5px; margin-top: 8px; font-style: italic; }
  .mg-mets { margin-top: 5px; display: flex; flex-wrap: wrap; gap: 4px 6px; }
  .mg-met { font-size: 10.5px; color: #4b5563; background: #f3f4f6;
            border: 1px solid #e5e7eb; border-radius: 4px; padding: 1px 5px;
            font-variant-numeric: tabular-nums; cursor: help; }
  .mg-met b { color: #111827; font-weight: 600; }
  .mg-lyr { margin-top: 8px; font-size: 12.5px; color: #374151; }
  .mg-lyr pre { white-space: pre-wrap; font-size: 12px; line-height: 1.45;
                background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
                padding: 8px 10px; margin: 6px 0 0; }
  .mg-grid { display: grid; gap: 16px;
             grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); }
  .mg-cell { border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden;
             background: #fff; display: flex; flex-direction: column; }
  .mg-spec { padding: 10px 11px 0; }
  .mg-spec img { width: 100%; height: auto; border-radius: 6px; display: block;
                 cursor: zoom-in; }
  .mg-audio { padding: 10px 11px 4px; }
  .mg-audio audio { width: 100%; display: block; }
  .mg-cap { padding: 6px 11px 12px; }
  .mg-model { font-weight: 600; font-size: 14px; }
  .mg-tag { display: inline-block; font-size: 11px; color: #374151; background: #f3f4f6;
            border-radius: 999px; padding: 1px 8px; margin: 4px 4px 0 0; }
  .mg-fast { display: inline-block; font-size: 11px; color: #065f46; background: #d1fae5;
             border-radius: 999px; padding: 1px 8px; margin: 4px 0 0; font-weight: 600; }
  .mg-set { color: #4338ca; font-size: 11.5px; margin-top: 6px; font-variant-numeric: tabular-nums; }
  .mg-sub { color: #6b7280; font-size: 12px; margin-top: 4px; line-height: 1.4;
            font-variant-numeric: tabular-nums; }
  .mg-repro { padding: 0 11px 10px; font-size: 12px; color: #4b5563; }
  .mg-repro summary { cursor: pointer; color: #6d28d9; font-weight: 600; }
  .mg-repro pre { white-space: pre-wrap; word-break: break-word; font-size: 11px;
                  line-height: 1.45; background: #faf5ff; border: 1px solid #e9d5ff;
                  border-radius: 6px; padding: 8px 10px; margin: 6px 0 0;
                  user-select: all; }
  .mg-repro .mg-hint { margin-top: 6px; color: #6b7280; font-size: 11px; }
  .mg-for { color: #4b5563; font-size: 11.5px; margin-top: 6px; padding-top: 6px;
            border-top: 1px dashed #e5e7eb; line-height: 1.4; }
  .mg-err { padding: 16px; color: #b91c1c; font-size: 13px; white-space: pre-wrap; }
  .mg-warn { padding: 8px 11px; color: #92400e; background: #fffbeb; font-size: 12px; }
  #mg-lb { position: fixed; inset: 0; z-index: 9999; display: none; cursor: zoom-out;
           flex-direction: column; align-items: center; justify-content: center;
           gap: 12px; padding: 24px; background: rgba(0,0,0,.88); }
  #mg-lb img { max-width: 96vw; max-height: 86vh; border-radius: 8px; }
  #mg-lb #mg-lb-cap { color: #e5e7eb; font-size: 14px; }
</style>
"""

_ZOOM = (
    "document.getElementById('mg-lb-img').src=this.src;"
    "document.getElementById('mg-lb-cap').textContent=this.dataset.cap;"
    "document.getElementById('mg-lb').style.display='flex'"
)
_LIGHTBOX = (
    "<div id=\"mg-lb\" onclick=\"this.style.display='none'\" style=\"display:none\">"
    '<img id="mg-lb-img" src="" alt="zoomed"/><div id="mg-lb-cap"></div></div>'
)


def _zoom_img(uri: str, cap: str) -> str:
    return (f'<img src="{uri}" alt="{html.escape(cap)}" '
            f'data-cap="{html.escape(cap, quote=True)}" onclick="{_ZOOM}"/>')


def _player(r: TrackResult) -> str:
    if not r.audio_uri:
        return ""
    # controls = native scrub/play/pause, no JS. Not autoplay: six tracks starting at
    # once is a wall of noise, and the whole point is to play them one at a time.
    return (f'<div class="mg-audio"><audio controls preload="metadata" '
            f'src="{r.audio_uri}"></audio></div>')


def _cell(r: TrackResult) -> str:
    if r.error:
        return (f'<div class="mg-cell"><div class="mg-err">⚠️ {html.escape(r.error)}</div>'
                f'<div class="mg-cap"><div class="mg-model">{html.escape(r.label)}</div>'
                f'</div></div>')

    img = (f'<div class="mg-spec">{_zoom_img(r.spec_uri, f"{r.label} · {r.settings}")}</div>'
           if r.spec_uri else "")
    note = f'<div class="mg-warn">{html.escape(r.embed_note)}</div>' if r.embed_note else ""
    fast = (f'<span class="mg-fast">{r.speedup:.1f}x real-time</span>'
            if r.speedup else "")
    tags = "".join(f'<span class="mg-tag">{html.escape(b)}</span>' for b in r.badges)
    settings = f'<div class="mg-set">{html.escape(r.settings)}</div>' if r.settings else ""
    peak = f' · peak {r.peak_gb:.1f}GB' if r.peak_gb else ""
    chan = {1: "mono", 2: "stereo"}.get(r.channels, f"{r.channels}ch")
    sub = (f'<div class="mg-sub">{r.seconds:.1f}s to render · {r.audio_seconds:.1f}s '
           f'{chan} @ {r.sample_rate/1000:.0f}kHz{peak}</div>')
    sublabel = (f'<div class="mg-sub">{html.escape(r.sublabel)}</div>'
                if r.sublabel else "")
    # What this model is FOR. Only meaningful once the grid spans model families, but
    # then it is the line that stops a comparison from being unfair by omission: a
    # 30s instrumental model losing a "sing me a chorus" prompt is not a quality result.
    intended = (f'<div class="mg-for">{html.escape(r.intended_for)}</div>'
                if r.intended_for else "")
    # Measurements, not a score. Rendered as a plain row with the units in the tooltip,
    # and deliberately NOT colour-coded or ranked: the moment one of these gets a green
    # tick the report is claiming to know which track is better, which it does not.
    met = ""
    if r.metrics:
        bits = "".join(
            f'<span class="mg-met" title="{html.escape(_METRIC_HELP.get(k, k))}">'
            f'{html.escape(k)} <b>{v:g}</b></span>'
            for k, v in r.metrics.items())
        met = f'<div class="mg-mets">{bits}</div>'
    cap = (f'<div class="mg-cap"><div class="mg-model">{html.escape(r.label)}</div>'
           f'{tags}{fast}{settings}{sub}{sublabel}{met}{intended}</div>')
    return f'<div class="mg-cell">{img}{_player(r)}{note}{cap}{_repro(r)}</div>'


def _repro(r: TrackResult) -> str:
    """The 'make this one again' block: the exact command, and the same as JSON."""
    if not r.repro:
        return ""
    return (
        '<details class="mg-repro"><summary>reproduce / tweak this one</summary>'
        f'<pre>{html.escape(r.repro.as_cli())}</pre>'
        '<div class="mg-hint">Change one flag to iterate. The settings above are what '
        'actually ran, not what was requested.</div>'
        f'<pre>{html.escape(r.repro.as_json())}</pre></details>'
    )


def _block(b: Block) -> str:
    note = f'<div class="mg-note">{html.escape(b.note)}</div>' if b.note else ""
    prompt = (f'<div class="mg-prompt">🎛️ {html.escape(b.prompt)}</div>'
              if b.prompt else "")
    if b.lyrics.strip():
        lyr = (f'<details class="mg-lyr">'
               f'<summary>{html.escape(b.lyrics_label)}</summary>'
               f'<pre>{html.escape(b.lyrics.strip())}</pre></details>')
    elif b.prompt:
        lyr = '<div class="mg-lyr">🎹 instrumental (no lyrics)</div>'
    else:
        lyr = ""
    cells = "".join(_cell(r) for r in b.results)
    return (f'<div class="mg-head"><div class="mg-title">{html.escape(b.heading)}</div>'
            f'{note}{prompt}{lyr}</div><div class="mg-grid">{cells}</div>')


def render_report(blocks: list[Block], *, title: str, meta: str = "") -> str:
    return (
        f'{REPORT_CSS}<div class="mg-wrap"><h2>{html.escape(title)}</h2>'
        f'<div class="mg-meta">{html.escape(meta)}</div>'
        + "".join(_block(b) for b in blocks) + "</div>" + _LIGHTBOX
    )


def render_status(title: str, body: str) -> str:
    return (f'{REPORT_CSS}<div class="mg-wrap"><h2>{html.escape(title)}</h2>'
            f'<div class="mg-meta">{html.escape(body)}</div></div>')
