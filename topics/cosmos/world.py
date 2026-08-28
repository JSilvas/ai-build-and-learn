"""Loading Cosmos 3 and rolling it forward, with the GB10-specific care it needs.

Split out of pipeline.py so the model code can be exercised from `smoke_test.py` on
the host without Flyte in the picture, which is the fast iteration loop.

── The three things that go wrong on this box ──────────────────────────────────
1. OVERSIZED LOADS HANG, THEY DO NOT OOM. The GB10 has one 119.7 GiB pool shared by
   the GPU, the OS and every other pod. Ask for more than is free and the box wedges
   rather than raising, so `guard_memory()` sets a hard cap BEFORE any weights are
   touched. It caps against FREE, not total: `set_per_process_memory_fraction` takes
   a share of TOTAL, and on a box where a leaky object store is already holding 40 GB
   a 0.9-of-total cap is not a cap at all. That mistake produces an OOM followed by a
   bare SIGSEGV (exit 139).

2. THE DOUBLE-COPY LOAD TRAP. `from_pretrained(...).to("cuda")` materializes the
   whole model on the host and then copies it, so it needs 2x the model on one pool.
   `device_map="cuda"` streams shards straight to the device instead, which is also
   what NVIDIA's own example runner does. Never add a `.to()` after this call.

3. NO torch.compile. Triton does not emit working SASS for sm_121a yet, so it fails
   or silently falls back to something slower than eager.

`faulthandler` is armed at import: a native crash inside a CUDA kernel or a codec
otherwise produces a bare exit 139 with no Python traceback at all, and the pod log
is the only place that information can come from.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import time

faulthandler.enable()

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# The canonical action-conditioning example ships INSIDE the checkpoint, which is
# why these are asset paths and not URLs: by the time we need them the snapshot is
# already on local disk, so there is no second network dependency to fail.
ACTION_FRAME = "assets/example_action_fd_agibotworld_first_frame.png"
ACTION_CHUNKS = "assets/example_action_fd_agibotworld_action_chunks.json"
I2V_FRAME = "assets/example_i2v_input.jpg"


# Cosmos3-Nano is 16B in BF16: ~30 GiB of weights resident, plus the VAE decode of a
# multi-second latent, which is the real peak. The failed rollout died asking for a
# single 14.13 GiB block, so the budget has to cover weights AND that spike.
NANO_NEEDS_GIB = 46.0

# Never hand the allocator more than this share of the pool however free the box
# looks. The GB10 pool is shared with the OS, so an oversized load wedges the whole
# box rather than raising, and there is no recovering from that without a reboot.
MAX_FRACTION = 0.90


def _host_available_gib() -> float | None:
    """MemAvailable in GiB, or None if /proc/meminfo is unreadable.

    This, not `cuMemGetInfo`, is the honest number on the GB10. There is one 119.7 GiB
    pool, and `cuMemGetInfo` reports only what is *unused* -- it counts the kernel's
    page cache as taken, even though page cache is clean and gets evicted the instant
    anything asks for the memory. With 64 GiB of cache from a 35 GB model download
    sitting there, cuMemGetInfo says 32 GiB free while MemAvailable says 99 GiB, and
    the second one is right.

    MemAvailable is also the number that correctly *excludes* the thing that genuinely
    does steal from us: a leaky rustfs holding 12 GiB of anonymous heap is not
    reclaimable, and MemAvailable does not count it. So this stays a real cap.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 2**30
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_limit_gib() -> float | None:
    """This pod's own memory cgroup ceiling, in GiB, or None if unlimited/absent.

    /proc/meminfo is NOT namespaced, so inside a Flyte pod it reports the whole host.
    That is what we want for the GPU pool, but the pod also has its own `memory=96Gi`
    limit, and being killed by the cgroup OOM-killer is just as fatal. Take the lower
    of the two.
    """
    try:
        with open("/sys/fs/cgroup/memory.max") as fh:
            raw = fh.read().strip()
        return None if raw == "max" else int(raw) / 2**30
    except (OSError, ValueError):
        return None


def guard_memory(needs_gib: float = NANO_NEEDS_GIB, headroom_gib: float = 8.0) -> str:
    """Cap this process against what is genuinely reclaimable, or refuse to start.

    Returns a human-readable line for the report, because "how much of the box was
    free when this ran" is the single most useful number when a run that worked
    yesterday OOMs today.

    Raises RuntimeError when the budget will not cover the load. That is deliberate,
    and it is the lesson from the first rollout attempt: the old version clamped a
    negative budget up to a 0.05 floor and carried on, so a guard whose whole job was
    to prevent an OOM produced one -- "Tried to allocate 14.13 GiB ... 5.98 GiB
    allowed", where 5.98 GiB is exactly 0.05 of the pool. Failing in two seconds with
    an actionable message beats failing in eight minutes with a CUDA traceback.
    """
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device"

    cuda_free, total = torch.cuda.mem_get_info()
    cuda_free_gib, total_gib = cuda_free / 2**30, total / 2**30

    # Take the most pessimistic real ceiling: host-reclaimable, the pod's own cgroup
    # limit, and the pool itself. Fall back to cuMemGetInfo only if /proc is missing.
    candidates = [c for c in (_host_available_gib(), _cgroup_limit_gib(), total_gib) if c]
    usable_gib = min(candidates) if candidates else cuda_free_gib
    budget_gib = usable_gib - headroom_gib

    if budget_gib < needs_gib:
        raise RuntimeError(
            f"only {usable_gib:.1f} GiB usable ({budget_gib:.1f} GiB after headroom), "
            f"and this load needs ~{needs_gib:.0f} GiB. Refusing to start rather than "
            f"OOM eight minutes in. Free the box first:\n"
            f"  kubectl rollout restart deploy/rustfs -n flyte   # the usual culprit\n"
            f"  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'\n"
            f"or run ./preflight.sh, which does both and re-checks."
        )

    # set_per_process_memory_fraction is a share of TOTAL, so convert.
    fraction = min(MAX_FRACTION, budget_gib / total_gib)
    torch.cuda.set_per_process_memory_fraction(fraction)
    line = (
        f"{usable_gib:.1f} GiB usable of {total_gib:.1f} GiB "
        f"(cuMemGetInfo says {cuda_free_gib:.1f} free; the difference is reclaimable "
        f"page cache); capped this process at {fraction:.2f} "
        f"({fraction * total_gib:.1f} GiB)"
    )
    log.info("memory guard: %s", line)
    return line


def snapshot(repo: str) -> str:
    """Download the checkpoint (resumable) and return the local path."""
    from huggingface_hub import snapshot_download

    t0 = time.monotonic()
    # max_workers=4 rather than the default 8. Fetches on this box are more reliable
    # serialized than parallel; a stalled worker is what produces a "download" that
    # sits at the same byte count until the task times out.
    path = snapshot_download(repo_id=repo, max_workers=4)
    log.info("snapshot %s -> %s (%.1f min)", repo, path, (time.monotonic() - t0) / 60)
    return path


def load(repo: str, *, sound: bool = False):
    """Build a Cosmos3OmniPipeline on the GPU, ready to call.

    `enable_safety_checker=False` is deliberate and is explained at length in
    config.py: the default True constructs a CosmosSafetyChecker, which is a separate
    `cosmos_guardrail` install that pulls a GATED Llama Guard checkpoint, so leaving
    it on fails at construction on any box whose token has not accepted that licence.
    These runs therefore have no content guardrail.
    """
    import torch
    from diffusers import Cosmos3OmniPipeline

    path = snapshot(repo)
    guard_memory()

    t0 = time.monotonic()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,   # the ONLY precision NVIDIA tests for Cosmos 3
        device_map="cuda",            # streams shards to device; see the module docstring
        enable_safety_checker=False,
    )
    log.info("loaded %s in %.1f min", repo, (time.monotonic() - t0) / 60)

    if not sound and getattr(pipe, "sound_tokenizer", None) is not None:
        # The audio branch is ~2 GB of weights that never get used unless
        # enable_sound=True, and on this box the activation budget is the binding
        # constraint. `sound_tokenizer` is in the pipeline's `_optional_components`,
        # so unregistering it is supported rather than a hack; every task reloads the
        # pipeline from scratch anyway, so nothing downstream sees a half-built one.
        pipe.sound_tokenizer = None
        torch.cuda.empty_cache()
        log.info("dropped the sound tokenizer (enable_sound is off), ~2 GB reclaimed")
    return pipe


def generate(
    pipe,
    prompt: str,
    *,
    image=None,
    negative_prompt: str | None = None,
    num_frames: int = 61,
    height: int = 480,
    width: int = 832,
    fps: int = 24,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int | None = 0,
    sound: bool = False,
):
    """Text-to-video, or image-to-video when `image` is given.

    The pipeline picks the mode from the inputs: `num_frames == 1` is text-to-image,
    an `image` anchors frame 0 and makes it image-to-video, otherwise it is
    text-to-video. There is no mode flag to set and no separate pipeline class.

    Defaults are deliberately smaller than NVIDIA's (189 frames at 720x1280). A clip
    is going to be base64'd into an HTML report, and 480p for ~2.5s is the largest
    thing that stays comfortably embeddable. Pass the bigger numbers when you want
    the quality shot rather than the report.
    """
    import torch

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    t0 = time.monotonic()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=float(fps),
        num_inference_steps=steps,
        guidance_scale=guidance,
        enable_sound=sound,
        generator=generator,
        # The checker was never constructed (enable_safety_checker=False at load), so
        # this is belt and braces: it keeps the call working if someone turns the
        # constructor flag back on to get the guardrail without editing every task.
        enable_safety_check=False,
    )
    secs = time.monotonic() - t0
    log.info("generated %s frames in %.1fs (%.1fs/step)", num_frames, secs, secs / steps)
    return result, secs


def load_action_example(model_path: str) -> dict:
    """Read the action-conditioning example that ships inside the checkpoint.

    Returns the parsed JSON plus the conditioning frame. Everything the rollout needs
    is self-describing in that file: prompt, embodiment domain, viewpoint, fps,
    resolution tier, chunk size, and the chunks themselves as [num_chunks, 16, 29].
    """
    import torch
    from PIL import Image

    meta = json.loads(open(os.path.join(model_path, ACTION_CHUNKS)).read())
    meta["chunks"] = torch.tensor(meta["action_chunks"], dtype=torch.float32)
    meta["first_frame"] = Image.open(os.path.join(model_path, ACTION_FRAME)).convert("RGB")
    log.info(
        "action example: %s, domain=%s, chunks=%s",
        meta.get("prompt"), meta.get("domain_name"), tuple(meta["chunks"].shape),
    )
    return meta


def rollout(
    pipe,
    meta: dict,
    *,
    num_chunks: int = 2,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int | None = 0,
):
    """Action-conditioned forward dynamics: first frame + actions -> future video.

    This is the world-model claim in its most literal form. The prompt names the task
    ("Pickup items in the supermarket") but says NOTHING about what happens next; the
    future comes entirely from the action sequence and the model's learned dynamics.

    Chunks are rolled AUTOREGRESSIVELY: chunk 0 is conditioned on the real observed
    frame, and every chunk after it is conditioned on the last frame the model itself
    predicted. That is the honest version of a rollout, and it is also where error
    compounds, which is worth watching in the output. Each chunk of 16 actions yields
    17 frames (`chunk_size + 1`).

    Note what is NOT passed: height, width and num_frames must all be None for an
    action run. Resolution comes from `action.resolution_tier` and the frame count
    from `action.chunk_size`; passing them raises.
    """
    import torch
    from diffusers import CosmosActionCondition

    chunks = meta["chunks"]
    frame = meta["first_frame"]
    frames: list = []
    per_chunk: list[float] = []

    for i in range(min(num_chunks, chunks.shape[0])):
        generator = torch.Generator().manual_seed(seed + i) if seed is not None else None
        t0 = time.monotonic()
        result = pipe(
            prompt=meta["prompt"],
            action=CosmosActionCondition(
                mode="forward_dynamics",
                chunk_size=int(meta["action_chunk_size"]),
                domain_name=meta["domain_name"],
                resolution_tier=int(meta["image_size"]),
                raw_actions=chunks[i],
                image=frame,
                view_point=meta.get("view_point", "ego_view"),
            ),
            fps=float(meta.get("fps", 10)),
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            # NVIDIA's runner passes use_system_prompt=False for action modes: the
            # action model was trained on structured captions without the generic
            # "you are a helpful assistant who will generate videos" preamble.
            use_system_prompt=False,
            enable_safety_check=False,
        )
        secs = time.monotonic() - t0
        per_chunk.append(secs)
        chunk_frames = list(result.video)
        # Drop the conditioning frame on every chunk after the first, or the seam
        # shows up as a duplicated frame in the middle of the rollout.
        frames.extend(chunk_frames if i == 0 else chunk_frames[1:])
        # Autoregress: the next chunk starts from what this one predicted.
        frame = chunk_frames[-1]
        log.info("chunk %s: %s frames in %.1fs", i, len(chunk_frames), secs)

    return frames, per_chunk
