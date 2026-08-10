"""NVIDIA Cosmos 3 on Flyte: a world model in a pod, with the video in the report.

    flyte run pipeline.py imagine                      # text -> predicted world
    flyte run pipeline.py imagine --scene forklift
    flyte run pipeline.py rollout                      # actions -> predicted future
    flyte run pipeline.py compare                      # short vs structured prompt
    flyte run pipeline.py world_models                 # all three, one run

Runs to the `world-models` project (.flyte/config.yaml), alongside topics/dreamerv3.
The pairing is the point of the event: Dreamer LEARNS a world model of one small
environment from its own experience, Cosmos 3 IS a pretrained world model of the
physical world that you condition and roll forward. The robotics demos that supply
the physics ground truth (topics/rl-mujoco, topics/isaac-sim) live in `physical-ai`.

── Why every task loads the pipeline itself, and only once ─────────────────────
There is no shared model cache across pods on this cluster, so each task pod pulls
the 35 GB snapshot into its own /tmp/hf. That download dominates the wall clock:
denoising 45 frames is a couple of minutes, fetching the weights is longer. So each
task loads ONE pipeline and generates everything it needs from it, and `compare`
exists as a single task rather than a fan-out for exactly that reason. Splitting it
into two tasks would double the download to parallelise the cheap part.

── Why the report gets painted before there is anything to show ────────────────
Same lesson as the DreamerV3 task next door. A pod that spends its first ten minutes
downloading and its next few loading a 16B transformer, while its report stays
blank, is indistinguishable from a hung pod. Each task repaints at every stage so
the report always says what it is doing.
"""

from __future__ import annotations

import logging

import flyte
import flyte.report

# Imported at top level so Flyte bundles these siblings into the pod. A deferred
# import inside a task body is exactly how you get ModuleNotFoundError in the pod
# while everything works on the host.
import media
import prompts
import reports
import world
from config import NANO, gpu_env, orch_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _paint(stage: str, detail: str, rows: list[tuple[str, str]]) -> None:
    flyte.report.replace(reports.progress_html(stage, detail, rows), do_flush=True)


@gpu_env.task(report=True)
async def imagine(
    scene: str = "box-topple",
    repo: str = NANO,
    frames: int = 45,
    height: int = 480,
    width: int = 832,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
    sound: bool = False,
) -> dict:
    """Generate a predicted world from text, and show it in the report.

    Defaults are 480p / 45 frames rather than NVIDIA's 720p / 189. The clip is going
    to be base64'd into an HTML report, so this is sized to embed rather than to
    impress; raise both for the quality shot.
    """
    rows = [("Model", repo), ("Scene", scene), ("Frames", f"{frames} at {width}x{height}")]
    _paint("Fetching weights", f"{repo} is ~35 GB and lands in this pod's /tmp/hf.", rows)

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo, sound=sound)

    prompt = prompts.get(scene)
    _paint("Generating", f"{steps} denoising steps.", rows)
    result, secs = world.generate(
        pipe, prompt,
        negative_prompt=prompts.NEGATIVE,
        num_frames=frames, height=height, width=width,
        steps=steps, guidance=guidance, seed=seed, sound=sound,
    )

    mp4 = media.encode(result.video, fps=24)
    probe = media.probe(mp4)
    log.info("clip: %s", probe)

    rows += [
        ("Denoise time", f"{secs / 60:.1f} min ({secs / steps:.1f}s/step)"),
        ("Clip", f"{len(mp4) / 1024:.0f} KB"),
    ]
    body = reports.clip_block(
        f"Predicted world: {scene}",
        media.video_html(mp4, f"{frames} frames at {width}x{height}, seed {seed}"),
        media.strip(result.video),
        probe,
        prompt,
    )
    flyte.report.replace(
        reports.final_html("Text to world", rows, body, reports.IMAGINE_EXPLAINER),
        do_flush=True,
    )
    return {"scene": scene, "seconds": round(secs, 1), "probe": probe, "kb": len(mp4) // 1024}


@gpu_env.task(report=True)
async def rollout(
    repo: str = NANO,
    chunks: int = 2,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Action-conditioned rollout: one observed frame plus robot actions -> the future.

    Uses the action example that ships INSIDE the checkpoint rather than one we
    invent: an AgiBotWorld humanoid picking items in a supermarket, with four chunks
    of sixteen 29-dimensional actions and the matching first frame. Using NVIDIA's own
    asset means the action encoding is unambiguously correct, so if the rollout looks
    wrong it is the model or the box, not our action tensor.

    `chunks` is the interesting knob. Chunk 0 is conditioned on the real observed
    frame; every chunk after it is conditioned on the last frame the model itself
    predicted, so raising this is how you watch prediction error compound.
    """
    rows = [("Model", repo), ("Task", "forward dynamics (action-conditioned)")]
    _paint("Fetching weights", f"{repo} is ~35 GB and lands in this pod's /tmp/hf.", rows)

    path = world.snapshot(repo)
    meta = world.load_action_example(path)
    rows += [
        ("Embodiment", f"{meta['domain_name']} ({meta['chunks'].shape[-1]}-D actions)"),
        ("Prompt", meta["prompt"]),
        ("Chunks", f"{chunks} x {meta['action_chunk_size']} actions"),
    ]

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    _paint("Rolling forward", "Each chunk is conditioned on the previous chunk's last frame.", rows)
    frames, per_chunk = world.rollout(
        pipe, meta, num_chunks=chunks, steps=steps, guidance=guidance, seed=seed
    )

    fps = int(meta.get("fps", 10))
    mp4 = media.encode(frames, fps=fps)
    probe = media.probe(mp4)
    total = sum(per_chunk)
    log.info("rollout: %s frames, %s", len(frames), probe)

    rows += [
        ("Rollout time", f"{total / 60:.1f} min ({', '.join(f'{s:.0f}s' for s in per_chunk)})"),
        ("Frames", f"{len(frames)} at {fps} fps"),
    ]
    body = reports.clip_block(
        "Predicted future, driven by actions",
        media.video_html(mp4, f"{len(frames)} frames, {chunks} chunk(s) autoregressive"),
        media.strip(frames, count=8),
        probe,
        meta["prompt"],
    )
    body = reports.side_by_side([
        ("Observed: the one frame the model was given", media.image_html(meta["first_frame"])),
        ("Predicted: everything after it", body),
    ])
    flyte.report.replace(
        reports.final_html("Actions to future", rows, body, reports.ROLLOUT_EXPLAINER),
        do_flush=True,
    )
    return {
        "embodiment": meta["domain_name"],
        "chunks": chunks,
        "frames": len(frames),
        "seconds": round(total, 1),
        "probe": probe,
    }


@gpu_env.task(report=True)
async def compare(
    scene: str = "box-topple",
    repo: str = NANO,
    frames: int = 45,
    height: int = 480,
    width: int = 832,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Same scene and seed, prompted twice: one sentence vs the structured caption.

    Cosmos 3 was trained on long structured JSON captions, and NVIDIA's guidance is
    to upsample a short prompt into that form with an LLM before generating. This
    task is the measurement of what that step is worth, and it costs one extra
    generation rather than one extra 35 GB download because both clips come from the
    same loaded pipeline.
    """
    rows = [("Model", repo), ("Scene", scene), ("Seed", str(seed))]
    _paint("Fetching weights", f"{repo} is ~35 GB and lands in this pod's /tmp/hf.", rows)

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    cells, timings = [], {}
    for style, label in (("short", "One sentence"), ("structured", "Structured caption")):
        prompt = prompts.get(scene, style)
        _paint("Generating", f"{label.lower()}, {steps} denoising steps.", rows)
        result, secs = world.generate(
            pipe, prompt,
            negative_prompt=prompts.NEGATIVE,
            num_frames=frames, height=height, width=width,
            steps=steps, guidance=guidance, seed=seed,
        )
        mp4 = media.encode(result.video, fps=24)
        timings[style] = round(secs, 1)
        cells.append((
            label,
            media.video_html(mp4, f"{len(mp4) / 1024:.0f} KB", max_width=440)
            + media.strip(result.video, count=4, width=104)
            + reports.note(media.probe(mp4))
            + reports.details("prompt", prompt),
        ))
        log.info("%s: %.1fs", style, secs)

    rows.append(("Denoise time", ", ".join(f"{k} {v}s" for k, v in timings.items())))
    flyte.report.replace(
        reports.final_html(
            "Prompt shape", rows, reports.side_by_side(cells), reports.COMPARE_EXPLAINER
        ),
        do_flush=True,
    )
    return {"scene": scene, "seconds": timings}


@orch_env.task(report=True)
async def world_models(scene: str = "box-topple", repo: str = NANO) -> dict:
    """Entry point. CPU-only orchestrator so it cannot deadlock its own GPU children.

    The children run in SEQUENCE, not in parallel, and that is not a style choice:
    there is one GPU on this box, so a second GPU task would sit Unschedulable on
    "Insufficient nvidia.com/gpu" until the first finished anyway.
    """
    generated = await imagine(scene=scene, repo=repo)
    predicted = await rollout(repo=repo)
    prompted = await compare(scene=scene, repo=repo)
    result = {"imagine": generated, "rollout": predicted, "compare": prompted}
    log.info("result: %s", result)
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(world_models))
