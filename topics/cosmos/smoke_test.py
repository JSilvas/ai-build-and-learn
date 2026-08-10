"""Prove Cosmos 3 works on this host, without Flyte in the way.

    ./.venv/bin/python smoke_test.py              # one short 480p clip
    ./.venv/bin/python smoke_test.py --image      # 1-frame text-to-image, ~30s
    ./.venv/bin/python smoke_test.py --action     # one action-conditioned chunk

This is the control number that the containerised run from pipeline.py gets compared
against, the same role smoke_test.py plays in topics/isaac-sim. If this passes and
the Flyte run does not, the problem is the pod, not the model.

Writes into out/ and prints a luminance probe of what it produced, because a black
clip is a valid mp4 and "it ran without an error" is not the same as "it rendered".
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import time

import media
import prompts
import world

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/Cosmos3-Nano")
    ap.add_argument("--scene", default="box-topple", choices=sorted(prompts.SCENES))
    ap.add_argument("--image", action="store_true", help="1 frame (text-to-image)")
    ap.add_argument("--action", action="store_true", help="action-conditioned rollout")
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)

    print(f"guard: {world.guard_memory()}")
    t0 = time.monotonic()
    pipe = world.load(args.repo)
    print(f"loaded in {(time.monotonic() - t0) / 60:.1f} min")

    if args.action:
        meta = world.load_action_example(world.snapshot(args.repo))
        frames, per_chunk = world.rollout(pipe, meta, num_chunks=1, steps=args.steps)
        mp4 = media.encode(frames, fps=int(meta.get("fps", 10)))
        (outdir / "action.mp4").write_bytes(mp4)
        print(f"action rollout: {sum(per_chunk):.1f}s -> out/action.mp4")
        print(f"probe: {media.probe(mp4)}")
        return

    prompt = prompts.get(args.scene)
    result, secs = world.generate(
        pipe,
        prompt,
        negative_prompt=prompts.NEGATIVE,
        num_frames=1 if args.image else args.frames,
        height=480,
        width=832,
        steps=args.steps,
    )

    if args.image:
        result.video[0].save(outdir / "sample.jpg", format="JPEG", quality=90)
        print(f"text-to-image: {secs:.1f}s -> out/sample.jpg")
        return

    mp4 = media.encode(result.video, fps=24)
    (outdir / "sample.mp4").write_bytes(mp4)
    print(f"text-to-video: {secs:.1f}s ({secs / args.steps:.1f}s/step) -> out/sample.mp4")
    print(f"probe: {media.probe(mp4)}")


if __name__ == "__main__":
    main()
