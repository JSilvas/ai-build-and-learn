"""Run the real code paths on the host before spending a pod on them.

    .venv/bin/python smoke_test.py            # ViT-L, one clip, ~2 min
    .venv/bin/python smoke_test.py --full     # also encodes all 100 clips for the probe

Deliberately exercises the parts that are easy to get silently wrong rather than the
parts that are easy to test: token layout, overlay alignment, and whether the scores
actually beat their own chance floor. A green run here means `flyte run` is worth it.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

import clips as clip_io
import jepa
import probing
import viz
from config import CLIPS_REPO, VITL

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
FRAMES = 32


def main(full: bool) -> int:
    fails = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    print("guard:", jepa.guard_memory())
    model, processor, params = jepa.load(VITL)
    print(f"loaded {VITL}: {params / 1e6:.0f}M params")

    catalog = clip_io.list_clips(CLIPS_REPO)
    labels = clip_io.labels_of(catalog)
    print(f"catalog: {len(catalog)} clips, classes {labels}")
    check("catalog non-empty", len(catalog) > 0)

    path = clip_io.pick(catalog, "bowling")
    pixel_values, shown = clip_io.load_clip(processor, CLIPS_REPO, path, FRAMES)
    tubelets, grid = jepa.grid_of(model, FRAMES)
    print(f"clip {path}: pv {tuple(pixel_values.shape)} shown {shown.shape}")

    # The overlay contract: one patch is exactly one (256/grid)^2 pixel block of `shown`.
    check("shown frames match the model input", shown.shape[0] == pixel_values.shape[1]
          and shown.shape[1] == model.config.crop_size,
          f"{shown.shape} vs crop {model.config.crop_size}")
    check("patch grid divides the frame", shown.shape[1] % grid == 0)

    seq = jepa.encode(model, pixel_values)
    jepa.check_layout(seq, tubelets, grid)          # raises if the raster ever changes
    check("token layout", True, f"{tubelets} tubelets x {grid}x{grid} = {seq.shape[0]}")

    aniso = jepa.anisotropy(seq)
    print(f"anisotropy: raw {aniso['raw']:.3f} -> centered {aniso['centered']:.3f}")
    check("centering removes the shared component", abs(aniso["centered"]) < 0.05
          and aniso["raw"] > aniso["centered"])

    loc = {}
    for name, mask3d in (
        ("tube", jepa.tube(tubelets, grid, blocks=2, size=8, seed=0)),
        ("future", jepa.future(tubelets, grid, context=0.5)),
    ):
        ctx_ids, tgt_ids = jepa.ids_of(mask3d)
        check(f"{name} mask ids partition the sequence",
              len(ctx_ids) + len(tgt_ids) == seq.shape[0] and len(tgt_ids) > 0,
              f"{len(ctx_ids)} context / {len(tgt_ids)} target")

        pred, true = jepa.predict(model, pixel_values, ctx_ids, tgt_ids)
        check(f"{name} predictor shape", pred.shape == true.shape,
              f"{tuple(pred.shape)}")

        s = jepa.score(pred, true, tgt_ids, grid)
        floor = jepa.shuffled_floor(pred, true, tgt_ids, grid)
        loc[name] = jepa.localization(s, floor)
        print(f"  {name}: masked {len(tgt_ids) / seq.shape[0]:.0%}  "
              f"cos {s['cos']:.3f} (chance {floor['cos']:.3f})  "
              f"top1 {s['top1']:.1%} (chance {floor['top1']:.1%})  "
              f"median dt {s['dt']:.1f} (chance {floor['dt']:.1f}) dh {s['dh']:.1f} "
              f"dw {s['dw']:.1f}  time-localised {loc[name]:.2f}")
        check(f"{name} beats its shuffled floor", s["cos"] > floor["cos"])

        field = jepa.per_patch_cos(pred, true, tgt_ids, tubelets, grid)
        check(f"{name} score map masks correctly",
              int(np.isfinite(field).sum()) == len(tgt_ids),
              f"{int(np.isfinite(field).sum())} finite of {field.size}")

        masked = viz.masked_video(shown, mask3d)
        check(f"{name} masked video differs from the input", not np.array_equal(masked, shown))
        pair = viz.side_by_side_video(masked, viz.heat_video(shown, field))
        mp4 = viz.encode_mp4(pair, fps=12)
        report = viz.probe(mp4)
        print(f"  {name} mp4: {len(mp4) / 1024:.0f} KB, {report}")
        check(f"{name} mp4 decodes and is not black", "black" in report
              and report.split("black")[0].strip().endswith("0"), report[:80])

    # The claim the whole `inpaint` task rests on. If this ever flips, the report's
    # verdict is wrong and the README needs rewriting, so it is a check and not a note.
    check("tube mask localises in time better than the future mask",
          loc["tube"] > loc["future"],
          f"tube {loc['tube']:.2f} vs future {loc['future']:.2f}")

    # Charts must render headless (Agg) or the report silently loses them.
    check("horizon chart renders",
          viz.horizon_chart({"x": [(1, 0.5), (2, 0.4)]}, {"chance": 0.3}).startswith("<img"))
    check("bar chart renders",
          viz.bar_chart("t", ["a"], {"s": [0.5]}, "y", floor=0.2).startswith("<img"))
    check("confusion chart renders",
          viz.confusion_chart(np.eye(3, dtype=int), list("abc"), "t").startswith("<img"))

    if full:
        print("encoding the full dataset for the probe...")
        data = probing.encode_dataset(
            model, processor, CLIPS_REPO, catalog, FRAMES,
            on_progress=lambda i, n: print(f"  {i}/{n}", end="\r"),
        )
        acc, pred = probing.linear_probe(data["X"], data["y"], data["train"])
        acc_px, _ = probing.linear_probe(data["P"], data["y"], data["train"])
        ret, _ = probing.retrieval(data["X"], data["y"])
        chance = 1 / len(data["labels"])
        print(f"\nprobe {acc:.1%}  pixels {acc_px:.1%}  1-NN {ret:.1%}  chance {chance:.0%}")
        check("probe beats the pixel baseline", acc > acc_px)
        check("retrieval beats chance", ret > 2 * chance)
        check("no clips lost", not data["failed"], f"{len(data['failed'])} failed")

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run the 100-clip probe")
    sys.exit(main(ap.parse_args().full))
