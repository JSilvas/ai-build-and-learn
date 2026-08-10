"""V-JEPA 2 on Flyte: a world model with no decoder, and what that costs.

    flyte run pipeline.py inpaint                  # the predictor, under two masks
    flyte run pipeline.py inpaint --clip archery --context 0.75
    flyte run pipeline.py probe                    # frozen features -> action recognition
    flyte run pipeline.py scale                    # ViT-L vs ViT-g, same measurements
    flyte run pipeline.py vjepa                    # all three, one run

Runs to the `world-models` project (.flyte/config.yaml), alongside topics/cosmos and
topics/dreamerv3. The three are the same question asked three ways: Cosmos predicts the
future in pixels, Dreamer learns a latent world model of one environment from its own
experience, and V-JEPA 2 predicts representations learned self-supervised from video.

── What this demo is careful about ─────────────────────────────────────────────
V-JEPA 2 has no decoder, so there is no honest way to render "what it predicted". Every
frame in these reports is either the model's literal input (with the masked patches
blacked out) or a per-patch number we computed painted onto those same pixels. Nothing
here is a vector dressed up as an image.

The second care is floors. A cosine similarity between two ViT tokens is a number
between 0 and 1 that always looks encouraging, so every score in `inpaint` is reported
next to a shuffled-pairing chance floor and a no-model baseline, and every score in
`probe` next to chance and a raw-pixel probe.

── Why the tasks run in sequence ───────────────────────────────────────────────
One GPU on this box, so a second GPU task would sit Unschedulable on "Insufficient
nvidia.com/gpu" until the first finished anyway. The orchestrator is CPU-only for the
same reason: a GPU-holding orchestrator deadlocks its own GPU child.
"""

from __future__ import annotations

import logging

import flyte
import flyte.report

# Imported at top level so Flyte bundles these siblings into the pod. A deferred import
# inside a task body is exactly how you get ModuleNotFoundError in the pod while
# everything works on the host.
import clips as clip_io
import jepa
import probing
import reports
import viz
from config import CLIPS_REPO, VITG, VITL, gpu_env, orch_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _paint(stage: str, detail: str, rows: list[tuple[str, str]]) -> None:
    flyte.report.replace(reports.progress_html(stage, detail, rows), do_flush=True)


@gpu_env.task(report=True)
async def inpaint(
    clip: str = "bowling",
    repo: str = VITL,
    frames: int = 64,
    context: float = 0.5,
    block: int = 8,
    seed: int = 0,
) -> dict:
    """Hide part of a clip, predict the hidden part IN LATENT SPACE, and score it twice.

    Once under a TUBE mask (spatial blocks removed across the whole clip), which is the
    masking V-JEPA 2 was pretrained on, and once under a FUTURE mask (everything after
    `context` of the way through), which it never saw during pretraining. Same clip,
    same predictor, same scoring, so the difference is the mask.

    The defaults are chosen so the two masks hide roughly the SAME fraction of tokens:
    two 8x8 blocks is ~48% and `context=0.5` is 50%. That matters, because most of
    these metrics move with how much was hidden, and an easier mask winning would prove
    nothing. `block` is the side length in patches of each of the two tube blocks;
    `context` only affects the future mask.
    """
    rows = [("Model", repo), ("Clip", clip), ("Frames", str(frames))]
    _paint("Fetching", "Pulling the checkpoint and one clip into this pod.", rows)

    guard = jepa.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "V-JEPA 2 encoder + predictor, BF16.", rows)
    model, processor, params = jepa.load(repo)
    rows.append(("Params", f"{params / 1e6:.0f}M"))

    catalog = clip_io.list_clips(CLIPS_REPO)
    path = clip_io.pick(catalog, clip)
    _paint("Encoding", f"{path}", rows)
    pixel_values, shown = clip_io.load_clip(processor, CLIPS_REPO, path, frames)
    tubelets, grid = jepa.grid_of(model, frames)
    seq = jepa.encode(model, pixel_values)
    jepa.check_layout(seq, tubelets, grid)

    aniso = jepa.anisotropy(seq)
    rows += [
        ("Tokens", f"{tubelets} tubelets x {grid}x{grid} = {seq.shape[0]}, dim {seq.shape[1]}"),
        ("Source", path),
        ("Random-pair cosine", f"raw {aniso['raw']:.3f} -> centered {aniso['centered']:.3f}"),
    ]

    masks = {
        "tube (pretraining mask)": jepa.tube(tubelets, grid, blocks=2, size=block, seed=seed),
        "future (never trained on)": jepa.future(tubelets, grid, context=context),
    }

    cells, results, hlines = [], {}, {}
    horizon_series = {}
    for name, mask3d in masks.items():
        _paint("Predicting", f"{name}: running the predictor over the masked tokens.", rows)
        context_ids, target_ids = jepa.ids_of(mask3d)
        pred, true = jepa.predict(model, pixel_values, context_ids, target_ids)

        scored = jepa.score(pred, true, target_ids, grid)
        floor = jepa.shuffled_floor(pred, true, target_ids, grid, seed=seed)
        ctx = jepa.context_floor(seq, context_ids, target_ids, true, grid)
        # Normalised against this mask's OWN chance level, which is the only way the
        # two masks can be put on the same axis. See jepa.localization.
        scored["loc"] = jepa.localization(scored, floor)
        ctx["loc"] = jepa.localization(ctx, floor)
        floor["loc"] = jepa.localization(floor, floor)
        results[name] = {"predictor": scored, "shuffled": floor, "context_mean": ctx}

        field = jepa.per_patch_cos(pred, true, target_ids, tubelets, grid)
        masked_frames = viz.masked_video(shown, mask3d)
        heat_frames = viz.heat_video(shown, field)
        pair = viz.side_by_side_video(masked_frames, heat_frames)
        mp4 = viz.encode_mp4(pair, fps=12)
        log.info("%s: %s | %s", name, scored, viz.probe(mp4))

        share = float(mask3d.float().mean())
        cells.append((
            name,
            viz.video_html(mp4, f"left: what the encoder saw ({share:.0%} of tokens hidden). "
                                f"right: per-patch prediction quality, grey = not masked.")
            + reports.note(viz.probe(mp4))
            + reports.score_table(
                [("predictor", scored), ("context mean (no model)", ctx),
                 ("shuffled pairing (chance)", floor)],
                highlight="predictor",
            ),
        ))
        horizon_series[name] = jepa.horizon(pred, true, mask3d, target_ids, tubelets, grid)
        if name.startswith("future"):
            hlines["shuffled (chance)"] = floor["cos"]

    tube_s = results["tube (pretraining mask)"]["predictor"]
    tube_f = results["tube (pretraining mask)"]["shuffled"]
    future_s = results["future (never trained on)"]["predictor"]
    future_f = results["future (never trained on)"]["shuffled"]
    tube_share = float(masks["tube (pretraining mask)"].float().mean())
    future_share = float(masks["future (never trained on)"].float().mean())
    verdict = reports.verdict(
        f"With <b>{tube_share:.0%}</b> and <b>{future_share:.0%}</b> of tokens hidden, so the two "
        f"masks are the same size and only the shape differs: under the mask it was pretrained "
        f"on, the predictor's nearest retrieved token is a median <b>{tube_s['dt']:.0f} "
        f"tubelets</b> from the truth in time, where shuffling the same predictions gives "
        f"{tube_f['dt']:.0f}. That is {tube_s['loc']:.0%} of the chance-level temporal error "
        f"removed: it finds the right moment. Asked to extrapolate forward in time instead, the "
        f"same predictor lands a median <b>{future_s['dt']:.0f} tubelets</b> away against a "
        f"chance level of {future_f['dt']:.0f}, or {future_s['loc']:.0%}. It is returning a "
        f"plausible representation of the scene at the wrong moment. This checkpoint inpaints in "
        f"latent space; it does not forecast.",
        good=tube_s["loc"] > future_s["loc"],
    )

    body = (
        reports.heading("What the encoder saw, and where the prediction landed")
        + reports.side_by_side(cells, basis=460)
        + reports.heading("Does it degrade with horizon, or was it never tracking time?")
        + viz.horizon_chart(
            {"future mask (predictor)": horizon_series["future (never trained on)"]}, hlines
        )
        + reports.note(
            "Only the future mask is plotted: 'tubelets ahead of the last visible frame' has no "
            "meaning for a tube mask, whose targets span the whole clip. The tube mask's numbers "
            "are in the scorecard above. Note also that the cosine axis is not comparable "
            "between the two masks, which is what the 'time localised' column exists to fix."
        )
        + reports.note(reports.GEOMETRY_NOTE)
        + verdict
    )
    flyte.report.replace(
        reports.final_html("Masked latent prediction", rows, body, reports.INPAINT_EXPLAINER),
        do_flush=True,
    )
    return {
        "clip": path,
        "repo": repo,
        "anisotropy": aniso,
        "scores": {k: {m: {kk: round(vv, 4) for kk, vv in s.items()} for m, s in v.items()}
                   for k, v in results.items()},
    }


@gpu_env.task(report=True)
async def probe(repo: str = VITL, frames: int = 32) -> dict:
    """Freeze the encoder, mean-pool every clip, and see what one linear layer can do.

    100 Kinetics clips over 5 classes, the dataset's own train/val split. Nothing is
    fine-tuned: the only trained parameters in this task are a 1024x5 matrix and its
    bias. The retrieval number below it involves no training at all.
    """
    rows = [("Model", repo), ("Dataset", CLIPS_REPO), ("Frames per clip", str(frames))]
    _paint("Fetching", "Pulling the checkpoint and the clip catalogue.", rows)

    guard = jepa.guard_memory()
    rows.append(("GPU", guard))
    model, processor, params = jepa.load(repo)
    rows.append(("Params", f"{params / 1e6:.0f}M"))

    catalog = clip_io.list_clips(CLIPS_REPO)
    rows.append(("Clips", f"{len(catalog)} over {len(clip_io.labels_of(catalog))} classes"))
    _paint("Encoding", "One forward pass per clip, encoder only, no gradients.", rows)

    data = probing.encode_dataset(
        model, processor, CLIPS_REPO, catalog, frames,
        on_progress=lambda i, n: _paint("Encoding", f"clip {i}/{n}", rows),
    )
    X, P, y, train = data["X"], data["P"], data["y"], data["train"]
    labels = data["labels"]
    if data["failed"]:
        rows.append(("Skipped", f"{len(data['failed'])} clip(s) failed to decode"))

    _paint("Probing", "Training one linear layer on the frozen features.", rows)
    acc, pred = probing.linear_probe(X, y, train)
    acc_pixels, _ = probing.linear_probe(P, y, train)
    ret, neighbours = probing.retrieval(X, y, centered=True)
    ret_raw, _ = probing.retrieval(X, y, centered=False)
    chance = 1.0 / len(labels)

    rows += [
        ("Split", f"{int(train.sum())} train / {int((~train).sum())} val"),
        ("Linear probe", f"{acc:.1%}"),
        ("Raw-pixel probe", f"{acc_pixels:.1%}"),
        ("1-NN retrieval", f"{ret:.1%} centered, {ret_raw:.1%} raw"),
        ("Chance", f"{chance:.0%}"),
    ]

    cm = probing.confusion(y[~train], pred, len(labels))
    charts = reports.side_by_side([
        ("Where the probe is wrong", viz.confusion_chart(cm, labels, "Val confusion")),
        ("Against the floors", viz.bar_chart(
            "Frozen features vs baselines", ["5-way action recognition"],
            {"V-JEPA 2 + linear": [acc], "raw pixels + linear": [acc_pixels],
             "V-JEPA 2 1-NN (no training)": [ret]},
            "accuracy", floor=chance, floor_label="chance",
        )),
    ], basis=340)

    _paint("Rendering", "Building the retrieval examples.", rows)
    retrieval_cells = []
    for qi in _retrieval_examples(y, neighbours, labels):
        q_split, q_label, q_path = data["clips"][qi]
        q_frames = clip_io.decode(clip_io.fetch(CLIPS_REPO, q_path), 24)
        inner = viz.video_html(viz.encode_mp4(q_frames, fps=10),
                               f"query: {q_label}", max_width=230)
        for rank, ni in enumerate(neighbours[qi].tolist()[:2]):
            n_split, n_label, n_path = data["clips"][ni]
            hit = "match" if n_label == q_label else "MISS"
            n_frames = clip_io.decode(clip_io.fetch(CLIPS_REPO, n_path), 24)
            inner += viz.video_html(viz.encode_mp4(n_frames, fps=10),
                                    f"#{rank + 1}: {n_label} ({hit})", max_width=230)
        retrieval_cells.append((f"{q_label}", inner))

    body = (
        reports.heading("How good are features nobody supervised?")
        + charts
        + reports.heading("Nearest neighbours in the frozen embedding space")
        + reports.note(
            "No classifier and no training: each query clip is shown with the clips whose "
            "mean-pooled embedding is closest to it. These are the same features the probe "
            "above sees."
        )
        + reports.side_by_side(retrieval_cells, basis=250)
        + reports.note(reports.GEOMETRY_NOTE)
        + reports.verdict(
            f"A single linear layer on frozen V-JEPA 2 features gets <b>{acc:.0%}</b> on 5-way "
            f"action recognition, against <b>{acc_pixels:.0%}</b> for the same probe on "
            f"downsampled pixels and {chance:.0%} chance. With no training at all, nearest-"
            f"neighbour retrieval gets {ret:.0%}. Centering matters: the same retrieval on raw "
            f"features scores {ret_raw:.0%}.",
            good=acc > acc_pixels,
        )
    )
    flyte.report.replace(
        reports.final_html("Frozen features, one linear layer", rows, body,
                           reports.PROBE_EXPLAINER),
        do_flush=True,
    )
    return {
        "repo": repo,
        "probe": round(acc, 4),
        "pixel_probe": round(acc_pixels, 4),
        "retrieval": round(ret, 4),
        "retrieval_raw": round(ret_raw, 4),
        "chance": round(chance, 4),
        "clips": len(data["clips"]),
    }


def _retrieval_examples(y, neighbours, labels, per: int = 3) -> list[int]:
    """Pick query clips to show: prefer a mix of hits and at least one miss.

    Showing only successes would be a demo of the report, not of the model.
    """
    hits = [i for i in range(len(y)) if y[neighbours[i][0]] == y[i]]
    misses = [i for i in range(len(y)) if y[neighbours[i][0]] != y[i]]
    picked = hits[:: max(1, len(hits) // max(per - 1, 1))][: per - 1]
    return picked + misses[:1] if misses else hits[:per]


@gpu_env.task(report=True)
async def scale(
    small: str = VITL,
    large: str = VITG,
    clip: str = "bowling",
    frames: int = 32,
    block: int = 8,
    seed: int = 0,
) -> dict:
    """Does a bigger self-supervised encoder carry more? Same clips, same masks, same probe.

    Both models are loaded and measured inside ONE task rather than fanned out, for the
    same reason the Cosmos demo does it: there is a single GPU here, so a fan-out would
    serialise anyway, and this way the two models are scored on identically decoded
    clips instead of two independent decodes.
    """
    rows = [("Models", f"{small} vs {large}"), ("Frames per clip", str(frames))]
    _paint("Starting", "Two encoders, measured on the same clips.", rows)
    rows.append(("GPU", jepa.guard_memory()))

    catalog = clip_io.list_clips(CLIPS_REPO)
    path = clip_io.pick(catalog, clip)
    out = {}
    for repo in (small, large):
        _paint("Loading", f"{repo}", rows)
        model, processor, params = jepa.load(repo)

        pixel_values, _ = clip_io.load_clip(processor, CLIPS_REPO, path, frames)
        tubelets, grid = jepa.grid_of(model, frames)
        seq = jepa.encode(model, pixel_values)
        jepa.check_layout(seq, tubelets, grid)

        masks = {
            "tube": jepa.tube(tubelets, grid, blocks=2, size=block, seed=seed),
            "future": jepa.future(tubelets, grid, context=0.5),
        }
        scores = {}
        for name, mask3d in masks.items():
            ctx_ids, tgt_ids = jepa.ids_of(mask3d)
            pred, true = jepa.predict(model, pixel_values, ctx_ids, tgt_ids)
            s = jepa.score(pred, true, tgt_ids, grid)
            floor = jepa.shuffled_floor(pred, true, tgt_ids, grid, seed=seed)
            s["loc"] = jepa.localization(s, floor)
            floor["loc"] = jepa.localization(floor, floor)
            scores[name], scores[f"{name}_floor"] = s, floor

        _paint("Encoding", f"{repo}: {len(catalog)} clips for the probe.", rows)
        data = probing.encode_dataset(
            model, processor, CLIPS_REPO, catalog, frames,
            on_progress=lambda i, n, r=repo: _paint("Encoding", f"{r}: clip {i}/{n}", rows),
        )
        acc, _ = probing.linear_probe(data["X"], data["y"], data["train"])
        ret, _ = probing.retrieval(data["X"], data["y"])
        out[repo] = {
            "params_m": round(params / 1e6),
            "hidden": int(seq.shape[1]),
            "probe": round(acc, 4),
            "retrieval": round(ret, 4),
            "scores": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in scores.items()},
        }
        rows.append((repo.split("/")[-1], f"{params / 1e6:.0f}M, probe {acc:.1%}, 1-NN {ret:.1%}"))
        log.info("%s -> %s", repo, out[repo])

        del model
        _free()

    names = [r.split("/")[-1] for r in out]
    chance = 1.0 / len(clip_io.labels_of(catalog))
    body = (
        reports.heading("Semantics: what a linear layer can read off")
        + viz.bar_chart(
            "Frozen-feature action recognition", names,
            {"linear probe": [out[r]["probe"] for r in out],
             "1-NN retrieval": [out[r]["retrieval"] for r in out]},
            "accuracy", floor=chance, floor_label="chance",
        )
        + reports.heading("Prediction: does it put the masked tokens at the right moment?")
        + viz.bar_chart(
            "Masked latent prediction, temporal localisation", names,
            {"tube mask (pretrained on)": [out[r]["scores"]["tube"]["loc"] for r in out],
             "future mask (never trained on)": [out[r]["scores"]["future"]["loc"] for r in out]},
            "fraction of chance error removed", floor=0.0, floor_label="chance",
        )
        + reports.note(
            "Plotted as the normalised localisation score rather than raw cosine, because raw "
            "cosine is not comparable between the two masks. 1.00 means the retrieved token is "
            "at exactly the right moment; 0.00 means no better than shuffling the predictions. "
            "The two charts are also different questions: the first is about the encoder's "
            "representation, the second about the predictor head on top of it, and a bigger "
            "encoder is not obliged to improve both."
        )
        + reports.details("raw", repr(out))
    )
    flyte.report.replace(
        reports.final_html("Encoder size", rows, body, reports.SCALE_EXPLAINER), do_flush=True
    )
    return out


def _free() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@orch_env.task(report=True)
async def vjepa(clip: str = "bowling", repo: str = VITL) -> dict:
    """Entry point. CPU-only orchestrator so it cannot deadlock its own GPU children."""
    predicted = await inpaint(clip=clip, repo=repo)
    probed = await probe(repo=repo)
    scaled = await scale(clip=clip)
    result = {"inpaint": predicted, "probe": probed, "scale": scaled}
    log.info("result: %s", result)
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(vjepa))
