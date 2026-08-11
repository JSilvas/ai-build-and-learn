"""The V-JEPA 2 encoder, its predictor, and honest ways to score what the predictor did.

── The one thing to understand about the token layout ──────────────────────────
`VJEPA2PatchEmbeddings3D` is a Conv3d with stride (tubelet, patch, patch) followed by
`.flatten(2).transpose(1, 2)`, so token index is

    i = t * G * G + h * G + w        G = crop_size // patch_size = 16
                                     t indexes TUBELETS, each 2 real frames

and `VJEPA2RopeAttention.get_position_ids` decodes exactly that arithmetic back out of
whatever index you hand it. Masks are therefore plain flat token ids into that raster,
and "mask the future" is literally "mask the tail of the sequence". Everything in this
file relies on that ordering; `check_layout()` is the tripwire if a future transformers
release changes it, and every task calls it right after the first encode.

── Why every similarity here is CENTERED ───────────────────────────────────────
Measured on this box: the mean cosine between two RANDOM patch tokens of the same clip
is 0.28 at the last layer and 0.92 at layer 8. The token cloud sits in a narrow cone,
so raw cosine is dominated by a component every token shares and tells you almost
nothing. Subtract the per-clip mean token first and the same number drops to 0.003,
while adjacent patches sit at 0.33 and distant patches at 0.08.

That is not cosmetic. Uncentered, 1-NN clip retrieval scores 76%; centered, 82%. So
`center()` is applied before any similarity, and `anisotropy()` exists to put the
before/after numbers in the report rather than asking anyone to take this on faith.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def guard_memory(fraction: float = 0.85) -> str:
    """Cap this process against FREE unified memory, not total, then say what it did.

    On the GB10 there is one 119.7 GiB pool shared by the OS, the page cache and every
    other pod. `set_per_process_memory_fraction` takes a share of TOTAL, so asking for
    0.85 of total when something else already holds 40 GB is how you get an OOM
    followed by a bare SIGSEGV. Scaling the request by what is actually free is the fix
    (learned the hard way in topics/music-generation).

    V-JEPA 2 does not need any of this: ViT-g peaks at 2.1 GiB. It is here because the
    same helper failing loudly is much better than a mystery exit 139, and because the
    free/total split is worth printing into the report either way.
    """
    if not torch.cuda.is_available():
        return "no CUDA device"
    free, total = torch.cuda.mem_get_info()
    share = max(0.10, min(fraction, fraction * free / total))
    torch.cuda.set_per_process_memory_fraction(share)
    return (
        f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB unified, "
        f"capped at {share:.0%}"
    )


def load(repo: str):
    """Load a pretrained V-JEPA 2 encoder+predictor and its video processor."""
    from transformers import AutoVideoProcessor, VJEPA2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    t0 = time.time()
    processor = AutoVideoProcessor.from_pretrained(repo)
    model = VJEPA2Model.from_pretrained(repo, dtype=dtype).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    log.info("loaded %s (%.0fM params) in %.0fs", repo, params / 1e6, time.time() - t0)
    return model, processor, params


def grid_of(model, num_frames: int) -> tuple[int, int]:
    """(tubelets, grid) for a clip of `num_frames` frames."""
    cfg = model.config
    patch = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
    return num_frames // cfg.tubelet_size, cfg.crop_size // patch


def check_layout(seq: torch.Tensor, tubelets: int, grid: int) -> None:
    """Fail loudly if the token count is not tubelets x grid x grid.

    Every mask in this file is a flat index into that raster. If transformers ever
    changes the patch-embedding flatten order or adds a prefix token, the masks would
    still be valid indices and would still produce plausible-looking numbers for the
    wrong tokens. This is the one cheap check that turns that into a crash.
    """
    expected = tubelets * grid * grid
    if seq.shape[0] != expected:
        raise RuntimeError(
            f"token layout changed: encoder returned {seq.shape[0]} tokens, expected "
            f"{expected} ({tubelets} tubelets x {grid}x{grid}). Masks in jepa.py assume "
            f"index = t*G*G + h*G + w and are no longer valid."
        )


@torch.inference_mode()
def encode(model, pixel_values) -> torch.Tensor:
    """Clip -> [N, D] patch tokens. No predictor, no head."""
    device = next(model.parameters()).device
    seq = model.get_vision_features(pixel_values.to(device))[0].float()
    return seq


@torch.inference_mode()
def predict(model, pixel_values, context_ids: torch.Tensor, target_ids: torch.Tensor):
    """Run the predictor: context tokens -> the masked target tokens.

    Returns (predicted, true) as [n_target, D]. `true` comes back from the model as
    `target_hidden_state`, which is the encoder's own output at those positions: the
    thing the predictor was trained to match.
    """
    device = next(model.parameters()).device
    out = model(
        pixel_values_videos=pixel_values.to(device),
        context_mask=[context_ids.to(device).unsqueeze(0)],
        target_mask=[target_ids.to(device).unsqueeze(0)],
    )
    po = out.predictor_output
    return po.last_hidden_state[0].float(), po.target_hidden_state[0].float()


# ── Masks ───────────────────────────────────────────────────────────────────────
#
# The two that matter, and the whole argument of the `inpaint` task:
#
#   tube()    A spatial block removed across the ENTIRE temporal extent. This is the
#             shape V-JEPA 2 was actually pretrained on (multiblock masking), so the
#             predictor is in-distribution and you are measuring what it learned.
#
#   future()  Everything after some timestep removed. This is what you would reach for
#             if you assumed "world model" meant "forecaster". The predictor has never
#             seen it: pretraining masks always span all time, so no token was ever
#             predicted from a strictly earlier one.
#
# Running both against the same chance floor is the measurement. See the README.


def tube(tubelets: int, grid: int, blocks: int = 2, size: int = 5, seed: int = 0):
    """In-distribution: `blocks` spatial squares removed across all time."""
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(tubelets, grid, grid, dtype=torch.bool)
    for _ in range(blocks):
        h = int(torch.randint(0, max(grid - size, 0) + 1, (1,), generator=g))
        w = int(torch.randint(0, max(grid - size, 0) + 1, (1,), generator=g))
        m[:, h : h + size, w : w + size] = True
    return m


def future(tubelets: int, grid: int, context: float = 0.5):
    """Out-of-distribution: every token after the first `context` fraction of time."""
    m = torch.zeros(tubelets, grid, grid, dtype=torch.bool)
    m[max(1, int(round(tubelets * context))) :] = True
    return m


def ids_of(mask3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[T, G, G] bool -> (context_ids, target_ids) as flat token indices."""
    flat = mask3d.reshape(-1)
    return torch.nonzero(~flat).squeeze(1), torch.nonzero(flat).squeeze(1)


# ── Scoring ─────────────────────────────────────────────────────────────────────


def center(x: torch.Tensor) -> torch.Tensor:
    """Remove the shared component, then L2-normalise. See the module docstring."""
    return F.normalize(x - x.mean(0, keepdim=True), dim=-1)


def anisotropy(seq: torch.Tensor, samples: int = 4000, seed: int = 0) -> dict:
    """Mean cosine between random token pairs, raw and centered."""
    g = torch.Generator(seq.device).manual_seed(seed)
    i = torch.randint(0, seq.shape[0], (samples,), device=seq.device, generator=g)
    j = torch.randint(0, seq.shape[0], (samples,), device=seq.device, generator=g)
    raw = F.normalize(seq, dim=-1)
    cen = center(seq)
    return {
        "raw": float((raw[i] * raw[j]).sum(-1).mean()),
        "centered": float((cen[i] * cen[j]).sum(-1).mean()),
    }


def _coords(ids: torch.Tensor, grid: int) -> torch.Tensor:
    ppf = grid * grid
    return torch.stack([ids // ppf, (ids % ppf) // grid, ids % grid], -1).float()


def score(pred: torch.Tensor, true: torch.Tensor, target_ids: torch.Tensor, grid: int) -> dict:
    """How good is a prediction of the masked tokens? Three numbers, and why each.

    cos       Mean cosine to the true token, on centered features. Easy to read, but on
              its own uninterpretable: you need the chance floor next to it, which is
              why every caller also scores a shuffled version of the same prediction.

    top1      Fraction of predictions whose nearest neighbour, AMONG THE MASKED TOKENS
              ONLY, is its own ground truth. Restricting the candidate set to the
              targets is what makes this honest: if visible tokens were candidates, a
              predictor could score well by returning a copy of the nearest visible
              patch, which is not prediction.

    dt/dh/dw  When it retrieves the wrong token, HOW wrong: the median gap in tubelets
              and in patches between the token it retrieved and the true one. This is
              the number that separates the two masks. Under the mask V-JEPA 2 was
              trained on the median dt is 0 (right moment, roughly right place); asked
              to extrapolate forward in time it is 3-4 tubelets, i.e. it returns
              something from the wrong moment entirely.
    """
    q = center(pred)
    cand = center(true)
    sims = q @ cand.T
    nn_local = sims.argmax(1)
    gold = torch.arange(len(target_ids), device=pred.device)

    retrieved = target_ids.to(pred.device)[nn_local]
    d = (_coords(retrieved, grid) - _coords(target_ids.to(pred.device), grid)).abs()
    med = d.median(0).values
    return {
        "cos": float(F.cosine_similarity(q, cand, dim=-1).mean()),
        "top1": float((nn_local == gold).float().mean()),
        "dt": float(med[0]),
        "dh": float(med[1]),
        "dw": float(med[2]),
    }


def localization(scored: dict, floor: dict) -> float:
    """How much of the chance temporal error the prediction removed. 1.0 perfect, 0 chance.

    `dt` on its own is not comparable between masks, because the two masks span
    different amounts of time: a tube mask's targets cover all 32 tubelets, a
    half-clip future mask's cover 16, so guessing at random scores a median dt of ~10
    under one and ~5 under the other. Dividing by each mask's OWN shuffled floor is
    what makes "did it find the right moment?" a single comparable number.

    Note this only normalises the chance level, not the difficulty. It is still the
    conservative direction for the argument in `inpaint`: the tube mask is the one
    with more room to be wrong, and it is the one that scores 1.0.
    """
    chance = max(floor.get("dt", 0.0), 1e-6)
    return float(max(0.0, 1.0 - scored["dt"] / chance))


def shuffled_floor(pred, true, target_ids, grid, seed: int = 0) -> dict:
    """The same score for a permutation of the same predictions: the chance floor.

    Better than comparing against random vectors, because it holds the marginal
    distribution of the predictions fixed and destroys only the pairing. Whatever a
    prediction scores, it has to beat this to have predicted anything at all.
    """
    perm = torch.randperm(pred.shape[0], device=pred.device,
                          generator=torch.Generator(pred.device).manual_seed(seed))
    return score(pred[perm], true, target_ids, grid)


def context_floor(seq, context_ids, target_ids, true, grid) -> dict:
    """"Answer with the average of what you can see." The no-model baseline."""
    mean = seq[context_ids.to(seq.device)].mean(0, keepdim=True)
    return score(mean.expand(len(target_ids), -1), true, target_ids, grid)


def per_patch_cos(pred, true, target_ids, tubelets, grid) -> np.ndarray:
    """[T, G, G] map of prediction quality, NaN where the token was visible."""
    cos = F.cosine_similarity(center(pred), center(true), dim=-1)
    field = torch.full((tubelets * grid * grid,), float("nan"), device=cos.device)
    field[target_ids.to(cos.device)] = cos
    return field.reshape(tubelets, grid, grid).cpu().numpy()


def horizon(pred, true, mask3d, target_ids, tubelets, grid) -> list[tuple[int, float]]:
    """Mean prediction quality per tubelet, for the temporal-extrapolation mask.

    x is "tubelets ahead of the last visible frame", so the curve answers the question
    the word "world model" invites: does it degrade with horizon, or was it never
    tracking time in the first place?
    """
    field = per_patch_cos(pred, true, target_ids, tubelets, grid)
    masked = mask3d.numpy()
    first = int(masked.any(axis=(1, 2)).argmax())
    out = []
    for t in range(tubelets):
        if not masked[t].any():
            continue
        vals = field[t][masked[t]]
        out.append((t - first + 1, float(np.nanmean(vals))))
    return out
