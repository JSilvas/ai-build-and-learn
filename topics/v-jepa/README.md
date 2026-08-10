# World Models with V-JEPA 2: prediction in representation space

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This one looks at world models from a different angle. Where [Cosmos](../cosmos) generates the future in pixels, [V-JEPA 2](https://github.com/facebookresearch/vjepa2) predicts in representation space: it learns by predicting the abstract embeddings of masked video rather than reconstructing every pixel. That is the core thesis of the JEPA line of work, Yann LeCun's argument that predicting representations is more efficient, and better for reasoning and planning, than generating pixels.

It runs on Flyte, on one DGX Spark, and the reports have video in them.

## The thing that makes this demo different

V-JEPA 2 **has no decoder**. The predictor emits 1024-dimensional vectors and there is no head in any released checkpoint that turns one back into a pixel. So "show me what it predicted" is not a screenshot anyone can take, and any image claiming to be one is a projection of a vector, not a prediction.

That constraint shapes everything here. Every frame in these reports is one of two things:

- the model's **literal input**, with the masked patches blacked out
- a **per-patch number we computed**, painted onto those same pixels

Nothing is a vector dressed up as an image. The corollary is that the demo has to carry its weight in measurements rather than in pretty pictures, so every score is reported next to its own chance floor.

## What it does

```
vjepa (orchestrator, CPU)
  ├── inpaint (GPU)   hide part of a clip, predict it in latent space, score it twice
  ├── probe   (GPU)   freeze the encoder, train one linear layer, 5-way action recognition
  └── scale   (GPU)   ViT-L vs ViT-g, same clips, same masks, same probe
```

```bash
./setup.sh                                  # local venv (the pods build their own image)
.venv/bin/python smoke_test.py --full       # ~4 min, checks the claims before spending a pod

.venv/bin/flyte run pipeline.py inpaint     # ~30s after the image is built
.venv/bin/flyte run pipeline.py inpaint --clip archery --context 0.75
.venv/bin/flyte run pipeline.py probe       # ~3 min
.venv/bin/flyte run pipeline.py scale       # ~8 min, two encoders
.venv/bin/flyte run pipeline.py vjepa       # all three
```

Runs to the `world-models` Flyte project, alongside [topics/cosmos](../cosmos) and [topics/dreamerv3](../dreamerv3). The three are the same question asked three ways: Cosmos predicts the future in pixels, Dreamer learns a latent world model of one environment from its own experience, V-JEPA 2 predicts representations learned self-supervised from internet video.

Data is [`nateraw/kinetics-mini`](https://huggingface.co/datasets/nateraw/kinetics-mini): 100 clips, 5 classes, ungated, under 200 MB, and already the dataset the transformers V-JEPA 2 docs use.

## The finding: this checkpoint inpaints, it does not forecast

This is the part worth the stream time, and it was not what I expected going in.

The obvious experiment for a "world model" is: show it the first half of a clip, ask it to predict the second half, measure. Do that and the numbers are bad. The interesting question is *why*, and the answer is that **it is the wrong question to ask this checkpoint**.

V-JEPA 2's pretraining masks are **tubes**: a spatial block removed across the *entire* temporal extent of the clip. Every masked token always had visible tokens from its own timestep to attend to. No token was ever predicted from a strictly earlier one. So temporal extrapolation is not something the predictor does badly, it is something it was never asked to do.

`inpaint` runs both masks against the same clip, same predictor, same scoring, with **the same fraction of tokens hidden** (two 8x8 tube blocks is ~48%; a half-clip future mask is 50%) so that the only difference is the *shape* of the hole.

Scoring is deliberately paranoid. Cosine similarity between two ViT tokens is a number that always looks encouraging, so the headline metric is a retrieval one: for each predicted token, find its nearest neighbour **among the masked tokens only** (restricting candidates to the targets is what stops a model scoring well by copying a visible neighbour), then measure *how far off in the raster* that neighbour is. `median dt` is the gap in tubelets. Each mask is also scored with its predictions shuffled, which gives the chance floor for that exact mask.

Measured on the Spark, `facebook/vjepa2-vitl-fpc64-256`, 64 frames of `val/bowling/--dVV4_CSvw`, run `rscs9t9hhlft7hsk5ddz`:

| mask | hidden | cosine | top-1 | median dt | chance dt | time localised |
|---|---|---|---|---|---|---|
| **tube** (pretraining mask) | 48% | 0.317 | 3.9% | **0.0** | 10.0 | **1.00** |
| **future** (never trained on) | 50% | 0.276 | 1.2% | **3.0** | 5.0 | **0.40** |

Under the mask it was trained on, the predictor's retrieved token is at **exactly the right moment**, median 0 tubelets off where chance is 10. Asked to extrapolate forward instead, it lands 3 tubelets away against a chance level of 5. It is returning a plausible representation of roughly the right scene at the *wrong time*.

Note the cosine column barely moves (0.317 vs 0.276) while the localisation collapses. That is the whole reason the report leads with `dt` and not with cosine: **cosine is the metric that would have let this pass unnoticed.**

The same pattern holds across every clip and both encoder sizes tested. It is also, reassuringly, exactly what the architecture predicts, and it is the gap that **V-JEPA 2-AC**, the action-conditioned post-train, exists to close.

> **Scope note.** V-JEPA 2-AC is *not* in this demo, and not because of time. It is not on the Hub under `facebook/` (checked against the API: only the six pretrain checkpoints plus the SSv2 and Diving48 classifiers exist), its weights ship via the `facebookresearch/vjepa2` repo, and it has no `transformers` class. The honest scope here is the pretrained predictor, and `inpaint` measures precisely where that predictor stops being a world model.

## The other finding: the features are strong, and you have to centre them

`probe` freezes the encoder, mean-pools each clip into one vector, and trains a single 1024x5 linear layer. Nothing else is trained anywhere.

Run `r8r29vr8db5gc25cgm6n`, 100 clips, the dataset's own 50/50 train/val split:

| | accuracy |
|---|---|
| **V-JEPA 2 features + linear probe** | **78%** |
| raw pixels + the same linear probe | 40% |
| V-JEPA 2 1-NN retrieval, no training at all | 82% |
| chance | 20% |

The pixel baseline is the one that matters. "78% on 5-way action recognition" is unreadable on its own; 78% where a downsampled space-time thumbnail of the same clips gets 40% is a claim about the *representation*.

The second result is a trap worth knowing about. **V-JEPA 2's token cloud sits in a narrow cone.** The mean cosine between two random patch tokens of the same clip is 0.28 at the last layer and 0.92 at layer 8: raw cosine is dominated by a component every token shares and tells you almost nothing. Subtract the per-clip mean token first and random pairs drop to ~0.00, while adjacent patches sit at 0.33 and distant patches at 0.08.

That is not cosmetic, and the demo puts a number on it rather than asserting it: the *same* retrieval scores **82% centred and 76% raw**. Every similarity in this repo is computed on centred features for that reason.

## What did not work, and is therefore not in the demo

Worth writing down, because the first version of this demo had all of it and it was all quietly wrong.

- **PCA-of-patch-embeddings feature videos.** The DINOv2 trick of projecting patch tokens to 3 components and rendering them as RGB. On V-JEPA 2 it is noise: the top 3 components explain ~20% of variance and the result is a rainbow with no visible object structure. Tried layers 8/14/20/24 and four normalisations; none were legible.
- **Query-patch correspondence tracking.** Cosine from one patch to all others, over time. Diffuse and uninformative once the anisotropy above is accounted for.
- **A "what is moving" salience map.** Hot on the *blank white wall*, because in flat regions the centred embedding has small norm and its direction is noise. An artifact of normalising, not a signal.
- **Overlays composited on the source clip.** `AutoVideoProcessor` resizes the shortest edge to 292 and centre-crops to 256, so a 480x270 clip loses most of its width. Painting a 16x16 patch grid onto a naive resize of the *original* is misaligned by tens of pixels, and the heatmaps looked plausible the whole time. `clips.shown_pixels()` inverts the normalisation and hands back exactly the tensor the patch embedding consumed; every overlay composites on that.

The general lesson: V-JEPA 2 is not DINOv2. It has no register tokens, its spatial grid is coarse (16x16), and its patch tokens do not give clean semantic maps out of the box. A demo whose centrepiece is a picture that does not hold up is worse than one that leads with a measurement.

## How it is put together

| file | what is in it |
|---|---|
| `config.py` | Flyte image and task environments, checkpoint ids, Spark tuning |
| `clips.py` | fetch/decode Kinetics clips, and recover the exact pixels the model saw |
| `jepa.py` | model load, the token raster, the two masks, and the scoring with its floors |
| `probing.py` | linear probe, retrieval, dataset encoding |
| `viz.py` | mp4 encode/probe, masked and heatmap videos, matplotlib charts |
| `reports.py` | the report HTML, shared palette with the other world-model demos |
| `pipeline.py` | the four Flyte tasks |
| `smoke_test.py` | runs the real code paths on the host, and asserts the finding above |

A few things that are load-bearing and non-obvious:

- **Token index is `t*G*G + h*G + w`**, temporal-major, where `G = 16` and `t` indexes *tubelets* of 2 frames. `VJEPA2RopeAttention.get_position_ids` decodes exactly that arithmetic back out of whatever index you hand it, so masks are plain flat indices. `jepa.check_layout()` is the tripwire if a transformers release ever changes it: bad masks would still be valid indices and would still produce plausible numbers for the wrong tokens.
- **Mean pooling, not the attentive pooler.** `VJEPA2AttentivePooler` exists in the architecture but its weights are only trained in the *classifier* checkpoints. In a pretrained-only checkpoint it is randomly initialised, so using it would be measuring noise.
- **Probe features are standardised on train statistics only.** A small effect at n=50, and exactly the kind of leak that makes a benchmark number quietly wrong.
- **The orchestrator is CPU-only.** One GPU on this box, and an orchestrator pod holds its resources while its children run, so a GPU-holding orchestrator deadlocks its own GPU child on "Insufficient nvidia.com/gpu". Same trap as the cosmos, videogen, mujoco and Isaac Sim demos.

## Hardware notes (DGX Spark, GB10, arm64)

This is by far the cheapest demo in the repo to run, which is itself the point: no generation means no diffusion loop.

| | |
|---|---|
| ViT-L encode, 64 frames | **0.58 s**, 1.2 GiB peak |
| ViT-g encode, 32 frames | 0.77 s, 2.1 GiB peak |
| `inpaint` end to end in a pod | **32 s** |
| `probe`, 100 clips | ~3 min, mostly download and decode |
| Largest checkpoint on disk | 4.4 GB (ViT-g, 1035M params) |

`torch` is installed on its own layer from the cu130 index; the plain-PyPI aarch64 wheel is CPU-only and the only symptom is `torch.cuda.is_available() == False` at encode time. PyAV rather than imageio-ffmpeg, because aarch64 wheels reliably exist for one and not the other. Do not `torch.compile` the encoder: Triton does not emit working SASS for sm_121a yet.

## Some things to look up to get started

**Model**
- V-JEPA 2 (Meta): https://github.com/facebookresearch/vjepa2
- Transformers docs and checkpoints: https://huggingface.co/docs/transformers/model_doc/vjepa2
- Paper, "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning": https://huggingface.co/papers/2506.09985

**Background**
- Meta AI overview: https://ai.meta.com/research/vjepa/
- The anisotropy problem in contextual embeddings, which is what `centre()` is fixing: https://arxiv.org/abs/1909.00512
