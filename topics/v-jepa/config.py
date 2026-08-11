"""Shared Flyte config for the V-JEPA 2 demo.

Runs to the `world-models` Flyte project, alongside topics/cosmos and topics/dreamerv3,
because all three are the same question asked three ways. Cosmos predicts the future in
PIXELS. DreamerV3 learns a latent world model of one small environment from its own
experience. V-JEPA 2 is the third answer: predict in REPRESENTATION space, learned
self-supervised from internet video, with no decoder anywhere in the model.

That last detail drives the whole demo. There is no pixel head to render, so "show me
what it predicted" is not a screenshot you can take. Everything visual here is either
the model's actual input (which we can show exactly) or a per-patch measurement painted
back onto that input.

DGX-Spark-pinned (GB10 Blackwell, arm64, cu130): aarch64 platform + the devbox-local
registry. Drop the pins for a generic Flyte 2 cluster.

── Why this image is boring, and that is the point ─────────────────────────────
V-JEPA 2 landed in Hugging Face Transformers proper (`VJEPA2Model`), so there is no
vendor runner, no CUDA extension, and nothing to build from source. Compare with
topics/isaac-sim, which had to give up on `pip install isaacsim` entirely, and
topics/cosmos, which had to route around natten. This is a plain Debian base plus pip,
and it builds in minutes.
"""

from __future__ import annotations

import flyte

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# GB10 is Blackwell (sm_121) on the cu130 stack. PyTorch publishes matching aarch64
# wheels only on this index; the plain-PyPI aarch64 wheel is CPU-only and the sole
# symptom of getting it is torch.cuda.is_available() == False at encode time.
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

# ── Checkpoints ─────────────────────────────────────────────────────────────────
#
# All ungated (verified against the HF API), so no licence click-through. HF_TOKEN
# still goes into the pods because unauthenticated pulls are the ones that get
# rate-limited.
#
# Every one of these is a PRETRAINED checkpoint: encoder + predictor, no task head.
# That is what we want, because the demo is about what self-supervision alone buys.
VITL = "facebook/vjepa2-vitl-fpc64-256"  # ViT-L/16,  326M, hidden 1024, 24 layers
VITH = "facebook/vjepa2-vith-fpc64-256"  # ViT-H/16,  632M, hidden 1280, 32 layers
VITG = "facebook/vjepa2-vitg-fpc64-256"  # ViT-g/16, 1035M, hidden 1408, 40 layers

# Not used here, and worth knowing why. V-JEPA 2-AC, the action-conditioned post-train
# that actually plans robot manipulation, is NOT on the Hub under facebook/ (checked:
# only the six pretrain + SSv2/Diving48 classifier repos exist). Its weights ship via
# the facebookresearch/vjepa2 repo and it has no `transformers` class. So the honest
# scope of this demo is the pretrained predictor, and `inpaint` measures exactly where
# that predictor stops being a world model. See the README.
AC_NOTE = "facebookresearch/vjepa2 (not on the Hub, no transformers class)"

# 5 classes x 20 clips, ~10s each at 15fps, train/ and val/ splits of 10 each.
# Ungated, tiny, and already the dataset the transformers V-JEPA 2 docs use.
CLIPS_REPO = "nateraw/kinetics-mini"

HF_HOME = "/tmp/hf"
HF_SECRET = flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")


VJEPA_SPEC = (
    # VJEPA2Model / VJEPA2Predictor and AutoVideoProcessor's fast video path.
    "transformers>=5.11",
    "accelerate>=1.10",
    "safetensors",
    "numpy",
    "pillow",
    "huggingface_hub",
    # PyAV decodes the source clips and encodes the report mp4s. aarch64 wheels exist
    # for PyAV and do not reliably for imageio-ffmpeg, which is the conclusion the
    # video-generation, Cosmos and Isaac Sim demos all reached independently.
    "av",
    "matplotlib",
    "flyte==2.2.1",
    # 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
    "connectrpc==0.10.*",
)

image = (
    flyte.Image.from_debian_base(name="vjepa2", registry=REGISTRY, platform=PLATFORM)
    .with_apt_packages("git", "ffmpeg")
    # torch on its OWN layer and from the cu130 index, before anything else can
    # resolve a plain-PyPI torch over the top of it.
    .with_pip_packages("torch", "torchvision", index_url=TORCH_INDEX)
    .with_pip_packages(*VJEPA_SPEC)
)


# ── DGX Spark tuning ────────────────────────────────────────────────────────────
#
# Carried over from the video-generation and Cosmos demos. Less load-bearing here than
# there: ViT-g peaks at 2.1 GiB, so this box is nowhere near its ceiling. Kept anyway
# because expandable_segments still matters when "GPU memory" IS the same unified
# 119.7 GiB pool the OS and every other pod share.
#
# Anti-pattern, same as next door: do NOT torch.compile the encoder. Triton does not
# emit working SASS for sm_121a yet, so it fails or silently falls back.
_SPARK_ENV = {
    "CUDA_CACHE_MAXSIZE": "4294967296",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_MODULE_LOADING": "EAGER",
}

# hf_transfer is OFF. It is a Rust downloader that does its own DNS and ignores socket
# timeouts, so a black-holed route to the HF CDN hangs forever instead of erroring. The
# plain Python downloader plus HF_HUB_DOWNLOAD_TIMEOUT (which bounds a stalled *read*,
# not the total) fails in ~60s and resumes from the .incomplete file.
_ENV_VARS = {
    "HF_HOME": HF_HOME,
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "HF_HUB_DOWNLOAD_TIMEOUT": "60",
    # Matplotlib writes a font cache on first import and $HOME is not writable in the pod.
    "MPLCONFIGDIR": "/tmp/mpl",
}

_GPU_ENV_VARS = {**_ENV_VARS, **_SPARK_ENV}


# ── Environments ────────────────────────────────────────────────────────────────
#
# One GPU on this box, and the orchestrator is CPU-only ON PURPOSE: an orchestrator pod
# holds its resources for as long as its children run, so a GPU-holding orchestrator
# deadlocks its own GPU child on "Insufficient nvidia.com/gpu". Same trap as the
# cosmos, videogen, mujoco, dreamer and Isaac Sim demos.
#
# memory=32Gi / disk=60Gi is generous for what this actually does. The largest
# checkpoint is ~4.4 GB on disk and 2.1 GiB resident, and the whole kinetics-mini
# dataset is under 200 MB. The headroom is for the 100-clip feature matrix and for
# decoding, not for the model.
gpu_env = flyte.TaskEnvironment(
    name="vjepa",
    image=image,
    resources=flyte.Resources(cpu="8", memory="32Gi", gpu=1, disk="60Gi"),
    secrets=[HF_SECRET],
    env_vars=_GPU_ENV_VARS,
)

orch_env = flyte.TaskEnvironment(
    name="vjepa-orch",
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="20Gi"),
    secrets=[HF_SECRET],
    env_vars=_ENV_VARS,
    depends_on=[gpu_env],
)
