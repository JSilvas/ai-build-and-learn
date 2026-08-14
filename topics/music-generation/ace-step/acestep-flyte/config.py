"""Shared Flyte config for the ACE-Step 1.5 music-generation demo.

Same shape as the demos next door (topics/text-to-speech, topics/video-generation),
so if you've read one of those you already know the layout:

  - compare_pipeline.py : GPU *tasks* that render prompts through ACE-Step and emit a
    single Flyte report whose tracks play inline.
  - music_core.py       : the Flyte-free engine room (load, generate, render).

DGX-Spark-pinned (GB10 Blackwell, arm64, cu130 stack): aarch64 platform + the
devbox-local registry. Drop the pins for a generic Flyte 2 cluster.

── ONE image, unlike the TTS demo ───────────────────────────────────────────────
The text-to-speech demo needed a separate image per model because every open TTS
package shipped its own mutually hostile pins. Music generation, at least for
ACE-Step, is back to the video demo's happy case: ACE-Step 1.5 was merged into
`diffusers` (PR #13095, released in 0.39.0) as `AceStepPipeline`, so every checkpoint
here loads through one `from_pretrained` call on one stack. One image, one GPU env,
and adding the next model is a registry entry rather than a new Dockerfile.

That is worth saying out loud on the stream: it is the difference between "the
ecosystem converged on a runtime" and "every model is its own science project".

The upstream `acestep` package (github.com/ace-step/ACE-Step-1.5) is the other way in
and carries features diffusers has not absorbed yet: the 5 Hz LM stage, Flow-Edit,
LoRA training, the Gradio studio. We deliberately take the diffusers path because it
keeps this demo one pip install wide and shares a mental model with the image and
video demos. See the README for what that trade costs.
"""

from __future__ import annotations

import flyte
from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# GB10 is Blackwell on the cu130 stack; PyTorch publishes matching aarch64 wheels.
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

# AceStepPipeline landed in diffusers 0.39.0. Below that the import does not exist,
# and the failure mode is a bare ImportError deep in a GPU pod, so floor it here.
DIFFUSERS_MIN = "diffusers>=0.39.0"

APP_NAME = "acestep-studio"
APP_PORT = 7865          # 7862 image-gen, 7863 videogen, 7864 tts, 7870 magenta

HF_HOME = "/tmp/hf"
HF_SECRET = flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")


# ── The one image ────────────────────────────────────────────────────────────────
#
# soundfile writes the wavs and encodes the OGG we embed in the report; matplotlib
# renders the waveform + spectrogram PNG that is the *visual* comparison surface (you
# can see a drop, a chorus lift, or a wall of clipping at a glance). transformers is
# for the Qwen3 text encoder the pipeline loads as a component; accelerate for the
# low-cpu-mem load path. The report itself needs no librosa (matplotlib's own specgram
# is enough); AudioLDM2 does, so it is on the generation image only, not here.
_COMMON = (
    "soundfile",
    "numpy",
    "matplotlib",
    "huggingface_hub",
    "kubernetes",     # config.py imports kubernetes.client at module top
)


def _base(name: str) -> flyte.Image:
    return (
        flyte.Image.from_debian_base(name=name, registry=REGISTRY, platform=PLATFORM)
        .with_apt_packages("git", "ffmpeg")
        .with_pip_packages("torch", "torchaudio", index_url=TORCH_INDEX)
        .with_pip_packages(*_COMMON)
    )


# The generation image. transformers floored at 4.51 because the pipeline's text
# encoder is a `Qwen3Model` and older transformers has no Qwen3 architecture; the
# failure is a KeyError inside from_pretrained, not an import error, so it survives
# the build and dies in the pod.
# librosa and sentencepiece are for AudioLDM2: diffusers guards librosa behind
# `is_librosa_available()`, and sentencepiece backs the T5 tokenizer in its text stack.
# Both are small next to torch and both are far cheaper to add here than to discover
# missing inside a GPU pod several minutes into a run.
# torchsde is for Stable Audio Open: its default scheduler is
# CosineDPMSolverMultistepScheduler, which diffusers guards behind a torchsde backend.
# Without it `from_pretrained` fails at load with a dummy-object ImportError, AFTER the
# 15.7GB download has already succeeded, which is a slow way to learn about a 2MB
# dependency. Optional scheduler backends are easy to miss precisely because the model
# card never mentions them.
#
# PINNED to 0.2.5, not floating. On torchsde 0.2.6 the generation dies with
# `RecursionError: maximum recursion depth exceeded` inside diffusers'
# `BatchedBrownianTree.__call__`, which drives the scheduler's stochastic noise. There
# is no way around it by configuration: CosineDPMSolverMultistepScheduler constructs a
# BrownianTreeNoiseSampler unconditionally (scheduling_cosine_dpmsolver_multistep.py
# line ~650), so there is no deterministic algorithm_type to fall back to.
gen_image = _base("acestep-gen").with_pip_packages(
    DIFFUSERS_MIN, "transformers>=4.51", "accelerate", "safetensors",
    "librosa", "sentencepiece", "torchsde==0.2.5",
    # The numba/llvmlite floor is load-bearing, and it exists because of the torchsde
    # pin above. Pinning torchsde makes uv backtrack librosa to an old release whose
    # numba is 0.53.1, which refuses to install on Python 3.12 at all:
    #   RuntimeError: Cannot install on Python version 3.12.13; only >=3.6,<3.10
    # Flooring both keeps the resolver on wheels that exist for aarch64/3.12. The TTS
    # demo next door hit the identical trap through Parler's audio deps.
    "numba>=0.60", "llvmlite>=0.43",
)

# The download task is model-agnostic: it only needs huggingface_hub, so it rides a
# torch-free image that builds in a fraction of the time.
fetch_image = (
    flyte.Image.from_debian_base(name="acestep-fetch", registry=REGISTRY, platform=PLATFORM)
    .with_pip_packages("huggingface_hub", "kubernetes", "numpy",
                       "soundfile", "matplotlib")
)


# ── DGX Spark tuning ─────────────────────────────────────────────────────────────
#
# ACE-Step XL is ~11GB of bf16 weights, small next to the GB10's 119.7GiB unified
# pool, so the video demo's fit machinery is not needed. We keep the JIT cache sizing
# (first-call kernel compiles are real) and the expandable allocator, which matters
# more here than the number suggests: the VAE decode of a 4-minute track allocates one
# very large contiguous activation, and a fragmented pool is exactly how that fails.
_SPARK_ENV = {
    "CUDA_CACHE_MAXSIZE": "2147483648",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_MODULE_LOADING": "EAGER",
}

# NOTE: no HF_HUB_ENABLE_HF_TRANSFER here, and that is deliberate.
#
# The image and video demos next door set it, and the TTS demo goes further: it turns
# hf_transfer OFF in the fetch task so that HF_HUB_DOWNLOAD_TIMEOUT can bound a stalled
# read. That lore is now DEAD. Current huggingface_hub routes downloads through Xet
# (content-addressed storage) and ignores hf_transfer entirely; setting the variable
# only earns a FutureWarning in every pod's logs:
#
#   The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated as
#   'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead.
#
# HF_HUB_DOWNLOAD_TIMEOUT does not bound the Xet path either, so the stall protection
# it was bought for no longer exists. What actually happens on a bad fetch is a hard
# error, observed on this pipeline's first run:
#
#   CAS Client Error: Request middleware error: error sending request for url (...)
#
# That is retryable and the retry resumes, so the protection now lives in
# fetch_weights' `retries=3` rather than in an environment variable.
_ENV_VARS = {"HF_HOME": HF_HOME}
_FETCH_ENV_VARS = dict(_ENV_VARS)

_GPU_ENV_VARS = {**_ENV_VARS, **_SPARK_ENV}


# ── Environments ─────────────────────────────────────────────────────────────────
#
#   cpu_env  (acestep-fetch) : fetch_weights, an HF download, no GPU.
#   gpu_env  (acestep-gen)   : the generation work. One env, because one image.
#   orch_env (acestep-orch)  : the orchestrator. CPU-only ON PURPOSE, because an
#                              orchestrator pod stays alive holding its resources
#                              while awaiting children, so if it held the box's one
#                              GPU its own GPU children would deadlock forever.

cpu_env = flyte.TaskEnvironment(
    name="acestep-fetch",
    image=fetch_image,
    resources=flyte.Resources(cpu="4", memory="8Gi", disk="60Gi"),
    secrets=[HF_SECRET],
    env_vars=_FETCH_ENV_VARS,
)

# memory="96Gi", up from 48Gi. On GB10 the GPU pool IS host memory, so every CUDA
# allocation is charged to this pod's cgroup: the ceiling has to be sized for the
# RENDER, not for the 11GB of weights. music_core.prepare_gpu caps torch against this
# same cgroup limit so the two can never disagree.
#
# Raised while chasing a run of SIGSEGVs on long tracks that turned out to be a
# libsndfile bug in the report renderer, not a cgroup overrun (see
# music_core.encode_audio). So this is headroom rather than a proven fix; 48Gi may well
# be sufficient. Left at 96Gi because long renders on a 120GB box have no reason to be
# tight, but do not treat this number as load-bearing evidence of anything.
gpu_env = flyte.TaskEnvironment(
    name="acestep-gen",
    image=gen_image,
    resources=flyte.Resources(cpu="8", memory="96Gi", gpu=1, disk="80Gi"),
    secrets=[HF_SECRET],
    env_vars=_GPU_ENV_VARS,
)

# ── DiffRhythm 2: the first model here that needs its OWN image ────────────────────
#
# Everything else in this demo shares one image because it loads through diffusers or
# transformers. DiffRhythm does neither: there is no PyPI package and no packaging
# metadata in the repo at all, so it is a git clone plus PYTHONPATH, and its
# requirements.txt hard-pins `torchaudio==2.6.0` / `transformers==4.49.0` / `numpy` in
# ways that would drag the whole registry backwards. Same call the TTS demo makes for
# Chatterbox: isolate it.
#
# Three things this build does that are not obvious:
#
#  1. NO requirements.txt. Installing it would pull torchaudio 2.6, which has no cu130
#     aarch64 wheel, and would fight the Spark's torch. We install the inference deps
#     by hand on top of the cu130 torch already in the base.
#  2. The trainer import is DELETED. `model/__init__.py` does
#     `from model.trainer import Trainer`, and trainer.py imports wandb, accelerate and
#     `dataset.dataset` (pylance, pyarrow, pandas). Inference needs only CFM and DiT, so
#     one sed removes an entire training dependency tree.
#  3. espeak-ng is apt-level and load-bearing: phonemizer shells out to it for the
#     lyric g2p, and a missing espeak-ng is a RUNTIME failure, not an import one.
#
# Deps below are exactly what `infer/infer.py` + `infer_utils.py` + `model/` reach for:
# muq (the MuQ-MuLan style encoder), x-transformers / torchdiffeq / einops (the CFM+DiT
# stack), mutagen + librosa (audio IO), and the g2p chain for lyrics.
DIFFRHYTHM_REPO = "https://github.com/ASLP-lab/DiffRhythm2"
DIFFRHYTHM_DIR = "/opt/diffrhythm2"

diffrhythm_image = (
    flyte.Image.from_debian_base(
        name="acestep-diffrhythm", registry=REGISTRY, platform=PLATFORM
    )
    .with_apt_packages("git", "ffmpeg", "espeak-ng", "cmake", "build-essential")
    .with_pip_packages("torch", "torchaudio", index_url=TORCH_INDEX)
    .with_pip_packages(*_COMMON)
    .with_pip_packages(
        # What inference.py + diffrhythm2/ + bigvgan/ actually reach for. NOT their
        # requirements.txt, which pins torch==2.7 (no cu130 aarch64 wheel) and
        # transformers==4.47.1.
        # transformers PINNED to their requirement. v2 imports `StaticCache` from
        # `transformers.models.llama.modeling_llama`, which moved to `cache_utils` in
        # transformers 5, so a modern version fails at import:
        #   ImportError: cannot import name 'StaticCache' ... Did you mean 'DynamicCache'?
        # (I had assumed v2 avoided transformers' Llama entirely because its own DiT and
        # BigVGAN do the generation. It does not: the cache class comes from there.)
        # Pinning is free here because this image is isolated from the other seven models.
        "einops", "torchdiffeq", "transformers==4.47.1", "muq", "pedalboard", "tqdm",
        "librosa", "mutagen", "accelerate", "onnxruntime",
        # the lyric g2p chain
        "phonemizer", "pypinyin", "jieba", "cn2an", "pykakasi",
        "unidecode", "inflect", "py3langid",
        # pyopenjtalk is NOT optional despite English-only lyrics: their CNENTokenizer
        # imports the whole g2p chain eagerly, and `g2p/g2p/japanese.py` pulls it in at
        # module level. Skipped it initially as a Japanese-only dep and a risky native
        # build on aarch64; it is neither optional nor avoidable. cmake and
        # build-essential are already in this image for exactly this.
        "pyopenjtalk",
        "numba>=0.60", "llvmlite>=0.43",
    )
    .with_commands([
        f"git clone --depth 1 {DIFFRHYTHM_REPO} {DIFFRHYTHM_DIR}",
        # Prove the model modules import at BUILD time rather than discovering it in a
        # GPU pod. v2 ships its own DiT and BigVGAN vocoder and never touches
        # transformers' Llama, which is precisely why it avoids the rotary-embedding
        # dimension bug that blocks v1 on any modern transformers.
        f"cd {DIFFRHYTHM_DIR} && PYTHONPATH={DIFFRHYTHM_DIR} python -c \""
        f"from diffrhythm2.cfm import CFM; from diffrhythm2.backbones.dit import DiT; "
        f"from bigvgan.model import Generator; from muq import MuQMuLan; "
        f"print('diffrhythm2 imports OK')\"",
    ])
)


diffrhythm_env = flyte.TaskEnvironment(
    name="acestep-diffrhythm",
    image=diffrhythm_image,
    resources=flyte.Resources(cpu="8", memory="96Gi", gpu=1, disk="80Gi"),
    secrets=[HF_SECRET],
    env_vars={**_GPU_ENV_VARS, "PYTHONPATH": DIFFRHYTHM_DIR},
)


# ── MiniMax-Music3 ───────────────────────────────────────────────────────────────
#
# The SECOND model here with its own image, and for the same class of reason as
# DiffRhythm: it needs a `diffusers` the rest of the registry must not have.
#
# MiniMax-Music3 loads through `ModularPipeline`, and support for it landed in a
# diffusers PR that has not shipped in a release. So this image installs diffusers from
# that exact commit. Putting an unreleased diffusers into the shared `gen_image` would
# put five working models (ACE-Step x3, AudioLDM 2, Stable Audio) on an unreleased
# dependency to gain one, which is the wrong trade; isolating it costs a build and
# risks nothing.
#
# Pinned to the COMMIT, not to a branch. A branch would silently change under us and
# turn a reproducible image into a moving target, and the failure would look like the
# model breaking rather than the dependency moving.
MINIMAX_DIFFUSERS_COMMIT = "dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d"

minimax_image = (
    flyte.Image.from_debian_base(
        name="acestep-minimax", registry=REGISTRY, platform=PLATFORM
    )
    .with_apt_packages("git", "ffmpeg")
    .with_pip_packages("torch", "torchaudio", index_url=TORCH_INDEX)
    .with_pip_packages(*_COMMON)
    .with_pip_packages(
        f"diffusers @ git+https://github.com/huggingface/diffusers.git"
        f"@{MINIMAX_DIFFUSERS_COMMIT}",
        "transformers", "accelerate", "sentencepiece", "librosa",
    )
    # Check the import at BUILD time, not in a GPU pod eleven minutes into a 57GB
    # download. This is the lesson DiffRhythm taught twice: a build-time
    # `RUN python -c "import ..."` catches a broken dependency tree in seconds, where
    # the same failure at run time costs a pod, a queue slot and the download.
    .with_commands([
        'python -c "from diffusers import ModularPipeline; '
        'print(\'ModularPipeline import OK\')"',
    ])
)

minimax_env = flyte.TaskEnvironment(
    name="acestep-minimax",
    image=minimax_image,
    resources=flyte.Resources(cpu="8", memory="96Gi", gpu=1, disk="120Gi"),
    secrets=[HF_SECRET],
    # 120Gi of disk, not the usual 80: the repo is ~57GB and the pod needs room for the
    # downloaded Dir plus whatever it unpacks alongside it.
    env_vars=_GPU_ENV_VARS,
)


orch_env = flyte.TaskEnvironment(
    name="acestep-orch",
    image=fetch_image,
    resources=flyte.Resources(cpu="2", memory="8Gi", disk="40Gi"),
    secrets=[HF_SECRET],
    env_vars=_ENV_VARS,
    # diffrhythm_env is in here because DiffRhythm runs in its OWN image, and an
    # orchestrator can only dispatch to environments it declares. Without it the run
    # dies at submit with `MissingEnvironment: 'acestep-diffrhythm' not found in image
    # cache`, which is a clear error but only if you already know to expect it.
    depends_on=[cpu_env, gpu_env, diffrhythm_env, minimax_env],
)


# ── The studio app ───────────────────────────────────────────────────────────────
#
# A thin LAUNCHER, exactly like the image and video studios: it submits runs and links
# the report. No torch, no diffusers, no GPU, no model. That is not minimalism for its
# own sake, it is the only shape that works on a one-GPU box: an app pod stays alive
# for as long as the app is up, so an app that generated in-process would hold the
# Spark's only GPU forever and every pipeline task would sit Unschedulable behind it.
# Launching runs means the GPU is held only while a track is actually rendering.
#
# connectrpc pinned to 0.10.x: 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
# kubernetes because the app imports THIS module for the name/port, and the pod
# templates below need kubernetes.client at import time.
studio_app_image = (
    flyte.Image.from_debian_base(
        name="acestep-studio-image", registry=REGISTRY, platform=PLATFORM
    )
    .with_pip_packages("flyte==2.2.1", "connectrpc==0.10.*", "gradio==5.42.0",
                       "python-dotenv", "kubernetes")
)

# Kept for the day the studio wants a model resident instead of launching runs.
# AppEnvironment does NOT honor flyte.Resources(gpu=1) on this SDK (it serializes to a
# bare `gpu` key k8s drops and the pod silently schedules CPU-only); a PodTemplate that
# sets nvidia.com/gpu directly is the fix.
app_gpu_pod = flyte.PodTemplate(
    primary_container_name="app",
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="app",
                resources=V1ResourceRequirements(
                    requests={"cpu": "8", "memory": "48Gi", "ephemeral-storage": "80Gi"},
                    limits={"nvidia.com/gpu": "1"},
                ),
            )
        ]
    ),
)
