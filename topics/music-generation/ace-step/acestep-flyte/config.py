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
# low-cpu-mem load path. No librosa: matplotlib's own specgram is enough for the
# report and keeps the image lighter.
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
gen_image = _base("acestep-gen").with_pip_packages(
    DIFFUSERS_MIN, "transformers>=4.51", "accelerate", "safetensors"
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

orch_env = flyte.TaskEnvironment(
    name="acestep-orch",
    image=fetch_image,
    resources=flyte.Resources(cpu="2", memory="8Gi", disk="40Gi"),
    secrets=[HF_SECRET],
    env_vars=_ENV_VARS,
    depends_on=[cpu_env, gpu_env],
)


# ── The studio app (not built yet) ───────────────────────────────────────────────
#
# Kept for parity with the image/video/TTS studios: when the Gradio launcher lands it
# will be a thin CPU app that submits runs and links the report, holding no GPU and
# loading no model. AppEnvironment does NOT honor flyte.Resources(gpu=1) on this SDK
# (it serializes to a bare `gpu` key k8s drops and the pod silently schedules CPU-
# only); a PodTemplate that sets nvidia.com/gpu directly is the fix, so it is here
# ready for the day the app wants to hold the model resident instead.
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
