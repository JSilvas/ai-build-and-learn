"""The LLM half of the voice app's `vllm` deployment: a vLLM server on Flyte.

Deliberately a near-copy of `topics/gemma4/gemma4-dgx-devbox/vllm_server.py`, because
that pattern already works on this box: the frozen-dataclass arm64 fix, installing the
plugin into the base image's system Python, `flyte.prefetch.hf_model` + `RunOutput` so
the weights are fetched once instead of on every cold start.

Deploy:
    python voice_vllm.py                      # qwen3-4b by default
    VOICE_LLM=gemma12b python voice_vllm.py   # any key from MODELS below
    # then point the voice app at the URL it prints:
    LLM_BACKEND=vllm VLLM_URL=<url> python voice_app.py

── How to pick the model ────────────────────────────────────────────────────────
The Spark is memory-bandwidth-bound, so tokens/sec tracks bytes-read-per-token. A voice
assistant only has to stay ahead of speech (~3 tok/s of actual talking) with margin, and
that bar is low: an 8GB model clears it roughly tenfold. So size DOWN. The capability you
give up is worth less here than the latency you buy, because replies are one to three
spoken sentences and the system prompt asks for exactly that.

Two failures on 2026-07-22 taught the rest, and both are recorded against MODELS below:
oversized `gpu-memory-utilization` will OOM the box (the reservation is system RAM on a
GB10), and vLLM's quantized-MoE kernels are the fragile path right now. Dense + bf16 has
no quantization code path at all, which is why the default is a small dense Qwen.
"""

from __future__ import annotations

import os

from flyteplugins.vllm import VLLMAppEnvironment

import flyte
import flyte.app

# Which model this app serves. `VOICE_LLM=<key> python voice_vllm.py`.
#
# The third field is gpu-memory-utilization, which MUST be sized per model: vLLM reserves
# `util * 119.7GiB` up front, so one constant across models either starves the big ones or
# leaves a small one squatting on most of the box's RAM. On the GB10 that reservation is
# system RAM everything else is also using, which is exactly how the first deploy died.
# Roughly: weights + a comfortable 8192-token KV cache, and nothing more, because a voice
# assistant serves ONE conversation and has no use for batch headroom.
#
# ── Learned the hard way, 2026-07-22 ─────────────────────────────────────────────
# Prefer DENSE + bf16 for anything that has to work today. Both deploy failures came from
# the exotic end of the stack, and the second one was specifically vLLM's quantized-MoE
# kernels:
#
#   gemma26b-nvfp4  BROKEN. vLLM 0.19.1.dev picks its VLLM_CUTLASS NvFp4 MoE backend and
#                   dies on `KeyError: layers.13.experts.100.down_proj.input_scale`. The
#                   tensor IS in the checkpoint (safetensors index shows 128 experts and
#                   3840 down_proj.input_scale entries) but under the full multimodal name
#                   `model.language_model.layers.13...`. So: a vLLM name-mapping bug in the
#                   experimental ModelOpt NVFP4 path, NOT a bad checkpoint and NOT the
#                   Flyte streaming loader (2 shards, both streamed fine). Retry after a
#                   vLLM bump.
#   gemma26b-fp8    UNTESTED. FP8-*dynamic* quantizes activations at runtime so it ships no
#                   static input_scale tensors, making the failure above structurally
#                   impossible; but it is still a quantized-MoE path, so treat with care.
#
# A dense bf16 model has no quantization path at all, which is why the small Qwen entries
# are the safe default. MoE is not a bad idea in principle: 26B-A4B reads only ~4B active
# params per token, so it can be FASTER than its size suggests while being far more
# capable. It is just the fragile path in vLLM right now.
MODELS = {
    # key              hf repo                                    served id           util
    "qwen3-1.7b":     ("Qwen/Qwen3-1.7B",                        "qwen3-1.7b",       "0.15"),
    "qwen3-4b":       ("Qwen/Qwen3-4B",                          "qwen3-4b",         "0.20"),
    "qwen3-8b":       ("Qwen/Qwen3-8B",                          "qwen3-8b",         "0.30"),
    "gemma12b":       ("google/gemma-4-12B-it",                  "gemma-4-12b-it",   "0.35"),
    "gemma26b-fp8":   ("RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic", "gemma-4-26b-fp8", "0.45"),
    "gemma26b-nvfp4": ("nvidia/Gemma-4-26B-A4B-NVFP4",           "gemma-4-26b-nvfp4", "0.35"),
    "gemma26b-bf16":  ("google/gemma-4-26B-A4B-it",              "gemma-4-26b-it",   "0.70"),
}
VOICE_LLM = os.environ.get("VOICE_LLM", "qwen3-4b")
HF_REPO, MODEL_ID, _DEFAULT_UTIL = MODELS[VOICE_LLM]

APP_NAME = f"tts-voice-llm-{VOICE_LLM.replace('.', '-')}"

# Override the per-model default only if you know why. The gemma4 devbox app hardcodes
# 0.75, which is correct for the bf16 52GB model IT serves and catastrophic here: copying
# it verbatim is what made the first deploy reserve ~90GB of the 119.7GiB pool and die with
# `torch.AcceleratorError: CUDA error: out of memory` (rustfs was sitting on 49GB of leaked
# heap at the time, so the pool was smaller than it looked; see the README recovery note).
GPU_MEMORY_UTILIZATION = os.environ.get("VOICE_GPU_MEM_UTIL", _DEFAULT_UTIL)

# from_base() returns a FROZEN dataclass pinned to linux/amd64 and clone() exposes no
# platform kwarg, so the freeze is bypassed to set arm64. Straight from vllm_server.py.
_base = flyte.Image.from_base("vllm/vllm-openai:gemma4-cu130")
object.__setattr__(_base, "platform", ("linux/arm64",))

image = (
    _base.clone(registry="localhost:30000", name="tts-voice-vllm-image", extendable=True)
    # Into the base image's system Python, where vllm + torch already live: installing
    # into Flyte's /opt/venv leaves vllm-fserve unable to import torch.
    .with_commands([
        "/usr/bin/python3 -m pip install --no-cache-dir --pre flyteplugins-vllm"
    ])
)

vllm_app = VLLMAppEnvironment(
    name=APP_NAME,
    image=image,
    model_hf_path=HF_REPO,      # replaced by model_path=RunOutput(...) on deploy
    model_id=MODEL_ID,
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu=1, disk="40Gi"),
    stream_model=True,
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        # Cold starts are minutes, and this app holds the box's only GPU while up, so the
        # window is a balance: long enough that a conversation doesn't pay a reload,
        # short enough that the compare/clone pipelines get the GPU back.
        scaledown_after=1800,
    ),
    requires_auth=False,
    extra_args=[
        # Voice replies are short; a big context just costs KV cache we would rather
        # leave free on a shared pool.
        "--max-model-len", "8192",
        "--trust-remote-code",
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        # sm_121 hangs during CUDA-graph capture on this box (same failure that deferred
        # voxtral). Eager skips capture at a small throughput cost.
        "--enforce-eager",
    ],
)


if __name__ == "__main__":
    flyte.init_from_config()

    # VOICE_PREFETCH_RUN=<run-name> reuses a known-good prefetch instead of re-fetching.
    if existing := os.environ.get("VOICE_PREFETCH_RUN"):
        run_name = existing
        print(f"Reusing prefetched weights from run: {run_name}")
    else:
        import flyte.prefetch
        from flyte.remote import Run

        print(f"Prefetching {HF_REPO} …")
        run: Run = flyte.prefetch.hf_model(repo=HF_REPO)
        run.wait()
        print(f"Prefetch run: {run.url}")
        run_name = run.name

    print(f"Deploying vLLM for {MODEL_ID} ({VOICE_LLM}, util={GPU_MEMORY_UTILIZATION}) …")
    app = flyte.serve(
        vllm_app.clone_with(
            name=vllm_app.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run_name),
            model_hf_path=None,
        )
    )
    print(f"vLLM app deployed. Console: {app.url}")
    print()
    # app.url is the CONSOLE link, not the serving endpoint: posting to it does not
    # reach vLLM. The endpoint is the Knative route, and which one you want depends on
    # who is calling. Resolve it rather than printing app.url and misleading the reader.
    svc = f"{APP_NAME}-{os.environ.get('FLYTE_PROJECT', 'text-to-speech')}-development"
    print(f"  OpenAI base URL, from INSIDE the cluster (what voice_app uses in a pod):")
    print(f"    http://{svc}.flyte.svc.cluster.local/v1")
    print(f"  From the devbox host (RUN_MODE=host, or curl):")
    print(f"    http://{svc}.localhost:30081/v1")
    print()
    print("Point the voice app at it:")
    print(f"  LLM_BACKEND=vllm VLLM_MODEL_ID={MODEL_ID} \\")
    print(f"    VLLM_URL=http://{svc}.flyte.svc.cluster.local python voice_app.py")
