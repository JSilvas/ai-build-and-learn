"""Shared Flyte config for the MuJoCo RL demo: teaching a Unitree G1 humanoid to walk.

Same shape as the video-generation demo next door (topics/video-generation), so if
you've read that one the layout is familiar:

  - train.py  : the GPU task. Brax PPO over an MJX (MuJoCo-on-GPU) G1 env, thousands
                of simulations stepping in parallel inside one XLA program.
  - render.py : a CPU task. MJX has no renderer, so the replay is played back through
                stock MuJoCo and lands in the Flyte report as a scrubable clip.
  - app.py    : a thin CPU Gradio *app* that launches runs and links the report. It
                holds no GPU and trains nothing.

── Why MJX and not the CPU PPO from the workshop tutorial ──────────────────────
The `rl-unitree-g1` tutorial ran Gymnasium + PyTorch PPO across 5 Flyte workers at
32 vectorised envs each: 160 simulations, and its own README's "what we tried" table
ends at run 6 = TBD. The humanoid stood, shuffled, and fell. That is not primarily a
reward-tuning failure, it is a sample-count failure: humanoid locomotion policies in
the literature are trained on 10^8-10^9 environment steps, and 160 CPU envs cannot
reach that in a stream-length run.

MJX compiles the physics into XLA and vmaps it, so the whole loop (simulate, score,
update) stays on the GPU with no Python in the inner loop. That is the difference
between 160 and several thousand parallel envs, and it is what makes the humanoid
reachable on one box.

── DGX Spark pins (GB10 Blackwell, arm64, CUDA 13.0) ───────────────────────────
1. `jax[cuda13]`, NOT `jax[cuda12]`. The workshop tutorial's requirements.txt says
   cuda12; this box's driver is CUDA 13.0 (verified: `nvidia-smi` reports 580.126.09
   / CUDA 13.0). jax-cuda13-plugin publishes manylinux_2_27_aarch64 wheels, so the
   arm64 install is clean. Verified working on this box: jax 0.10.2 + cuda13 plugin
   enumerates `[CudaDevice(id=0)]`.
2. `mujoco` ships aarch64 wheels; `mujoco-mjx`, `brax` and `playground` are pure
   Python. Nothing here needs to be built from source.
3. The PyPI distribution of MuJoCo Playground is named **`playground`**, but it
   imports as `mujoco_playground`. There is no `mujoco-playground` on PyPI (404).
4. **jax is capped at 0.9.x.** brax 0.14.2 (current release) still calls
   `jax.device_put_replicated` in `ppo/train.py:756`, and jax removed it in 0.10. An
   unpinned install resolves to 0.10.2 and every run dies at the first training step
   with `AttributeError: jax.device_put_replicated is deprecated`. 0.9.2 still ships
   it and its cuda13 plugin still enumerates the GB10. Lift the pin when brax moves
   off the old pmap API.
"""

from __future__ import annotations

import flyte
from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

APP_NAME = "g1-walk-studio"
APP_PORT = 7864          # 7862 image-gen, 7863 videogen; don't collide


# ── The Spark's real memory ceiling ─────────────────────────────────────────────
#
# On GB10 there is ONE 119.7GiB pool shared by the OS, the page cache and the GPU.
# Two consequences, both of which cost real debugging time on this box:
#
# 1. XLA_PYTHON_CLIENT_PREALLOCATE=false is MANDATORY. JAX's default is to
#    preallocate 75% of "GPU memory" on first use. On a discrete card that reserves
#    VRAM nobody else wanted; here it claims ~90GB of the same RAM the OS is living
#    in. Set it false and let the allocator grow.
#
# 2. PAGE CACHE STARVES CUDA INIT, and the error message actively lies about it.
#    Measured on this box 2026-08-04: with 86GB in buff/cache, `cuMemGetInfo`
#    reported only 18.0 GiB free of 119.7 GiB, and JAX died with
#
#        Failed to create stream executor for device CUDA:0: CUDA_ERROR_OUT_OF_MEMORY
#        RuntimeError: Unable to initialize backend 'cuda': INTERNAL: no supported
#        devices found for platform CUDA (you may need to uninstall the failing plugin)
#
#    That reads like a broken install. It is not: it is transient memory pressure,
#    and the suggested fix (uninstall the plugin) is exactly wrong. `free` is no help
#    either, since it cheerfully reported 99GB "available" throughout. The tell is
#    cuMemGetInfo, not free. Dropping the pressure (finishing whatever is churning
#    the cache, or `sync && sysctl -w vm.drop_caches=3`) brings the device straight
#    back. Lowering XLA_PYTHON_CLIENT_MEM_FRACTION does NOT help; the failure is at
#    stream-executor creation, before any arena is sized.
#
#    Practical rule: don't start a training run while a big blob scan, model fetch or
#    docker build is hammering the disk.
_SPARK_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    # A ceiling, not a reservation, once PREALLOCATE is false. Leaves ~25GB of the
    # unified pool for the OS, the page cache and the render task.
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.80",
    # Video/physics kernels JIT a lot on first use; the default JIT cache is tiny and
    # evicts, so every step re-JITs. Same knob the videogen demo needed.
    "CUDA_CACHE_MAXSIZE": "4294967296",
    "CUDA_MODULE_LOADING": "EAGER",
}

# MuJoCo's renderer needs a GL context. In a container there is no display, so EGL is
# the only option that works headless. Without this, `mujoco.Renderer` raises on
# construction rather than falling back.
_RENDER_ENV = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}


# ── Python deps ─────────────────────────────────────────────────────────────────
#
# The GL runtime libs are apt packages, not pip: `mujoco.Renderer` dlopens libEGL at
# runtime and fails with a bare ImportError if they're missing.
_GL_APT = (
    "libegl1", "libegl-mesa0", "libgl1", "libgl1-mesa-dri",
    "libgles2", "libglx-mesa0", "libosmesa6",
)

MJX_SPEC = (
    # cuda13 not cuda12 (module docstring), and capped at 0.9.x (pin note below).
    "jax[cuda13]==0.9.2",
    "mujoco",
    "mujoco-mjx",
    "brax",
    "playground",         # imports as `mujoco_playground`
    "numpy",
    "Pillow",
    "matplotlib",
    "mediapy",            # Playground's own examples use it; cheap, and handy for mp4
    # PyAV encodes the replay to mp4. Same reasoning as the videogen demo: PyAV ships
    # manylinux aarch64 wheels, imageio-ffmpeg does not reliably install here.
    "av",
    "imageio",
    # config.py imports kubernetes.client at module top for the app pod template, and
    # task pods import config too, so every image needs it.
    "kubernetes",
)


def _mjx_image(name: str) -> flyte.Image:
    return (
        flyte.Image.from_debian_base(name=name, registry=REGISTRY, platform=PLATFORM)
        .with_apt_packages("git", "ffmpeg", *_GL_APT)
        .with_pip_packages(*MJX_SPEC)
        # Playground fetches the robot XMLs (mujoco_menagerie) on first use, which
        # would otherwise be a git clone inside EVERY task pod, on every run.
        #
        # ⚠️ It must land in Playground's OWN package directory. MENAGERIE_PATH is
        # `<site-packages>/mujoco_playground/external_deps/mujoco_menagerie`, hardcoded
        # relative to the module file, with no env var to redirect it. An earlier
        # version of this file cloned to /opt/mujoco_menagerie and it did nothing at
        # all: every pod still printed "mujoco_menagerie not found. Downloading..."
        # and re-cloned, which is minutes of wall clock per task and a network
        # dependency in a job that should have none.
        #
        # Calling Playground's own `ensure_menagerie_exists()` rather than doing the
        # clone by hand means we also inherit its pinned commit
        # (MENAGERIE_COMMIT_SHA), so the robot XMLs cannot drift from the version the
        # env code expects.
        .with_commands([
            "python -c 'from mujoco_playground._src import mjx_env; "
            "mjx_env.ensure_menagerie_exists()'",
        ])
    )


image = _mjx_image("g1-mjx-image")

# The studio app is a LAUNCHER: it submits runs and links reports, so it needs no
# jax/mujoco at all. Keeping it tiny means the app pod never holds the GPU.
# connectrpc pinned to 0.10.x: 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
studio_app_image = (
    flyte.Image.from_debian_base(
        name="g1-studio-image", registry=REGISTRY, platform=PLATFORM
    )
    .with_pip_packages("flyte==2.2.1", "connectrpc==0.10.*", "gradio==5.42.0", "python-dotenv")
)


# ── Environments ────────────────────────────────────────────────────────────────
#
#   gpu_env   (g1-train)  : train_policy. Physics AND gradients, both on the GPU.
#   cpu_env   (g1-render) : render_replay. CPU MuJoCo, because MJX cannot render.
#   orch_env  (g1-orch)   : the orchestrator. CPU-only ON PURPOSE: an orchestrator pod
#                           stays alive holding its resources while awaiting children,
#                           so if it asked for the GPU it would hold the box's only one
#                           and deadlock its own GPU child on "Insufficient
#                           nvidia.com/gpu" forever. (Learned the hard way in the
#                           videogen demo; same trap here.)
#
# Cross-env calls need the caller to `depends_on` the callee's env or `flyte run` won't
# build the callee's image ("Environment '…' not found in image cache").

# memory=96Gi: MJX's memory is dominated by num_envs x the per-env state, and the whole
# rollout buffer is resident. 64Gi was enough for 4096 envs (verified) but 8192 is the
# value DeepMind's config actually uses, and env state scales with it. The node reports
# 125.5GiB allocatable with under 1GiB reserved by anything else, so 96Gi schedules
# fine and still leaves the OS and the render task room in the shared pool.
# If a pod goes Unschedulable, this is the knob to turn down.
gpu_env = flyte.TaskEnvironment(
    name="g1-train",
    image=image,
    resources=flyte.Resources(cpu="8", memory="96Gi", gpu=1, disk="50Gi"),
    env_vars={**_SPARK_ENV, **_RENDER_ENV},
)

cpu_env = flyte.TaskEnvironment(
    name="g1-render",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", disk="20Gi"),
    # The render task runs the policy too, but on CPU. Pin JAX to CPU so it cannot
    # grab the GPU out from under a concurrent training run.
    env_vars={**_RENDER_ENV, "JAX_PLATFORMS": "cpu"},
)

orch_env = flyte.TaskEnvironment(
    name="g1-orch",
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="20Gi"),
    env_vars={"JAX_PLATFORMS": "cpu"},
    depends_on=[gpu_env, cpu_env],
)

# AppEnvironment does NOT honor flyte.Resources(gpu=1) on this SDK: the serializer maps
# it to a bare `gpu` name that k8s drops, so the pod silently schedules CPU-only. The
# fix (verified in the magenta + imagegen + videogen demos) is a PodTemplate that sets
# nvidia.com/gpu directly, passing NO resources=. Unused while the studio is a pure
# launcher; kept for in-pod experiments.
app_gpu_pod = flyte.PodTemplate(
    primary_container_name="app",
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="app",
                resources=V1ResourceRequirements(
                    requests={"cpu": "8", "memory": "64Gi", "ephemeral-storage": "50Gi"},
                    limits={"nvidia.com/gpu": "1"},
                ),
            )
        ]
    ),
)
