# Reinforcement Learning in MuJoCo: teaching a humanoid to walk

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This one is about reinforcement learning in [MuJoCo](https://github.com/google-deepmind/mujoco), the physics engine that most robotics RL runs on. The goal is narrow and honest: get a **Unitree G1 humanoid to actually walk**, on one DGX Spark, and be able to watch it happen.

Humanoid locomotion is the standard hard case. A quadruped is stable enough that a mediocre policy still looks like walking; a humanoid falls over, and every shortcut in your reward function shows up immediately as a robot that shuffles, hops, or dives forward to farm the velocity term.

## The thing this is fixing

A previous version of this demo trained the G1 with Gymnasium + PyTorch PPO across 5 workers at 32 vectorized environments each: **160 simulations**, on CPU. Its own notes end with a table of six attempts, the last one marked TBD. The robot stood, shuffled, and fell over.

The reward function was not really the problem. Sample count was. Published humanoid locomotion policies are trained on 10^8 to 10^9 environment steps, and 160 CPU environments cannot get there in a session.

So this version changes the axis that actually mattered:

| | Old (CPU PPO) | This (MJX + Brax) |
|---|---|---|
| Physics | CPU MuJoCo | GPU, compiled (MJX on Warp) |
| Parallel envs | 160 (5 workers × 32) | 4,096 on one GPU |
| Inner loop | Python | one XLA program, no Python per step |
| Reward | hand-written, 10 terms | DeepMind's tuned `G1JoystickFlatTerrain` |
| Measured throughput | — | **59,757 env-steps/sec** at 2,048 envs |

Measured on this box (GB10, sm_121, arm64) on 2026-08-04 with mujoco 3.11.0 and jax 0.10.2:

```
bare env.step
  1024 envs   32,005 env-steps/sec
  2048 envs   59,757 env-steps/sec      scales close to linearly

end-to-end PPO (gradient updates included), 1024 envs
  2,129,920 steps in 66s  ->  ~32,300 steps/sec
  one-off compile         ~26s before the first eval
```

At 1024 envs the gradient updates are essentially free next to the simulation, so the end-to-end rate matches the bare simulation rate. 200M steps lands on the order of an hour at 4096 envs.

## What it does

```
walk (orchestrator, CPU)
  ├── train_policy  (GPU)   Brax PPO over MJX, 4096 envs, live reward curve
  └── render_replay (CPU) ×2  trained policy AND a random policy, both -> mp4
```

**Mid-training video.** Every ~15% of the run (`--snapshot_every_pct`, 0 to disable) the current policy is filmed for 250 steps and dropped into the *training task's* live report, overwriting the previous clip, so you can watch the gait develop instead of waiting an hour to find out it learned to hop. Underneath the clip is a filmstrip of one still per snapshot: the reward curve tells you the number went up, the filmstrip tells you whether it went up for the right reason.

This hangs off brax's `policy_params_fn(step, make_policy, params)`, which fires at every eval boundary and is the only hook that hands over the live weights (`progress_fn` gets just `(step, metrics)`, so it can't render anything). Cost is roughly 10s per snapshot after a one-off single-env compile, under 2% of a full run. The whole thing is wrapped in a `try/except` that logs and continues, because no rendering problem should ever kill an hour of training.

The reward curve streams into the Flyte report while training runs. Two replay clips land there when it finishes: the trained policy and, next to it, a **random policy on the same command and camera**. The trained clip on its own is unevaluable, because "is that good walking?" has no answer without the before-picture.

Verified end to end on the devbox (project `physical-ai`, run `rfjnchxkvtbv2tztgf69`, 2026-08-04). A deliberately tiny 4M-step run produced:

```
best eval reward   -3.08
trained policy     survived 33 / 300 steps
random policy      survived  8 / 300 steps
```

4x longer upright after 4M steps, which is 2% of the tuned recipe. It is not walking yet, and it is not supposed to be: that run exists to prove the pipeline, not the policy. The report came back at 296KB with both clips embedded and playable.

## Run it

Runs go to the **`physical-ai`** Flyte project (`.flyte/config.yaml`).

```bash
# Confirm the curve is climbing (~10 min)
flyte run pipeline.py walk --num_timesteps 20000000 --preset baseline

# The real run. DeepMind's recipe for this env is 200M steps.
flyte run pipeline.py walk --num_timesteps 200000000 --num_envs 8192 --preset baseline

# Several reward presets, one chart. These SERIALIZE: one GPU.
flyte run pipeline.py compare_presets --presets '["baseline","high-step"]'
```

Or launch from the studio:

```bash
python app.py        # deploys the Gradio launcher, prints its URL
```

| Flag | Default | What it does |
|---|---|---|
| `--num_timesteps` | 20,000,000 | Total environment steps. The tuned recipe is 200M |
| `--num_envs` | 4096 | Parallel simulations on the GPU. Scales nearly linearly |
| `--preset` | `baseline` | Reward preset, see below |
| `--num_evals` | 10 | Evaluation points, so also how often the chart updates |
| `--domain_randomization` | true | Per-env friction, mass and motor gains |
| `--replay_steps` | 500 | Length of the replay clip |

## The environment

We do not hand-roll the humanoid. `mujoco_playground` ships DeepMind's own `G1JoystickFlatTerrain`, which is the config their published G1 results come from. Starting from something that demonstrably walks and *then* turning knobs is the whole lesson of the previous attempt.

**"Joystick"** means the env samples a random velocity command each episode (x in [-1, 1] m/s, y in [-0.5, 0.5], yaw in [-1, 1] rad/s) and rewards *tracking* it. The policy is not learning "walk forward", it is learning "go the speed and direction I am told". That is harder, and the result is steerable.

**Asymmetric actor-critic.** The observation is a dict, not a vector:

```
state             (103,)   what the POLICY sees      onboard-sensor shaped
privileged_state  (216,)   what the VALUE net sees   ground-truth sim state
```

The critic gets contact forces and true body velocities that a real robot could never measure, which makes the value function much easier to learn, while the policy stays deployable. This is why `train.py` passes `policy_obs_key="state"` and `value_obs_key="privileged_state"`; swap them and you either cripple the critic or train a policy that cannot run on hardware.

## Terrain

Two Playground envs, same code path, same tuned PPO config, same observation and action shapes. They differ only in the scene:

| `--terrain` | Scene | Notes |
|---|---|---|
| `rough` (default) | `scene_mjx_feetonly_rough_terrain.xml` | 10x10m heightfield, 0.05m relief, rocky texture |
| `flat` | `scene_mjx_feetonly_flat_terrain.xml` | a bare plane |

`rough` is the better default on both counts: it's a genuinely harder problem, and it doesn't look like the robot is walking on nothing. The terrain is stored in the checkpoint, and `render.py` reads it from there rather than from a default — replaying a rough-terrain policy on flat ground would silently compare two different problems.

Known cosmetic wart: the rough scene ships no skybox, so its background renders black where flat's is grey. Affects the replay's looks, not the physics.

## Reward presets

Each preset is a small diff on top of DeepMind's scales, not a rewrite. Only the named keys change.

| Preset | Diff | Why |
|---|---|---|
| `baseline` | none | The control. Whatever this scores is the number to beat |
| `high-step` | `feet_clearance: 1.0`, `feet_air_time: 3.0` | Anti-shuffle. Playground ships `feet_clearance` at **0.0**, so it's off by default; turning it on is the most direct answer to a robot that slides its feet |
| `smooth` | `action_rate: -0.75`, `energy: -0.001`, `dof_acc: -2.5e-7` | Penalize jerk. Slower, cleaner gait |
| `no-perturb` | pushes off, sensor noise off | Learns fastest, generalizes worst. Good for a first end-to-end check |

## Things that cost real debugging time on this box

**`njmax` overflows.** Playground ships `njmax=90` for this env. On the Spark the first rollout printed `nefc overflow - please increase njmax to 93` repeatedly. `njmax` caps the constraint rows the solver can hold, and a humanoid in ground contact makes more than 90 of them. On overflow MJX **silently drops constraints** rather than erroring, so feet sink through the floor and you are no longer training on the physics you think you are. We set `njmax=128`. Fixing it also roughly doubled throughput.

**Page cache starves CUDA, and the error lies about it.** With 86GB in `buff/cache`, `cuMemGetInfo` reported only **18.0 GiB free of 119.7 GiB** and JAX died with:

```
Failed to create stream executor for device CUDA:0: CUDA_ERROR_OUT_OF_MEMORY
RuntimeError: Unable to initialize backend 'cuda': INTERNAL: no supported devices
found for platform CUDA (you may need to uninstall the failing plugin package)
```

That reads like a broken install. It is not, and the suggested fix is exactly wrong. `free` is no help either: it reported 99GB "available" the whole time. The tell is `cuMemGetInfo`, not `free`. Lowering `XLA_PYTHON_CLIENT_MEM_FRACTION` does **not** help, because the failure is at stream-executor creation, before any arena is sized.

This is not hypothetical: it killed a 200M-step run at minute zero, an hour after the failure mode was first written down here.

Two things now guard against it.

**`train.py` waits instead of dying.** `_wait_for_gpu()` probes the CUDA driver through `ctypes` and retries for up to 15 minutes before giving up. It runs **before `import jax`**, and that ordering is load-bearing: jax caches a failed backend init for the life of the process, so a task that imports jax during a starved window stays broken even after the box recovers.

**Freeing the cache needs no root.** `drop_caches` wants sudo, but briefly allocating and releasing a large anonymous block makes the kernel evict reclaimable cache to satisfy it:

```python
N = 45 * 2**30
buf = bytearray(N)
for i in range(0, N, 1 << 22):   # touch every 4MB to actually commit
    buf[i] = 1
del buf
```

Measured: `buff/cache` 98GB → 54GB, free 1GB → 45GB, `cuCtxCreate` 2 → 0.

Practical rule stands anyway: do not start a run while a big blob scan, model fetch, or docker build is hammering the disk.

**`XLA_PYTHON_CLIENT_PREALLOCATE=false` is mandatory.** JAX preallocates 75% of "GPU memory" by default. On a discrete card that reserves VRAM nobody wanted. On GB10 there is one 119.7GiB pool shared with the OS, so the default claims ~90GB of the RAM the OS is living in.

**`jax[cuda13]`, not `cuda12`.** The GB10 driver is CUDA 13.0. `jax-cuda13-plugin` publishes `manylinux_2_27_aarch64` wheels, so the arm64 install is clean.

**jax is pinned to 0.9.2, and it has to be.** brax 0.14.2 (the current release) still calls `jax.device_put_replicated` in `ppo/train.py:756`. jax removed it in 0.10, so an unpinned install resolves to 0.10.2 and every run dies the moment training starts:

```
AttributeError: jax.device_put_replicated is deprecated; use jax.device_put instead.
```

Nothing in the install warns you; it fails after the env has already built and compiled. 0.9.2 still ships the function (as a `DeprecationWarning`) and its cuda13 plugin still enumerates the GB10. Lift the pin when brax moves off the old pmap API.

**MJX runs on Warp now.** MuJoCo 3.11's MJX uses `mujoco_warp` (`impl="warp"`), not pure JAX/XLA. It reports `CUDA Toolkit 12.9, Driver 13.0`, sees the GB10 as `sm_121, 120 GiB, mempool enabled`, and JIT-compiles a long list of kernels on first use (~40s). That is a one-off, but it is why the first iteration looks hung.

**`mujoco_menagerie` must be baked into Playground's own package directory.** `MENAGERIE_PATH` is `<site-packages>/mujoco_playground/external_deps/mujoco_menagerie`, hardcoded relative to the module file, with no env var to redirect it. An earlier version of `config.py` cloned it to `/opt/mujoco_menagerie` and that did nothing whatsoever: every task pod still printed `mujoco_menagerie not found. Downloading...` and re-cloned at runtime, costing minutes per task and putting a network dependency inside a job that should have none. The fix is to call Playground's own installer at image build time:

```python
.with_commands([
    "python -c 'from mujoco_playground._src import mjx_env; mjx_env.ensure_menagerie_exists()'",
])
```

That also inherits its pinned `MENAGERIE_COMMIT_SHA`, so the robot XMLs can't drift from the version the env code expects.

**Rendering is CPU, and a different backend.** MJX's fast path needs the GPU and MJX has no renderer. The replay rebuilds the same env with `impl="jax"` (loads in 0.5s on CPU) and draws through headless EGL. So the policy is *trained* on Warp and *replayed* on JAX. Those are two implementations of one model and should agree closely, but they are not bit-identical: if a replay disagrees badly with the eval reward, suspect this first.

## Project structure

```
config.py     Flyte envs and images, all the Spark-specific pins and knobs
envs.py       the G1 env, the njmax fix, the reward presets
train.py      the GPU task: Brax PPO over MJX
render.py     the CPU task: replay -> mp4
reports.py    charts, summary tables, the video block
pipeline.py   orchestrators: `walk` and `compare_presets`
app.py        Gradio launcher (holds no GPU, trains nothing)
```

## Where to take it next

- **8192 envs.** DeepMind's config uses it and we default to 4096. Worth measuring whether the scaling holds.
- **Rough terrain.** `G1JoystickRoughTerrain` is the same code path and the same PPO config, one string away.
- **Reward sweeps as a fan-out.** `compare_presets` already does the shape; with more than one GPU it would actually parallelize.
- **Sim-to-real gap.** The policy only ever sees `state`, so it is deployable in principle. What breaks first on hardware is the interesting question.

## Tooling

- MuJoCo: https://github.com/google-deepmind/mujoco
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
- MJX (MuJoCo on GPU): https://mujoco.readthedocs.io/en/stable/mjx.html
- Brax (JAX RL algorithms): https://github.com/google/brax
- mujoco_menagerie (the robot models): https://github.com/google-deepmind/mujoco_menagerie
