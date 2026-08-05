"""Train the G1 to walk: Brax PPO over an MJX environment, all on one GB10.

The whole training loop is a single XLA program. Simulate, score, compute advantages,
update the networks: none of it returns to Python between steps. That is what buys the
throughput, and it is also why the first call is slow (everything compiles) and every
call after it is fast.

── Measured on this box ────────────────────────────────────────────────────────
DGX Spark (GB10, sm_121, arm64), mujoco 3.11.0 / MJX-Warp, jax 0.10.2 + cuda13,
2026-08-04:

    1024 envs   32,005 env-steps/sec
    2048 envs   59,757 env-steps/sec      (scales nearly linearly, so go wider)
    env build + first compile   ~40s one-off

Those are bare `env.step` rates. The end-to-end figure, PPO gradient updates included,
measured through this exact code path at 1024 envs:

    2,129,920 steps in 66s   ->  ~32,300 steps/sec end to end
    one-off compile          ~26s before the first eval

So at 1024 envs the gradient updates are essentially free next to the simulation, and
throughput scales close to linearly with num_envs up to at least 2048. That is the
argument for the 4096 default and for trying 8192 (what DeepMind's config actually
uses) once a run is known to work.

DeepMind's tuned recipe for this env is 200M steps: on the order of an hour at 4096
envs. Start with `--num_timesteps 20_000_000` to confirm the reward curve is climbing
before committing to the full run.

── Why we take Playground's PPO config wholesale ───────────────────────────────
`mujoco_playground.config.locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")`
is DeepMind's own tuned config for exactly this env: 200M steps, 8192 envs, entropy
0.005, clipping 0.2, a (512,256,128) policy and value net, and the asymmetric obs keys.
Reading it out of the package rather than copying the numbers here means a Playground
upgrade that re-tunes the env re-tunes us too, instead of leaving us on a stale copy
that silently stops matching the env it was tuned for.

Usage:
    flyte run train.py train_policy --num_timesteps 20000000 --preset baseline
    flyte run train.py train_policy --num_timesteps 200000000 --preset high-step
"""

from __future__ import annotations

import json
import logging
import pickle
import time

import flyte
import flyte.report
from flyte.io import File

from config import gpu_env
import envs
import reports

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _wait_for_gpu(timeout_s: int = 900, poll_s: int = 20) -> None:
    """Block until CUDA can actually initialize, instead of dying on a transient.

    This exists because of a failure mode specific to this box: GB10 shares one
    119.7GiB pool between the OS page cache and the GPU, so heavy disk I/O elsewhere
    (a blob scan, a model fetch, a docker build) can leave CUDA unable to create a
    context at all. JAX reports it as:

        Unable to initialize backend 'cuda': INTERNAL: no supported devices found
        for platform CUDA (you may need to uninstall the failing plugin package)

    which reads like a broken install and is nothing of the sort. It killed a 200M-step
    run at minute zero once; the whole point of this function is that it never does so
    again. We probe the driver directly with ctypes rather than importing jax, because
    jax caches its backend-init failure process-wide: once it has failed, every later
    call in the same process keeps failing even after the box recovers.
    """
    import ctypes
    import time

    cuda = ctypes.CDLL("libcuda.so.1")
    deadline = time.time() + timeout_s
    attempt = 0
    while True:
        attempt += 1
        cuda.cuInit(0)
        dev = ctypes.c_int()
        ctx = ctypes.c_void_p()
        if cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0 and \
                cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0:
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
            cuda.cuCtxDestroy_v2(ctx)
            log.info(
                f"GPU ready after {attempt} probe(s): "
                f"{free.value / 2**30:.1f} GiB free of {total.value / 2**30:.1f} GiB"
            )
            return
        if time.time() >= deadline:
            raise RuntimeError(
                f"CUDA could not initialize within {timeout_s}s. On GB10 this is "
                f"almost always page-cache pressure, not a broken driver: check "
                f"cuMemGetInfo (NOT `free`, which counts reclaimable cache as "
                f"available) and let the disk I/O finish, or "
                f"`sync && sysctl -w vm.drop_caches=3`."
            )
        log.warning(
            f"  CUDA unavailable (probe {attempt}); page-cache pressure? "
            f"retrying in {poll_s}s"
        )
        time.sleep(poll_s)


def _ppo_params(num_timesteps: int, num_envs: int, num_evals: int, terrain: str):
    """Playground's tuned G1 PPO config, with the knobs we expose overridden.

    Flat and rough share one tuned config in Playground, but we read it per-terrain
    rather than hardcoding flat, so a future divergence upstream carries through.
    """
    from mujoco_playground.config import locomotion_params

    params = locomotion_params.brax_ppo_config(envs.env_name(terrain))
    params.num_timesteps = num_timesteps
    params.num_envs = num_envs
    params.num_evals = num_evals
    return params


# The command the mid-training snapshots are filmed on. Fixed across every snapshot
# so the clips are comparable to each other: a walk straight at the camera at the
# env's top commanded speed.
SNAPSHOT_COMMAND = (1.0, 0.0, 0.0)


@gpu_env.task(report=True, retries=1)
async def train_policy(
    num_timesteps: int = 20_000_000,
    num_envs: int = 4096,
    num_evals: int = 10,
    preset: str = envs.DEFAULT_PRESET,
    seed: int = 0,
    domain_randomization: bool = True,
    snapshot_every_pct: float = 15.0,
    snapshot_steps: int = 250,
    terrain: str = envs.DEFAULT_TERRAIN,
) -> File:
    """Train one G1 walking policy and return its checkpoint.

    Returns a pickle holding the Brax params plus the eval history, so `render.py`
    can replay it and the orchestrator can chart it without re-reading the GPU.

    `snapshot_every_pct` films the CURRENT policy periodically and drops the clip
    into the live report, overwriting the previous one, so you can watch the gait
    develop instead of waiting an hour to find out it learned to hop. Set 0 to
    disable. Roughly 10s per snapshot after a one-off single-env compile, so at the
    default (~7 snapshots over a run) the overhead is under 2%.
    """
    import functools

    # BEFORE importing jax: jax caches a failed backend init for the life of the
    # process, so if we let it import while the box is starved it stays broken even
    # after the box recovers. Probe the driver first, wait it out, then import.
    _wait_for_gpu()

    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    spec = envs.get_preset(preset)
    log.info(f"device: {jax.devices()}")
    log.info(f"preset: {spec.key} -- {spec.notes}")

    env, cfg = envs.build_env(preset, terrain=terrain)
    log.info(f"terrain: {terrain} ({envs.env_name(terrain)})")
    log.info(f"env: action_size={env.action_size} obs={env.observation_size} njmax={cfg.njmax}")

    params = _ppo_params(num_timesteps, num_envs, num_evals, terrain)

    # The asymmetric actor-critic wiring. See the envs.py docstring: the policy is
    # restricted to onboard-sensor-shaped observations, the critic gets ground truth.
    # Playground's config already carries the right keys; we assert rather than assume,
    # because silently swapping them trains a policy that cannot run on hardware.
    net_cfg = dict(params.network_factory)
    assert net_cfg["policy_obs_key"] == envs.POLICY_OBS_KEY, net_cfg
    assert net_cfg["value_obs_key"] == envs.VALUE_OBS_KEY, net_cfg
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_cfg)

    randomizer = envs.get_randomizer(terrain) if domain_randomization else None
    log.info(f"domain randomization: {'on' if randomizer else 'off'}")

    # Progress is streamed into the Flyte report as it arrives rather than dumped at
    # the end: a 200M-step run is long enough that watching the reward curve climb is
    # the whole point of having a live report.
    history: list[dict] = []
    started = time.time()

    # ── Mid-training video ──────────────────────────────────────────────────────
    #
    # brax calls `policy_params_fn(step, make_policy, params)` at every eval boundary
    # (plus once at step 0). It is the ONLY hook that hands over the live weights:
    # progress_fn gets just (step, metrics), so it cannot render anything.
    #
    # We roll out on `env` directly. brax trains on its own wrapped, vmapped copy, so
    # the single un-vmapped env here is untouched by training and safe to step.
    _every_n = (
        max(1, round(num_evals * snapshot_every_pct / 100.0))
        if snapshot_every_pct > 0
        else 0
    )
    _snap_latest: dict = {}
    _snap_timeline: list[dict] = []
    _snap_state = {"evals": 0, "reset": None, "step": None, "cam": None}

    def _render_snapshot(step: int, make_policy, params) -> None:
        import base64
        import io as _io

        import jax.numpy as jp
        import numpy as np
        from PIL import Image

        from render import _encode_mp4, _pick_camera

        if _snap_state["reset"] is None:
            _snap_state["reset"] = jax.jit(env.reset)
            _snap_state["step"] = jax.jit(env.step)
            _snap_state["cam"] = _pick_camera(env.mj_model)

        inference = jax.jit(make_policy(params, deterministic=True))
        cmd = jp.array(SNAPSHOT_COMMAND, dtype=jp.float32)
        rng = jax.random.PRNGKey(0)          # same seed every snapshot: comparable clips
        state = _snap_state["reset"](rng)
        state.info["command"] = cmd

        traj = [state]
        for _ in range(snapshot_steps):
            rng, act_rng = jax.random.split(rng)
            action, _ = inference(state.obs, act_rng)
            state = _snap_state["step"](state, action)
            state.info["command"] = cmd       # the env would resample it mid-episode
            traj.append(state)
            if float(state.done) > 0.5:
                break

        frames = env.render(traj, height=270, width=360, camera=_snap_state["cam"])
        mp4 = _encode_mp4(frames, int(round(1.0 / env.dt)))
        reward = history[-1]["reward"] if history else 0.0

        _snap_latest.clear()
        _snap_latest.update(
            step=step, reward=reward, steps=len(traj) - 1,
            max_steps=snapshot_steps, mp4=mp4,
        )
        # One still per snapshot for the filmstrip. Middle frame, because the first is
        # always the same standing pose and tells you nothing.
        buf = _io.BytesIO()
        mid = np.asarray(frames[len(frames) // 2], dtype=np.uint8)
        Image.fromarray(mid).save(buf, "JPEG", quality=70)
        _snap_timeline.append(
            {"step": step, "reward": reward,
             "frame": base64.b64encode(buf.getvalue()).decode()}
        )
        log.info(
            f"  snapshot @ {step:,}: survived {len(traj) - 1}/{snapshot_steps} steps, "
            f"{len(mp4) / 1024:.0f}KB clip"
        )

    def policy_params_fn(step: int, make_policy, params) -> None:
        """Never let a rendering problem kill an hour of training."""
        _snap_state["evals"] += 1
        if not _every_n or (_snap_state["evals"] - 1) % _every_n:
            return
        try:
            _render_snapshot(step, make_policy, params)
        except Exception as exc:
            log.warning(f"  snapshot at step {step:,} failed ({exc}); training continues")

    def progress(step: int, metrics: dict) -> None:
        row = {
            "step": int(step),
            "reward": float(metrics.get("eval/episode_reward", 0.0)),
            "reward_std": float(metrics.get("eval/episode_reward_std", 0.0)),
            "elapsed_s": round(time.time() - started, 1),
        }
        # Tracking reward is the one that actually says "is it walking where it was
        # told", as opposed to total reward which mixes in all the shaping terms.
        for name in ("eval/episode_reward/tracking_lin_vel", "eval/episode_reward/feet_air_time"):
            if name in metrics:
                row[name.rsplit("/", 1)[-1]] = float(metrics[name])
        history.append(row)
        sps = row["step"] / max(row["elapsed_s"], 1e-9)
        log.info(
            f"  step {row['step']:>11,} | reward {row['reward']:8.1f}"
            f" +/- {row['reward_std']:6.1f} | {sps:>9,.0f} steps/s"
        )
        # Fire and forget; the orchestrator redraws the full chart at the end anyway.
        try:
            flyte.report.replace(
                reports.progress_html(
                    spec, params, history,
                    extra=reports.training_snapshot_html(_snap_latest, _snap_timeline),
                )
            )
            flyte.report.flush()
        except Exception:
            pass

    log.info(f"training {num_timesteps:,} steps across {num_envs:,} envs...")
    make_inference_fn, trained_params, _ = ppo.train(
        environment=env,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        network_factory=network_factory,
        randomization_fn=randomizer,
        # Playground's wrapper adds the auto-reset and domain-randomization vmap that
        # Brax's own wrapper does not know about. Using Brax's default here silently
        # drops the randomization.
        wrap_env_fn=wrapper.wrap_for_brax_training,
        seed=seed,
        **{k: v for k, v in params.items() if k != "network_factory"},
    )
    elapsed = time.time() - started
    log.info(f"done in {elapsed / 60:.1f} min")

    path = "/tmp/g1_checkpoint.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "params": jax.device_get(trained_params),
                "history": history,
                "preset": spec.key,
                "env_name": envs.env_name(terrain),
                "terrain": terrain,
                "network_factory": net_cfg,
                "num_timesteps": num_timesteps,
                "num_envs": num_envs,
                "seed": seed,
                "domain_randomization": bool(randomizer),
                "elapsed_s": round(elapsed, 1),
            },
            f,
        )
    return await File.from_local(path)
