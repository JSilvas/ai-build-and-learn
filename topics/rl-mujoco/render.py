"""Replay a trained policy and turn it into a clip the Flyte report can play.

This is a CPU task on purpose. MJX's fast path (`impl="warp"`) needs the GPU, but the
GPU is the scarce resource on this box and rendering does not need it: we rebuild the
same env with `impl="jax"`, which runs on CPU, and MuJoCo's own renderer draws the
frames through headless EGL.

Verified on this box (2026-08-04): `impl="jax"` on CPU loads the G1 in 0.5s, the first
jitted step compiles in ~4.9s, and `env.render` returns real frames headless.

── One honest caveat ───────────────────────────────────────────────────────────
The policy is TRAINED against `impl="warp"` and REPLAYED against `impl="jax"`. These
are two implementations of the same model, so the trajectories should match closely,
but they are not bit-identical and a policy right at the edge of stability can look
slightly different here than it did in training. If a replay disagrees with the eval
reward badly, that is the first thing to suspect: set `impl="warp"` and run this on
the GPU env to check.

── Pinning the joystick command ────────────────────────────────────────────────
`G1JoystickFlatTerrain` resamples a random velocity command mid-episode, which is
right for training and wrong for a demo clip: the robot would randomly change
direction halfway through. We overwrite `state.info["command"]` after every step so
the replay holds one chosen command for its whole length.
"""

from __future__ import annotations

import base64
import json
import logging

import flyte
import flyte.report
from flyte.io import File

from config import cpu_env
import envs
import reports

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# [forward m/s, lateral m/s, yaw rad/s]. A steady walk straight at the camera reads
# best on stream; the env is trained over x in [-1, 1] so 1.0 is its top commanded speed.
DEFAULT_COMMAND = (1.0, 0.0, 0.0)


def _encode_mp4(frames, fps: int) -> bytes:
    """Frames (H, W, 3) uint8 -> H.264 mp4 bytes, via PyAV.

    PyAV rather than imageio-ffmpeg for the same reason the videogen demo chose it:
    PyAV publishes manylinux aarch64 wheels and imageio-ffmpeg does not install
    reliably here. yuv420p because anything else fails to play in a browser.
    """
    import io

    import av
    import numpy as np

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        h, w = np.asarray(frames[0]).shape[:2]
        # H.264 requires even dimensions; odd ones fail at encoder open with a
        # message that does not mention the size.
        stream.width = w - (w % 2)
        stream.height = h - (h % 2)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23"}
        for frame in frames:
            arr = np.asarray(frame, dtype=np.uint8)[: stream.height, : stream.width]
            container.mux(stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
        container.mux(stream.encode())  # flush
    return buf.getvalue()


def _pick_camera(mj_model) -> str | int:
    """Prefer a tracking camera so the robot doesn't walk out of frame."""
    import mujoco

    names = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(mj_model.ncam)
    ]
    for preferred in ("track", "tracking", "side", "front"):
        if preferred in names:
            return preferred
    return names[0] if names else -1


@cpu_env.task(report=True, retries=2)
async def render_replay(
    checkpoint: File,
    steps: int = 500,
    command: tuple[float, float, float] = DEFAULT_COMMAND,
    width: int = 480,
    height: int = 360,
    seed: int = 0,
    policy: str = "trained",
) -> File:
    """Roll a policy out once and return the clip plus its stats.

    `policy="random"` ignores the checkpoint's weights and samples actions uniformly
    from [-1, 1]. That is the before-picture: it is what "no training" looks like, and
    without it a viewer has no scale for whether the trained clip is good. It costs one
    extra CPU rollout and it is the difference between a report you can *evaluate* and
    one you can only admire.

    A true random policy rather than an untrained network on purpose: an untrained
    network needs its observation normalizer initialised to the right pytree shape,
    which is fiddly and, when you get it subtly wrong, fails deep inside brax rather
    than at the call site. Uniform actions have no such failure mode and answer the
    same question.
    """
    import functools
    import pickle

    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry

    local = await checkpoint.download()
    with open(local, "rb") as f:
        ckpt = pickle.load(f)

    if policy not in ("trained", "random"):
        raise ValueError(f"policy must be 'trained' or 'random', got {policy!r}")

    preset = ckpt.get("preset", envs.DEFAULT_PRESET)
    # Read the terrain off the CHECKPOINT, never a default: replaying a
    # rough-terrain policy on flat ground (or vice versa) silently compares two
    # different problems. Older checkpoints predate the field, hence the fallback.
    terrain = ckpt.get("terrain", "flat")
    log.info(f"replaying policy={policy} preset={preset} terrain={terrain} ({ckpt.get('num_timesteps', 0):,} steps trained)")

    # Same env, CPU backend. See the module docstring on why this is impl="jax".
    cfg = envs.build_env_config(preset, terrain=terrain)
    cfg.impl = "jax"
    env = registry.load(envs.env_name(terrain), config=cfg)

    if policy == "trained":
        # Rebuild the exact network the checkpoint was trained with.
        # `normalize_observations` was on during training, so the normalizer must be
        # reattached here or the policy sees inputs on a completely different scale
        # and flails.
        net_cfg = ckpt["network_factory"]
        ppo_network = functools.partial(ppo_networks.make_ppo_networks, **net_cfg)(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
        # deterministic=True: replay the policy's mean action, not a sample from it.
        # The stochastic policy is for exploration during training; a demo clip should
        # show what the policy actually believes.
        _policy_fn = jax.jit(
            ppo_networks.make_inference_fn(ppo_network)(ckpt["params"], deterministic=True)
        )

        def act_fn(obs, rng):
            action, _ = _policy_fn(obs, rng)
            return action
    else:
        action_size = env.action_size

        def act_fn(obs, rng):
            return jax.random.uniform(rng, (action_size,), minval=-1.0, maxval=1.0)

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    cmd = jp.array(command, dtype=jp.float32)
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    state.info["command"] = cmd

    trajectory = [state]
    total_reward = 0.0
    for i in range(steps):
        rng, act_rng = jax.random.split(rng)
        action = act_fn(state.obs, act_rng)
        state = step(state, action)
        # Pin the command: the env would otherwise resample it mid-episode.
        state.info["command"] = cmd
        total_reward += float(state.reward)
        trajectory.append(state)
        if float(state.done) > 0.5:
            log.info(f"  episode terminated (fell) at step {i + 1}")
            break

    walked = len(trajectory) - 1
    log.info(f"  {walked} steps, total reward {total_reward:.1f}")

    frames = env.render(trajectory, height=height, width=width, camera=_pick_camera(env.mj_model))
    fps = int(round(1.0 / env.dt))

    label = "Trained policy" if policy == "trained" else "Random policy (before training)"
    caption = (
        f"{label} | preset={preset} | terrain={terrain} | "
        f"command=({command[0]:.1f}, {command[1]:.1f}, {command[2]:.1f}) | "
        f"{walked}/{steps} steps survived | reward {total_reward:.1f}"
    )
    try:
        mp4 = _encode_mp4(frames, fps)
        block = reports.video_html(mp4, caption)
        mp4_b64 = base64.b64encode(mp4).decode()
    except Exception as exc:
        # A failed encode should not lose the run. Fall back to stills.
        log.warning(f"  mp4 encode failed ({exc}); falling back to a frame strip")
        import io as _io

        from PIL import Image

        strip = []
        for frame in frames[:: max(len(frames) // 12, 1)]:
            buf = _io.BytesIO()
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, "JPEG", quality=75)
            strip.append(base64.b64encode(buf.getvalue()).decode())
        block = reports.frame_strip_html(strip, caption)
        mp4_b64 = ""

    await flyte.report.replace.aio(f"<h3>{label}</h3>{block}")
    await flyte.report.flush.aio()

    out = "/tmp/g1_replay.json"
    with open(out, "w") as f:
        json.dump(
            {
                "policy": policy,
                "label": label,
                "preset": preset,
                "terrain": terrain,
                "steps": walked,
                "reward": total_reward,
                "fell": walked < steps,
                "fps": fps,
                "command": list(command),
                "mp4_b64": mp4_b64,
                "html": block,
            },
            f,
        )
    return await File.from_local(out)
