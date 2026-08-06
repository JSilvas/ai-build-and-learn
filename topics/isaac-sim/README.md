# Isaac Sim on a DGX Spark: getting the simulator to exist

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This picks up where the [MuJoCo event](../rl-mujoco) left off. There we got a Unitree G1 walking with MJX + Brax PPO at 4,096 parallel environments on one GPU. Here we step up to [NVIDIA Isaac Sim](https://github.com/isaac-sim/IsaacSim), the GPU-accelerated, photorealistic robotics simulator built on Omniverse, and [Isaac Lab](https://github.com/isaac-sim/IsaacLab), the RL framework on top of it.

The pitch for Isaac Sim over MuJoCo is sim-to-real. MuJoCo gives you fast, accurate rigid-body physics and a rendering story that is fine for watching a gait. Isaac Sim gives you ray-traced sensors, USD scenes, articulated robots with real actuator models, and domain randomization built for transferring a policy onto actual hardware. You pay for it in weight.

**This event is about the weight.** Before any of the interesting parts, Isaac Sim has to run on this box, and on a DGX Spark that is genuinely non-trivial: there are no aarch64 wheels, so `pip install isaacsim` (which every x86 tutorial opens with) does not exist here. The only supported path on Grace-Blackwell is a source build. That is what this directory automates and verifies.

## The Spark-specific problem

| | x86 workstation | DGX Spark (GB10) |
|---|---|---|
| Install | `pip install isaacsim[all]` | no wheel exists, build from source |
| Compiler | whatever | **gcc-11**, not the gcc-13 DGX OS 7 ships |
| OpenMP | just works | must preload the system `libgomp`, or Kit dies |
| Disk | a few GB | ~50 GB of build artifacts |
| Build time | none | 10 to 15 min compile |

Three of those five rows are silent failures. The compiler one fails loudly. The libgomp one fails about twenty seconds into startup with a symbol lookup error that looks like a broken build but is not: PyTorch's aarch64 wheels bundle their own OpenMP runtime, Isaac Sim's native extensions want the system one, and whichever the loader sees first wins the whole process. `env.sh` fixes that in one line and explains why.

## Layout

There are two ways to run Isaac Sim here, and both are wired up:

```
bare metal on the host           in a Flyte pod
─────────────────────            ──────────────
setup.sh    builds from source   Dockerfile   FROM nvcr.io/nvidia/isaac-sim
env.sh      source before use    config.py    the Flyte image + task envs
smoke_test.py  a shim over ↓     pipeline.py  `flyte run pipeline.py smoke`
              checks.py  ← the physics, shared by both, so the numbers compare
```

The source build is the fast iteration loop: edit, rerun, no image rebuild. The container is what ships, and it is the one that proves the GPU reaches a pod. `checks.py` is deliberately the only place the physics lives, because two paths measuring different things would make the comparison worthless.

Nothing installs into this repo. Both trees go to `$ISAAC_ROOT` (default `~/isaac`), because 50 GB of build output has no business inside git. The local `.venv` here holds `flyte` only; there is no `isaacsim` in it and there cannot be.

## Run it

```bash
cd topics/isaac-sim

# Stage 1. Needs sudo and a human: it installs gcc-11 and shows you the
# NVIDIA Omniverse license, which you accept yourself.
./setup.sh prereqs

# Stage 2 and 3. No prompts, so let them run.
./setup.sh isaacsim     # clone + build, 10-15 min
./setup.sh isaaclab     # clone + pip install rsl_rl / rl_games / skrl / sb3

# Did it work?
source env.sh
$ISAACSIM_PYTHON_EXE smoke_test.py
```

And in a pod, against the `physical-ai` project:

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
./.venv/bin/flyte run pipeline.py smoke
```

The same four checks run inside `nvcr.io/nvidia/isaac-sim:6.0.0` under Flyte's device plugin, and land in a Flyte report. No source build needed for this path.

`smoke_test.py` checks four things, in the order they actually fail:

1. **Kit boots headless.** Catches libgomp, missing X11 headers, a bad build.
2. **PhysX is on the GB10.** This is the one worth having. PhysX falling back to CPU does not raise; it just runs, quietly, roughly 50x slower. Every parallel-env benchmark you take afterwards is then meaningless, and you have no idea.
3. **Gravity is real.** A cube dropped from 2 m falls, and falls no further than `0.5*g*t^2` allows. Cheap, but it separates "Kit started" from "physics is actually stepping".
4. **Throughput.** Physics steps/sec, so there is a number to compare against when something later feels slow.

Exit code is 0 only if all four pass, so it drops into a shell script.

## Status: it runs

Verified end to end on this box (DGX Spark, GB10, DGX OS 7.2.3 / Ubuntu 24.04, driver 580.126.09, CUDA 13.0, aarch64) on 2026-08-05. Every number below is measured, not quoted.

```
Isaac Sim 6.0.1
  clone (shallow + submodules + LFS)   1.3 GB, ~90s
  build.sh                             460s  (BUILD (RELEASE) SUCCEEDED)
  built tree                           11 GB
  disk consumed incl. packman cache    ~36 GB
  headless Kit startup                 ~7s

smoke_test.py                          4/4 passed
  kit boots        SimulationApp came up headless
  physx on gpu     physics device = cuda:0
  gravity is real  cube fell 1.900 m from 2.00 m in 240 steps
  throughput       719 physics steps/sec
```

Warp reports the GPU as `"cuda:0" : "NVIDIA GB10" (120 GiB, sm_121, mempool enabled)`, against CUDA Toolkit 12.9 with driver 13.0.

And the same four checks in a Flyte pod, run `r5fskgrgkkchgzkvnbsk`, both pods Completed in about 4 minutes:

```
isaac_smoke: 4/4 passed on NVIDIA GB10 | arch sm_121 | 120 GiB
  kit boots        SimulationApp came up headless
  physx on gpu     physics device = cuda:0
  gravity is real  cube fell 1.900 m from 2.00 m in 240 steps
  throughput       338 physics steps/sec
```

Identical fall distance to the millimetre, and PhysX gets `cuda:0` through the device plugin. That is the result worth having: **the GPU reaches the pod, and the container simulates.** Throughput is lower in the pod (338 vs 430-720 on the host) but the host figure swings by 40% run to run at this scale, so with one rigid body the honest reading is "same order, don't infer a container penalty from it". A real comparison needs thousands of environments, which is Isaac Lab's job.

That 719 steps/sec is **one** rigid body paying full Kit overhead per step, so it is a floor and a regression canary, not a benchmark. The number that matters for RL is thousands of environments stepped together, which is Isaac Lab's job and the next section's problem.

`gravity is real` earns its place: the cube starts at 2.00 m, is 0.2 m on a side, and comes to rest with its centre at 0.10 m. 1.900 m is exactly right, and it is the difference between "Kit started" and "physics is actually stepping".

Host prerequisites confirmed **missing** out of the box, all installed by `setup.sh prereqs`: gcc-11 (DGX OS ships 13.3), git-lfs, `python3.12-dev`, and all six X11/GL dev packages. Already present: CUDA 13.0 and `libgomp.so.1`.

## Things that cost time, so you do not pay twice

**`sim_app.close()` never returns.** Kit runs with `--/app/fastShutdown=True`, so `close()` calls `os._exit()` internally. Anything after it is dead code that silently does not run. Worse, it exits **0** regardless: the first version of this smoke test died on an `AttributeError` mid-scene and still reported success. The fix is `sim_app.close(exit_code=...)`, which is the supported way to exit nonzero, and printing your summary *before* the call.

**`isaacsim.core.api` object kwargs want `np.array`, not lists.** `position`, `scale` and `color` are handed to USD via `.tolist()`, so a plain list dies with `'list' object has no attribute 'tolist'` four frames deep inside `PreviewSurface`. The signatures say `Optional[np.ndarray]` and they mean it.

**`isaacsim.core.api` ships under `extsDeprecated/` in 6.0.** It works, and it is still what every tutorial uses, so it is what this smoke-tests against. `isaacsim.core.experimental` is the replacement when it goes.

**`GLFW initialization failed` on startup is noise.** Headless has no display; Kit logs it three times as a warning and carries on.

**No `pip install isaacsim` on aarch64.** Worth repeating, because it is the first thing every guide tells you to do and it will send you looking for a broken index.

## Getting it into a Flyte pod

Four things bite, in this order. All four are commented at the site in `Dockerfile`, `config.py` and `pipeline.py`.

**1. The container has no system python.** Not an old one, none: `which python3` returns nothing. Kit's embedded CPython 3.12 at `/isaac-sim/kit/python/bin/python3` is the only interpreter, it is not on `PATH`, and running it directly does *not* find `isaacsim`. `python.sh` is just a wrapper that exports `PYTHONPATH`, `LD_LIBRARY_PATH`, `LD_PRELOAD=libcarb.so`, `CARB_APP_PATH`, `EXP_PATH` and `ISAAC_PATH` and then execs that same binary. Bake those into the image and Flyte's own process can `import isaacsim` directly. Extract them from the container rather than reading the script:

```bash
docker run --rm --entrypoint bash nvcr.io/nvidia/isaac-sim:6.0.0 -c \
  '/isaac-sim/python.sh -c "import os; print(os.environ[\"PYTHONPATH\"])"'
```

**2. `from_base()` cannot build this image on arm64.** The obvious spelling fails twice. First `Cannot add additional layers to a non-extendable image`, fixed by `.clone(..., extendable=True)`. Then the kubelet rejects the result with `no match for platform in manifest sha256:...: not found`, which does not say "wrong architecture" but means exactly that: `from_base()` leaves platform at the dataclass default `linux/amd64`, flyte 2.2.1's `clone()` takes no `platform`, and `dataclasses.replace` is refused by `Image.__post_init__`. `from_dockerfile()` is the only 2.2.1 constructor taking both an arbitrary base and a platform, which is why there is a Dockerfile here.

**3. The base image's ENTRYPOINT hijacks the task.** Flyte serialises tasks with `command=[]` and `args=["a0", ...]`, so the *image* entrypoint runs. Isaac Sim's is `/isaac-sim/runheadless.sh`, which would be handed Flyte's args and do something surreal. Clear it: `ENTRYPOINT []`, matching Flyte's own base. `a0` is a console script from the flyte pip install, so it lands on `PATH` automatically once Kit's python bin is there.

**4. Kit and Flyte fight over the asyncio event loop, and this is the subtle one.** `SimulationApp.close()` has two modes and in a Flyte pod both are wrong:

| | what it does | in a pod |
|---|---|---|
| `fast_shutdown=True` (default) | calls `os._exit()` internally | process vanishes before Flyte records a result; run looks like an unexplained pod exit |
| `fast_shutdown=False` | cancels every asyncio task to shut down gracefully | cancels **Flyte's own** `load_and_run_task()` and `Controller.watch_for_errors()` |

The second failure is worth seeing, because nothing about it says "Isaac Sim did this":

```
Cancelling <Task pending name='Task-26' coro=<load_and_run_task() ... entrypoints.py:244>>
Cancelling <Task pending name='Task-25' coro=<Controller.watch_for_errors() ... _core.py:143>>
RuntimeError: Cannot enter into task ... while another task is being executed
```

That pod sat `Running` for 28 minutes holding the GPU until it was aborted by hand.

The fix is to run the simulator in a **child process**, which gives Kit its own event loop to tear down however it likes while Flyte's is never touched. Note this is not about imports: `flyte` and `isaacsim` coexist in one interpreter perfectly well. It is purely about who owns the loop at shutdown. It also happens to be the shape Isaac Lab wants, since its RL entry points are scripts you invoke rather than libraries you drive.

One consequence worth knowing: Flyte only bundles modules the task module imports **at top level**. `smoke_test.py` cannot be imported (it builds a `SimulationApp` at import time), so it would never reach the pod. That is why the runner lives in `checks.py`, which `pipeline.py` imports and then spawns by path.

**pip will warn, and you can ignore it.** flyte's pydantic (2.13.4) and typing_extensions (4.16.0) violate `isaaclab`'s declared pins (`<2.12`, `==4.12.2`). Measured, not assumed: `isaacsim` imports, `isaaclab` enumerates its full task registry, and the physics checks still pass 4/4. Those pins are metadata, not runtime constraints. Revisit if RL training misbehaves.

## Where this goes next

Once the simulator exists, the actual event is the comparison the MuJoCo run sets up:

- Isaac Lab RL on the same class of problem (humanoid or quadruped locomotion), against the MJX + Brax numbers from [rl-mujoco](../rl-mujoco): 59,757 env-steps/sec at 2,048 envs.
- Where the extra fidelity earns its cost, and where it does not. A quadruped that walks in MuJoCo is not obviously better served by ray tracing.
- Sim-to-real: domain randomization, actuator models, and what "transfer" means when you do not have the robot.

## Reference

- [Isaac Sim](https://github.com/isaac-sim/IsaacSim) and [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [NVIDIA DGX Spark playbook for Isaac](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/isaac), the source of the build recipe
- [Arm learning path: Isaac Sim + Isaac Lab on DGX Spark](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_isaac_robotics/1_introduction_isaac/)
- RL libraries Isaac Lab wires up: RSL-RL, rl_games, skrl, Stable-Baselines3
- Physics: PhysX 5, and the newer Newton engine
