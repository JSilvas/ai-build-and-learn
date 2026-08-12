# World Models with DreamerV3

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event kicks off a run on world models: models that learn an internal representation of how an environment works, then use it to predict what happens next and to plan. It's a natural next step after the generation and RL events, tying both threads together.

We start with [DreamerV3](https://github.com/danijar/dreamerv3), which is the best on-ramp to the idea. It runs on the same MuJoCo physics as the [MuJoCo event](../rl-mujoco) next door, so the comparison between model-free and model-based RL is direct rather than hand-waved.

## What DreamerV3 actually is

Most RL you have seen is **model-free**. PPO, the algorithm behind the walking G1 in
[rl-mujoco](../rl-mujoco), never learns what the environment *is*. It tries things,
sees what scored well, and shifts its policy toward whatever worked. That is why it
needs enormous amounts of experience: the only way to find out what an action does is
to do it, in the real simulator, millions of times.

DreamerV3 is **model-based**. It spends its experience learning a compact model of the
environment's dynamics, and then trains its policy almost entirely *inside that model*,
on imagined rollouts, without touching the simulator. The environment is used to
collect data and to check the model, not to run the millions of trials the policy needs.

That is the whole pitch, and it buys two things:

- **Sample efficiency.** Real environment steps are the expensive resource, whether
  that is a slow simulator or an actual robot. Dreamer replaces most of them with
  imagined ones, which are cheap because they are a forward pass through a small
  neural network rather than a physics engine.
- **Generality.** DreamerV3's headline claim is that one fixed set of hyperparameters
  works across very different domains, from continuous control to Atari to Minecraft,
  which is unusual in RL and is what the "Mastering Diverse Domains through World
  Models" paper is named for.

## How it works

There are two halves, trained together but doing different jobs.

### The world model, an RSSM

The world model is a **Recurrent State-Space Model**. At each timestep it maintains a
latent state in two parts:

- `h`, a **deterministic** recurrent state carried forward by a GRU. This is the
  model's memory of everything that has happened.
- `z`, a **stochastic** state, a small set of discrete categorical variables. This is
  what the model is uncertain about.

Around that sit four learned pieces:

| piece | what it does |
|---|---|
| **encoder** | turns the observation (proprioception, or pixels) into a `z` |
| **dynamics** | predicts the *next* `z` from `h` alone, without seeing the observation |
| **decoder** | reconstructs the observation from `(h, z)` |
| **reward / continue heads** | predict the reward, and whether the episode keeps going |

The interesting one is **dynamics**. It has to guess what the world will look like
next *before* seeing it. Training pushes its prediction and the encoder's posterior
together, and that gap is the KL loss that appears in the logs as `dyn` and `rep`.
They are the same divergence measured with the gradient flowing to different sides:
`dyn` trains the predictor toward the encoder, `rep` trains the encoder toward
something the predictor can actually anticipate. Both are clipped by "free bits" so
the model does not waste capacity driving an already-small KL to zero.

Once dynamics is good, the model can roll forward in latent space on its own. That is
the dream.

### The actor and critic, trained inside the dream

The policy never trains on real trajectories. Instead:

1. Take real states from the replay buffer as starting points.
2. Roll the world model forward ~15 steps in latent space, with the actor choosing
   actions and the reward and continue heads supplying rewards and terminations.
3. Train the critic on the returns of those imagined trajectories, and the actor to
   maximise them.

No decoding to pixels is needed for this, since everything happens in latent space,
which is what makes imagination cheap.

### The tricks that made v3 work

DreamerV2 needed per-domain tuning. V3's contribution is mostly a set of robustness
tricks that let one configuration work everywhere: **symlog** squashing of rewards and
observations so wildly different magnitudes stop mattering, **two-hot** encoding that
turns value regression into classification over a fixed set of bins, **free bits** on
the KL, and normalising returns by a percentile range so the actor's gradient scale is
stable whether rewards are sparse or dense.

### Reading the losses in the report

The report this repo produces shows the model's own loss curves, and they map straight
onto the parts above:

```
dyn, rep              the two sides of the dynamics KL
height, orientations, velocity    decoder heads, one per DMC observation
rew, con              reward and continuation heads
policy, value         actor and critic, trained on imagined rollouts
```

If `rew` falls, the model is learning what earns reward. If `dyn` falls, it is
learning what happens next. Those two are the world model working; `policy` and
`value` are the agent exploiting it.

## Why MuJoCo, and why next to rl-mujoco

DreamerV3's standard continuous-control benchmark is the **DeepMind Control Suite**
(`dm_control`), which runs on MuJoCo. That is the same physics engine as the
[rl-mujoco](../rl-mujoco) event, so this is a genuine like-for-like: same simulator,
same class of locomotion problem, two opposite approaches. PPO there brute-forces
through 4,096 parallel environments; Dreamer here runs a *single* environment and
spends its compute on learning the model instead.

Note the shape of the two workloads is completely different, and it shows up in the
numbers. MJX steps thousands of environments in parallel on the GPU. Dreamer steps one
environment on the CPU and trains a small network on the GPU, so its bottleneck is
policy throughput, not simulation throughput.

## Run it

```bash
cd topics/dreamerv3
./setup.sh                      # venv, upstream checkout, the Blackwell patch
```

On the host:

```bash
export PYTHONPATH=~/dreamerv3
./.venv/bin/python $PYTHONPATH/dreamerv3/main.py \
  --configs dmc_proprio --task dmc_walker_walk \
  --logdir ~/logdir/walker --run.log_every 10
```

In a Flyte pod, against the `world-models` project:

```bash
./.venv/bin/flyte run pipeline.py dream                    # 100k steps
./.venv/bin/flyte run pipeline.py dream --steps 20000      # is the plumbing alive?
```

`dmc_proprio` learns from state, `dmc_vision` learns from pixels. Vision is the more
impressive demo and needs the renderer working inside the pod, which is its own
problem (see the note on driver capabilities below).

Every run ends by filming the trained policy, and that clip lands in the report.
Because the agent here observes proprioception and never sees an image, there is no
video to save by accident: `replay.py` renders the MuJoCo scene *alongside* the policy,
asking dm_control's physics for a camera frame at each step while the agent acts on the
same state vector it trained on. Rendering does not change what the agent observes, so
the clip is an honest recording rather than a separate run.

## Things that cost time, so you do not pay twice

**Upstream's pinned JAX cannot use this GPU.** `requirements.txt` upstream says
`jax[cuda12]==0.4.33`. The GB10 is Blackwell, compute capability sm_121, and CUDA 12
builds of that vintage ship no kernels for it. This uses `jax[cuda13]==0.9.2`, the
version [rl-mujoco](../rl-mujoco) already proved on this box.

**Moving JAX forward breaks `jax.jit`, in exactly six places.** Newer JAX made every
argument after `fun` keyword-only, and upstream still calls it positionally:

```
TypeError: jit() takes from 0 to 1 positional arguments but 5 were given
```

`patches/0001-jax-jit-keyword-only.patch` converts those six call sites to keyword
arguments. No behaviour change. It is applied at image build time against a pinned
upstream commit, with `git apply --check` first so a future bump fails the build
loudly instead of producing an image that dies at agent init.

**The `numpy<2` cap does not apply to us.** Upstream caps numpy below 2 and its own
comment says why: `DMLab: <2, MineRLv1.0: <1.24`. We run neither, and jax 0.9.2
requires numpy>=2, so the cap is both unnecessary and unsatisfiable here.

**`run.log_every` is a wall clock timer, not a step count, and it defaults to
minutes.** A short smoke run finishes before it ever fires, writes an empty logdir,
and looks exactly like a run that did nothing. It does not help that training also
only *starts* once the replay buffer holds `batch_size * batch_length` transitions, so
a very short run legitimately has no losses to show. Pass `--run.log_every 10` when
smoke testing, and do not conclude anything from an empty logdir.

**Do not name a Flyte task parameter `task`.** It collides with the runner's own
argument and the run dies before it starts, with `_Runner.run() got multiple values
for argument 'task'`. This uses `task_id`.

**`PLATFORM` must be a tuple.** flyte hands it to `docker buildx build --platform`, so
a bare `"linux/arm64"` string is iterated character by character into
`l,i,n,u,x,/,a,r,m,6,4` and buildx rejects `"l"` as an unknown operating system.

**Filming needs the envs in process.** embodied's `Driver` defaults to one subprocess
per environment. You can step a subprocess env perfectly well, but you cannot reach its
`physics` object from the parent, so there is nothing to render. `replay.py` builds its
Driver with `parallel=False`.

**The checkpoint directory contains a pointer file, not just checkpoints.** Alongside
the timestamped directories sits a 22-byte `latest` file. Taking the last entry by name
grabs that pointer and dies on `assert exists(path)`. Hand `elements.Checkpoint` the
directory and let it resolve which checkpoint to load.

**Pixel-based runs need graphics driver libraries the devbox does not inject.** The
flyte devbox runs with `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, which is enough
for CUDA training and not enough for any renderer. `dmc_proprio` is unaffected;
`dmc_vision` and any replay video are not. The full diagnosis and the fix are in
[topics/isaac-sim](../isaac-sim/README.md#3-the-video-was-black-and-the-cause-was-the-cluster).

## Status: it dreams

Verified on this box (DGX Spark, GB10, aarch64, driver 580.126.09, CUDA 13.0).

On the host, `dmc_walker_walk`, proprio:

```
jax 0.9.2 + cuda13        devices: [CudaDevice(id=0)]
agent                     640,867 params
                          364,416 dyn | 66,111 val | 57,663 rew | 51,096 dec
                          50,316 pol  | 41,153 con | 10,112 enc
reward loss               5.53 -> 0.58 over 4,736 steps
fps/train                 32,250
```

In a Flyte pod, run `rxhb6r2tk6plkh4l9fl7`, `world-models` project:

```
dmc_walker_walk / dmc_proprio, 20,000 steps, 14.7 min
16 episodes, score 40.28 -> 34.44 (best 50.04)
losses logged: dyn, rep, con, rew, policy, value, repval
               + decoder heads: height, orientations, velocity
replay         500 frames, luminance mean 68.0, 0 black
```

Read those scores as plumbing, not performance. Walker-walk saturates near 950 and
takes on the order of a million environment steps to get there, so 20,000 steps is the
first 2% of the curve. What it demonstrates is that the world model trains, imagines,
and reports inside a pod.

## Where this goes next

- A run to actual convergence, one million steps, and the honest wall-clock cost of it
  on this box.
- `dmc_vision`: learning the world model from pixels, which is where a world model
  gets visually convincing, and which needs the renderer working in the pod.
- **Video of the dream.** The report already films the trained policy in the real
  environment. The far more interesting clip is the model's *imagination*: the decoder
  can reconstruct observations from imagined latent states, so you can render what the
  model thinks will happen next beside what actually happened. That is the single best
  way to show what a world model is, and it is the thing to build next. It needs
  `dmc_vision`, since reconstructing a proprioceptive state vector gives you numbers to
  look at rather than pictures.
- The comparison against [rl-mujoco](../rl-mujoco): sample efficiency (score per
  environment step) rather than wall clock, which is the axis where model-based
  methods are supposed to win.

## Reference

- [DreamerV3](https://github.com/danijar/dreamerv3) (Danijar Hafner), and the paper
  "Mastering Diverse Domains through World Models"
- The original [World Models](https://worldmodels.github.io/) paper (Ha and Schmidhuber)
- [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), the MuJoCo
  task set used here
