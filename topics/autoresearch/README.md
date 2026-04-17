# AutoResearch

[Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — an
autonomous LLM-pretraining research loop. You give an AI agent a small but
real GPT training setup; it edits `train.py`, trains for ~5 minutes, checks
whether `val_bpb` improved, keeps or reverts, and repeats. Wake up the next
morning to a log of experiments and (hopefully) a better model.

This folder wraps the upstream repo four ways:

1. **Mode A — Claude Code (faithful Karpathy)** — open Claude Code in
   `upstream/`, point it at `program.md`, walk away.
2. **Mode B — Headless driver loop** — `python driver.py --tag demo` runs a
   fixed number of iterations from the terminal. Same agent, no UI babysitting.
3. **Mode C — Flyte workflow with TUI reports** — `flyte run --local --tui workflow.py`
   wraps each iteration as a Flyte task so the Flyte TUI shows per-iteration
   change descriptions, val_bpb, status, and log tails as they happen.
4. **Mode D — Local-LLM agent (Ollama)** — `python driver.py --agent local`
   swaps Claude for any local model served by Ollama (default `qwen3-coder-next`,
   easy to swap to Gemma 4, DeepSeek-Coder, etc.). Honest about the gap — small
   models will fail more, and *seeing how they fail* is the demo.

```
topics/autoresearch/
├── README.md                ← you are here
├── requirements.txt         ← deps for the wrapper (driver/workflow), not training
├── driver.py                ← one iteration = propose → train → keep/revert
├── workflow.py              ← Flyte wrapper around driver, renders HTML reports
├── local_agent.py           ← Mode D: Ollama HTTP client + diff sanitize + git apply
├── instructions/
│   ├── karpathy.md          ← default agent brief (copy of upstream/program.md)
│   └── karpathy_verbose.md  ← verbose variant for small local models
└── upstream/                ← cloned karpathy/autoresearch (its own git repo)
    ├── prepare.py
    ├── train.py             ← the file the agent edits
    ├── program.md           ← the agent's brief
    └── pyproject.toml
```

## What is this really doing?

Stripped of the hype, autoresearch is small:

- A fixed training harness (`prepare.py` + a read-only eval).
- One file the agent is allowed to edit (`train.py`).
- One markdown file (`program.md`) telling an off-the-shelf coding agent what
  to do.
- A logbook convention (`results.tsv` + per-experiment git commits).
- A `while true:` loop with permission prompts disabled.

There is no novel agent framework, no clever ML, no new optimization
technique. The agent is Claude (or Codex) in a loop, hitting a metric.

So why does anyone care? Three reasons, in decreasing order of substance:

1. **The constraints are the insight, not the code.** Single editable file +
   read-only eval + fixed 5-minute budget = experiments are *comparable* and
   *uncheatable*. Most "AI does science" demos handwave this and the agent
   ends up gaming the metric or growing the surface area until results aren't
   apples-to-apples. The discipline is the thing — autoresearch is a
   methodology demo dressed up as a tool.

2. **It actually works.** Karpathy's `progress.png` shows monotonic `val_bpb`
   improvement over ~100 iterations. That's the surprising empirical claim:
   with a tight enough sandbox, current frontier models can run closed-loop
   ML research and produce real gains, today, unattended. Not a 2027
   prediction — a thing that runs overnight on one GPU.

3. **`program.md` as the program.** Karpathy's bet is that "research org
   code" lives in markdown, not Python. Today's `program.md` is bare;
   tomorrow's encodes taste, multiple agent roles, branch strategies,
   paper-reading subloops. The repo is a wedge for "you program research
   orgs by writing prompts." Whether that pays off is a different question.

The takeaway most people miss: this isn't really about LLM pretraining.
It's a 700-line lesson in how to bound an agent so it can do useful
closed-loop work — the shape generalizes well past `train.py`.

### Bonus subplot: agents do platform engineering

A side effect we ran into the first time we pointed this at a brand-new GPU
(NVIDIA GB10 / Blackwell on a DGX Spark, CUDA 13): the first ~4 iterations
weren't research at all — they were platform shakedown. The agent
diagnosed and worked around real, current-day kernel issues entirely on its
own:

1. Triton's bundled `ptxas` was CUDA 12.8; GB10 needs 13.0 →
   set `TRITON_PTXAS_PATH` to the system binary. Commit, retry.
2. `kernels-community/flash-attn3` ships no prebuilt `sm_121a` kernel image →
   the agent swapped FA3 for `flex_attention`. Commit, retry.
3. `flex_attention` ran but at **1.21% MFU** (vs ~40% on H100) — the agent
   correctly diagnosed kernel overhead as the bottleneck and tried SDPA next.

This is the *same* read-log → form-hypothesis → make-one-change → measure →
keep-or-revert loop that program.md describes for ML research, applied to
hardware compatibility instead. The bounded-loop methodology is hardware
agnostic — anywhere you have a tight read/measure/decide cycle, this shape
works. Watching it debug a GPU you yourself don't fully understand is, frankly,
a more compelling demo than watching it tune `n_embd`.

…and then it actually did the research. Once the kernel situation
stabilized, the loop took val_bpb from **1.819 → 1.395** (–23% relative)
across six kept commits, plus a clean discard that proved the
reject-and-revert path works:

| # | commit  | val_bpb  | VRAM   | status   | description                                          |
|---|---------|----------|--------|----------|------------------------------------------------------|
| 0 | 567db9f | 1.818913 | 44.0 G | keep     | baseline (FA3 → flex_attention for sm_121a compat)   |
| 1 | e1181b8 | 1.821681 | 44.0 G | keep     | SDPA instead of flex, drop sliding window — *kept despite a +0.003 regression because the code was simpler and SDPA is faster downstream* |
| 2 | 9dfa100 | 1.793186 | 87.2 G | keep     | DEVICE_BATCH_SIZE 128→256 (grad_accum 2→1)           |
| 3 | 3b9d79d | **1.467645** | 27.9 G | keep     | DEPTH 8→4 (50M → 11.5M params; **110 steps vs 39**) |
| 4 | 4d93489 | 1.424601 | 13.1 G | keep     | DEPTH 4→2 (3.5M params; 247 steps) — kept exploring "smaller = more steps wins" |
| 5 | 29c26a9 | **1.395246** | 23.2 G | keep     | DEPTH 2→3 (10.7M params; 132 steps) — backed off, found the sweet spot |
| 6 | 5a815c4 | 1.503361 | 30.8 G | **discard** | ASPECT_RATIO 64→128 (17.9M params, only 96 steps) — wider model lost the step-budget tradeoff, reverted |

Three things worth pointing out:

1. **The biggest single jump** (–0.32 val_bpb at iter 3) came from
   *shrinking* the model. The agent figured out that on GB10's
   throughput budget the model was severely undertrained (only 39 steps
   in 5 minutes), so a smaller model that takes more steps wins over a
   bigger one that barely moves. Karpathy notes this exact tradeoff in
   the upstream README's "Platform support" hints, but the agent
   rediscovered it empirically from the run logs without being told.

2. **Iters 4–5 are a clean two-step hill climb.** Iter 4 pushed the
   smaller-is-better hypothesis further (DEPTH 4→2: still wins). Iter 5
   *backed off* (2→3) and discovered the local optimum sat between
   them. That's binary-search behaviour over a single hyperparameter,
   executed across two commits — exactly what a careful human would do,
   except the agent didn't need a Jupyter notebook to run the sweep.

3. **Iter 6 is the first discard** — proves the reject-and-revert path
   isn't theoretical. Widening (`ASPECT_RATIO 64→128`) lost the same
   step-budget tradeoff the agent had just been winning on the depth
   axis. Run logged, branch reset, kept val_bpb unchanged. Without this
   guardrail the loop drifts; with it, the metric is monotonic.

The val_bpb / memory / status / description quad in `results.tsv` does
the narrative work — you can read the agent's reasoning in 7 rows
without ever opening a diff.

## Hardware

Karpathy's defaults assume a single H100. The DGX Spark (GB10) works, but
expect different absolute val_bpb numbers — the 5-minute budget is fair within
your machine, not across machines. If you're on smaller hardware, check the
**Platform support** section of `upstream/README.md` for tuning hints
(`DEPTH`, `MAX_SEQ_LEN`, `WINDOW_PATTERN=L`, `TOTAL_BATCH_SIZE`).

## One-time setup

Same convention as the other build-learn topics: one shared venv at the repo
root, install per-topic `requirements.txt` into it. The training stack is
bundled into our requirements (mirrored from `upstream/pyproject.toml`) so you
don't need a second venv inside `upstream/`.

```bash
# From the build-learn repo root:
uv venv .venv --python 3.11
source .venv/bin/activate

# --index-strategy lets uv fall back to PyPI for packages that also exist on
# the PyTorch cu128 index (e.g. requests). Without it, uv refuses to mix
# indexes for safety reasons.
uv pip install --index-strategy unsafe-best-match -r topics/autoresearch/requirements.txt

# Clone karpathy's repo as upstream/ (gitignored from build-learn).
cd topics/autoresearch
git clone https://github.com/karpathy/autoresearch.git upstream

# One-time data prep + a baseline sanity run (each ~2–5 min).
cd upstream
python prepare.py        # downloads data + trains BPE tokenizer
python train.py          # one baseline training run
cd ..
```

You also need `claude` (Claude Code CLI) on PATH for Modes B and C.

> **Alternative:** if you'd rather use karpathy's isolated `uv sync` env
> (creates `upstream/.venv` from `upstream/pyproject.toml`), do
> `cd upstream && uv sync && uv run prepare.py && uv run train.py`. The driver
> calls `python train.py` against whichever venv is active, so just activate
> `upstream/.venv` first if you go that route.

## Mode A — Claude Code (no driver, no Flyte)

The bare karpathy flow. From inside `upstream/`:

```bash
cd upstream
claude --permission-mode bypassPermissions
```

Then prompt:

> Hi, have a look at program.md and let's kick off a new experiment. Let's do
> the setup first.

The agent creates a branch, runs the baseline, and starts looping until you
stop it. Best for an unattended overnight run.

## Mode B — Headless driver loop

`driver.py` runs a fixed number of iterations from the terminal. One iteration
= ask Claude to make ONE edit + commit, run training, parse val_bpb,
keep-or-revert, append to `results.tsv`.

```bash
# Default: Claude Sonnet, the karpathy.md instructions
python driver.py --tag demo --iterations 3

# Pick a different Claude model
python driver.py --tag opus-test --iterations 3 --model opus

# Use a custom instructions file (e.g. one focused on optimizer tuning)
python driver.py --tag opt-focus --iterations 3 \
    --instructions instructions/my_optimizer_brief.md
```

Useful when you want a finite run with no UI, e.g. a quick demo or a CI smoke.

## Mode C — Flyte workflow with TUI reports

Same loop, but each iteration is a Flyte task with a live HTML report. Run
locally so you don't need a cluster. The `--tui` flag is what actually pops
the live status panel — without it, Flyte just runs the workflow and prints
to stdout:

```bash
flyte run --local --tui workflow.py run_autoresearch --tag demo --iterations 3
```

The Flyte TUI shows:

- A node per iteration (baseline + N agent iterations) executing sequentially.
- Per-iteration HTML report: change description, status badge
  (BASELINE / KEEP / DISCARD / CRASH), val_bpb, peak VRAM, log tail.
- A rolling summary table on the workflow node that updates after each
  iteration completes.

Why Flyte for what is essentially a sequential loop? Two reasons that matter
for a stream demo: (1) the TUI gives clean per-step visibility without writing
your own UI; (2) the same workflow can be deployed to a remote Flyte cluster
later without changing the iteration code.

## Mode D — Local-LLM agent (Ollama)

Swap Claude for any model served by Ollama. Useful for: keeping the loop
fully offline, comparing how different small open models reason about ML
research ("Qwen vs Gemma 4 vs DeepSeek bake-off"), or testing how much of
the demo is the model vs the methodology.

### One-time Ollama setup

```bash
# 1. Install Ollama (if not already): https://ollama.com/download
# 2. Start the server (default port 11434)
ollama serve &

# 3. Pull the default model — qwen3-coder-next is the strongest open coder
#    that fits on a single GPU and the one that worked well in our auto-rel
#    pipeline. ~50GB download, one-time.
ollama pull qwen3-coder-next
```

### Keep the model loaded (important!)

By default, Ollama **evicts the model from GPU** after 5 minutes of
inactivity. Since autoresearch alternates between agent calls (model on
GPU) and training runs (PyTorch on GPU), the model gets evicted every
iteration — and reloading a 31B model from disk takes ~4.5 minutes. This
turns every iteration into a 10+ minute affair and causes timeouts.

Fix: tell Ollama to keep the model loaded indefinitely. No restart needed:

```bash
# Tell Ollama to never unload gemma4 (works on the already-running server)
curl -s http://localhost:11434/api/generate -d '{"model":"gemma4:31b","keep_alive":-1}'

# Warm the model (first load takes ~4 min, subsequent calls are instant)
ollama run gemma4:31b "say hello"
```

The DGX Spark has 128 GB unified VRAM — more than enough for both the
local model (~17 GB for gemma4:31b) and training (~23 GB) to coexist.
After warming, each agent call drops from ~5 minutes to ~10 seconds.

If Ollama is installed as a system service (common), you can't `pkill`
it without sudo. Use `sudo systemctl stop ollama` first if you need to
restart it, or just use the `keep_alive` curl above on the running
instance.

To apply keep-alive globally (all models, all sessions), restart with:

```bash
sudo systemctl stop ollama
OLLAMA_KEEP_ALIVE=-1 ollama serve &
```

### Run with the local agent

Same `driver.py` and `workflow.py`, just add `--agent local`:

```bash
# Headless (Mode B-style) with the default local model
python driver.py --tag local-demo --iterations 3 --agent local

# Headless with a specific model
python driver.py --tag gemma4-test --iterations 3 \
    --agent local --model gemma4:something

# Flyte TUI version (Mode C-style) with local agent — note the --tui flag
flyte run --local --tui workflow.py run_autoresearch \
    --tag local-flyte --iterations 3 --agent local
```

### What's different from Mode B/C

- **Different prompt** — `instructions/karpathy_verbose.md` is loaded by
  default for `--agent local`. Small models need explicit edit format,
  concrete examples, and a "common mistakes" list. Override with
  `--instructions <path>` for a different brief.
- **Single-shot, no tool use** — the local model gets ONE shot per iteration
  to output one or more `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` blocks
  (aider-style). We do literal substitution against `train.py`. If any block
  has no unique match or the model's output is malformed, the iteration is
  logged as `discard` with a `[TAG]` describing the failure, and the loop
  moves on. No retry, no reflection (see "Why no reflection?" below).
- **Why search/replace, not unified diff** — first attempt used `git apply`
  on a unified diff. Result on `qwen3-coder-next`: **0/3 iterations applied**
  ("corrupt patch", "patch failed at line 599"). The *reasoning* in the
  descriptions was correct ("reduce DEPTH from 3 to 2"), but small models
  can't reliably compute `@@ -line,count @@` headers against a 700-line file.
  Switching to search/replace got 3/3 iterations to apply on the next test.
  This is also why aider, Cursor, and similar tools use this format.
- **No `bypassPermissions` worry** — the local agent doesn't have general
  shell or filesystem access. It outputs text; we validate and substitute.
- **Configurable via env vars** — `OLLAMA_URL` (default
  `http://localhost:11434`), `AUTORESEARCH_LOCAL_MODEL` (default
  `qwen3-coder-next`).

### Honest expectations

Small local models are noticeably worse than Sonnet at this task. Expect:

- **Higher discard rate** — both from "experiment was a bad idea" and from
  output-format failures (e.g. `[SEARCH NOT FOUND]` when the model retypes
  context from memory instead of copying it).
- **Limited cross-iteration coherence** — the model receives the full
  results history each iteration, but small models tend not to use it well.
  In our first qwen3-coder-next run on a depth=3 baseline, the model tried
  depth 3→2 *twice* in 3 iterations even though the first attempt was
  already in the discard log. Claude doesn't do this.
- **Some genuinely interesting moves anyway** — especially when the local
  optimum is non-obvious. In the same test, the model independently
  rediscovered that depth=3 is a local optimum on GB10 (tested both 2 and
  4, both worse) — same conclusion the Claude run reached, just via
  brute-force exploration instead of binary search.
- **Failure modes that *differ* from Claude's** — which is the demo.

### Why no reflection / retry?

Auto-rel (the cousin project this borrows from) uses up to 10 fix
iterations per project because the project must work. Autoresearch is the
opposite — failure is normal, expected, and informative. Karpathy's
program.md literally says *"If the idea itself is fundamentally broken,
just skip it, log 'crash', and move on."* Adding reflection would mask
the local model's failure modes, which are the most interesting part of
the comparison.

### Reading failure tags in results.tsv

When the local agent fails to produce a valid edit, the iteration is
logged as `discard` and the description gets a leading `[TAG]` so you can
diagnose what went wrong without opening the run log:

| Tag | Meaning |
|-----|---------|
| `[OLLAMA UNREACHABLE]`  | Couldn't reach `OLLAMA_URL`. Check `ollama serve` is running. |
| `[INVALID OUTPUT: no SEARCH/REPLACE blocks]` | Model returned text but no parseable blocks. Often the model emitted prose instead of following the format. |
| `[SEARCH NOT FOUND ...]` | Model produced a SEARCH block whose text doesn't appear in `train.py` — usually because the model retyped from memory instead of copying exactly (whitespace, indentation drift). |
| `[AMBIGUOUS MATCH ...]`  | SEARCH text appears multiple times in `train.py`. Model picked too-short context; needs a unique surrounding line. |
| `[git add failed: …]` / `[git commit failed: …]` | Edit applied but git balked. Usually means the model touched something it shouldn't have. |

Same `[TIMEOUT]` and `[crash]` tags from Mode B/C also apply when
training itself misbehaves — those are agent-agnostic.

## Customizing the agent's brief (instructions/)

`instructions/` is the customization surface — drop in a new markdown
file and point the driver at it with `--instructions path/to/yours.md`.
Two ship by default:

- `karpathy.md` — verbatim copy of `upstream/program.md`. Default for `--agent claude`.
- `karpathy_verbose.md` — adds explicit diff-format rules, concrete edit
  examples, and a "common mistakes" list. Default for `--agent local`.

Things you might want a custom brief for:
- Steering research direction ("focus on optimizer tweaks only")
- Demoing program.md as the real programming surface (run two iterations
  of the loop with two different briefs, compare arcs)
- Adapting the loop to a non-pretraining target (e.g. shrink an existing
  model to match throughput)

The brief is the actual lever — same code, different markdown, different
research behavior. That's the point karpathy is making with program.md.

## Where results land

- `upstream/results.tsv` — the canonical log
  (`commit  val_bpb  memory_gb  status  description`). Untracked by upstream's
  git on purpose — it's a per-machine artifact.
- `upstream/run.log` — last training run's stdout/stderr.
- Git history on branch `autoresearch/<tag>` inside `upstream/` — one commit
  per kept experiment. `git log --oneline autoresearch/<tag>` is the human
  view of what stuck.

## Resetting a run

```bash
cd upstream
git checkout master
git branch -D autoresearch/<tag>     # nuke the experiment branch
rm -f results.tsv run.log
```

## Notes

- `upstream/` is its own git repo — agent commits never pollute `build-learn`.
- The driver passes the recent `results.tsv` rows to the agent each iteration
  so the agent has memory of what's been tried, even though each `claude -p`
  call is a fresh session.
- Hard caps in `driver.py`: agent has 5 min to propose, training has 10 min
  before it's killed and counted as a crash.
