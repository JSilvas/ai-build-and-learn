# Track B — Kyutai Moshi (MLX)

The open-weights full-duplex model: user and system audio are modeled on
parallel streams by a single joint speech-text model, no separate
STT → LLM → TTS hop. Kyutai's own numbers put end-to-end latency around
~200ms — this is the one that's actually trying to be "as fast as a phone
call," not just "fast for a pipeline."

Runs on a single Apple Silicon GPU via the `moshi_mlx` port from Kyutai's
own repo — no CUDA needed, which is why it's a fit for the M4 Max.

## Setup

Moshi ships its own local client — this track leans on that directly rather
than reimplementing the streaming protocol, since re-deriving the duplex
framing correctly is exactly the hard part they've already solved.

```bash
cd topics/voxtral/realtime-voice-showdown/track-b-moshi-mlx
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
python -m moshi_mlx.local_web
```

This starts a local web server (defaults to `http://localhost:8998`) with a
browser-based mic/speaker UI — open it in Safari/Chrome and talk. First run
downloads the model weights from Hugging Face (several GB).

**I have not been able to verify the exact current CLI entrypoint/flags** —
`moshi_mlx`'s local-web command has moved around between releases (I've
seen it invoked both as a module and via a `moshi-mlx` console script in
different versions). If `python -m moshi_mlx.local_web` 404s or isn't
found:

1. `pip show moshi_mlx` and check `moshi_mlx --help` / look for a
   `moshi-mlx` console script installed alongside it.
2. Check the "Rust/MLX" section of the
   [kyutai-labs/moshi README](https://github.com/kyutai-labs/moshi) for the
   current invocation — it documents PyTorch, Rust, and MLX clients
   separately and the flags differ between them.

## Measuring latency here

Because the reference client is a browser UI (mic/speaker handled in-page,
not over a script you control), you can't easily hook `common/latency.py`
into it automatically. Two options:

- **Manual stopwatch**: time from when you stop talking to when audio
  starts back, average a handful of turns by hand into `results.md`.
- **Instrument it properly**: if you want a real TTFA number like the other
  two tracks, you'd need to either patch the local-web server to log
  timestamps server-side, or use `moshi_mlx`'s lower-level Python API
  directly (bypassing the web UI) and drive it the way `track-a` drives the
  Realtime API. That's a good "if you have time" extension — check the repo
  for whether it exposes a streaming Python generator you can call directly
  rather than only the packaged web server.

## What to double check

- Model size/download — Moshi's weights are multiple GB; confirm you have
  disk headroom before kicking off the first run.
- Whether the MLX build in `requirements.txt` needs pinning against your
  exact macOS/Python version — MLX wheels are Apple-Silicon-specific and
  occasionally lag new macOS releases.
