# Track A — OpenAI Realtime API (`gpt-realtime`)

The cloud baseline. One WebSocket, speech in, speech out, no local model at
all. This is the number the local tracks are trying to beat (or at least
get close to) without leaving your machine.

- Model: `gpt-realtime` (GA Realtime API — chosen over the old
  `gpt-4o-realtime-preview`, which OpenAI has been pointing people off of)
- Reported average latency: ~300ms round trip
- Cost: roughly $0.05/min on `gpt-realtime`, ~$0.016/min if you swap the
  `model=` query param for the mini tier (`gpt-realtime-2.1-mini`) — cheaper
  to iterate against while you're just testing the harness

## Setup

```bash
cd topics/voxtral/realtime-voice-showdown/track-a-openai-realtime
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY
```

## Run

```bash
python realtime_client.py
```

Talk into the mic; it should start playing the reply back as soon as audio
starts arriving. Ctrl-C to stop — it'll print the time-to-first-audio for
that turn and append a line to `../results.jsonl`.

## What to double check against current docs

I wrote this from the documented Realtime API event shapes
(`session.update`, `input_audio_buffer.append`, `response.audio.delta`,
etc.), but the API has moved fast — before debugging your own code, diff
this against the current [Realtime API reference](https://platform.openai.com/docs/api-reference/realtime)
for:

- Whether the `OpenAI-Beta: realtime=v1` header is still needed now that
  the API is GA (left it out below since GA endpoints usually don't require
  it — add it back if the connection gets rejected).
- The exact turn-detection config shape (`server_vad` vs newer semantic VAD
  options).
- Audio format defaults — this assumes 24 kHz mono PCM16 in and out.
