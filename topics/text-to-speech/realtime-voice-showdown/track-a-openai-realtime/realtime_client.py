"""Mic -> gpt-realtime -> speaker, with time-to-first-audio logging.

Untested against a live mic/speaker (written in a headless sandbox) — see
the README's "what to double check" section before assuming a bug in here
over an API drift issue.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from latency import LatencyLog  # noqa: E402

load_dotenv()

API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.environ.get("REALTIME_MODEL", "gpt-realtime")
URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

SAMPLE_RATE = 24_000
CHANNELS = 1
CHUNK_MS = 40
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000


async def mic_to_socket(ws, log: LatencyLog) -> None:
    """Stream mic audio to the server as base64 PCM16 chunks."""
    loop = asyncio.get_event_loop()
    q: asyncio.Queue[bytes] = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
        callback=callback,
    ):
        while True:
            chunk = await q.get()
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )


async def socket_to_speaker(ws, log: LatencyLog) -> None:
    """Read server events, play audio deltas, mark latency timestamps."""
    out_stream = sd.RawOutputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16"
    )
    out_stream.start()
    turn_active = False

    async for raw in ws:
        event = json.loads(raw)
        etype = event.get("type")

        if etype == "input_audio_buffer.speech_stopped":
            log.mark("user_stopped_speaking")
            turn_active = True

        elif etype == "response.audio.delta":
            audio = base64.b64decode(event["delta"])
            if turn_active:
                log.mark("first_audio_out")
                log.finish()
                turn_active = False
            out_stream.write(np.frombuffer(audio, dtype=np.int16))

        elif etype == "response.done":
            pass  # one turn complete; loop continues for the next one

        elif etype == "error":
            print("server error:", event, file=sys.stderr)


async def main() -> None:
    async with websockets.connect(URL, additional_headers={
        "Authorization": f"Bearer {API_KEY}",
    }) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],
                        "voice": "alloy",
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "turn_detection": {"type": "server_vad"},
                    },
                }
            )
        )

        log = LatencyLog("track-a-openai-realtime")
        print("Listening... speak, then pause. Ctrl-C to stop.")
        await asyncio.gather(
            mic_to_socket(ws, log),
            socket_to_speaker(ws, log),
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
