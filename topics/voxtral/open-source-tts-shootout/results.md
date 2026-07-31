# Results log

One row per model per prompt category. Real-time factor (RTF) = audio
seconds produced / wall-clock synth seconds — from `common/benchmark.py`.

| Model | Prompt | RTF | Cloning quality (1-5, n/a if unsupported) | Naturalness (1-5) | License OK for your use case? | Notes |
|---|---|---|---|---|---|---|
| Kokoro-82M | narration | | n/a | | Apache-2.0 — yes | |
| Kokoro-82M | expressive | | n/a | | | |
| Chatterbox | narration | | | | MIT — yes | |
| Chatterbox | expressive | | | | | |
| Chatterbox | cloning | | | | | |
| XTTS v2 | narration | | | | Non-commercial — check use case | |
| XTTS v2 | cloning | | | | | |
| XTTS v2 | multilingual | | n/a | | | |
| F5-TTS | narration | | | | Non-commercial — check use case | |
| F5-TTS | cloning | | | | | |
| Orpheus 3B | narration | | | | Apache/MIT-ish — verify per finetune | |
| Orpheus 3B | expressive | | | | | |
| Piper | narration | | n/a | | MIT — yes | |
| Piper | multilingual | | n/a | | | |

## Notes

Free-form observations — which model surprised you, where quality vs. speed
actually trades off, any hardware/dependency friction that mattered more
than the model itself.
