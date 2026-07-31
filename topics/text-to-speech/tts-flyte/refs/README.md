# Reference voices

One `.wav` plus one `.txt` per voice, same stem: `sage.wav` + `sage.txt`. The `.txt` is
the **exact** transcript of the wav, and it is load-bearing, not documentation: Qwen,
Dia and CSM all condition on it, so three of the five cloners degrade if it drifts from
what was actually said. That is why the reference is a fixed script read aloud rather
than a found clip that gets Whisper-transcribed: an ASR error in the reference silently
becomes a worse clone in every model that reads it.

## Recording one

Read `sage.txt` (or write your own and save both files) and save as:

- **mono WAV**, 24kHz or higher
- **8-15 seconds.** Every model here accepts 3s, all of them clone better at 8-15s,
  and most only read the first ~15s, so past ~30s you are storing audio nobody uses.
- No music, no room echo, no background noise. The models clone the *recording* as much
  as the voice: reverb in the reference comes back as reverb in every clip.
- Do not normalize to full scale. A clipped reference distorts the cloned timbre, and
  `RefVoice.warnings()` will call it out in the report.

The pipeline surfaces these as warnings on the report's reference card rather than
failing, so a marginal clip still produces a run you can look at.

## The wavs are gitignored on purpose

`.gitignore` here excludes `*.wav`. A reference clip is a **voiceprint**: it is the
exact input someone else would need to clone that voice with this same pipeline, and
committing one to a public repo hands it over permanently, in a repo whose whole point
is demonstrating how easy the cloning is. The transcripts are committed (they are just
sentences), the audio is not.

If you deliberately want a shareable reference so the demo runs for other people, use a
public-domain clip with a clear license (LibriSpeech, or a Common Voice sample) and
force-add it with a note about where it came from. Do not commit anyone's voice,
including your own, without deciding you meant to.

## `librispeech.wav`, the shipped stand-in

So the pipeline runs for someone who just cloned the repo, and so a run does not need
anyone's real voiceprint to be green.

- **Source**: LibriSpeech `dev-clean`, speaker 1272, chapter 128104, utterances 0000 and
  0001, concatenated with a 0.2s gap. Fetched from the `hf-internal-testing/
  librispeech_asr_dummy` mirror of the corpus.
- **License**: LibriSpeech is CC BY 4.0, derived from public-domain LibriVox recordings.
- **Transcript**: `librispeech.txt` is the corpus's own transcript, which is why this
  clip is a good stand-in specifically: the words are verified, not ASR-guessed, which is
  the whole reason this directory pairs wavs with hand-written text.
  Casing and punctuation were restored (the corpus stores transcripts uppercased and
  unpunctuated); **not one word was changed**.
- **10.9s, mono, peak 0.62** — inside the 8-15s window and nowhere near clipping.

One caveat to read the numbers with: it is **16kHz**, because that is the rate
LibriSpeech was recorded at. Every model here takes it (`RefVoice.at()` resamples to
whatever each one wants) and the speaker scorer runs at 16kHz anyway, but upsampling
invents no detail, so a clone off this clip has less high-frequency material to work with
than one off a 24kHz recording of your own voice. Fine as a control, not the best case.

The `*.wav` rule above still hides it, so committing it takes a deliberate
`git add -f refs/librispeech.wav`. Do that: this one is a published corpus recording of
someone who consented to exactly that, not a voiceprint anyone here has a claim to
protect, and force-adding it is what makes the demo runnable for someone who just cloned
the repo. To rebuild it instead of committing it, the recipe is the source note above:
rows 0 and 1 of `hf-internal-testing/librispeech_asr_dummy`, concatenated with a 0.2s gap.
