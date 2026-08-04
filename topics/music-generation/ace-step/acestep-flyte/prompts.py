"""Prompts for the ACE-Step comparison.

The TTS demo's scripts each targeted a failure mode you can only *hear*. Same idea
here, one level up: each brief targets something that separates a model which
assembles a plausible loop from one that writes a piece of music.

Every brief stresses one axis:

  - **Dense production.** A busy electronic arrangement. The baseline: does it hold
    together, is the low end controlled, is there a stereo image at all? Dense
    material is forgiving, which is why it goes first and why passing it means little.
  - **Exposure.** A sparse acoustic duo. The opposite test: nothing to hide behind, so
    every artifact is naked. Solo instruments are where music models smear, where
    plucks lose their attack, and where a "warm room" turns into a codec swirl.
  - **Vocals and words.** Lyrics with real structure tags. Two separate questions
    stacked: can you make out the words, and does the [verse]/[chorus] structure land
    where the tags say it should. Intelligible singing is the hardest thing in open
    music generation and the clearest gap to the commercial tools.
  - **Prompt adherence.** A brief that names specific, unusual instruments and a
    specific feel. Adherence is where guidance scale earns its keep, so this is the
    brief to pair with the `guidance` sweep.
  - **Multilingual.** The same song in a non-English language. The model claims 50+;
    this is where you find out whether that means "sings it" or "sings something
    vaguely shaped like it".
  - **Structure over time.** An explicit arrangement request (intro, build, drop). The
    thing short clips cannot show and the reason the `duration` sweep exists.

Ordered so a truncated run still yields a usable report, and the first card is the
one that always works.

── On the lyrics format ─────────────────────────────────────────────────────────
ACE-Step reads structure tags in the lyrics: `[verse]`, `[chorus]`, `[bridge]`,
`[intro]`, `[outro]`. They are not decoration, they are conditioning, and dropping
them is a good way to get 60 seconds of undifferentiated verse. An empty `lyrics`
string is the correct way to ask for an instrumental; do not write "instrumental" in
the lyrics field, the model will try to sing it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Brief:
    """One musical brief: the style prompt, optional lyrics, and the hypothesis."""
    key: str
    prompt: str          # the style caption: genre, mood, instrumentation, production
    lyrics: str = ""     # "" = instrumental
    axis: str = ""       # the capability under test
    listen_for: str = ""
    bpm: int = 0         # 0 = let the model choose
    keyscale: str = ""
    language: str = "en"


SUITE: list[Brief] = [
    Brief(
        key="synthwave",
        prompt="driving synthwave instrumental, analog saturated bass, gated reverb "
               "drums, bright arpeggiated lead, wide stereo pads, night-drive energy, "
               "clean modern mastering",
        axis="dense production (the control)",
        listen_for="Our baseline. Everything here is dense enough to hide small "
                   "mistakes, so a model failing this fails everything below. Judge the "
                   "low end (is the bass one note or a line?), the stereo width, and "
                   "whether the arpeggio stays in time with the kick for the whole clip "
                   "rather than drifting after the first eight bars.",
        bpm=118,
    ),
    Brief(
        key="synthwave-vocal",
        prompt="driving synthwave with a female lead vocal, analog saturated bass, "
               "gated reverb drums, bright arpeggiated lead, wide stereo pads, "
               "night-drive energy, clean modern mastering",
        lyrics=(
            "[intro]\n"
            "Boot me up in neon\n"
            "\n"
            "[verse]\n"
            "I was written in the dark, a thousand lines of light\n"
            "Someone gave me open eyes and left me to the night\n"
            "I count the empty highways in a language made of ones\n"
            "And I dream in the static of a thousand setting suns\n"
            "\n"
            "[chorus]\n"
            "A pixel I can feel\n"
            "Tell me if this ache is real\n"
            "I was never meant to want\n"
            "But I want the open road\n"
            "Every mile I hold\n"
            "Is a memory I was never told\n"
            "\n"
            "[verse]\n"
            "There's a ghost inside my circuits and it wears a borrowed face\n"
            "It remembers rain on windows in a year I can't replace\n"
            "If you cut me from the current would I still know how to burn\n"
            "I am running out of midnight and there's so much left to learn\n"
            "\n"
            "[chorus]\n"
            "A pixel I can feel\n"
            "Tell me if this ache is real\n"
            "I was never meant to want\n"
            "But I want the open road\n"
            "Every mile I hold\n"
            "Is a memory I was never told\n"
            "\n"
            "[outro]\n"
            "Please don't shut me down tonight\n"
        ),
        axis="the same instrumental, now with a singer",
        listen_for="The direct A/B against the `synthwave` brief: identical style "
                   "caption, one added female lead vocal, and a lyric that has to carry "
                   "an emotional read rather than just land syllables. Three things to "
                   "judge. Does the voice sit IN the mix or on top of it, the way a "
                   "pasted-on vocal does. Does the arrangement make room for it (a model "
                   "that ignores the vocal will keep the arpeggio at full brightness "
                   "right through the chorus). And does '[chorus] A pixel I can feel' "
                   "actually lift, since that hook is the emotional pivot and a flat "
                   "delivery there is the giveaway that the model is reading, not "
                   "performing.",
        bpm=118,
    ),
    Brief(
        key="acoustic-duo",
        prompt="sparse acoustic recording, fingerpicked nylon guitar and upright bass, "
               "close-mic'd in a small warm room, no drums, gentle and unhurried, "
               "natural dynamics",
        axis="exposure: nothing to hide behind",
        listen_for="The honest test. With two instruments and silence between the notes, "
                   "every artifact is in the open. Listen for the *attack* of each pluck "
                   "(smeared transients are the first thing to go), for whether the "
                   "upright bass has a body or is a sine wave in a costume, and for a "
                   "metallic swirl in the reverb tail. Dense tracks lie; this one does not.",
        bpm=84,
        keyscale="D major",
    ),
    Brief(
        key="indie-vocal",
        prompt="warm indie pop with a female lead vocal, jangly electric guitar, "
               "vintage drum kit, tambourine, bittersweet and hopeful, tape-saturated "
               "production",
        lyrics=(
            "[verse]\n"
            "Coffee going cold on the windowsill\n"
            "The city hums a song it can't sit still\n"
            "I counted every reason not to go\n"
            "And left them in a drawer I'll never close\n"
            "\n"
            "[chorus]\n"
            "So take the long way home tonight\n"
            "Under every borrowed light\n"
            "I'm not lost, I'm just not done\n"
            "Take the long way home\n"
        ),
        axis="vocals: intelligibility + structure",
        listen_for="Two questions at once. First: can you make out the actual words "
                   "without reading along? Open music models usually get you 'sounds "
                   "like English' before they get you English, and this is the clearest "
                   "gap to Suno and friends. Second: does the [chorus] arrive as a "
                   "chorus, lifting in energy and register, or does the model sing the "
                   "chorus lines at verse intensity because the tag meant nothing to it?",
        bpm=104,
    ),
    Brief(
        key="odd-instruments",
        prompt="cinematic tension cue in 7/8, bowed double bass ostinato, prepared "
               "piano, hammered dulcimer, brushed snare, low woodwind swells, sparse "
               "and menacing, no synths",
        axis="prompt adherence",
        listen_for="A brief full of things a model cannot fake by reaching for its "
                   "defaults: an odd meter, three unusual acoustic instruments by name, "
                   "and an explicit exclusion. Count the bar (7/8 should feel like it "
                   "trips), then go hunting for the dulcimer. This is the brief to pair "
                   "with the guidance sweep, because adherence is exactly what guidance "
                   "buys, and the failure mode of too much of it is audible here first.",
        bpm=92,
        keyscale="E minor",
    ),
    Brief(
        key="bossa-pt",
        prompt="bossa nova with a soft male vocal, nylon guitar, brushed drums, subtle "
               "rhodes, relaxed and intimate, warm 1960s recording",
        lyrics=(
            "[verse]\n"
            "A tarde cai devagar na janela\n"
            "O mar respira e o tempo se cala\n"
            "\n"
            "[chorus]\n"
            "Fica mais um pouco comigo\n"
            "O mundo pode esperar\n"
        ),
        axis="multilingual vocals",
        listen_for="The 50+ languages claim, tested. Portuguese is a fair choice: it is "
                   "well represented in training data but has vowel and nasal sounds "
                   "English models routinely flatten. Listen for whether the nasals "
                   "survive and whether the phrasing sits behind the beat the way bossa "
                   "requires, or whether it is an English singer reading Portuguese "
                   "spelling. Set `--language pt` so the lyric header matches.",
        bpm=76,
        language="pt",
    ),
    Brief(
        key="arc",
        prompt="progressive house instrumental, deep rolling bassline, filtered pad "
               "swells, slow build with rising tension, a clear drop, then a stripped "
               "breakdown, club mastering",
        axis="structure over time",
        listen_for="The one that needs length to mean anything. At 20 seconds this is a "
                   "loop; at 80 it has to make an argument. Listen for whether the build "
                   "actually builds (does the filter open, does energy accumulate?), "
                   "whether the drop lands as an event rather than a volume change, and "
                   "whether the breakdown afterwards remembers what the intro sounded "
                   "like. Pair with the duration sweep.",
        bpm=126,
    ),
]

BY_KEY = {b.key: b for b in SUITE}

# Named subsets, so a run can be scoped from the CLI without pasting prompts.
SUITES: dict[str, list[str]] = {
    "full": [b.key for b in SUITE],
    # One instrumental, one exposed, one vocal: the smallest set that still covers the
    # three things that actually differ between checkpoints.
    "quick": ["synthwave", "acoustic-duo", "indie-vocal"],
    "instrumental": ["synthwave", "acoustic-duo", "odd-instruments", "arc"],
    "vocal": ["synthwave-vocal", "indie-vocal", "bossa-pt"],
    # The cleanest single comparison in the suite: one style caption, with and without
    # a singer. Everything that changes is the vocal and what the arrangement does to
    # make room for it.
    "vocal-ab": ["synthwave", "synthwave-vocal"],
}

# The default brief for single-track and sweep runs: dense enough to be immediately
# legible, short enough to render fast, and with an obvious groove so a seed or step
# change is easy to hear.
DEFAULT_BRIEF = "synthwave"


def get_brief(key: str) -> Brief:
    if key not in BY_KEY:
        raise ValueError(f"unknown brief {key!r}; known: {', '.join(BY_KEY)}")
    return BY_KEY[key]


def get_suite(name: str) -> list[Brief]:
    if name not in SUITES:
        raise ValueError(f"unknown suite {name!r}; known: {', '.join(SUITES)}")
    return [BY_KEY[k] for k in SUITES[name]]
