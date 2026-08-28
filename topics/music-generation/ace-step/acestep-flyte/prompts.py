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

Those seven are `CORE`. A second block, `GENRE_SWAP`, runs the opposite experiment:
instead of moving the musical ask to find a failure mode, it holds ONE lyric perfectly
still and moves only the genre around it, ten times. See its own header below.

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


CORE: list[Brief] = [
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

# ── The genre swap ───────────────────────────────────────────────────────────────
#
# Ten briefs, one lyric. The seven above each move the *musical ask* to find a failure
# mode; these hold the words completely still and move only the world around them.
#
# The question is sharper than "what does it sound like as metal". A model that has
# learned genre as a set of TIMBRES will hand you the same tune ten times wearing
# different clothes: same melody, same phrasing, same places to breathe, with the
# guitars swapped for a pedal steel. A model that has learned genre as a set of
# CONVENTIONS will rewrite the melody, move the stresses onto different syllables, and
# change where the line ends. Play the chorus of any two of these back to back; if the
# tune survives the swap, you have your answer, and it is the less impressive one.
#
# The lyric is deliberately plain: no genre markers, no proper nouns, nothing that
# points at a decade or a place. Anything specific in it ("neon", "highway", "whiskey")
# would do half the model's work and quietly rig the test.
#
# Every caption names instruments, a vocal delivery and a production era, because that
# is what these models actually condition on. Four of the ten also carry an explicit
# EXCLUSION ("no instruments", "no percussion"). Those are the interesting ones: a bare
# genre tag can be satisfied by vibes, but a negative constraint can only be satisfied
# by obeying it, and adding a drum kit to plainchant is the single most common way a
# music model tells you it is pattern-matching rather than listening.

GENRE_SWAP_LYRIC = (
    "[verse]\n"
    "Cold water and a borrowed coat\n"
    "The last train out, the last I wrote\n"
    "I packed up every word I'd said\n"
    "And left the light on overhead\n"
    "\n"
    "[chorus]\n"
    "Don't wait up, don't wait up\n"
    "I'm burning through the dark\n"
    "Don't wait up, don't wait up\n"
    "I'll find you by the spark\n"
)

def _swap(key: str, prompt: str, listen_for: str, bpm: int = 0,
          keyscale: str = "") -> Brief:
    """A genre-swap brief. Only `prompt` differs; the lyric is shared by construction.

    The caption's FIRST CLAUSE must name the world ("1970s outlaw country, weathered
    male baritone, ..."), because that clause becomes the row's axis label and so the
    report heading. That is also how these captions should be written anyway: these
    models weight the head of the prompt most heavily, so burying the genre behind three
    instruments is bad prompting as well as a bad heading. `_check_swaps` enforces it.
    """
    world = prompt.split(",")[0].strip()
    return Brief(key=key, prompt=prompt, lyrics=GENRE_SWAP_LYRIC,
                 axis=f"genre swap · {world}",
                 listen_for=listen_for, bpm=bpm, keyscale=keyscale)


GENRE_SWAP: list[Brief] = [
    _swap(
        "gs-outlaw",
        "1970s outlaw country, weathered male baritone, pedal steel, honky-tonk piano, "
        "telecaster twang, brushed drums, dry Nashville room",
        "The pedal steel is the easy tell, so listen past it. Country lives in the "
        "PHRASING: the voice should sit fractionally behind the beat and put a turn on "
        "the end of each line. If the melody is on the beat and dead straight with a "
        "steel guitar behind it, the model has dressed up a pop vocal.",
        bpm=96, keyscale="G major",
    ),
    _swap(
        "gs-blackmetal",
        "Norwegian black metal, shrieked rasping vocals, tremolo-picked guitar wall, "
        "blast beat drums, cold lo-fi rehearsal room recording, no polish",
        "The hardest ask in the grid, because the words are gentle and the delivery is "
        "not. Two failure modes to watch for, and they are opposites. It may SOFTEN: "
        "keep a sung melody and add distorted guitars, because the lyric reads tender. "
        "Or it may refuse the production note and give you a clean modern metal mix "
        "instead of a room recorded on one microphone. Intelligibility should COLLAPSE "
        "here by design, so if you can still make out 'don't wait up', that is the "
        "model declining the brief, not succeeding at it.",
        bpm=170, keyscale="E minor",
    ),
    _swap(
        "gs-chant",
        "sacred plainchant, male schola singing in unison, free unmeasured rhythm, "
        "enormous stone cathedral reverb, no instruments, no percussion",
        "Three exclusions and no pulse, which makes this the strictest brief here. "
        "Count the failures in order: is there a drum (there must not be), is there an "
        "instrument at all, is the choir in UNISON or has it drifted into comfortable "
        "Western harmony, and is the reverb a genuinely long tail or a short plate "
        "turned up. Note bpm is left unset on purpose: chant has no metre, and asking "
        "for one would be asking the model to break the brief.",
        keyscale="D dorian",
    ),
    _swap(
        "gs-funkbr",
        "baile funk from Rio, tamborzao beat, blown out distorted sub bass, chanted "
        "call and response vocals, raw phone-speaker mastering, party energy",
        "Two specifics to check. The tamborzao is a named rhythm, not a vibe, so either "
        "the pattern is there or it is generic club percussion. And 'chanted call and "
        "response' asks the model to stop singing a melody and start trading shouts, "
        "which is a structural change to how the lyric is delivered. Bonus: is 'blown "
        "out' real distortion, or clean audio with the fader up?",
        bpm=130,
    ),
    _swap(
        "gs-boyband",
        "late 1990s boy band pop ballad, four part male harmony, glossy DX7 electric "
        "piano, finger snaps, huge chorus, polished radio mastering",
        "The harmony stack is the whole test. Does the chorus actually split into parts "
        "that move independently, or is it one voice doubled and widened? Stacked "
        "unison is what most models do when asked for harmony, and it sounds thick "
        "rather than harmonised. Listen also for whether the last chorus lifts a "
        "semitone, because the model has certainly heard enough of these to know.",
        bpm=92,
    ),
    _swap(
        "gs-delta",
        "1930s delta blues, single weathered male voice, resonator slide guitar, foot "
        "stomp, recorded with one ribbon microphone, shellac surface noise",
        "This one asks for an ERA, not a genre, and that is the harder half. The common "
        "failure is a technically excellent modern blues recording: clean, wide, "
        "full-range, which is exactly wrong. Listen for a narrow mono-ish image, rolled "
        "off top and bottom, and surface noise that is continuous rather than an effect "
        "sprinkled on top. The slide should be a real gliss between notes, not vibrato.",
        bpm=72,
    ),
    _swap(
        "gs-bulgarian",
        "Bulgarian women's choir, close dissonant harmony in open fifths and seconds, "
        "straight tone with no vibrato, village hall reverb, no instruments",
        "A deliberate trap for a model that learned harmony from pop. Two things almost "
        "every model gets wrong here: it adds VIBRATO (the style is defined by its "
        "absence) and it resolves the dissonance into comfortable thirds. The sound you "
        "want is a second that sits there and grinds. If it comes back as a pretty "
        "Western choir, the model knows 'women's choir' and nothing about this one.",
    ),
    _swap(
        "gs-drill",
        "UK drill, sliding 808 glides, sparse dark piano motif, triplet hi hats, "
        "deadpan half-sung flow, wide sub bass, club mastering",
        "The 808 either glides between pitches or it does not, and that single detail "
        "is most of what separates drill from every other trap derivative. The subtler "
        "test is the vocal: drill is spoken-adjacent, so a model that only knows how to "
        "SING will give you a pop topline over a drill beat. This is the clearest case "
        "in the grid where the backing can be right and the delivery still wrong.",
        bpm=142,
    ),
    _swap(
        "gs-shanty",
        "sea shanty, unaccompanied male chorus in rough unison, boot stomps and hand "
        "claps, dockside room, call and response, no instruments",
        "'Unaccompanied' plus 'no instruments' means the stomp is the only percussion "
        "allowed, so any kick drum is a fail. The interesting question is the crowd: a "
        "shanty needs several men who are not quite together, and what models usually "
        "produce is one clean voice copied and detuned. Roughness is the feature here, "
        "and it is the thing a model trained to sound good will smooth away.",
        bpm=88,
    ),
    _swap(
        "gs-hyperpop",
        "hyperpop, pitch shifted vocals, hard clipped saturated 808s, glitch edits, "
        "sugary and abrasive at once, brickwalled and distorted",
        "The modern-artifact test, and the mirror image of the delta blues brief: there "
        "the artifacts had to sound accidental, here they have to sound chosen. Listen "
        "for whether the clipping is a deliberate texture with shape to it or just a "
        "loud mix, and whether the pitch shift is committed to. A tasteful hyperpop "
        "track is a failed hyperpop track.",
        bpm=160,
    ),
]

# ── The genre swap, batch two ────────────────────────────────────────────────────
#
# Fourteen more worlds for the same lyric. Batch one was mostly Anglophone popular
# music plus two choral outliers, and a model trained on the internet is going to be
# comfortable in almost all of it. This batch is chosen where the training data thins
# out and where the genre is defined by something a timbre swap cannot fake:
#
#   - a VOCAL TECHNIQUE that is the genre (qawwali melisma, enka's kobushi, throat
#     singing's overtones, flamenco's cracked cante). You cannot arrive at these by
#     picking instruments; the model either has the technique or it substitutes a
#     normal singing voice over the right backing, which is the tell.
#   - a NAMED RHYTHM (the amen break, the polka oompah, the highlife guitar pattern).
#     Either the pattern is there or it is generic percussion in the right costume.
#   - an ENSEMBLE ROLE structure (doo-wop's bass singer, gospel's call and response,
#     mariachi's trumpet answers). These need several voices doing different jobs,
#     which is exactly what a model that widens one voice cannot do.
#
# Kept separate from batch one rather than appended to it so the first report stays a
# reproducible artifact: `genre-swap` is still those ten, in that order.

GENRE_SWAP_2: list[Brief] = [
    _swap(
        "gs-mariachi",
        "mariachi, massed trumpets answering the vocal, vihuela and guitarron, full "
        "chested male voice, festive plaza recording",
        "The trumpets have a JOB here: mariachi trumpets answer the singer in the gaps "
        "at the end of lines, they do not play through. If they sit under the vocal "
        "like a pad, the model has the instruments and not the arrangement. Second "
        "thing: the guitarron is a specific fat plucked bass, and a synthesised or "
        "bowed low end in its place is the easiest substitution to miss.",
        bpm=120,
    ),
    _swap(
        "gs-qawwali",
        "qawwali, harmonium drone, tabla, group handclaps, melismatic male lead voice "
        "building in intensity, devotional, live shrine recording",
        "The one thing that matters is the MELISMA: qawwali runs many notes over a "
        "single syllable, and the lyric here is plain English with short words, so the "
        "model has to stretch them to do it. A model that sings one note per syllable "
        "over a harmonium has produced the setting and not the style. Second: qawwali "
        "escalates, so the last line should be more intense than the first.",
        bpm=96,
    ),
    _swap(
        "gs-afrobeats",
        "afrobeats, log drum bass, syncopated shakers and rim clicks, laid back "
        "melodic vocal, glossy modern Lagos production",
        "The groove is the test and it is a subtle one: afrobeats percussion is "
        "syncopated against a steady pulse, so the shakers and rims should pull "
        "AGAINST the kick rather than lock to it. A straight quantised version of this "
        "sounds like pop with a shaker on it. Listen also for the vocal sitting lazily "
        "behind the beat, which is most of the genre's feel.",
        bpm=104,
    ),
    _swap(
        "gs-klezmer",
        "klezmer, wailing clarinet lead with bent notes, accordion, tsimbl, upright "
        "bass, freylekhs dance feel, wedding band recorded in a hall",
        "The clarinet has to CRY: bends, slides and a sob at the top of phrases. A "
        "clean legato clarinet playing the melody straight is a classical clarinet in a "
        "klezmer band. The other half is the mode, which should sound neither major nor "
        "minor in the usual way, and a model that flattens it to plain minor has lost "
        "the thing that makes this music identifiable in two bars.",
        bpm=132,
    ),
    _swap(
        "gs-gospel",
        "southern gospel choir, hammond organ with rotary speaker, tambourine, piano, "
        "lead voice with a full choir answering, huge live church room",
        "Call and response, so listen for a genuine dialogue: lead sings, choir answers, "
        "and they overlap at the seams. The common failure is a lead voice with a choir "
        "singing the same words at the same time, which is doubling, not answering. The "
        "Hammond should have audible rotary movement rather than being a generic organ "
        "patch, and this brief is the best chance in the whole grid of hearing real "
        "vocal harmony rather than a widened single take.",
        bpm=88,
    ),
    _swap(
        "gs-jungle",
        "1990s jungle, chopped and rearranged amen break at double time, deep sub bass, "
        "ragga vocal chat, rave stabs, dark and rolling",
        "The amen break is a specific six second drum loop that everyone chops, and "
        "'chopped and rearranged' is the ask: the pattern should move and stutter, not "
        "loop identically. Also watch the TEMPO TRICK that defines jungle: drums at "
        "roughly double the speed of the bassline and the vocal, so it should feel fast "
        "and slow simultaneously. A model that renders everything at one tempo has "
        "produced drum and bass wallpaper.",
        bpm=170,
    ),
    _swap(
        "gs-flamenco",
        "flamenco, spanish guitar with rasgueado, palmas handclaps, cajon, raspy "
        "anguished cante jondo male voice, close intimate room",
        "The voice should sound DAMAGED: cante jondo is cracked, strained and pushed to "
        "the edge, and a model that renders a pleasant Spanish-flavoured vocal has "
        "missed the entire point. The palmas are the second tell, because they are a "
        "played rhythm with syncopation and accents, not a metronome clap on the beat.",
        bpm=100,
    ),
    _swap(
        "gs-bollywood",
        "1970s Bollywood filmi, lush string orchestra, tabla and dholak, female "
        "playback vocal with rapid ornaments, bright reverb, analog tape",
        "Ornamentation is the marker: the little rapid turns and grace notes around each "
        "note. A straight pop vocal over strings and tabla is the failure. Also listen "
        "for the ERA in the recording, because 1970s filmi has a very particular bright "
        "compressed sound, and a modern clean orchestral mix here is the same mistake as "
        "a modern blues recording in `gs-delta`.",
        bpm=112,
    ),
    _swap(
        "gs-doowop",
        "1950s doo-wop, lead tenor with backing group singing nonsense syllables, "
        "prominent bass singer, upright bass, brushed drums, tape slap echo",
        "The bass SINGER is the whole test and it is a role no other brief here asks "
        "for: a human voice on the bottom singing the low part, not a bass instrument. "
        "Listen for whether the backing group sings syllables (sha-la-la, doo-wop) "
        "rather than the lyric, and for slap echo, which is a single short repeat and "
        "not reverb.",
        bpm=76,
    ),
    _swap(
        "gs-enka",
        "Japanese enka, dramatic female voice with heavy kobushi vibrato, shakuhachi "
        "flute, lush strings, slow and melancholic, 1980s recording",
        "Kobushi is a specific ornamented wobble at the ends of notes, quite unlike "
        "Western vibrato, and it is what makes enka enka. This brief is also the "
        "grid's quiet language test: the lyric is English, so a model that has only "
        "learned this style attached to Japanese may not transfer it. If it comes back "
        "as a Western ballad with a flute, the style did not survive the language.",
        bpm=68,
    ),
    _swap(
        "gs-polka",
        "Bavarian oompah polka, tuba on the downbeat, accordion, clarinet, snare, "
        "beer hall crowd, relentlessly cheerful",
        "The oompah is a named pattern: bass note on one, chord on two, forever. It is "
        "trivially simple and therefore a clean check on whether the model renders "
        "requested rhythms or approximates them. The interesting friction is emotional, "
        "because the lyric is about leaving in the dark and this arrangement refuses to "
        "be sad about it. Does the model let that contradiction stand?",
        bpm=124,
    ),
    _swap(
        "gs-throat",
        "Tuvan throat singing, khoomei overtone voice with a low fundamental drone, "
        "igil horsehead fiddle, frame drum, wide open steppe, sparse",
        "The most extreme ask in either batch: overtone singing produces two audible "
        "pitches from one voice at once, a low drone and a whistling harmonic above it. "
        "Almost nothing in the training data does this. The likely outcome is a low "
        "growly normal voice, which is a fair and interesting failure, and worth "
        "keeping in the grid precisely because it marks the edge of what is in there.",
        bpm=0,
    ),
    _swap(
        "gs-highlife",
        "Ghanaian highlife, interlocking palm-wine guitar lines, horn section, "
        "cowbell and congas, relaxed swung groove, warm 1970s recording",
        "'Interlocking' is the ask: two guitar parts that lock into one pattern neither "
        "plays alone. Most models give you one guitar and a rhythm part instead, which "
        "is the difference between the genre and a warm guitar band. The cowbell keeps "
        "the time reference, so if the guitars drift away from it the interlock was "
        "never there.",
        bpm=108,
    ),
    _swap(
        "gs-vaporwave",
        "vaporwave, heavily pitched down sample loop, tape wobble and flutter, glassy "
        "mallets, cavernous 1980s mall reverb, hazy and nostalgic",
        "The mirror of `gs-hyperpop`: both are defined by processing rather than "
        "instruments, but this one asks the model to sound DEGRADED and slow rather "
        "than loud and bright. Listen for genuine pitch-down artifacts (a slower, "
        "duller, thicker sound) rather than a slow track played normally, and for tape "
        "wobble as real pitch instability. A clean lo-fi beat is the near miss here.",
        bpm=60,
    ),
]

# ── Performance callouts: does the model take direction? ─────────────────────────
#
# ACE-Step documents FIVE structure tags as real conditioning: [intro], [verse],
# [chorus], [bridge], [outro]. It documents nothing about performance directions, the
# things a producer writes in a margin: [belted], [whispered], [guitar solo],
# [breakdown], [half-time]. So there are three plausible outcomes and no way to know
# which without rendering both:
#
#   1. They condition. The model has seen enough annotated lyrics that "belted" moves
#      the delivery, and this becomes a genuinely useful control surface.
#   2. They are ignored. The tokeniser drops unknown bracketed tags and the two takes
#      are the same song with a different seed's worth of variation.
#   3. THEY GET SUNG. The README's existing warning is that writing "instrumental" in
#      the lyrics field makes the model sing the word "instrumental". If unknown tags
#      fall through to the vocal path, a singer earnestly performing the words "guitar
#      solo" is the funniest possible result and also the most informative.
#
# The control has to be airtight or the comparison means nothing, so the plain lyric is
# DERIVED from the annotated one by stripping callouts rather than written twice. Two
# hand-written versions would inevitably differ by a word somewhere and that word would
# be indistinguishable from the effect.
#
# Structure tags survive stripping, in BOTH variants. They are documented conditioning
# and holding them constant is the point; this experiment is about the undocumented
# ones. Callouts therefore live on their OWN lines, never inline, so that `sung_lines`
# counts the same number of sung lines in both variants: an inline "[belted] I am
# durable" would start with a bracket and silently drop out of the count.

_STRUCTURAL_TAGS = {"intro", "verse", "chorus", "bridge", "outro"}


def strip_callouts(lyrics: str) -> str:
    """The same lyric with performance directions removed, structure tags kept."""
    out = []
    for line in lyrics.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            if s[1:-1].strip().lower().split(" - ")[0] not in _STRUCTURAL_TAGS:
                continue
        out.append(line)
    return "\n".join(out)


# The joke version. It works because it is played completely straight: the comedy is in
# treating a retry policy as heartbreak, not in winking at the audience. "I retried you
# exponentially" only lands if the singer means it.
BALLAD_FLYTE = """[intro]
[softly, clean electric guitar, no drums]
[verse]
[restrained, close mic]
I've been running since a Tuesday in the rain
Half my memory is gone but I remember your name
They pulled the power on me, said the run was lost
I came back at step forty-seven and I paid the cost
[chorus]
[belted, full band, huge]
I am durable, darling
You can kill me, I'll come back
Every step I ever finished
Is a step I'll never lack
Pull the plug out of the wall
Let the whole cluster fall
I'll resume from where we were
And I'll remember it all
[verse]
[restrained again, drums half-time]
There's a checkpoint in the object store with your face on it
Twelve gigs of everything I couldn't quit
The scheduler said to let you go, the queue was getting long
I retried you exponentially and I got it wrong
[breakdown]
[drums drop out, whispered]
Three attempts, then it fails for good
[bridge]
[building, full voice]
That's the policy, I understood
But nobody wrote the backoff for a heart
So I keep on coming back to where we start
[guitar solo]
[outro]
[quiet, single voice]
I'll resume from where we were
"""

# The straight version. Deliberately the SAME skeleton, verse for verse and almost line
# for line, with every literal term turned back into the human thing it was borrowed
# from. Worth hearing next to the other one: the joke song and the sincere song are the
# same song, which is either a point about metaphor or a point about infrastructure.
BALLAD_SERIOUS = """[intro]
[softly, clean electric guitar, no drums]
[verse]
[restrained, close mic]
They shut the power off the winter I was born
I lost the better half of every word I'd sworn
I woke up in the middle of a night I couldn't place
With nothing left to hold but the shape of your face
[chorus]
[belted, full band, huge]
I am still here, I am still here
Take the ground out from beneath
I will find the thread I dropped
And I'll gather up the rest
Break me in the middle
Let the whole thing fall apart
I'll begin from where we stopped
I remember, I restart
[verse]
[restrained again, drums half-time]
There's a photograph of you in a drawer I never lost
Everything I couldn't finish, everything it cost
They told me I should let it go, the waiting was too long
But I kept on coming back to where we both belong
[breakdown]
[drums drop out, whispered]
Three times trying, then you're meant to stop
[bridge]
[building, full voice]
That's the wisdom, that's the drop
But no one writes the rule for a heart
So I keep returning to the start
[guitar solo]
[outro]
[quiet, single voice]
I'll begin from where we stopped
"""

# A rewrite of BALLAD_SERIOUS, kept ALONGSIDE it rather than replacing it, because the
# original is the structural twin of BALLAD_FLYTE and the `ballad-registers` comparison
# depends on that parallel surviving.
#
# What was wrong with the first one: it inherited the joke version's rhyme scheme, and
# it shows. "That's the wisdom, that's the drop" exists only because "drop" rhymed in a
# song about retry policies; in the sincere register it means nothing. "Take the ground
# out from beneath" is contorted, "I'll gather up the rest" is vague, and "I am still
# here, I am still here" is the kind of line that sounds like emotion without carrying
# any.
#
# What this one does differently: concrete sensory detail instead of abstraction (a
# third stair, a drawer, a coat), and one hook line the whole song is built to deliver.
# "You can break the line, you can't break the thread" is the argument of the song, and
# a power ballad needs its argument to fit in one breath. It is also 26 sung lines to
# the original's 21, which is deliberate: a real chorus repeat makes it a song rather
# than a demo, and it gives the length ladder more to work with.
BALLAD_RETURN = """[intro]
[softly, clean electric guitar, no drums]
[verse]
[restrained, close mic]
They cut the power out the winter I was ten
And I learned the house by touching walls again
I know the sound the third stair used to make
I know which door you closed and didn't take
[chorus]
[belted, full band, huge]
Burn it to the ground, I know the way back
Take the years from me, I'll walk them twice
You can break the line, you can't break the thread
I remember, I remember
Everything you thought was lost
I will carry it across
[verse]
[restrained again, drums half-time]
There's a photograph I never had to find
It kept its corner of the drawer, and I kept mine
They told me that the years would take the weight of it
I let them try. They didn't.
[breakdown]
[drums drop out, whispered]
Three times trying, then they say you let it go
[bridge]
[building, full voice]
Whoever wrote that rule has never known
A thing worth coming back for on its own
I have counted every door I never opened
And I am not done counting
[chorus]
[belted, full band, huge, final]
Burn it to the ground, I know the way back
Take the years from me, I'll walk them twice
You can break the line, you can't break the thread
I remember, I remember
Everything you thought was lost
I will carry it across
[guitar solo]
[outro]
[quiet, single voice]
I know the way back
"""

BALLAD_PROMPT = ("1980s arena power ballad, clean chorused electric guitar in the "
                 "verses, huge distorted guitars in the chorus, big gated snare, "
                 "soaring male lead vocal, anthemic, wide reverb")
BALLAD_BPM = 74

_BALLAD_LISTEN = (
    "The A/B is the point, so play the two cards in this pair back to back. The "
    "annotated take carries directions ACE-Step does not document: [belted], "
    "[whispered], [breakdown], [guitar solo], [half-time]. Three things could happen "
    "and all three are worth knowing. They might CONDITION, in which case the chorus "
    "should be visibly bigger than the verse and the breakdown should actually drop "
    "the drums, and you have found an undocumented control surface. They might be "
    "IGNORED, in which case the pair differs only as much as two seeds would. Or they "
    "might be SUNG, because a stray word in the lyrics field is a word this model will "
    "happily perform, and a singer solemnly delivering the words 'guitar solo' is the "
    "outcome to hope for. Structure tags are in BOTH takes, so anything you hear is "
    "the callouts and not the sections.")


def _ballad(key: str, lyrics: str, extra: str) -> Brief:
    return Brief(key=key, prompt=BALLAD_PROMPT, lyrics=lyrics,
                 axis="performance callouts", listen_for=extra + " " + _BALLAD_LISTEN,
                 bpm=BALLAD_BPM)


CALLOUTS: list[Brief] = [
    _ballad("ballad-flyte-plain", strip_callouts(BALLAD_FLYTE),
            "The joke ballad with NO performance directions: structure tags only, "
            "which is what every other brief in this repo uses."),
    _ballad("ballad-flyte-callouts", BALLAD_FLYTE,
            "The same words with a producer's margin notes added."),
    _ballad("ballad-serious-plain", strip_callouts(BALLAD_SERIOUS),
            "The sincere ballad, structure tags only. Same skeleton as the Flyte one "
            "verse for verse, with every literal term turned back into the human thing "
            "it was borrowed from."),
    _ballad("ballad-serious-callouts", BALLAD_SERIOUS,
            "The sincere ballad with the same margin notes."),
    _ballad("ballad-return-plain", strip_callouts(BALLAD_RETURN),
            "'The Way Back', the rewrite. Structure tags only."),
    _ballad("ballad-return-callouts", BALLAD_RETURN,
            "'The Way Back', the rewrite, with margin notes. 26 sung lines including a "
            "full chorus repeat, so it is a song rather than a demo and it gives the "
            "length ladder something to stretch."),
]


# ── Production language: does asking for a ROOM beat asking for a SOUND? ─────────
#
# `BALLAD_PROMPT` is maximalist by design: huge, gated, wide, anthemic. Every term in
# it asks for more processing, and processing is what a generative model is least able
# to fake convincingly. Reverb tails go metallic, brickwalled loudness leaves no
# dynamics for the ear to read as real, and "wide" on a model that narrows its stereo
# image under pressure is a request it will half-fulfil.
#
# The alternative is to describe a RECORDING rather than a SOUND: name the room, the
# mic placement, the tape, and ask explicitly for dynamics and against limiting. The
# hypothesis is that a model asked to imitate a documented recording chain lands closer
# to something that sounds recorded, while a model asked for adjectives lands in the
# uncanny middle. It is a hypothesis and not a fact, which is why it is an A/B with the
# lyric and everything else held fixed.
BALLAD_PROMPT_DRY = (
    "1970s rock ballad recorded live to tape, close-mic'd drum kit in a wood-panelled "
    "room, warm lightly overdriven electric guitar, upright piano, electric bass, one "
    "male lead vocal sung with natural dynamics, minimal reverb, no compression on the "
    "vocal, analog tape saturation, quiet verses and loud choruses")

PRODUCTION: list[Brief] = [
    Brief(key="ballad-flyte-dry", prompt=BALLAD_PROMPT_DRY, lyrics=BALLAD_FLYTE,
          axis="production language", bpm=BALLAD_BPM,
          listen_for=(
              "Identical words and identical callouts to `ballad-flyte-callouts`; the "
              "only difference is that the caption describes a recording instead of a "
              "sound. Listen for DYNAMICS above all: the maximalist caption asks for "
              "everything to be huge all the time, which leaves the ear nothing to "
              "read as human, while this one asks for quiet verses and loud choruses. "
              "Then listen to the reverb tails, which is where the other caption's "
              "'wide' and 'arena' most easily turn metallic, and to whether the vocal "
              "has audible breath and level variation now that nothing asked to "
              "compress it.")),
]


# ── A dreamy cinematic brief, for the three-way vocal comparison ─────────────────
#
# Written for the ACE-Step vs MiniMax head-to-head rather than to probe a failure
# mode, so unlike the CORE briefs it is not trying to break anything. It is here
# because both models read the SAME structure tags, which makes it the first brief in
# this repo that can go to two different families with no conversion and no asterisk on
# the card.
#
# The caption is deliberately production-heavy rather than adjective-heavy: "tape
# delay", "sub bass swell", "filtered arpeggio" and a named tempo give the model
# something to render, where "emotional" and "dreamy" on their own give it a mood and
# no instructions. The emotional read is supposed to come from the WORDS.

DREAMSCAPE_LYRIC = """[intro]
[soft pads, distant piano]
[verse]
I woke up somewhere over the Atlantic
Half a dream still holding to my eyes
The window held a thin blue line of morning
And everything below us was disguised
[verse]
You said that love is mostly just attention
A held breath and a hand that doesn't move
I have been learning how to give it slowly
I have nothing left to prove
[chorus]
So let it fall, let it fall
Like light across the water
I will find you in the quiet
When the noise has all gone under
Let it fall, let it fall
I am not afraid of falling
If the dark is where you are
Then the dark is where I'm going
[verse]
There's a satellite that's been up there for decades
Still transmitting to a room that isn't there
I think about it circling in silence
Saying something beautiful to empty air
[chorus]
Let it fall, let it fall
Like light across the water
I will find you in the quiet
When the noise has all gone under
Let it fall, let it fall
I am not afraid of falling
If the dark is where you are
Then the dark is where I'm going
[outro]
Then the dark is where I'm going
"""

DREAMSCAPE: list[Brief] = [
    Brief(
        key="dreamscape",
        prompt="cinematic electronic ballad, breathy female lead vocal, wide analog "
               "pads, deep sub bass swells, sparse felt piano, filtered arpeggio, tape "
               "delay, slow build into a euphoric wash, spacious reverb, emotional and "
               "dreamy, modern film-score production",
        lyrics=DREAMSCAPE_LYRIC,
        axis="the fair three-way vocal comparison",
        bpm=90,
        keyscale="A minor",
        listen_for=(
            "The first brief here that goes to two model FAMILIES with no conversion "
            "and no asterisk: ACE-Step and MiniMax read the same [verse]/[chorus] tags, "
            "where DiffRhythm would need the structure thrown away and the lines spread "
            "evenly. So any difference you hear is the model, not the handicap.\n\n"
            "Judge the voice first, since that is where sft beat turbo most clearly and "
            "where these two differ most: breath, consonants, and whether it sounds "
            "performed or read. Then judge the BUILD, because the caption asks for a "
            "slow rise into a euphoric wash and that is a structural request a model "
            "either executes over four minutes or ignores. Third: does the chorus "
            "actually lift? 'Let it fall' is the emotional pivot and a flat delivery "
            "there is the giveaway."),
    ),
]


def _check_callouts() -> None:
    """The control has to be exact, and the tag placement has to keep line counts equal."""
    for annotated, plain in ((BALLAD_FLYTE, "ballad-flyte-plain"),
                             (BALLAD_SERIOUS, "ballad-serious-plain"),
                             (BALLAD_RETURN, "ballad-return-plain")):
        stripped = strip_callouts(annotated)
        assert sung_lines(stripped) == sung_lines(annotated), (
            "a callout is inline rather than on its own line, so it is being counted "
            "as a sung line in one variant and not the other")
        assert BY_KEY[plain].lyrics == stripped
        # Every callout must actually be removed, or the "plain" take is not plain.
        for line in stripped.splitlines():
            s = line.strip()
            if s.startswith("["):
                assert s[1:-1].strip().lower() in _STRUCTURAL_TAGS, f"leaked: {s}"


# ── Lyric density: how much room does a line of words need? ──────────────────────
#
# The observation that started this: short tracks sound more compressed and more
# obviously synthetic. The README has always asserted a rule of thumb here, "budget
# roughly 4 seconds of track per sung line", and nobody ever measured it.
#
# ACE-Step paces the WHOLE lyric to fit `audio_duration` rather than truncating it, so
# a long lyric in a short render is not cut off, it is compressed: syllables shorten,
# breaths disappear, and the model spends its capacity fitting words in rather than
# singing them. That predicts the interesting variable is not duration and not line
# count but the RATIO, seconds of track per sung line.
#
# There is a competing explanation, though, and it is why this is a grid rather than a
# row. Short renders may simply be worse regardless of lyrics: less room to establish
# an arrangement, and fewer latent frames to work with. Duration and density are
# confounded in the everyday case (same lyric, shorter track) and only a grid can
# separate them. See the constant-density diagonal in `density`'s report note.
#
# The four lyrics below are NESTED: each is the previous one plus another section of
# the same song, in a real form (v c v c b c). If they were four different lyrics, any
# difference heard could be the words rather than the length, and the experiment would
# be worthless.

_D_V1 = ("[verse]\n"
         "Cold water and a borrowed coat\n"
         "The last train out, the last I wrote\n"
         "I packed up every word I'd said\n"
         "And left the light on overhead\n")

_D_CHORUS = ("[chorus]\n"
             "Don't wait up, don't wait up\n"
             "I'm burning through the dark\n"
             "Don't wait up, don't wait up\n"
             "I'll find you by the spark\n")

_D_V2 = ("[verse]\n"
         "The station clock is running slow\n"
         "It knows the way I couldn't go\n"
         "I traded morning for the road\n"
         "And everything I couldn't hold\n")

_D_BRIDGE = ("[bridge]\n"
             "If you are counting, I am counting too\n"
             "Every mile I go is one more mile from you\n"
             "I never learned the trick of saying this\n"
             "So I say it to the dark and let it miss\n")

# line count -> lyric. Keys are SUNG lines, which is what the ratio is per.
DENSITY_LYRICS: dict[int, str] = {
    4:  _D_V1,
    8:  _D_V1 + "\n" + _D_CHORUS,
    16: _D_V1 + "\n" + _D_CHORUS + "\n" + _D_V2 + "\n" + _D_CHORUS,
    24: (_D_V1 + "\n" + _D_CHORUS + "\n" + _D_V2 + "\n" + _D_CHORUS + "\n"
         + _D_BRIDGE + "\n" + _D_CHORUS),
}

# One caption for the whole grid, chosen so the VOICE is the loudest thing in the mix
# and the words are judgeable. A dense production would hide exactly the compression
# artifact we are trying to hear, which would make the experiment comfortable and
# useless.
DENSITY_PROMPT = ("warm indie pop with a clear female lead vocal, jangly electric "
                  "guitar, vintage drum kit, tambourine, unhurried, tape-saturated "
                  "production, vocal well forward in the mix")
DENSITY_BPM = 104


def sung_lines(lyrics: str) -> int:
    """Lines that will actually be sung: structure tags and blanks are not lyrics.

    The denominator of the whole experiment, so it lives next to the lyrics rather
    than in the pipeline: a report that divides by the wrong number of lines produces
    a plausible seconds-per-line figure that is quietly wrong.
    """
    return len([l for l in (lyrics or "").splitlines()
                if l.strip() and not l.strip().startswith("[")])


# The provisional seconds-per-line target, and the honest status of the number: it is
# ONE listening pass on the `density` grid, where 24 lines at 144s (6.0s/line) sounded
# the most natural. That cell is the corner of the grid, so it is also the longest
# track and nothing bounds it from above, and the follow-up runs that would settle it
# have not been judged yet.
#
# It is used here anyway, and defensibly, because nothing depends on it being right:
# it is the CENTRE of a bracket, not a setting. `suggest_durations` spans roughly 3x
# around it, so if the true knee is at 4 or at 9 the ladder still contains it and you
# pick by ear. Do not promote this to a default duration anywhere until the fixed-length
# comparison has been listened to.
SECONDS_PER_LINE = 6.0

_LADDER = (0.6, 1.0, 1.7)          # multiples of the suggestion, spanning ~3x
_INSTRUMENTAL_LADDER = (30.0, 60.0, 120.0)


def suggest_durations(lyrics: str, rungs: tuple[float, ...] = _LADDER) -> list[float]:
    """A ladder of lengths worth hearing for this lyric, shortest first.

    The point of a ladder rather than a single suggested number is that the right
    length is a judgement, not a calculation: the lyric sets roughly how much room the
    words need, but whether a song wants to breathe past that is taste. So this returns
    a bracket around the estimate and lets the ear pick, which also means it stays
    useful while [[SECONDS_PER_LINE]] is still provisional.

    An empty lyric gets a fixed ladder instead. Density is undefined without words, and
    for an instrumental the length is a compositional choice rather than a constraint.
    """
    n = sung_lines(lyrics)
    if n == 0:
        return list(_INSTRUMENTAL_LADDER)
    base = n * SECONDS_PER_LINE
    out: list[float] = []
    for r in rungs:
        # 10s is the model's floor and 600s its ceiling; a long lyric on the top rung
        # can reach the latter, and silently exceeding either is how you get a run that
        # fails minutes into a GPU pod.
        v = round(min(max(base * r, 10.0), 600.0))
        if v not in out:
            out.append(float(v))
    return out


def _check_density() -> None:
    """Nesting and line counts, both of which are load-bearing for the conclusion."""
    for n, lyr in DENSITY_LYRICS.items():
        assert sung_lines(lyr) == n, f"{n}-line lyric has {sung_lines(lyr)} sung lines"
    ordered = [DENSITY_LYRICS[k] for k in sorted(DENSITY_LYRICS)]
    for shorter, longer in zip(ordered, ordered[1:]):
        assert longer.startswith(shorter), (
            "density lyrics must be NESTED, or length is confounded with content")


def _check_swaps() -> None:
    """The invariants that make the swap an experiment rather than a playlist.

    Cheap enough to run at import: a broken invariant here (a lyric that drifted, a
    duplicated caption) would not raise anywhere, it would quietly produce a report
    that looks fine and answers nothing.
    """
    swaps = [*GENRE_SWAP, *GENRE_SWAP_2]
    assert len({b.lyrics for b in swaps}) == 1, "swap briefs must share ONE lyric"
    assert len({b.prompt for b in swaps}) == len(swaps), "swap captions must all differ"
    for b in swaps:
        world = b.prompt.split(",")[0].strip()
        assert world and len(world.split()) <= 6, (
            f"{b.key}: the caption's first clause is the row heading, so it must be a "
            f"short world name, got {world!r}")


# Every brief in the repo, assembled here at the bottom because the blocks above are
# independent experiments and each needs the ones before it only for its own checks.
SUITE: list[Brief] = [*CORE, *GENRE_SWAP, *GENRE_SWAP_2, *CALLOUTS, *PRODUCTION,
                      *DREAMSCAPE]

BY_KEY = {b.key: b for b in SUITE}

_check_swaps()
_check_density()
_check_callouts()      # needs BY_KEY, so it runs after it

# Named subsets, so a run can be scoped from the CLI without pasting prompts.
SUITES: dict[str, list[str]] = {
    # `full` stays the seven CAPABILITY briefs. The genre swap is a different kind of
    # experiment (one lyric, ten worlds) and folding it in here would turn every
    # `--suite full` run into seventeen renders of two unrelated questions.
    "full": [b.key for b in CORE],
    # One instrumental, one exposed, one vocal: the smallest set that still covers the
    # three things that actually differ between checkpoints.
    "quick": ["synthwave", "acoustic-duo", "indie-vocal"],
    "instrumental": ["synthwave", "acoustic-duo", "odd-instruments", "arc"],
    "vocal": ["synthwave-vocal", "indie-vocal", "bossa-pt"],
    # The cleanest single comparison in the suite: one style caption, with and without
    # a singer. Everything that changes is the vocal and what the arrangement does to
    # make room for it.
    "vocal-ab": ["synthwave", "synthwave-vocal"],

    # The genre swap. One GPU task, one loaded pipeline, so the twenty-fourth row costs
    # a render (~3s) rather than another 11GB load.
    "genre-swap": [b.key for b in GENRE_SWAP],
    "genre-swap-more": [b.key for b in GENRE_SWAP_2],
    "genre-swap-all": [b.key for b in (*GENRE_SWAP, *GENRE_SWAP_2)],
    # Four with the widest distance between them, for a warm-up or a stream segment
    # that has to fit in a break: a twangy one, a violent one, an unaccompanied one
    # with two exclusions to obey, and one that is barely sung at all.
    "genre-swap-quick": ["gs-outlaw", "gs-blackmetal", "gs-chant", "gs-drill"],

    # The callout A/B. Each pair is plain then annotated, so the report reads as two
    # rows you play against each other rather than four unrelated ballads.
    "callouts": [b.key for b in CALLOUTS],
    "callouts-flyte": ["ballad-flyte-plain", "ballad-flyte-callouts"],
    "callouts-serious": ["ballad-serious-plain", "ballad-serious-callouts"],
    # The other comparison hiding in these four: same skeleton, one literal and one
    # sincere. Both annotated, so the callouts are held constant instead.
    "ballad-registers": ["ballad-flyte-callouts", "ballad-serious-callouts"],
    # Same words, same callouts, same everything: one caption asks for a SOUND and the
    # other for a RECORDING.
    "production-ab": ["ballad-flyte-callouts", "ballad-flyte-dry"],
    # The two families that read the same lyric format, on one brief.
    "dreamscape": ["dreamscape"],
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
