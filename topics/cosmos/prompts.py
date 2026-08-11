"""Prompts for Cosmos 3, in the shape the model was actually trained on.

── Why this file exists at all ─────────────────────────────────────────────────
Cosmos 3 is not prompted like Stable Diffusion. It was trained on long, STRUCTURED
JSON captions describing subjects, background, lighting, cinematography and a
per-segment temporal breakdown, and the checkpoint ships one as
`assets/example_t2v_prompt.json` next to the one-line version of the same scene in
`assets/example_t2v_prompt_short.txt` ("A robot arm is cleaning a plate in the
kitchen"). NVIDIA's own guidance is to *upsample* the short form into the long one
with an LLM before generation, via the `cosmos-framework` package.

We do not want a second model in the loop just to write a caption, so this file
takes the middle road: `structured()` assembles the same JSON schema from a handful
of plain fields, and the scenes below fill it in. The pipeline still accepts a raw
string, so a bare sentence works too; it is simply further from the training
distribution and looks it.

The prompt is passed to the pipeline as a plain `str`. Cosmos 3 tokenizes it into a
chat template, so a JSON *document* and a sentence are the same kind of input as far
as the API is concerned.

── The scenes ──────────────────────────────────────────────────────────────────
All four are physical-AI scenes on purpose, not scenery. A world model earns its
name by getting contact, occlusion and momentum right, and a drone shot over a
mountain range tests none of those. Each one names a specific physical event you can
check in the output: does the sponge deform, does the box tip, does the pallet stay
put when the forks lift.
"""

from __future__ import annotations

import json

# The canonical negative prompt from the checkpoint (`assets/negative_prompt.json`),
# reduced to prose. The full asset is another structured JSON document; its whole
# content is "everything that makes a generated video look generated", and the
# pipeline concatenates the resolution/duration templates onto it, so a readable
# paragraph behaves the same as the JSON here.
NEGATIVE = (
    "Blurry, poorly defined subjects with inconsistent shapes and unrealistic "
    "proportions. Distorted features, visible compression artifacts, muddy textures, "
    "color bleeding. Subjects float and are improperly grounded, without correct "
    "occlusion or spatial coherence. Incoherent motion with visible frame-to-frame "
    "discontinuities; movement appears as a slideshow rather than smooth animation. "
    "Objects intersect each other. Physically impossible motion, objects that "
    "accelerate without a cause or come to rest without contact. Static, frozen "
    "frames. Overexposed, washed out, low quality, jpeg artifacts."
)


def structured(
    *,
    subject: str,
    subject_action: str,
    background: str,
    temporal: str,
    camera_motion: str = "Static tripod shot, no camera movement",
    framing: str = "Medium shot",
    lighting: str = "Bright, even, diffuse indoor lighting with soft shadows",
    style: str = "Live-action video, realistic, documentary camera",
    duration_s: int = 5,
    fps: int = 24,
) -> str:
    """Build a Cosmos 3 structured caption from plain fields.

    The key names and nesting match `assets/example_t2v_prompt.json` in the
    checkpoint. Fields the example leaves blank for non-human subjects (clothing,
    expression, gender, ...) are omitted rather than sent empty: the example ships
    them as "" and the tokenizer sees the same nothing either way, but a shorter
    document leaves more of the context window for the parts that carry signal.

    `temporal` is the one field worth writing carefully. It is the beat-by-beat
    description of what happens over the clip, and it is what makes the difference
    between a video model and a world model: it is where you state the physical
    event you want predicted.
    """
    doc = {
        "subjects": [
            {
                "description": subject,
                "action": subject_action,
                "location": "Center of frame",
                "relative_size": "Large within frame",
                "number_of_subjects": 1,
            }
        ],
        "background_setting": background,
        "lighting": {"conditions": lighting},
        "cinematography": {
            "camera_motion": camera_motion,
            "framing": framing,
            "focus": "Sharp focus on the primary subject throughout",
        },
        "style_medium": style,
        "temporal_caption": temporal,
        "duration": f"{duration_s}s",
        "fps": fps,
    }
    return json.dumps(doc, indent=2)


# ── Scenes ──────────────────────────────────────────────────────────────────────

SCENES: dict[str, str] = {
    "arm-sponge": structured(
        subject=(
            "A modern industrial robotic arm with a silver and dark gray metallic "
            "body, multiple articulated joints, and a rubber-padded parallel gripper "
            "holding a yellow and green kitchen sponge."
        ),
        subject_action="Wiping a dirty white ceramic plate with slow circular strokes",
        background=(
            "A residential kitchen countertop in light gray granite, white cabinetry "
            "behind, a stainless steel sink at the left edge of frame."
        ),
        temporal=(
            "The arm lowers the sponge until it contacts the plate and the sponge "
            "visibly compresses against the ceramic. It then sweeps left to right "
            "across the plate, the sponge staying in continuous contact and deforming "
            "around the raised rim as it crosses it. Dried sauce clears behind each "
            "stroke, revealing clean white ceramic. The arm lifts away at the end and "
            "the sponge springs back to its original shape."
        ),
        framing="Medium close-up on the plate and gripper",
    ),
    "box-topple": structured(
        subject=(
            "A cardboard shipping box, roughly 40 cm on a side, with printed shipping "
            "labels and visible tape seams."
        ),
        subject_action="Being pushed from one edge until it tips over",
        background=(
            "A concrete warehouse floor with painted yellow lane markings, metal "
            "shelving racks receding into the background."
        ),
        temporal=(
            "The box sits still on the concrete. A steady horizontal push begins at "
            "its top edge; the box slides slightly, then pivots about its bottom front "
            "edge as the push continues. It passes the balance point, accelerates "
            "under gravity, and lands flat on its side with a small bounce and a puff "
            "of dust. It comes to rest and does not move again."
        ),
        camera_motion="Static tripod shot at box height, no camera movement",
    ),
    "forklift": structured(
        subject=(
            "An autonomous warehouse forklift, orange and black, with a mast, two "
            "steel forks, and a sensor pod mounted above the cab."
        ),
        subject_action="Lifting a loaded wooden pallet off the floor",
        background=(
            "A large distribution warehouse with tall pallet racking, sealed concrete "
            "floor, and overhead fluorescent lighting."
        ),
        temporal=(
            "The forklift approaches the pallet in a straight line and slows to a stop "
            "with the forks under the pallet slats. The mast raises; the pallet lifts "
            "clear of the floor and the stacked boxes on it sway briefly and settle. "
            "The forklift reverses in a straight line, the load staying level."
        ),
        camera_motion="Slow dolly right, following the forklift",
        framing="Wide shot",
        lighting="Overhead industrial fluorescent lighting, cool white, hard shadows",
    ),
    "dashcam": structured(
        subject=(
            "The interior of a car seen from the driver's seat, dark dashboard and "
            "windshield frame visible at the bottom of the frame."
        ),
        subject_action="Driving forward along a two-lane road and braking",
        background=(
            "A coastal highway carved into steep rocky cliffs, ocean to the right, "
            "guardrail along the road edge, clear afternoon sky."
        ),
        temporal=(
            "The car travels forward at steady speed, the road curving gently right "
            "and the guardrail sweeping past. A rock tumbles down the cliff onto the "
            "lane ahead. The car brakes hard: the view pitches forward as the "
            "suspension compresses, the scene slows, and the car comes to a complete "
            "stop short of the rock."
        ),
        camera_motion="Fixed dashcam, rigidly mounted, moving with the vehicle",
        framing="Ego view through the windshield",
        lighting="Bright natural daylight, sun high and slightly behind the camera",
    ),
}

# The one-line versions, for the side-by-side that shows what the structured caption
# is actually buying you. Same scenes, written the way you would prompt any other
# video model.
SHORT: dict[str, str] = {
    "arm-sponge": "A robot arm is cleaning a plate in the kitchen",
    "box-topple": "A cardboard box is pushed over on a warehouse floor",
    "forklift": "An autonomous forklift lifts a pallet in a warehouse",
    "dashcam": "Dashcam view of a car braking for a rock on a coastal road",
}


def get(scene: str, style: str = "structured") -> str:
    """Look up a scene by name. `style` is "structured" or "short"."""
    table = SHORT if style == "short" else SCENES
    if scene not in table:
        raise KeyError(f"unknown scene {scene!r}; have {sorted(table)}")
    return table[scene]
