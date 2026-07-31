"""Shared prompts so every model in the shootout is judged on the same text.

Import PROMPTS in each model's synth.py and loop over the categories that
model actually supports (e.g. skip "multilingual" for English-only models,
skip "cloning" for models with no cloning support).
"""

PROMPTS = {
    "narration": (
        "The quarterly report shows a steady increase in adoption, "
        "with three new regions coming online next month."
    ),
    "expressive": (
        "Wait — you're telling me it actually worked? After all that?! "
        "I can't believe it."
    ),
    "multilingual": (
        "Buenos días, hoy vamos a comparar varios modelos de "
        "síntesis de voz de código abierto."
    ),
    "cloning": (
        "This is the sentence we're rendering in the cloned voice, "
        "so keep it a little longer than the others."
    ),
}
