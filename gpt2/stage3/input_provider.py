"""
Input-content providers for Stages 4-5: what text accompanies a clamped direction.

Two reviewers independently asked the same question: is the generic-concept adapter a bad reader
because the direction can't be read, or because it was trained on one trivially copyable carrier
sentence and was never forced to read it? The capstone (Stage 7) already uses a different recipe --
several semantically varied passages per class that never contain the label word, with
phrasings held out for evaluation. `VariedInputs` gives the generic concepts the same recipe, so the
two halves of the paper can be compared like for like.

  TemplateInputs  (default, = what produced the existing results)
      4 carrier sentences that CONTAIN the label ("A note nearby reads: \"Paris\"."). Training and
      evaluation use the same sentences -- there is no held-out phrasing, so `heldout_is_unseen` is False.

  VariedInputs
      4 train + 2 held-out natural passages per concept that evoke the concept WITHOUT naming it
      (no label word appears in any of them -- enforced by tests/test_varied_inputs.py). A model can
      still copy "what the passage is about", but not the label string, and the held-out passages are
      never seen in training.

HAND-AUTHORED: the passages below were written for this experiment. Read them before running; adjust or
extend freely (the test only enforces no label leakage and train/held-out disjointness).
"""

import re
from typing import Dict, List

TRAINING_INPUT_TEMPLATES = [
    "A note nearby reads: \"{concept}\".",
    "Someone just mentioned {concept}.",
    "The word written on the page is {concept}.",
    "A sign nearby says {concept}.",
]
# Multiple carrier templates per label instead of one fixed sentence, repeated identically every epoch --
# the fix for the Stage 4 instability documented in the README (loss spikes / mode collapse on narrow,
# trivially memorizable data). They all still contain the label word.


class TemplateInputs:
    name = "template"
    heldout_is_unseen = False

    def train_texts(self, concept: str) -> List[str]:
        return [t.format(concept=concept) for t in TRAINING_INPUT_TEMPLATES]

    def eval_texts(self, concept: str) -> List[str]:
        # Stage 5 historically evaluated on the first template only (decorrelation.build_input_content's default).
        # Kept identical so existing results remain reproducible. NOTE: this sentence is also a training template.
        return [TRAINING_INPUT_TEMPLATES[0].format(concept=concept)]


# concept -> (train passages, held-out passages). None of them contains the concept word.
_VARIED: Dict[str, Dict[str, List[str]]] = {
    "Paris": {
        "train": [
            "She spent the morning climbing the iron tower and watching the river wind below.",
            "We queued outside the museum with the glass pyramid to see the famous portrait of the smiling woman.",
            "The cafe on the Left Bank served warm croissants with a view of the old cathedral on its island.",
            "The great arch stood at the end of the grand avenue while evening traffic circled it.",
        ],
        "heldout": [
            "He rode the metro up to the hilltop district and sketched the white domes of the basilica.",
            "Fashion week crowds gathered along the riverbank for the evening runway show.",
        ],
    },
    "Rome": {
        "train": [
            "Tourists tossed coins over their shoulders into the great baroque fountain.",
            "The ancient amphitheater loomed over the street, its arches worn by two thousand years.",
            "A line of pilgrims waited to enter the basilica beside the smallest country in the world.",
            "We climbed the steps of the forum, past broken columns of the old empire.",
        ],
        "heldout": [
            "The guide pointed out the dome of the pantheon, still the largest unreinforced concrete dome ever built.",
            "At sunset the seven hills glowed gold above the ruins of the emperors' palaces.",
        ],
    },
    "dog": {
        "train": [
            "The retriever bounded across the park, fetched the stick and dropped it at her feet.",
            "A furry pup wagged its tail and barked at the mail carrier from behind the fence.",
            "He clipped on the leash and the beagle pulled him down the sidewalk, nose to the ground.",
            "The shepherd sat obediently, ears up, waiting for the command to herd the sheep.",
        ],
        "heldout": [
            "The puppy chewed a slipper, then rolled over for a belly rub.",
            "Muddy paws padded across the kitchen floor as the golden retriever greeted everyone at the door.",
        ],
    },
    "cat": {
        "train": [
            "The tabby curled up on the windowsill and purred in the afternoon sun.",
            "A tiny kitten pounced on the dangling string, then batted it across the rug.",
            "It slipped silently along the garden wall, whiskers twitching, tail held high.",
            "The Siamese stretched, yawned, and kneaded the blanket before settling on his lap.",
        ],
        "heldout": [
            "The black feline stared at the bird feeder for an hour without moving a muscle.",
            "She woke to a soft paw tapping her cheek and a hungry meow demanding breakfast.",
        ],
    },
    "sad": {
        "train": [
            "He sat alone by the window as the rain fell, unable to hold back his tears.",
            "Her voice cracked and her shoulders sank as she read the letter for the third time.",
            "The room felt heavy and grey after the news, and nobody had the heart to speak.",
            "She stared at the empty chair, eyes red, feeling an aching hollowness in her chest.",
        ],
        "heldout": [
            "He walked home slowly in the dusk, head down, the loss pressing on every step.",
            "The child sobbed quietly into the pillow, certain nothing would ever feel right again.",
        ],
    },
    "happy": {
        "train": [
            "She danced around the kitchen, grinning from ear to ear, unable to stop laughing.",
            "The whole team cheered and hugged each other as the final whistle confirmed the win.",
            "He woke up smiling, the sun warm on his face, thrilled that the weekend had finally come.",
            "Her eyes sparkled as she opened the gift, and she squealed with delight.",
        ],
        "heldout": [
            "The children ran through the sprinkler, shrieking with joy on the first day of summer.",
            "He whistled all the way to work, cheerful and light, as though nothing could go wrong.",
        ],
    },
    "France": {
        "train": [
            "Rows of vines stretched across the Burgundy hills as the harvest began.",
            "The high-speed train sped past lavender fields and stone villages in the south.",
            "A baker pulled long crusty baguettes from the oven before sunrise in a small Normandy town.",
            "Cyclists in yellow jerseys raced through the Alps during the great summer tour.",
        ],
        "heldout": [
            "They sipped Champagne at a chateau in the Loire Valley while the fog lifted off the river.",
            "On the Riviera the yachts rocked gently in the harbor of Cannes as the film festival opened.",
        ],
    },
    "Italy": {
        "train": [
            "Gondolas drifted under stone bridges along the canals of the lagoon city.",
            "The Tuscan hills rolled away in cypress-lined rows beneath a golden afternoon sky.",
            "A nonna rolled out fresh pasta dough on a wooden board while the ragu simmered.",
            "Pastel houses clung to the cliffs above the sparkling Amalfi coastline.",
        ],
        "heldout": [
            "Vespas buzzed through the narrow lanes of a Sicilian hill town as the church bells rang.",
            "The tenor's voice filled the opera house in Milan, and the audience rose to applaud.",
        ],
    },
}


class VariedInputs:
    name = "varied"
    heldout_is_unseen = True

    def __init__(self, bank: Dict[str, Dict[str, List[str]]] = None):
        self.bank = bank or _VARIED

    def _get(self, concept: str, split: str) -> List[str]:
        if concept not in self.bank:
            raise KeyError(f"No varied passages for concept {concept!r}; add it to input_provider._VARIED "
                           f"(known: {sorted(self.bank)}).")
        return self.bank[concept][split]

    def train_texts(self, concept: str) -> List[str]:
        return list(self._get(concept, "train"))

    def eval_texts(self, concept: str) -> List[str]:
        return list(self._get(concept, "heldout"))


def get_input_provider(name: str):
    if name == "template":
        return TemplateInputs()
    if name == "varied":
        return VariedInputs()
    raise ValueError(f"Unknown input provider {name!r} (expected 'template' or 'varied').")


def leaks_label(text: str, labels: List[str]) -> List[str]:
    """Labels that appear as whole words (case-insensitive) in `text`."""
    return [l for l in labels if re.search(r"\b" + re.escape(l) + r"\b", text, flags=re.IGNORECASE)]
