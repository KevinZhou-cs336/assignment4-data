from functools import cache
from pathlib import Path

import fasttext


# Resolve relative to this file, not the current working directory
MODELS_DIR = Path(__file__).parent / "models"
NSFW_MODEL_PATH = MODELS_DIR / "jigsaw_fasttext_bigrams_nsfw_final.bin"
TOXIC_MODEL_PATH = MODELS_DIR / "jigsaw_fasttext_bigrams_hatespeech_final.bin"

Label = str
Score = float


@cache
def _load_model(path: Path):
    # Each model is ~1GB, so load lazily on first use and keep it around
    return fasttext.load_model(str(path))


def _classify(model_path: Path, text: str) -> tuple[Label, Score]:
    # fastText predicts on a single line, so newlines must be removed first
    labels, scores = _load_model(model_path).predict(text.replace("\n", " "), k=1)
    label = labels[0].replace("__label__", "")
    # Convert numpy.float64 to a plain float; softmax rounding can slightly exceed 1
    return label, min(float(scores[0]), 1.0)


def classify_nsfw(text: str) -> tuple[Label, Score]:
    """Return ("nsfw" or "non-nsfw", confidence in [0, 1])."""
    return _classify(NSFW_MODEL_PATH, text)


def classify_toxic_speech(text: str) -> tuple[Label, Score]:
    """Return ("toxic" or "non-toxic", confidence in [0, 1])."""
    return _classify(TOXIC_MODEL_PATH, text)
