from functools import cache
from pathlib import Path

import fasttext


# Resolve relative to this file, not the current working directory
QUALITY_MODEL_PATH = Path(__file__).parent / "models" / "quality_classifier.bin"

Label = str
Score = float


@cache
def _load_model():
    # Load lazily so importing this module doesn't require the trained model
    return fasttext.load_model(str(QUALITY_MODEL_PATH))


def classify_quality(text: str) -> tuple[Label, Score]:
    """Return ("wiki" for high quality or "cc" for low quality, confidence in [0, 1])."""
    # fastText predicts on a single line, so newlines must be removed first
    labels, scores = _load_model().predict(" ".join(text.split()), k=1)
    label = labels[0].replace("__label__", "")
    # Convert numpy.float64 to a plain float; softmax rounding can slightly exceed 1
    return label, min(float(scores[0]), 1.0)


def quality_score(text: str) -> Score:
    """Return P(high quality) in [0, 1], for ranking or thresholding pages."""
    label, score = classify_quality(text)
    return score if label == "wiki" else 1 - score
