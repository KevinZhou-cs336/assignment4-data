from pathlib import Path

import fasttext


# Resolve relative to this file, not the current working directory
model_path = str(Path(__file__).parent / "models" / "lid.176.bin")

# 2. Load the classifier
model = fasttext.load_model(model_path)

LanguageCode = str
Score = float


def identify_language(input_string: str) -> tuple[LanguageCode, Score]:
    """Return (language code like "en"/"zh", confidence in [0, 1])."""
    # fastText predicts on a single line, so newlines must be removed first
    predictions = model.predict(input_string.replace("\n", " "), k=1)
    # Output: (('__label__en',), array([0.966046]))

    # Clean up the output formatting
    language_code = predictions[0][0].replace("__label__", "")
    # Convert numpy.float64 to a plain float; softmax rounding can slightly exceed 1
    confidence = min(float(predictions[1][0]), 1.0)
    return (language_code, confidence)