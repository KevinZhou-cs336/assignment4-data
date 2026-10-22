MIN_WORDS = 50
MAX_WORDS = 100_000
MIN_MEAN_WORD_LENGTH = 3
MAX_MEAN_WORD_LENGTH = 10
MAX_ELLIPSIS_LINE_FRACTION = 0.3
MIN_ALPHA_WORD_FRACTION = 0.8


def gopher_failure_reason(text: str) -> str | None:
    """Return the first Gopher rule (Rae et al., 2021) that text violates, or None if it passes.

    A document is rejected if it:
      - has fewer than 50 or more than 100,000 words
      - has a mean word length outside [3, 10] characters
      - has more than 30% of lines ending with an ellipsis
      - has fewer than 80% of words containing at least one alphabetic character
    """
    # Split on whitespace, so punctuation stays attached to words (e.g. "text.")
    words = text.split()
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        return "word_count"

    mean_word_length = sum(len(w) for w in words) / len(words)
    if not MIN_MEAN_WORD_LENGTH <= mean_word_length <= MAX_MEAN_WORD_LENGTH:
        return "mean_word_length"

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    ellipsis_lines = sum(line.endswith(("...", "…")) for line in lines)
    if ellipsis_lines / len(lines) > MAX_ELLIPSIS_LINE_FRACTION:
        return "ellipsis_lines"

    alpha_words = sum(any(c.isalpha() for c in w) for w in words)
    if alpha_words / len(words) < MIN_ALPHA_WORD_FRACTION:
        return "alpha_words"

    return None


def gopher_quality_filter(text: str) -> bool:
    """Return True if text passes the subset of Gopher quality rules."""
    return gopher_failure_reason(text) is None
