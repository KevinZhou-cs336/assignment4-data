"""Heuristic cleaning rules used to build C4 (Raffel et al., 2020, Section 2.2)."""

import re

TERMINAL_PUNCTUATION = (".", "!", "?", '"', "”")
MIN_WORDS_PER_LINE = 5
MIN_SENTENCES = 3
POLICY_PHRASES = ("terms of use", "privacy policy", "cookie policy", "uses cookies", "use of cookies", "use cookies")

SENTENCE_END_RE = re.compile(r"[.!?]+[\"”]?(?=\s|$)")


def _keep_line(line: str) -> bool:
    lowered = line.lower()
    return (
        line.endswith(TERMINAL_PUNCTUATION)
        and len(line.split()) >= MIN_WORDS_PER_LINE
        and "javascript" not in lowered
        and not any(phrase in lowered for phrase in POLICY_PHRASES)
    )


def c4_clean(text: str) -> tuple[str | None, str | None]:
    """Return (cleaned text, None), or (None, reason) if the whole page is dropped.

    Lines that fail the C4 line rules are removed; pages with lorem ipsum, code
    (a curly brace), or fewer than MIN_SENTENCES sentences left are dropped.
    """
    lowered = text.lower()
    if "lorem ipsum" in lowered:
        return None, "lorem_ipsum"
    if "{" in text:
        return None, "curly_brace"

    kept = [line.strip() for line in text.splitlines() if _keep_line(line.strip())]
    cleaned = "\n".join(kept)
    if len(SENTENCE_END_RE.findall(cleaned)) < MIN_SENTENCES:
        return None, "too_few_sentences"
    return cleaned, None
