import os
import shutil
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import mmh3


def normalize_text(text: str) -> str:
    # NFD splits accented letters into base letter + combining mark: "é" -> "e" + U+0301
    text = unicodedata.normalize("NFD", text)
    # Drop the combining marks, which removes accents
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    # Replace every Unicode punctuation char (category P*) with a space
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    # Collapse runs of whitespace into a single space
    return " ".join(text.split())


def _read_text(path: os.PathLike) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _ngrams(text: str, n: int) -> set[str]:
    words = normalize_text(text).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _minhash_signature(ngrams: set[str], num_of_hash: int) -> tuple[int, ...]:
    # Hash function i is murmurhash with seed i; keep the minimum over all n-grams
    return tuple(min(mmh3.hash(g, seed, signed=False) for g in ngrams) for seed in range(num_of_hash))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b)


def minhash_deduplication(
    paths: list[os.PathLike],
    num_of_hash: int,
    num_of_bands: int,
    ngram_len: int,
    jaccard_threshold: float,
    output_dir: os.PathLike,
):
    """Copy paths into output_dir, keeping one document from each cluster of fuzzy duplicates.

    Candidates come from LSH over minhash signatures; a candidate pair counts as a
    duplicate only if its true n-gram Jaccard similarity is >= jaccard_threshold.
    """
    if num_of_hash % num_of_bands != 0:
        raise ValueError("num_of_hash must be divisible by num_of_bands")
    rows_per_band = num_of_hash // num_of_bands

    # Normalize only for comparison; the files on disk stay untouched
    doc_ngrams = [_ngrams(_read_text(path), ngram_len) for path in paths]

    # LSH: documents whose signatures agree on every row of some band share a bucket
    buckets = defaultdict(list)
    for doc_id, ngrams in enumerate(doc_ngrams):
        if not ngrams:  # shorter than ngram_len words: nothing to compare, treat as unique
            continue
        signature = _minhash_signature(ngrams, num_of_hash)
        for band in range(num_of_bands):
            band_rows = signature[band * rows_per_band : (band + 1) * rows_per_band]
            buckets[(band, band_rows)].append(doc_id)

    candidate_pairs = set()
    for doc_ids in buckets.values():
        candidate_pairs.update(combinations(doc_ids, 2))

    # Union-find: duplicates of duplicates end up in the same cluster
    parent = list(range(len(paths)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in candidate_pairs:
        if _jaccard(doc_ngrams[a], doc_ngrams[b]) >= jaccard_threshold:
            # Root at the smaller id, so the earliest document in a cluster is the one kept
            ra, rb = sorted((find(a), find(b)))
            parent[rb] = ra

    # Keep each cluster's root, copying the original bytes
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, path in enumerate(paths):
        if find(doc_id) == doc_id:
            shutil.copy(path, output_dir / Path(path).name)
