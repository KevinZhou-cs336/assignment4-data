"""Fuzzy-deduplicate filtered documents across all files with MinHash + LSH.

Scales to many millions of documents by never holding n-gram sets or full signatures in
memory: each worker returns only one 64-bit key per LSH band per document. Documents that
share any band key are treated as duplicates (no exact Jaccard check, which would need
re-reading candidate pairs), and only the first document of each cluster is kept.

Usage:
    uv run python scripts/dedup_filtered.py data/filtered --out data/deduped
"""

import argparse
import concurrent.futures
import gzip
import json
import os
import time
from pathlib import Path

import mmh3
import numpy as np
from tqdm import tqdm

from cs336_data.minhash_deduplication import normalize_text


def _mix(x: np.ndarray) -> np.ndarray:
    # splitmix64 finalizer; uint64 multiplication wraps mod 2^64, which is what scrambles the bits
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _band_keys(text: str, seeds: np.ndarray, num_bands: int, ngram_len: int) -> np.ndarray:
    words = normalize_text(text).split()
    ngrams = {" ".join(words[i : i + ngram_len]) for i in range(len(words) - ngram_len + 1)}
    if not ngrams:
        # Too short for n-grams: key every band on the whole text, so only exact copies collide
        return np.array([mmh3.hash64(text, band)[0] for band in range(num_bands)], dtype=np.int64)

    # Hash each n-gram once, then derive one hash function per seed by mixing (h + seed)
    h = np.fromiter((mmh3.hash64(g, signed=False)[0] for g in ngrams), dtype=np.uint64, count=len(ngrams))
    signature = _mix(h[None, :] + seeds[:, None]).min(axis=1)
    bands = signature.reshape(num_bands, -1)
    return np.array([mmh3.hash64(band.tobytes(), i)[0] for i, band in enumerate(bands)], dtype=np.int64)


def signatures_for_file(path: str, num_hashes: int, num_bands: int, ngram_len: int) -> np.ndarray:
    rng = np.random.default_rng(0)  # same hash functions in every worker
    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=num_hashes, dtype=np.uint64)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        keys = [_band_keys(json.loads(line)["text"], seeds, num_bands, ngram_len) for line in f]
    return np.array(keys, dtype=np.int64).reshape(-1, num_bands)


def rewrite_file(input_path: str, output_path: str, drop: np.ndarray) -> int:
    kept = 0
    with gzip.open(input_path, "rt", encoding="utf-8") as f_in, gzip.open(output_path, "wt", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if not drop[i]:
                f_out.write(line)
                kept += 1
    return kept


def find_duplicates(keys: np.ndarray) -> np.ndarray:
    """Union documents sharing a band key; return a mask of every document that isn't its cluster's root."""
    parent = np.arange(len(keys))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for band in range(keys.shape[1]):
        order = np.argsort(keys[:, band], kind="stable")
        sorted_keys = keys[order, band]
        # Neighbours in sorted order with equal keys share a bucket
        for i in np.nonzero(sorted_keys[1:] == sorted_keys[:-1])[0]:
            ra, rb = sorted((find(order[i]), find(order[i + 1])))
            parent[rb] = ra

    roots = np.array([find(x) for x in range(len(keys))])
    return roots != np.arange(len(keys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-hashes", type=int, default=100)
    parser.add_argument("--num-bands", type=int, default=10)
    parser.add_argument("--ngram-len", type=int, default=5)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    paths = sorted(str(p) for p in Path(args.input_dir).glob("*.jsonl.gz"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Pass 1: band keys per file; map keeps file order so global ids are stable
        jobs = executor.map(
            signatures_for_file,
            paths,
            [args.num_hashes] * len(paths),
            [args.num_bands] * len(paths),
            [args.ngram_len] * len(paths),
        )
        per_file = list(tqdm(jobs, total=len(paths), desc="signatures"))

        # Pass 2: cluster globally
        keys = np.concatenate(per_file)
        drop = find_duplicates(keys)

        # Pass 3: rewrite each file without its duplicates
        offsets = np.cumsum([0] + [len(k) for k in per_file])
        futures = [
            executor.submit(rewrite_file, path, str(out_dir / Path(path).name), drop[offsets[i] : offsets[i + 1]])
            for i, path in enumerate(paths)
        ]
        kept = sum(f.result() for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="rewrite"))

    total = len(keys)
    print(f"{total} documents, removed {total - kept} fuzzy duplicates ({(total - kept) / total:.1%}), "
          f"kept {kept}, {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
