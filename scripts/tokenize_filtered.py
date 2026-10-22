"""Tokenize filtered documents with the GPT-2 tokenizer into one uint16 .bin for training.

Each document is followed by <|endoftext|>. Files are tokenized in parallel and
appended to the output in input order.

Usage:
    uv run python scripts/tokenize_filtered.py data/deduped --out data/tokenized/train.bin
"""

import argparse
import gzip
import json
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = None


def _init_worker():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")


def tokenize_file(path: str) -> np.ndarray:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    ids = []
    for doc_ids in tokenizer(texts)["input_ids"]:  # fast tokenizer encodes the batch in Rust
        ids.extend(doc_ids)
        ids.append(tokenizer.eos_token_id)
    return np.array(ids, dtype=np.uint16)  # GPT-2 vocab is 50257 < 65536


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    paths = sorted(str(p) for p in Path(args.input_dir).glob("*.jsonl.gz"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with multiprocessing.Pool(args.workers, initializer=_init_worker) as pool, open(args.out, "wb") as f_out:
        for ids in tqdm(pool.imap(tokenize_file, paths), total=len(paths), desc="Tokenizing files"):
            ids.tofile(f_out)
            total += len(ids)
    print(f"Tokenized {len(paths)} files into {total} tokens -> {args.out}")


if __name__ == "__main__":
    main()
