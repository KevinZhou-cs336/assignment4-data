"""Run language identification on text extracted from a WARC file.

Usage:
    uv run python scripts/langid_warc.py data/xxx.warc.gz --pool 2000 --n 20
"""

import argparse
import random
from collections import Counter

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language


def iter_warc_text(warc_path, limit):
    # yields (url, extracted text) for HTML responses
    count = 0
    with open(warc_path, "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            content_type = record.http_headers.get("Content-Type", "") if record.http_headers else ""
            if "html" not in content_type:
                continue
            yield record.headers.get("WARC-Target-URI"), extract_text_from_html_bytes(record.reader.read())
            count += 1
            if count >= limit:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("warc")
    parser.add_argument("--pool", type=int, default=2000, help="number of pages to classify")
    parser.add_argument("--n", type=int, default=20, help="number of random examples to print")
    parser.add_argument("--chars", type=int, default=300, help="max chars shown per example")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = []  # (url, text, lang, score)
    for url, text in iter_warc_text(args.warc, args.pool):
        lang, score = identify_language(text)
        results.append((url, text, lang, score))

    # Overall language distribution
    langs = Counter(lang for _, _, lang, _ in results)
    total = len(results)
    print(f"{total} pages classified")
    print(f"English: {langs['en']} ({langs['en'] / total:.1%})")
    print("Top languages:", ", ".join(f"{k} {v / total:.1%}" for k, v in langs.most_common(10)))

    # Score distribution, to help choose a threshold
    scores = sorted(score for *_, score in results)
    for t in (0.3, 0.5, 0.65, 0.8, 0.9):
        kept = sum(s >= t for s in scores)
        print(f"score >= {t}: {kept / total:.1%} of pages kept")

    # Random examples for manual labeling
    random.seed(args.seed)
    for i, (url, text, lang, score) in enumerate(random.sample(results, min(args.n, total)), 1):
        snippet = " ".join(text.split())[: args.chars]
        print("=" * 100)
        print(f"[{i}] pred={lang} score={score:.3f}  {url}")
        print(snippet)


if __name__ == "__main__":
    main()
