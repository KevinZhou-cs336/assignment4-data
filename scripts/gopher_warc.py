"""Run the Gopher quality filters on text extracted from a WARC file.

Usage:
    uv run python scripts/gopher_warc.py data/xxx.warc.gz --pool 2000 --n 20
"""

import argparse
import random
from collections import Counter

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language
from cs336_data.quality_filters import gopher_failure_reason


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


def report(name, rows):
    total = len(rows)
    reasons = Counter(r["reason"] for r in rows)
    print(f"--- {name}: {total} pages, {reasons[None] / total:.1%} pass ---")
    for reason, count in reasons.most_common():
        if reason is not None:
            print(f"  rejected by {reason}: {count / total:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("warc")
    parser.add_argument("--pool", type=int, default=2000, help="number of pages to filter")
    parser.add_argument("--n", type=int, default=20, help="number of random examples to print")
    parser.add_argument("--chars", type=int, default=300, help="max chars shown per example")
    parser.add_argument("--english-only", action="store_true", help="sample only English pages")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for url, text in iter_warc_text(args.warc, args.pool):
        words = text.split()
        rows.append({
            "url": url,
            "text": text,
            "lang": identify_language(text)[0],
            "reason": gopher_failure_reason(text),
            "n_words": len(words),
        })

    report("all languages", rows)
    english = [r for r in rows if r["lang"] == "en"]
    report("English only", english)

    pool = english if args.english_only else rows
    random.seed(args.seed)
    for i, r in enumerate(random.sample(pool, min(args.n, len(pool))), 1):
        snippet = " ".join(r["text"].split())[: args.chars]
        verdict = "PASS" if r["reason"] is None else f"FAIL({r['reason']})"
        print("=" * 100)
        print(f"[{i}] {verdict} lang={r['lang']} words={r['n_words']}  {r['url']}")
        print(snippet)


if __name__ == "__main__":
    main()
