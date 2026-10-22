"""Run the NSFW and toxic speech classifiers on text extracted from a WARC file.

Usage:
    uv run python scripts/harmful_warc.py data/xxx.warc.gz --pool 2000 --n 20
"""

import argparse
import random

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import identify_language

THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)


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


def harmful_prob(label, score, positive):
    # Convert (label, confidence) into P(positive label)
    return score if label == positive else 1 - score


def report(name, rows):
    total = len(rows)
    print(f"--- {name}: {total} pages ---")
    for key, positive in (("nsfw", "nsfw"), ("toxic", "toxic")):
        flagged = sum(r[key][0] == positive for r in rows)
        line = ", ".join(f">={t}: {sum(r[key + '_p'] >= t for r in rows) / total:.1%}" for t in THRESHOLDS)
        print(f"{key:>5}: predicted {positive} {flagged / total:.1%} | P({positive}) {line}")
    either = sum(r["nsfw"][0] == "nsfw" or r["toxic"][0] == "toxic" for r in rows)
    print(f"either harmful: {either / total:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("warc")
    parser.add_argument("--pool", type=int, default=2000, help="number of pages to classify")
    parser.add_argument("--n", type=int, default=20, help="number of random examples to print")
    parser.add_argument("--chars", type=int, default=300, help="max chars shown per example")
    parser.add_argument("--only-flagged", action="store_true", help="sample only pages flagged as harmful")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for url, text in iter_warc_text(args.warc, args.pool):
        nsfw = classify_nsfw(text)
        toxic = classify_toxic_speech(text)
        rows.append({
            "url": url,
            "text": text,
            "lang": identify_language(text)[0],
            "nsfw": nsfw,
            "toxic": toxic,
            "nsfw_p": harmful_prob(*nsfw, "nsfw"),
            "toxic_p": harmful_prob(*toxic, "toxic"),
        })

    report("all languages", rows)
    report("English only", [r for r in rows if r["lang"] == "en"])

    pool = rows
    if args.only_flagged:
        pool = [r for r in rows if r["nsfw"][0] == "nsfw" or r["toxic"][0] == "toxic"]
    random.seed(args.seed)
    for i, r in enumerate(random.sample(pool, min(args.n, len(pool))), 1):
        snippet = " ".join(r["text"].split())[: args.chars]
        print("=" * 100)
        print(f"[{i}] lang={r['lang']} nsfw={r['nsfw'][0]}({r['nsfw'][1]:.3f}) "
              f"toxic={r['toxic'][0]}({r['toxic'][1]:.3f})  {r['url']}")
        print(snippet)


if __name__ == "__main__":
    main()
