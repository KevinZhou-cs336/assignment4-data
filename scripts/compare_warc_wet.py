"""Compare our HTML->text extraction on a WARC file against Common Crawl's WET text.

Usage:
    uv run python scripts/compare_warc_wet.py data/xxx.warc.gz data/xxx.warc.wet.gz --n 5
"""

import argparse
import random

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract import extract_text_from_html_bytes


def load_warc_html(warc_path, limit):
    # url -> html bytes, only for HTML responses
    pages = {}
    with open(warc_path, "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            content_type = record.http_headers.get("Content-Type", "") if record.http_headers else ""
            if "html" not in content_type:
                continue
            pages[record.headers.get("WARC-Target-URI")] = record.reader.read()
            if len(pages) >= limit:
                break
    return pages


def load_wet_text(wet_path, urls):
    # url -> extracted text from the WET conversion records
    texts = {}
    with open(wet_path, "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.conversion):
            url = record.headers.get("WARC-Target-URI")
            if url in urls:
                texts[url] = record.reader.read().decode("utf-8", errors="replace")
                if len(texts) == len(urls):
                    break
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("warc")
    parser.add_argument("wet")
    parser.add_argument("--n", type=int, default=5, help="number of pages to print")
    parser.add_argument("--pool", type=int, default=500, help="number of WARC pages to sample from")
    parser.add_argument("--chars", type=int, default=1500, help="max chars shown per text")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pages = load_warc_html(args.warc, args.pool)
    wet = load_wet_text(args.wet, set(pages))
    matched = [url for url in pages if url in wet]
    print(f"{len(pages)} HTML pages from WARC, {len(matched)} found in WET\n")

    random.seed(args.seed)
    for url in random.sample(matched, min(args.n, len(matched))):
        ours = extract_text_from_html_bytes(pages[url])
        theirs = wet[url]
        print("=" * 100)
        print(url)
        print(f"ours: {len(ours)} chars, {ours.count(chr(10)) + 1} lines | "
              f"WET: {len(theirs)} chars, {theirs.count(chr(10)) + 1} lines")
        print("-" * 40, "ours (resiliparse)", "-" * 40)
        print(ours[: args.chars])
        print("-" * 40, "WET", "-" * 40)
        print(theirs[: args.chars])
        print()


if __name__ == "__main__":
    main()
