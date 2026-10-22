"""Run PII masking on text extracted from a WARC file and show random matches.

Usage:
    uv run python scripts/pii_warc.py data/xxx.warc.gz --pool 3000 --n 20
"""

import argparse
import random

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.pii import EMAIL_RE, IPV4_RE, PHONE_RE

PATTERNS = {"email": EMAIL_RE, "phone": PHONE_RE, "ip": IPV4_RE}


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
    parser.add_argument("--pool", type=int, default=3000, help="number of pages to scan")
    parser.add_argument("--n", type=int, default=20, help="number of random matches to print per type")
    parser.add_argument("--context", type=int, default=60, help="chars of context on each side")
    parser.add_argument("--type", choices=[*PATTERNS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    types = list(PATTERNS) if args.type == "all" else [args.type]
    matches = {t: [] for t in types}  # type -> [(url, context with match marked)]
    pages_with_match = {t: 0 for t in types}
    total = 0
    for url, text in iter_warc_text(args.warc, args.pool):
        total += 1
        for t in types:
            found = False
            for m in PATTERNS[t].finditer(text):
                found = True
                left = text[max(0, m.start() - args.context) : m.start()]
                right = text[m.end() : m.end() + args.context]
                context = " ".join(f"{left}[[{m.group()}]]{right}".split())
                matches[t].append((url, context))
            pages_with_match[t] += found

    random.seed(args.seed)
    for t in types:
        print("#" * 100)
        print(f"{t}: {len(matches[t])} matches on {pages_with_match[t]}/{total} pages")
        for i, (url, context) in enumerate(random.sample(matches[t], min(args.n, len(matches[t]))), 1):
            print(f"[{i}] {url}")
            print(f"    {context}")


if __name__ == "__main__":
    main()
