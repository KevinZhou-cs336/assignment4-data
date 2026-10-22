"""Uniformly sample URLs from a (gzipped) URL list with reservoir sampling.

Usage:
    uv run python scripts/sample_urls.py data/wiki/enwiki-20240420-extracted_urls.txt.gz \
        data/wiki/sampled_urls.txt --n 5000
"""

import argparse
import gzip
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", help="input URL list, one per line (.gz ok)")
    parser.add_argument("out", help="output file with sampled URLs")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    opener = gzip.open if args.urls.endswith(".gz") else open
    reservoir = []
    total = 0
    with opener(args.urls, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            url = line.strip()
            # wget only handles http(s); skip ftp://, mailto:, etc.
            if not url.startswith(("http://", "https://")):
                continue
            total += 1
            if len(reservoir) < args.n:
                reservoir.append(url)
            else:
                # Keep each URL with probability n / total
                j = random.randrange(total)
                if j < args.n:
                    reservoir[j] = url

    with open(args.out, "w") as f:
        f.write("\n".join(reservoir) + "\n")
    print(f"sampled {len(reservoir)} of {total} http(s) URLs -> {args.out}")


if __name__ == "__main__":
    main()
