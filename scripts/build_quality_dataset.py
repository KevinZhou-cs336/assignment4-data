"""Build fastText train/valid files for the quality classifier.

Positives (__label__wiki): pages linked from Wikipedia, fetched with scripts/fetch_urls.sh
and cleaned with our language ID, Gopher, and harmful content filters.
Negatives (__label__cc): random Common Crawl pages, kept only if English so the classifier
learns quality rather than language.

Usage:
    uv run python scripts/build_quality_dataset.py \
        --positive-warcs data/wiki/warc/*.warc.gz \
        --negative-warc data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz \
        --out-dir data/quality
"""

import argparse
import random
from collections import Counter
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import identify_language
from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality_filters import gopher_failure_reason

LANG_THRESHOLD = 0.65
HARMFUL_THRESHOLD = 0.9
MIN_WORDS = 50


def decode_chunked(body):
    # wget stores the raw HTTP body, so chunked responses still contain "<hex size>\r\n" markers
    out = bytearray()
    pos = 0
    while True:
        line_end = body.find(b"\r\n", pos)
        if line_end == -1:
            return body  # not valid chunked data; keep the original bytes
        try:
            # The size line may carry extensions after ";"
            size = int(body[pos:line_end].split(b";")[0].strip(), 16)
        except ValueError:
            return body
        if size == 0:
            return bytes(out)
        start = line_end + 2
        out += body[start : start + size]
        pos = start + size + 2  # skip the chunk's trailing \r\n


def iter_warc_text(warc_paths):
    # yields extracted text for successful HTML responses
    for path in warc_paths:
        with open(path, "rb") as f:
            for record in ArchiveIterator(f, record_types=WarcRecordType.response):
                http = record.http_headers
                if http is None or http.status_code != 200 or "html" not in http.get("Content-Type", ""):
                    continue
                body = record.reader.read()
                if "chunked" in (http.get("Transfer-Encoding") or "").lower():
                    body = decode_chunked(body)
                yield extract_text_from_html_bytes(body)


def is_english(text):
    lang, score = identify_language(text)
    return lang == "en" and score >= LANG_THRESHOLD


def positive_rejection(text):
    # Return why a Wikipedia-linked page is dropped, or None to keep it
    if not is_english(text):
        return "not_english"
    reason = gopher_failure_reason(text)
    if reason is not None:
        return f"gopher_{reason}"
    label, score = classify_nsfw(text)
    if label == "nsfw" and score >= HARMFUL_THRESHOLD:
        return "nsfw"
    label, score = classify_toxic_speech(text)
    if label == "toxic" and score >= HARMFUL_THRESHOLD:
        return "toxic"
    return None


def to_fasttext_line(label, text):
    # Mask PII so the classifier can't key on specific emails/phones/IPs
    text, _ = mask_emails(text)
    text, _ = mask_phone_numbers(text)
    text, _ = mask_ips(text)
    # fastText expects one example per line
    return f"__label__{label} " + " ".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-warcs", nargs="+", required=True)
    parser.add_argument("--negative-warc", required=True)
    parser.add_argument("--out-dir", default="data/quality")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    positives = []
    rejections = Counter()
    for text in iter_warc_text(args.positive_warcs):
        reason = positive_rejection(text)
        rejections[reason or "kept"] += 1
        if reason is None:
            positives.append(text)
    total = sum(rejections.values())
    print(f"positives: kept {len(positives)} of {total} HTML pages")
    for reason, count in rejections.most_common():
        print(f"  {reason}: {count} ({count / total:.1%})")

    # Take as many English CC pages as positives to keep the classes balanced
    negatives = []
    for text in iter_warc_text([args.negative_warc]):
        if len(text.split()) >= MIN_WORDS and is_english(text):
            negatives.append(text)
            if len(negatives) >= len(positives):
                break
    print(f"negatives: {len(negatives)} English CC pages")

    examples = [to_fasttext_line("wiki", t) for t in positives] + [to_fasttext_line("cc", t) for t in negatives]
    random.seed(args.seed)
    random.shuffle(examples)
    n_valid = int(len(examples) * args.valid_fraction)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality.valid").write_text("\n".join(examples[:n_valid]) + "\n")
    (out_dir / "quality.train").write_text("\n".join(examples[n_valid:]) + "\n")
    print(f"wrote {len(examples) - n_valid} train / {n_valid} valid examples to {out_dir}")


if __name__ == "__main__":
    main()
