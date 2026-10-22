"""Train the fastText quality classifier (wiki vs cc).

Usage:
    uv run python scripts/train_quality_classifier.py --data-dir data/quality \
        --out cs336_data/models/quality_classifier.bin
"""

import argparse
from pathlib import Path

import fasttext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/quality")
    parser.add_argument("--out", default="cs336_data/models/quality_classifier.bin")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    # fastText's default of 2M buckets gives a ~900MB model; far more than our data needs
    parser.add_argument("--bucket", type=int, default=200_000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model = fasttext.train_supervised(
        input=str(data_dir / "quality.train"),
        epoch=args.epoch,
        lr=args.lr,
        wordNgrams=args.word_ngrams,
        dim=args.dim,
        bucket=args.bucket,
    )
    n, precision, recall = model.test(str(data_dir / "quality.valid"))
    print(f"valid: {n} examples, precision@1 {precision:.3f}, recall@1 {recall:.3f}")
    for label in ("__label__wiki", "__label__cc"):
        stats = model.test_label(str(data_dir / "quality.valid"))[label]
        print(f"  {label}: precision {stats['precision']:.3f}, recall {stats['recall']:.3f}, f1 {stats['f1score']:.3f}")

    model.save_model(args.out)
    print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
