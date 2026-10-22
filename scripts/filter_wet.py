"""Filter Common Crawl WET files into language modeling data, one process per WET file.

Each WET file becomes a .jsonl.gz of {"url", "text"} documents in the output directory,
and per-filter counts are summed over all files and printed at the end.

Usage:
    # local, one process per CPU
    uv run python scripts/filter_wet.py data/*.warc.wet.gz --out data/filtered
    # Slurm array on the Together cluster
    uv run python scripts/filter_wet.py "/data/CC/CC*.warc.wet.gz" --out /data/$USER/filtered --slurm
"""

import argparse
import concurrent.futures
import glob
import gzip
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

from cs336_data.c4_filters import c4_clean
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import identify_language
from cs336_data.quality_filters import gopher_failure_reason

# Filters run in this order, cheapest first, so expensive ones see fewer documents
STAGES = ["language", "c4_rules", "gopher", "harmful", "quality"]


def filter_document(text: str, args) -> tuple[str | None, str | None, str | None]:
    """Return (kept text, None, None), or (None, stage, detail) for the stage that dropped it."""
    language, score = identify_language(text)
    if language != "en" or score < args.min_lang_score:
        return None, "language", "not_english" if language != "en" else "low_score"

    text, reason = c4_clean(text)
    if text is None:
        return None, "c4_rules", reason

    reason = gopher_failure_reason(text)
    if reason is not None:
        return None, "gopher", reason

    for name, classify in (("nsfw", classify_nsfw), ("toxic", classify_toxic_speech)):
        label, score = classify(text)
        if label == name and score >= args.max_harmful_score:
            return None, "harmful", name

    if args.min_quality is not None:
        from cs336_data.quality_classifier import quality_score

        if quality_score(text) < args.min_quality:
            return None, "quality", "low_quality"

    return text, None, None


def process_single_wet_file(input_path: str, output_path: str, args) -> Counter:
    counts = Counter()
    # A sample of removed/modified documents goes to removed/<name>, for inspecting the filters
    sample_path = Path(output_path).parent / "removed" / Path(output_path).name
    sample_path.parent.mkdir(exist_ok=True)
    rng = random.Random(Path(input_path).name)
    with (
        open(input_path, "rb") as f_in,
        gzip.open(output_path, "wt", encoding="utf-8") as f_out,
        gzip.open(sample_path, "wt", encoding="utf-8") as f_sample,
    ):
        for record in ArchiveIterator(f_in, record_types=WarcRecordType.conversion):
            text = record.reader.read().decode("utf-8", errors="replace")
            url = record.headers.get("WARC-Target-URI")
            counts["total"] += 1
            counts["chars_in"] += len(text)

            kept, stage, detail = filter_document(text, args)
            if kept is None:
                counts[f"removed:{stage}"] += 1
                counts[f"removed:{stage}:{detail}"] += 1
                if rng.random() < args.sample_removed:
                    row = {"url": url, "stage": stage, "detail": detail, "text": text}
                    f_sample.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            counts["kept"] += 1
            counts["chars_out"] += len(kept)
            if kept != text and rng.random() < args.sample_removed:
                row = {"url": url, "stage": "modified", "detail": "c4_rules", "text": text, "kept": kept}
                f_sample.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.write(json.dumps({"url": url, "text": kept}, ensure_ascii=False) + "\n")
    return counts


def make_executor(args):
    if not args.slurm:
        return concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)

    import submitit

    executor = submitit.AutoExecutor(folder="slurm_logs")
    executor.update_parameters(
        slurm_array_parallelism=args.workers,
        timeout_min=60,
        mem_gb=8,  # the two Jigsaw fastText models alone take ~2GB per process
        cpus_per_task=2,
        slurm_account="student",
        slurm_partition="a4-cpu",
        slurm_qos="a4-cpu-qos",
    )
    return executor


def report(counts: Counter, elapsed: float, num_files: int):
    total, kept = counts["total"], counts["kept"]
    removed = total - kept
    print(f"\n{num_files} WET files, {total} documents, {elapsed:.0f}s ({elapsed / num_files:.1f}s per file)")
    print(f"kept {kept} ({kept / total:.1%}), chars kept {counts['chars_out'] / counts['chars_in']:.1%}")

    print("\nremoved by stage (share of all removed documents | of documents reaching the stage):")
    reaching = total
    for stage in STAGES:
        n = counts[f"removed:{stage}"]
        if reaching == 0:
            break
        print(f"  {stage:10s} {n:8d}  {n / max(removed, 1):6.1%} | {n / reaching:6.1%}")
        details = sorted(
            ((k.split(":")[2], v) for k, v in counts.items() if k.startswith(f"removed:{stage}:")),
            key=lambda kv: -kv[1],
        )
        for detail, v in details:
            print(f"    {detail:20s} {v:8d}")
        reaching -= n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="WET files or glob patterns")
    parser.add_argument("--out", required=True)
    parser.add_argument("--slurm", action="store_true", help="submit a Slurm job array via submitit")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="processes, or max parallel Slurm jobs")
    parser.add_argument("--min-lang-score", type=float, default=0.65)
    parser.add_argument("--max-harmful-score", type=float, default=0.9)
    parser.add_argument("--min-quality", type=float, default=None, help="P(wiki) threshold; off by default")
    parser.add_argument("--sample-removed", type=float, default=0.01, help="fraction of removed docs to save")
    args = parser.parse_args()

    wet_paths = sorted(p for pattern in args.inputs for p in glob.glob(pattern))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    executor = make_executor(args)
    futures = []
    if args.slurm:
        import submitit

        # One Slurm job array instead of len(wet_paths) separate jobs
        with executor.batch():
            for path in wet_paths:
                output_path = str(out_dir / (Path(path).name.removesuffix(".warc.wet.gz") + ".jsonl.gz"))
                futures.append(executor.submit(process_single_wet_file, path, output_path, args))
        completed = submitit.helpers.as_completed(futures)
    else:
        for path in wet_paths:
            output_path = str(out_dir / (Path(path).name.removesuffix(".warc.wet.gz") + ".jsonl.gz"))
            futures.append(executor.submit(process_single_wet_file, path, output_path, args))
        completed = concurrent.futures.as_completed(futures)

    counts = Counter()
    for future in tqdm(completed, total=len(futures)):
        counts.update(future.result())
    report(counts, time.time() - start, len(wet_paths))


if __name__ == "__main__":
    main()
