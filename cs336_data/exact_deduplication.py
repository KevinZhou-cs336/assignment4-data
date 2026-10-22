import hashlib
import os
from collections import Counter
from pathlib import Path


def _line_key(line: bytes) -> bytes:
    # Fixed-size key, so memory doesn't grow with line length.
    # Strip the newline so a last line without "\n" matches the same line elsewhere.
    return hashlib.md5(line.rstrip(b"\r\n")).digest()


def exact_line_deduplication(paths: list[os.PathLike], output_dir: os.PathLike):
    """Rewrite each file into output_dir, keeping only lines that occur once in the whole corpus.

    A line that appears more than once (in the same file or across files) is removed
    everywhere, not kept once.
    """
    # Pass 1: count how often each line occurs across all files
    counts = Counter()
    for path in paths:
        with open(path, "rb") as f:
            for line in f:
                counts[_line_key(line)] += 1

    # Pass 2: rewrite each file with only its unique lines
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with open(path, "rb") as f_in, open(output_dir / Path(path).name, "wb") as f_out:
            for line in f_in:
                if counts[_line_key(line)] == 1:
                    f_out.write(line)
