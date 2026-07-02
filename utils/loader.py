"""
loader.py — ARIA Data Loader
Streams candidates.jsonl efficiently. Handles both JSON array and JSONL.
"""

import gzip
import json
import os
import sys


def stream_candidates(filepath: str):
    is_gz = filepath.endswith(".gz")
    opener = gzip.open if is_gz else open
    mode = "rt" if is_gz else "r"
    with opener(filepath, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_all_candidates(filepath: str, max_count: int = None, verbose: bool = True) -> list:
    candidates = []

    # Auto-detect JSON array vs JSONL
    if not filepath.endswith(".gz"):
        with open(filepath, "r", encoding="utf-8") as f:
            first_char = f.read(1)
        if first_char == "[":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            candidates = data[:max_count] if max_count else data
            if verbose:
                print(f"  Loaded {len(candidates):,} candidates (JSON array)", file=sys.stderr)
            return candidates

    for i, c in enumerate(stream_candidates(filepath)):
        candidates.append(c)
        if max_count and i + 1 >= max_count:
            break
        if verbose and (i + 1) % 10000 == 0:
            print(f"  Loaded {i+1:,} candidates...", file=sys.stderr)

    if verbose:
        print(f"  Total loaded: {len(candidates):,}", file=sys.stderr)
    return candidates


def find_candidates_file(base_dir: str = ".") -> str:
    for name in ["candidates.jsonl", "candidates.jsonl.gz", "sample_candidates.json"]:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No candidates file found in {base_dir}")
