"""Eval guard -- ensures benchmark integrity.

Benchmarks are immutable by design. The evolver cannot modify
eval files. Hashes are stored in config/ (separate from evaluation/)
so modifying benchmarks AND hashes requires touching two directories.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


# Hashes live in config/, NOT alongside the mutable benchmark files.
# This means an evolver that writes to evaluation/ cannot also update hashes.
HASH_FILENAME = "benchmark_hashes.json"


def _default_hash_path() -> str:
    """config/benchmark_hashes.json relative to repo root."""
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "config" / HASH_FILENAME)


def compute_file_hash(path: str) -> str:
    """SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_hashes(eval_dir: str) -> dict:
    """Compute hashes for all JSON files in the evaluation directory."""
    hashes = {}
    for name in os.listdir(eval_dir):
        if name.endswith(".json") and name != HASH_FILENAME:
            path = os.path.join(eval_dir, name)
            hashes[name] = compute_file_hash(path)
    return hashes


def save_hashes(eval_dir: str, hash_path: Optional[str] = None):
    """Write hash manifest to config/. Run once after authoring benchmarks."""
    if hash_path is None:
        hash_path = _default_hash_path()
    hashes = generate_hashes(eval_dir)
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)


def verify_hashes(eval_dir: str, hash_path: Optional[str] = None) -> Optional[str]:
    """
    Verify all benchmark files match their recorded hashes.
    Returns None if OK, or an error string describing the problem.

    REFUSES to run if hash file is missing (no silent pass-through).
    """
    if hash_path is None:
        hash_path = _default_hash_path()

    if not os.path.exists(hash_path):
        return (
            f"Hash manifest not found: {hash_path}. "
            f"Run 'jcoder seal-benchmarks' to generate it."
        )

    with open(hash_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    if not expected:
        return "Hash manifest is empty -- no benchmarks registered."

    for name, expected_hash in expected.items():
        path = os.path.join(eval_dir, name)
        if not os.path.exists(path):
            return f"Missing benchmark file: {name}"
        actual = compute_file_hash(path)
        if actual != expected_hash:
            return (
                f"Benchmark tampered: {name} "
                f"(expected {expected_hash[:12]}..., got {actual[:12]}...)"
            )

    # Detect unlisted JSON files (prevents sneaking in new benchmarks)
    listed = set(expected.keys())
    for name in os.listdir(eval_dir):
        if name.endswith(".json") and name != HASH_FILENAME and name not in listed:
            return (
                f"Unlisted benchmark file: {name}. "
                f"Re-run 'jcoder seal-benchmarks' to register it."
            )

    return None
