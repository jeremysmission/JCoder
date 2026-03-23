"""JCoder demo script -- showcase key capabilities.

Runs a scripted walkthrough of JCoder features using mock backends
so it works on any machine without GPU, Ollama, or network access.

Usage:
    cd D:\\JCoder
    python scripts/demo.py
    python scripts/demo.py --live    # use real Ollama model (if available)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(str(_ROOT))

# Fix Windows stdout encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Suppress per-engine FAISS warnings (expected on CPU-only machines)
import logging
logging.getLogger("core.index_engine").setLevel(logging.ERROR)


def _banner(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _step(num: int, desc: str) -> None:
    print(f"\n--- Step {num}: {desc} ---")


def _pause(seconds: float = 1.0) -> None:
    time.sleep(seconds)


def demo_doctor() -> bool:
    """Quick environment probe (no subprocess, no Ollama check)."""
    _step(1, "Environment Health Check")
    checks = []

    # Python version
    v = sys.version_info
    ok = v >= (3, 10)
    checks.append(("Python", ok, f"{v.major}.{v.minor}.{v.micro}"))

    # Key packages
    for pkg, pip_name in [("yaml", "pyyaml"), ("click", "click"),
                          ("rich", "rich"), ("numpy", "numpy"),
                          ("httpx", "httpx")]:
        try:
            __import__(pkg)
            checks.append((pip_name, True, "installed"))
        except ImportError:
            checks.append((pip_name, False, "MISSING"))

    # FAISS (optional)
    try:
        __import__("faiss")
        checks.append(("faiss-cpu", True, "installed"))
    except ImportError:
        checks.append(("faiss-cpu", None, "not installed (FTS5 fallback OK)"))

    # Config files
    for cfg in ["config/agent.yaml", "config/memory.yaml"]:
        exists = (_ROOT / cfg).exists()
        checks.append((cfg, exists, "found" if exists else "MISSING"))

    for name, ok, detail in checks:
        tag = "[OK]" if ok else ("[WARN]" if ok is None else "[FAIL]")
        print(f"  {tag} {name}: {detail}")

    passed = sum(1 for _, ok, _ in checks if ok is not False)
    total = len(checks)
    print(f"\n  {passed}/{total} checks passed")
    return all(ok is not False for _, ok, _ in checks)


def demo_federated_discovery() -> int:
    """Show federated index discovery."""
    _step(2, "Federated Index Discovery")
    from core.index_discovery import discover_fts5_indexes

    # Check known index locations
    index_dirs = [
        Path("data/indexes"),
        Path(os.environ.get("JCODER_DATA", "data")) / "indexes",
    ]

    total_indexes = 0
    total_size_mb = 0.0

    for idx_dir in index_dirs:
        if not idx_dir.exists():
            continue
        try:
            indexes = discover_fts5_indexes(str(idx_dir))
        except FileNotFoundError:
            continue
        if not indexes:
            continue

        print(f"\n  Directory: {idx_dir}")
        print(f"  Found {len(indexes)} FTS5 indexes:")
        for info in indexes[:10]:
            print(f"    {info['name']:40s} {info['size_mb']:>8.1f} MB")
        if len(indexes) > 10:
            print(f"    ... and {len(indexes) - 10} more")
        total_indexes += len(indexes)
        total_size_mb += sum(i["size_mb"] for i in indexes)

    if total_indexes == 0:
        print("  No FTS5 indexes found (run ingestion first)")
    else:
        print(f"\n  Total: {total_indexes} indexes, {total_size_mb:.0f} MB")
    return total_indexes


def demo_federated_search(mock: bool = True) -> None:
    """Run sample queries across federated indexes."""
    _step(3, "Federated Search (RRF across all indexes)")

    data_dir = Path(os.environ.get("JCODER_DATA", "data"))
    idx_dir = data_dir / "indexes"

    if not idx_dir.exists() or not list(idx_dir.glob("*.fts5.db")):
        print("  [SKIP] No FTS5 indexes available for search demo")
        return

    from core.index_discovery import (
        discover_fts5_indexes,
        load_federated_config,
        build_federated_from_config,
    )

    # Load config
    memory_yaml = _ROOT / "config" / "memory.yaml"
    if memory_yaml.exists():
        config = load_federated_config(str(memory_yaml))
    else:
        config = {"rrf_k": 60, "index_dir": str(idx_dir), "indexes": {}}

    # Build federated search (cap at 15 indexes for demo speed)
    print(f"  Building federated search from {idx_dir}...")
    t0 = time.perf_counter()
    fed = build_federated_from_config(config, str(idx_dir))
    build_ms = (time.perf_counter() - t0) * 1000
    info = fed.list_indexes()

    # If too many indexes, keep only the 15 largest for demo responsiveness
    total_loaded = len(info)
    if total_loaded > 15:
        keep = {e["name"] for e in sorted(info, key=lambda x: -x.get("count", 0))[:15]}
        for e in info:
            if e["name"] not in keep:
                fed.remove_index(e["name"])
        info = fed.list_indexes()
        print(f"  Using {len(info)} of {total_loaded} indexes (capped for demo speed), loaded in {build_ms:.0f}ms")
    else:
        print(f"  Loaded {len(info)} indexes in {build_ms:.0f}ms")

    if not info:
        print("  [SKIP] No indexes loaded successfully")
        return

    # Sample queries
    queries = [
        "how to read a file in python",
        "SQL injection prevention",
        "git merge conflict resolution",
        "async await javascript promise",
        "docker container networking",
    ]

    print(f"\n  Running {len(queries)} sample queries (top_k=5):\n")
    for q in queries:
        t0 = time.perf_counter()
        results = fed.search(q, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  Q: {q}")
        print(f"     {len(results)} results in {elapsed_ms:.0f}ms")
        if results:
            top = results[0]
            preview = top.content[:80].replace("\n", " ")
            print(f"     Top: [{top.index_name}] {preview}...")
        print()

    fed.close()


def demo_eval_validation() -> None:
    """Validate the eval set structure."""
    _step(4, "Eval Set Validation (200-question benchmark)")
    import json

    eval_files = [
        _ROOT / "evaluation" / "agent_eval_set.json",
        _ROOT / "evaluation" / "agent_eval_set_200.json",
    ]

    for ef in eval_files:
        if not ef.exists():
            continue
        with open(ef, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        questions = data if isinstance(data, list) else data.get("questions", [])
        categories = {}
        for q in questions:
            cat = q.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\n  {ef.name}: {len(questions)} questions")
        for cat, count in sorted(categories.items()):
            print(f"    {cat:25s}: {count}")


def demo_bench_search() -> None:
    """Run a quick search benchmark."""
    _step(5, "Search Latency Benchmark")

    data_dir = Path(os.environ.get("JCODER_DATA", "data"))
    idx_dir = data_dir / "indexes"

    if not idx_dir.exists() or not list(idx_dir.glob("*.fts5.db")):
        print("  [SKIP] No FTS5 indexes for benchmarking")
        return

    from core.config import StorageConfig
    from core.index_engine import IndexEngine
    from core.federated_search import FederatedSearch

    db_files = sorted(idx_dir.glob("*.fts5.db"))[:10]  # cap at 10 for demo
    storage = StorageConfig(data_dir=str(idx_dir.parent), index_dir=str(idx_dir))
    fed = FederatedSearch(embedding_engine=None, max_workers=8)

    for db in db_files:
        name = db.stem.replace(".fts5", "")
        eng = IndexEngine(dimension=768, storage=storage, sparse_only=True)
        eng._db_path = str(db)
        fed.add_index(name, eng, weight=1.0)

    queries = [
        "binary search algorithm",
        "python decorator pattern",
        "memory leak detection",
    ]

    # Warm up
    fed.search(queries[0], top_k=5)

    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        results = fed.search(q, top_k=5)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        print(f"  {q:35s} -> {len(results)} results in {ms:.0f}ms")

    avg = sum(latencies) / len(latencies)
    print(f"\n  Avg latency: {avg:.0f}ms across {len(db_files)} indexes")
    fed.close()


def demo_test_count() -> None:
    """Show test suite size."""
    _step(6, "Test Suite Summary")
    test_dir = _ROOT / "tests"
    test_files = sorted(test_dir.glob("test_*.py"))
    print(f"  {len(test_files)} test files in tests/")
    for tf in test_files:
        print(f"    {tf.name}")


def main():
    parser = argparse.ArgumentParser(description="JCoder demo walkthrough")
    parser.add_argument("--live", action="store_true",
                        help="Use real Ollama model instead of mocks")
    args = parser.parse_args()

    _banner("JCoder Demo -- Local AI Coding Assistant")
    print("  Mode:", "LIVE (Ollama)" if args.live else "MOCK (no model needed)")
    print("  Platform:", sys.platform)
    print("  Python:", sys.version.split()[0])
    print("  Project:", _ROOT)

    t0 = time.time()

    demo_doctor()
    count = demo_federated_discovery()
    if count > 0:
        demo_federated_search(mock=not args.live)
        demo_bench_search()
    demo_eval_validation()
    demo_test_count()

    elapsed = time.time() - t0
    _banner(f"Demo Complete ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
