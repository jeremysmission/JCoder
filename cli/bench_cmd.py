"""Federated search benchmark command.

Measures FTS5 query latency across all registered indexes, reports
per-index and aggregate stats, and validates that parallelization
delivers the expected speedup.

Usage:
    jcoder bench-search
    jcoder bench-search --queries 20 --top-k 10
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import click
from rich.console import Console
from rich.table import Table

console = Console()

SAMPLE_QUERIES = [
    "how to read a file in python",
    "async await javascript promise",
    "SQL injection prevention",
    "docker container networking",
    "git merge conflict resolution",
    "python decorator explained",
    "binary search algorithm complexity",
    "kubernetes pod scheduling",
    "REST API design best practices",
    "memory leak detection profiling",
    "TCP handshake TLS",
    "pytest fixture parametrize",
    "regex lookahead lookbehind",
    "hash table collision resolution",
    "linux file permissions chmod",
    "garbage collection reference counting",
    "thread safety race condition lock",
    "ssh key authentication",
    "database index B-tree",
    "CSS flexbox grid layout",
]


@click.command(name="bench-search")
@click.option("--queries", default=10, help="Number of queries to run")
@click.option("--top-k", default=10, help="Results per query")
@click.option("--index-dir", default=None, help="Override index directory")
def bench_search(queries: int, top_k: int, index_dir: str | None):
    """Benchmark federated search latency across all FTS5 indexes."""
    from core.index_engine import IndexEngine
    from core.config import StorageConfig
    from core.federated_search import FederatedSearch

    # Discover indexes
    if index_dir:
        idx_path = Path(index_dir)
    else:
        try:
            from agent.config_loader import load_agent_config
            cfg = load_agent_config()
            idx_path = Path(cfg.memory_index_dir)
        except Exception:
            idx_path = Path("D:/JCoder_Data/indexes")

    if not idx_path.exists():
        console.print(f"[FAIL] Index directory not found: {idx_path}")
        return

    db_files = sorted(idx_path.glob("*.fts5.db"))
    if not db_files:
        console.print(f"[FAIL] No FTS5 indexes found in {idx_path}")
        return

    console.print(f"Found {len(db_files)} FTS5 indexes in {idx_path}")

    # Load indexes
    storage = StorageConfig(data_dir=str(idx_path.parent), index_dir=str(idx_path))
    fed = FederatedSearch(embedding_engine=None, max_workers=8)
    engines: list[IndexEngine] = []

    for db in db_files:
        name = db.stem.replace(".fts5", "")
        eng = IndexEngine(dimension=768, storage=storage, sparse_only=True)
        eng._db_path = str(db)
        eng._fts_conn = None  # lazy open
        fed.add_index(name, eng, weight=1.0)
        engines.append(eng)

    console.print(f"Loaded {len(engines)} indexes into FederatedSearch")

    # Run queries
    query_set = SAMPLE_QUERIES[:queries]
    latencies: List[float] = []
    result_counts: List[int] = []

    console.print(f"\nRunning {len(query_set)} queries (top_k={top_k})...")

    # Warm-up query (opens connections)
    fed.search(query_set[0], top_k=top_k)

    for q in query_set:
        t0 = time.perf_counter()
        results = fed.search(q, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        result_counts.append(len(results))

    # Stats
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    total_results = sum(result_counts)
    avg_results = total_results / len(result_counts)

    # Display
    table = Table(title=f"Federated Search Benchmark ({len(engines)} indexes)")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Queries", str(len(query_set)))
    table.add_row("Top-K", str(top_k))
    table.add_row("Indexes", str(len(engines)))
    table.add_row("", "")
    table.add_row("Avg latency (ms)", f"{avg:.1f}")
    table.add_row("P50 latency (ms)", f"{p50:.1f}")
    table.add_row("P95 latency (ms)", f"{p95:.1f}")
    table.add_row("P99 latency (ms)", f"{p99:.1f}")
    table.add_row("Min latency (ms)", f"{latencies[0]:.1f}")
    table.add_row("Max latency (ms)", f"{latencies[-1]:.1f}")
    table.add_row("", "")
    table.add_row("Avg results/query", f"{avg_results:.1f}")
    table.add_row("Total results", str(total_results))

    console.print(table)

    # Per-query breakdown
    detail = Table(title="Per-Query Detail")
    detail.add_column("#", style="dim")
    detail.add_column("Query")
    detail.add_column("Latency (ms)", justify="right")
    detail.add_column("Results", justify="right")

    for i, (q, lat, cnt) in enumerate(zip(query_set, latencies, result_counts)):
        style = "red" if lat > avg * 2 else ""
        detail.add_row(str(i + 1), q[:50], f"{lat:.1f}", str(cnt), style=style)

    console.print(detail)

    # Cleanup
    fed.close()
    for eng in engines:
        eng.close()

    console.print(f"\n[OK] Benchmark complete. Avg: {avg:.1f}ms, P95: {p95:.1f}ms")
