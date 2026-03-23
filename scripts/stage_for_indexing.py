"""
stage_for_indexing.py -- Comprehensive preprocessing, sanitization, and staging
script that prepares ALL data sources under D:\\JCoder_Data\\clean_source for
vector indexing.

Phases:
  1. Inventory     -- scan directories, count files, report sizes
  2. PII Scan      -- sample-based PII/secret detection per source
  3. Dedup Stats   -- MinHash near-duplicate estimation per source
  4. Staging Report -- emit staging_manifest.json with readiness flags
  5. FTS5 Build    -- ingest ready sources into FTS5 indexes (no embeddings)

Usage:
  python scripts/stage_for_indexing.py
  python scripts/stage_for_indexing.py --skip-indexing
  python scripts/stage_for_indexing.py --source codesearchnet/python
  python scripts/stage_for_indexing.py --pii-full --max-files 500
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# sys.path -- same pattern as smoke_test.py
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Windows stdout safety
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from core.config import StorageConfig
from ingestion.dedup import MinHashDedup
from ingestion.pii_scanner import PIIScanner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CLEAN_SOURCE = Path(os.environ.get(
    "JCODER_CLEAN_SOURCE", r"data\clean_source"
))
_DATA_ROOT = Path(os.environ.get("JCODER_DATA_DIR", "data"))
_INDEX_DIR = _DATA_ROOT / "indexes"
_MANIFEST_PATH = _DATA_ROOT / "staging_manifest.json"

_PII_SAMPLE_SIZE = 100
_DEDUP_SAMPLE_SIZE = 500

# Sources that contain Q&A-style markdown (SO / CSN format)
_QA_SOURCES: Set[str] = {
    "codesearchnet/python", "codesearchnet/go", "codesearchnet/java",
    "codesearchnet/javascript", "codesearchnet/php", "codesearchnet/ruby",
    "python", "javascript", "bash", "cpp", "csharp",
}

# Sources that contain documentation markdown (heading-based split)
_DOC_SOURCES: Set[str] = {"rfc", "python_docs"}

# Sources that contain raw code files
_CODE_SOURCES: Set[str] = {"stack_v2"}

# How each source maps to an index name
_INDEX_NAME_MAP: Dict[str, str] = {
    "codesearchnet/python": "codesearchnet",
    "codesearchnet/go": "codesearchnet",
    "codesearchnet/java": "codesearchnet",
    "codesearchnet/javascript": "codesearchnet",
    "codesearchnet/php": "codesearchnet",
    "codesearchnet/ruby": "codesearchnet",
    "rfc": "docs",
    "python_docs": "docs",
    "python": "stackoverflow",
    "javascript": "stackoverflow",
    "bash": "stackoverflow",
    "cpp": "stackoverflow",
    "csharp": "stackoverflow",
    "stack_v2": "code",
}

# Graceful interrupt handling
_INTERRUPTED = False


def _sigint_handler(sig, frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    print("\n[WARN] Interrupt received -- finishing current operation...")


signal.signal(signal.SIGINT, _sigint_handler)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SourceInfo:
    """Inventory and analysis data for a single source directory."""
    name: str
    path: str
    file_count: int = 0
    total_size_bytes: int = 0
    extensions: Dict[str, int] = field(default_factory=dict)
    # PII scan
    pii_files_sampled: int = 0
    pii_total_findings: int = 0
    pii_finding_types: Dict[str, int] = field(default_factory=dict)
    pii_rate: float = 0.0
    pii_needs_full_scan: bool = False
    # Dedup
    dedup_files_sampled: int = 0
    dedup_total_seen: int = 0
    dedup_duplicates: int = 0
    dedup_pct: float = 0.0
    # Indexing
    ready_for_indexing: bool = False
    index_name: str = ""
    ingest_method: str = ""
    chunks_created: int = 0
    index_time_s: float = 0.0
    status: str = "pending"

    @property
    def total_size_mb(self) -> float:
        return round(self.total_size_bytes / (1024 * 1024), 2)


# ---------------------------------------------------------------------------
# Phase 1: Inventory
# ---------------------------------------------------------------------------

def _discover_sources(root: Path) -> List[SourceInfo]:
    """Walk clean_source, discover all source directories (1 or 2 levels deep)."""
    sources: List[SourceInfo] = []
    if not root.exists():
        print(f"[FAIL] Clean source root does not exist: {root}")
        return sources

    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue

        # Check for sub-sources (e.g. codesearchnet/python)
        subdirs = [d for d in entry.iterdir()
                   if d.is_dir() and not d.name.startswith(("_", "."))]

        if subdirs and entry.name in ("codesearchnet", "stack_v2", "unknown"):
            # Multi-level source
            for sub in sorted(subdirs):
                name = f"{entry.name}/{sub.name}"
                sources.append(SourceInfo(name=name, path=str(sub)))
        else:
            sources.append(SourceInfo(name=entry.name, path=str(entry)))

    return sources


def _collect_files(source_path: str, max_files: int = 0) -> List[str]:
    """Recursively collect all files under a source path."""
    files: List[str] = []
    for dirpath, _dirs, filenames in os.walk(source_path):
        _dirs[:] = [d for d in _dirs if not d.startswith(("_", ".", "__"))]
        for fn in filenames:
            if fn.startswith("."):
                continue
            files.append(os.path.join(dirpath, fn))
            if max_files and len(files) >= max_files:
                return files
    files.sort()
    return files


def _run_inventory(sources: List[SourceInfo], max_files: int = 0) -> None:
    """Phase 1: count files and sizes for each source."""
    print("\n" + "=" * 72)
    print("PHASE 1: INVENTORY")
    print("=" * 72)

    for src in sources:
        if _INTERRUPTED:
            break
        files = _collect_files(src.path, max_files)
        src.file_count = len(files)
        total_bytes = 0
        ext_counts: Dict[str, int] = defaultdict(int)
        for fpath in files:
            try:
                sz = os.path.getsize(fpath)
                total_bytes += sz
            except OSError:
                pass
            ext = os.path.splitext(fpath)[1].lower() or "(none)"
            ext_counts[ext] += 1
        src.total_size_bytes = total_bytes
        src.extensions = dict(ext_counts)

        # Determine ingest method and index name
        if src.name in _QA_SOURCES:
            src.ingest_method = "ingest_stackoverflow"
            src.index_name = _INDEX_NAME_MAP.get(src.name, "stackoverflow")
        elif src.name in _DOC_SOURCES:
            src.ingest_method = "ingest_markdown_docs"
            src.index_name = _INDEX_NAME_MAP.get(src.name, "docs")
        elif src.name in _CODE_SOURCES or src.name.startswith("stack_v2/"):
            src.ingest_method = "ingest_code_files"
            src.index_name = _INDEX_NAME_MAP.get(src.name, "code")
        else:
            # Unknown sources -- treat as markdown docs if they contain .md
            if ".md" in ext_counts:
                src.ingest_method = "ingest_markdown_docs"
                src.index_name = f"misc_{src.name.replace('/', '_')}"
            else:
                src.ingest_method = "ingest_code_files"
                src.index_name = f"misc_{src.name.replace('/', '_')}"

        if src.file_count == 0:
            src.status = "empty"

    # Print table
    print(f"\n{'Source':<30} {'Files':>10} {'Size (MB)':>12} {'Status':<15}")
    print("-" * 72)
    for src in sources:
        status = src.status if src.status != "pending" else "ok"
        print(f"{src.name:<30} {src.file_count:>10,} {src.total_size_mb:>12.2f} {status:<15}")

    total_files = sum(s.file_count for s in sources)
    total_mb = sum(s.total_size_mb for s in sources)
    print("-" * 72)
    print(f"{'TOTAL':<30} {total_files:>10,} {total_mb:>12.2f}")
    print(f"[OK] Inventory complete: {len(sources)} sources, "
          f"{total_files:,} files, {total_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Phase 2: PII Scan
# ---------------------------------------------------------------------------

def _run_pii_scan(
    sources: List[SourceInfo],
    full_scan: bool = False,
    max_files: int = 0,
) -> None:
    """Phase 2: sample-based PII scanning per source."""
    print("\n" + "=" * 72)
    print("PHASE 2: PII SCAN" + (" (FULL)" if full_scan else " (SAMPLE)"))
    print("=" * 72)

    scanner = PIIScanner(redact=False)  # scan-only, no redaction needed

    for src in sources:
        if _INTERRUPTED:
            break
        if src.file_count == 0:
            print(f"[OK] {src.name}: skipped (empty)")
            continue

        files = _collect_files(src.path, max_files)
        if not full_scan and len(files) > _PII_SAMPLE_SIZE:
            sample = random.sample(files, _PII_SAMPLE_SIZE)
        else:
            sample = files

        src.pii_files_sampled = len(sample)
        finding_types: Dict[str, int] = defaultdict(int)
        files_with_findings = 0

        for i, fpath in enumerate(sample):
            if _INTERRUPTED:
                break
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read(512_000)  # cap at 500KB per file
                result = scanner.scan(text)
                if result.findings:
                    files_with_findings += 1
                    src.pii_total_findings += len(result.findings)
                    for finding in result.findings:
                        finding_types[finding.type] += 1
            except Exception:
                pass  # binary files, permission errors, etc.

            if (i + 1) % 1000 == 0:
                print(f"  [{src.name}] PII scanned {i + 1}/{len(sample)}...")

        src.pii_finding_types = dict(finding_types)
        if src.pii_files_sampled > 0:
            src.pii_rate = round(files_with_findings / src.pii_files_sampled, 4)
        src.pii_needs_full_scan = src.pii_rate > 0.05

        tag = "[WARN]" if src.pii_needs_full_scan else "[OK]"
        extra = " ** NEEDS FULL SCAN **" if src.pii_needs_full_scan else ""
        types_str = ", ".join(f"{k}={v}" for k, v in sorted(finding_types.items()))
        print(f"{tag} {src.name}: sampled={src.pii_files_sampled}, "
              f"findings={src.pii_total_findings}, "
              f"rate={src.pii_rate:.2%}, types=[{types_str}]{extra}")


# ---------------------------------------------------------------------------
# Phase 3: Dedup Statistics
# ---------------------------------------------------------------------------

def _run_dedup_stats(sources: List[SourceInfo], max_files: int = 0) -> None:
    """Phase 3: MinHash near-duplicate estimation per source."""
    print("\n" + "=" * 72)
    print("PHASE 3: DEDUP STATISTICS")
    print("=" * 72)

    for src in sources:
        if _INTERRUPTED:
            break
        if src.file_count == 0:
            print(f"[OK] {src.name}: skipped (empty)")
            continue

        dedup = MinHashDedup(num_perm=64, threshold=0.8)
        files = _collect_files(src.path, max_files)
        if len(files) > _DEDUP_SAMPLE_SIZE:
            sample = random.sample(files, _DEDUP_SAMPLE_SIZE)
        else:
            sample = files

        src.dedup_files_sampled = len(sample)
        texts_seen = 0

        for i, fpath in enumerate(sample):
            if _INTERRUPTED:
                break
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read(128_000)  # cap at 128KB for dedup hashing
                if not text.strip():
                    continue
                dedup.add(text, doc_id=fpath)
                texts_seen += 1
            except Exception:
                pass

            if (i + 1) % 1000 == 0:
                print(f"  [{src.name}] Dedup checked {i + 1}/{len(sample)}...")

        stats = dedup.stats()
        src.dedup_total_seen = stats.total_seen
        src.dedup_duplicates = stats.exact_dupes + stats.near_dupes
        if stats.total_seen > 0:
            src.dedup_pct = round(
                (src.dedup_duplicates / stats.total_seen) * 100, 2
            )

        tag = "[WARN]" if src.dedup_pct > 10 else "[OK]"
        print(f"{tag} {src.name}: sampled={src.dedup_files_sampled}, "
              f"seen={stats.total_seen}, dupes={src.dedup_duplicates} "
              f"(exact={stats.exact_dupes}, near={stats.near_dupes}), "
              f"est_dup_pct={src.dedup_pct:.1f}%")


# ---------------------------------------------------------------------------
# Phase 4: Staging Manifest
# ---------------------------------------------------------------------------

def _determine_readiness(sources: List[SourceInfo]) -> None:
    """Set ready_for_indexing based on scan results."""
    for src in sources:
        if src.file_count == 0:
            src.ready_for_indexing = False
            src.status = "empty"
        elif src.pii_needs_full_scan:
            src.ready_for_indexing = False
            src.status = "pii_review_needed"
        else:
            src.ready_for_indexing = True
            src.status = "ready"


def _write_manifest(sources: List[SourceInfo]) -> str:
    """Phase 4: write staging_manifest.json."""
    print("\n" + "=" * 72)
    print("PHASE 4: STAGING MANIFEST")
    print("=" * 72)

    _determine_readiness(sources)

    source_entries = []
    for src in sources:
        source_entries.append({
            "name": src.name,
            "path": src.path,
            "file_count": src.file_count,
            "total_size_mb": src.total_size_mb,
            "extensions": src.extensions,
            "sample_pii_rate": src.pii_rate,
            "pii_total_findings": src.pii_total_findings,
            "pii_finding_types": src.pii_finding_types,
            "pii_needs_full_scan": src.pii_needs_full_scan,
            "estimated_dedup_pct": src.dedup_pct,
            "ready_for_indexing": src.ready_for_indexing,
            "index_name": src.index_name,
            "ingest_method": src.ingest_method,
            "status": src.status,
        })

    total_files = sum(s.file_count for s in sources)
    total_mb = round(sum(s.total_size_mb for s in sources), 2)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clean_source_root": str(_CLEAN_SOURCE),
        "sources": source_entries,
        "totals": {
            "files": total_files,
            "size_mb": total_mb,
            "sources_ready": sum(1 for s in sources if s.ready_for_indexing),
            "sources_total": len(sources),
        },
    }

    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[OK] Manifest written: {_MANIFEST_PATH}")
    print(f"     Sources ready: {manifest['totals']['sources_ready']}"
          f"/{manifest['totals']['sources_total']}")
    print(f"     Total files:   {total_files:,}")
    print(f"     Total size:    {total_mb:.2f} MB")

    # Summary table
    print(f"\n{'Source':<30} {'Ready':<8} {'PII%':>8} {'Dup%':>8} "
          f"{'Method':<25} {'Index':<15}")
    print("-" * 100)
    for src in sources:
        ready_str = "YES" if src.ready_for_indexing else "NO"
        print(f"{src.name:<30} {ready_str:<8} {src.pii_rate:>7.2%} "
              f"{src.dedup_pct:>7.1f}% {src.ingest_method:<25} {src.index_name:<15}")

    return str(_MANIFEST_PATH)


# ---------------------------------------------------------------------------
# Phase 5: FTS5 Indexing
# ---------------------------------------------------------------------------

def _run_indexing(sources: List[SourceInfo], max_files: int = 0) -> None:
    """Phase 5: build FTS5 indexes for ready sources using CorpusPipeline."""
    print("\n" + "=" * 72)
    print("PHASE 5: FTS5 INDEX BUILD")
    print("=" * 72)

    # Lazy import -- only needed when actually indexing
    from ingestion.corpus_pipeline import CorpusPipeline

    ready = [s for s in sources if s.ready_for_indexing]
    if not ready:
        print("[WARN] No sources ready for indexing.")
        return

    # Group sources by index_name so we ingest into shared indexes
    by_index: Dict[str, List[SourceInfo]] = defaultdict(list)
    for src in ready:
        by_index[src.index_name].append(src)

    storage = StorageConfig(
        data_dir=str(_DATA_ROOT),
        index_dir=str(_INDEX_DIR),
    )

    for index_name, group in sorted(by_index.items()):
        if _INTERRUPTED:
            print("[WARN] Interrupted -- skipping remaining indexes.")
            break

        print(f"\n--- Building index: {index_name} ---")
        pipeline = CorpusPipeline(
            embedding_engine=None,  # FTS5-only, no embeddings
            storage_config=storage,
            batch_size=128,
            pii_scanner=PIIScanner(redact=True),
            dedup=MinHashDedup(num_perm=128, threshold=0.8),
        )

        for src in group:
            if _INTERRUPTED:
                break

            method_name = src.ingest_method
            method = getattr(pipeline, method_name, None)
            if method is None:
                print(f"[FAIL] Unknown ingest method '{method_name}' "
                      f"for {src.name}")
                src.status = "method_error"
                continue

            print(f"  Ingesting {src.name} via {method_name}...")
            t0 = time.monotonic()
            try:
                stats = method(
                    source_dir=src.path,
                    index_name=index_name,
                    max_files=max_files,
                    resume=True,
                )
                elapsed = time.monotonic() - t0
                src.chunks_created = stats.chunks_created
                src.index_time_s = round(elapsed, 1)
                src.status = "indexed"
                print(f"  [OK] {src.name}: {stats.chunks_created} chunks, "
                      f"{elapsed:.1f}s, {len(stats.errors)} errors")
                if stats.errors:
                    for err in stats.errors[:5]:
                        print(f"    [WARN] {err}")
                    if len(stats.errors) > 5:
                        print(f"    ... and {len(stats.errors) - 5} more errors")
            except Exception as exc:
                elapsed = time.monotonic() - t0
                src.status = "index_error"
                src.index_time_s = round(elapsed, 1)
                print(f"  [FAIL] {src.name}: {exc}")

    # Index summary
    print(f"\n{'Source':<30} {'Chunks':>10} {'Time (s)':>10} {'Status':<15}")
    print("-" * 70)
    for src in ready:
        print(f"{src.name:<30} {src.chunks_created:>10,} "
              f"{src.index_time_s:>10.1f} {src.status:<15}")


# ---------------------------------------------------------------------------
# Update manifest after indexing
# ---------------------------------------------------------------------------

def _update_manifest_post_index(sources: List[SourceInfo]) -> None:
    """Re-write the manifest with indexing results appended."""
    if not _MANIFEST_PATH.exists():
        return

    try:
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    # Merge indexing results into source entries
    src_map = {s.name: s for s in sources}
    for entry in manifest.get("sources", []):
        name = entry.get("name", "")
        if name in src_map:
            src = src_map[name]
            entry["chunks_created"] = src.chunks_created
            entry["index_time_s"] = src.index_time_s
            entry["status"] = src.status

    manifest["indexing_completed_at"] = datetime.now(timezone.utc).isoformat()

    with open(_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[OK] Manifest updated with indexing results: {_MANIFEST_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage JCoder data sources for vector indexing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/stage_for_indexing.py\n"
            "  python scripts/stage_for_indexing.py --skip-indexing\n"
            "  python scripts/stage_for_indexing.py --source codesearchnet/python\n"
            "  python scripts/stage_for_indexing.py --pii-full --max-files 200\n"
        ),
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Only run phases 1-4 (inventory, PII, dedup, manifest). "
             "Do not build FTS5 indexes.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Process only a specific source (e.g. codesearchnet/python, rfc).",
    )
    parser.add_argument(
        "--pii-full",
        action="store_true",
        help="Run full PII scan instead of sampling 100 files per source.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit files processed per source (0 = unlimited). "
             "Useful for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    random.seed(args.seed)
    t_start = time.monotonic()

    print("=" * 72)
    print("JCoder -- stage_for_indexing.py")
    print(f"Clean source: {_CLEAN_SOURCE}")
    print(f"Index output: {_INDEX_DIR}")
    print(f"Manifest:     {_MANIFEST_PATH}")
    if args.source:
        print(f"Filter:       --source {args.source}")
    if args.max_files:
        print(f"Max files:    {args.max_files}")
    print("=" * 72)

    # --- Discover sources ---
    sources = _discover_sources(_CLEAN_SOURCE)
    if not sources:
        print("[FAIL] No sources found under {_CLEAN_SOURCE}")
        return 1

    # --- Filter to single source if requested ---
    if args.source:
        filtered = [s for s in sources if s.name == args.source]
        if not filtered:
            # Try partial match
            filtered = [s for s in sources if args.source in s.name]
        if not filtered:
            print(f"[FAIL] Source '{args.source}' not found. Available sources:")
            for s in sources:
                print(f"  - {s.name}")
            return 1
        sources = filtered

    # --- Phase 1: Inventory ---
    _run_inventory(sources, max_files=args.max_files)
    if _INTERRUPTED:
        print("[WARN] Interrupted during inventory.")
        return 130

    # --- Phase 2: PII Scan ---
    _run_pii_scan(sources, full_scan=args.pii_full, max_files=args.max_files)
    if _INTERRUPTED:
        print("[WARN] Interrupted during PII scan.")
        # Still write partial manifest
        _write_manifest(sources)
        return 130

    # --- Phase 3: Dedup Statistics ---
    _run_dedup_stats(sources, max_files=args.max_files)
    if _INTERRUPTED:
        print("[WARN] Interrupted during dedup scan.")
        _write_manifest(sources)
        return 130

    # --- Phase 4: Staging Manifest ---
    _write_manifest(sources)

    # --- Phase 5: FTS5 Indexing (unless --skip-indexing) ---
    if not args.skip_indexing:
        _run_indexing(sources, max_files=args.max_files)
        _update_manifest_post_index(sources)
    else:
        print("\n[OK] --skip-indexing: phases 1-4 complete, skipping FTS5 build.")

    elapsed = time.monotonic() - t_start
    print(f"\n{'=' * 72}")
    print(f"[OK] stage_for_indexing complete in {elapsed:.1f}s")
    print(f"{'=' * 72}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
