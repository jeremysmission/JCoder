"""
Prep-only staging pipeline for JCoder.

Purpose:
- Inventory raw dump roots and validate archive headers.
- Run sanitizer when needed (or reuse latest sanitize run).
- Compute chunk-readiness stats without embedding/indexing.
- Emit JSON/CSV reports for beast-machine indexing handoff.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config
from ingestion.chunker import Chunker
from ingestion.repo_loader import RepoLoader
from ingestion.sanitizer import SanitizationConfig, SanitizationPipeline


_DEFAULT_ROOTS_STR = os.environ.get("JCODER_DATA_ROOTS", os.pathsep.join([
    r"D:\Projects\KnowledgeBase\stackexchange_20251231",
    r"D:\Projects\reddit",
    r"D:\Projects\KnowledgeBase\sources\ragas",
    r"D:\RAG Source Data\stackexchange_20251231",
    r"D:\Projects\softwarerecs.stackexchange.com",
    r"D:\Misc\ZippedDownloadsDumps",
    r"D:\Archive\HybridRAG3_VariousVersions\HybridRAG3_Archive\2026-02-16\orphans",
]))
DEFAULT_ROOTS = [r for r in _DEFAULT_ROOTS_STR.split(os.pathsep) if r]

CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".cc",
    ".cxx", ".cs", ".rb", ".php", ".kt",
}
TEXT_EXTS = {".xml", ".json", ".jsonl", ".zst", ".md", ".txt"}
ARCHIVE_EXTS = {".7z", ".zip", ".tar", ".gz", ".xz", ".parquet"}
MAGIC_7Z = bytes.fromhex("377ABCAF271C")
MAGIC_ZST = bytes.fromhex("28B52FFD")


@dataclass
class RootPrepResult:
    raw_root: str
    exists: bool
    total_files: int
    candidate_files: int
    by_ext: Dict[str, int]
    archive_header_ok: int
    archive_header_bad: int
    sanitize_run_dir: str
    sanitize_log: str
    sanitize_entries: int
    sanitize_code_blocks: int
    chunk_files: int
    chunk_count: int
    status: str
    notes: str


def _iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "venv", "__pycache__"}]
        for fn in filenames:
            yield Path(dirpath) / fn


def _check_archive_header(path: Path) -> Tuple[bool, str]:
    ext = path.suffix.lower()
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except Exception as e:
        return False, f"read_error:{e}"
    if ext == ".7z":
        return head.startswith(MAGIC_7Z), head.hex()
    if ext == ".zst":
        return head.startswith(MAGIC_ZST), head.hex()
    return True, head.hex()


def _latest_sanitize_logs(logs_root: Path) -> Dict[str, Dict]:
    latest: Dict[str, Dict] = {}
    for p in sorted(logs_root.glob("sanitize_*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rr = payload.get("raw_root")
        if not rr:
            continue
        payload["_log_path"] = str(p)
        latest[rr] = payload
    return latest


def _inventory(root: Path) -> Tuple[int, int, Dict[str, int], int, int]:
    total = 0
    candidates = 0
    by_ext: Counter = Counter()
    ok = 0
    bad = 0
    for fp in _iter_files(root):
        total += 1
        ext = fp.suffix.lower()
        if ext in CODE_EXTS or ext in TEXT_EXTS or ext in ARCHIVE_EXTS:
            candidates += 1
            by_ext[ext] += 1
        if ext in {".7z", ".zst"}:
            valid, _sig = _check_archive_header(fp)
            if valid:
                ok += 1
            else:
                bad += 1
    return total, candidates, dict(by_ext), ok, bad


def _sanitize_or_reuse(
    raw_root: str,
    latest_logs: Dict[str, Dict],
    sanitizer: SanitizationPipeline,
) -> Tuple[str, str, int, int, str]:
    if raw_root in latest_logs:
        payload = latest_logs[raw_root]
        run_dir = str(payload.get("run_dir", ""))
        log_path = str(payload.get("_log_path", ""))
        if run_dir and Path(run_dir).exists():
            st = payload.get("stats", {})
            return (
                run_dir,
                log_path,
                int(st.get("entries_written", 0)),
                int(st.get("code_blocks_kept", 0)),
                "reused_latest_sanitize",
            )

    idx_name = "prep_" + Path(raw_root).name.replace(" ", "_")
    run_dir, stats, log_path = sanitizer.run(raw_root, index_name=idx_name)
    return run_dir, log_path, stats.entries_written, stats.code_blocks_kept, "sanitized_now"


def _chunk_stats(run_dir: str, chunk_max_chars: int) -> Tuple[int, int]:
    chunker = Chunker(max_chars=chunk_max_chars)
    loader = RepoLoader(chunker)
    chunks = loader.load(run_dir)
    unique_files = len({c.get("source_path", "") for c in chunks})
    return unique_files, len(chunks)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare staged raw roots for indexing readiness.")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=None,
        help="Optional explicit list of raw roots to process. If omitted, DEFAULT_ROOTS is used.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-sanitization even if a cached run exists for the root.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(None)
    s_cfg = SanitizationConfig()
    sanitizer = SanitizationPipeline(s_cfg)
    latest_logs = _latest_sanitize_logs(Path(s_cfg.clean_archive_dir) / "_logs")
    if args.force:
        latest_logs.clear()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _jcoder_data = Path(os.environ.get("JCODER_DATA_DIR", r"D:\JCoder_Data"))
    prep_root = _jcoder_data / "prep_stage" / f"prep_{ts}"
    prep_root.mkdir(parents=True, exist_ok=True)
    report_json = prep_root / "prep_report.json"
    report_csv = prep_root / "prep_report.csv"

    target_roots = args.roots if args.roots else DEFAULT_ROOTS

    results: List[RootPrepResult] = []
    for raw_root in target_roots:
        root = Path(raw_root)
        if not root.exists():
            results.append(
                RootPrepResult(
                    raw_root=raw_root,
                    exists=False,
                    total_files=0,
                    candidate_files=0,
                    by_ext={},
                    archive_header_ok=0,
                    archive_header_bad=0,
                    sanitize_run_dir="",
                    sanitize_log="",
                    sanitize_entries=0,
                    sanitize_code_blocks=0,
                    chunk_files=0,
                    chunk_count=0,
                    status="missing",
                    notes="path_not_found",
                )
            )
            continue

        total, candidates, by_ext, arch_ok, arch_bad = _inventory(root)
        run_dir, log_path, entries, code_blocks, status = _sanitize_or_reuse(
            raw_root, latest_logs, sanitizer
        )
        chunk_files = 0
        chunk_count = 0
        notes = ""
        if run_dir and Path(run_dir).exists():
            try:
                chunk_files, chunk_count = _chunk_stats(run_dir, cfg.chunking.max_chars)
            except Exception as e:
                notes = f"chunk_stats_error:{e}"
        else:
            notes = "sanitize_run_dir_missing"

        if arch_bad > 0:
            notes = (notes + "; " if notes else "") + f"bad_archive_headers={arch_bad}"

        results.append(
            RootPrepResult(
                raw_root=raw_root,
                exists=True,
                total_files=total,
                candidate_files=candidates,
                by_ext=by_ext,
                archive_header_ok=arch_ok,
                archive_header_bad=arch_bad,
                sanitize_run_dir=run_dir,
                sanitize_log=log_path,
                sanitize_entries=entries,
                sanitize_code_blocks=code_blocks,
                chunk_files=chunk_files,
                chunk_count=chunk_count,
                status=status,
                notes=notes,
            )
        )
        print(
            f"[prep] {raw_root} | status={status} | entries={entries} | "
            f"chunks={chunk_count} | bad_headers={arch_bad}"
        )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "prep_root": str(prep_root),
        "results": [r.__dict__ for r in results],
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "raw_root", "exists", "total_files", "candidate_files",
            "archive_header_ok", "archive_header_bad", "sanitize_run_dir",
            "sanitize_log", "sanitize_entries", "sanitize_code_blocks",
            "chunk_files", "chunk_count", "status", "notes",
        ])
        for r in results:
            w.writerow([
                r.raw_root, r.exists, r.total_files, r.candidate_files,
                r.archive_header_ok, r.archive_header_bad, r.sanitize_run_dir,
                r.sanitize_log, r.sanitize_entries, r.sanitize_code_blocks,
                r.chunk_files, r.chunk_count, r.status, r.notes,
            ])

    print(f"[prep] wrote {report_json}")
    print(f"[prep] wrote {report_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
