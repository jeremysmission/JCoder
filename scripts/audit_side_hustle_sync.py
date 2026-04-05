"""Audit the Side Hustle verified mirror that feeds JCoder research ingestion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_SIDE_HUSTLE_ROOT = Path(
    os.environ.get("SIDE_HUSTLE_ROOT", "C:/Users/jerem/Side Hustle RAG Business"),
)
DEFAULT_JCODER_ROOT = Path(os.environ.get("JCODER_ROOT", "C:/Users/jerem/JCoder"))
DEFAULT_SUMMARY_PATH = (
    DEFAULT_SIDE_HUSTLE_ROOT / "research" / "download_state" / "jcoder_sync_summary.json"
)
DEFAULT_OUTPUT_JSON = (
    DEFAULT_JCODER_ROOT / "data" / "audits" / "side_hustle_sync_audit.json"
)
DEFAULT_OUTPUT_MD = DEFAULT_OUTPUT_JSON.with_suffix(".md")

DEFAULT_QUERIES = [
    {
        "id": "hipaa",
        "query": "HIPAA security risk analysis requirements",
        "keywords": ["hipaa", "security", "risk", "analysis"],
        "min_score": 0.5,
    },
    {
        "id": "privilege",
        "query": "attorney client privilege and AI confidentiality",
        "keywords": ["attorney", "privilege", "ai", "confidentiality"],
        "min_score": 0.5,
    },
    {
        "id": "startup",
        "query": "Colorado LLC startup filing and annual report requirements",
        "keywords": ["colorado", "llc", "startup", "annual", "report"],
        "min_score": 0.4,
    },
    {
        "id": "roi",
        "query": "medical chronology productivity ROI evidence",
        "keywords": ["medical", "chronology", "productivity", "roi"],
        "min_score": 0.5,
    },
    {
        "id": "audit",
        "query": "security audit logging and evidence tracking",
        "keywords": ["security", "audit", "logging", "evidence"],
        "min_score": 0.5,
    },
]


@dataclass
class FileRecord:
    relpath: str
    size_bytes: int
    modified_at_utc: str
    sha256: str


@dataclass
class QueryHit:
    relpath: str
    score: float
    matched_keywords: List[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_from_ts(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def build_file_index(root: Path) -> Dict[str, FileRecord]:
    index: Dict[str, FileRecord] = {}
    for path in _iter_files(root):
        relpath = path.relative_to(root).as_posix()
        stat = path.stat()
        index[relpath] = FileRecord(
            relpath=relpath,
            size_bytes=stat.st_size,
            modified_at_utc=_iso_from_ts(stat.st_mtime),
            sha256=_sha256_file(path),
        )
    return index


def compare_file_indexes(
    source_index: Dict[str, FileRecord],
    dest_index: Dict[str, FileRecord],
) -> Dict[str, object]:
    source_paths = set(source_index)
    dest_paths = set(dest_index)
    mirrored = sorted(source_paths & dest_paths)
    missing = sorted(source_paths - dest_paths)
    extra = sorted(dest_paths - source_paths)
    drifted = sorted(
        relpath
        for relpath in mirrored
        if source_index[relpath].sha256 != dest_index[relpath].sha256
    )
    return {
        "source_file_count": len(source_index),
        "dest_file_count": len(dest_index),
        "mirrored_file_count": len(mirrored),
        "missing_in_jcoder": missing,
        "extra_in_jcoder": extra,
        "hash_mismatches": drifted,
        "needs_resync": bool(missing or drifted),
    }


def summarize_lanes(index: Dict[str, FileRecord]) -> List[Dict[str, object]]:
    lanes: Dict[str, int] = {}
    for relpath in index:
        parts = Path(relpath).parts
        lane = next((part for part in parts if part.startswith("lane_")), "unbucketed")
        lanes[lane] = lanes.get(lane, 0) + 1
    return [
        {"lane": lane, "files": count}
        for lane, count in sorted(lanes.items())
    ]


def _load_text_corpus(root: Path) -> Dict[str, str]:
    corpus: Dict[str, str] = {}
    for path in _iter_files(root):
        try:
            corpus[path.relative_to(root).as_posix()] = path.read_text(
                encoding="utf-8", errors="replace",
            ).lower()
        except OSError:
            continue
    return corpus


def run_query_audit(root: Path, queries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    corpus = _load_text_corpus(root)
    results: List[Dict[str, object]] = []
    for query in queries:
        keywords = [str(keyword).lower() for keyword in query["keywords"]]
        hits: List[QueryHit] = []
        for relpath, text in corpus.items():
            matched = [keyword for keyword in keywords if keyword in text]
            if not matched:
                continue
            score = len(matched) / max(1, len(keywords))
            hits.append(QueryHit(relpath=relpath, score=score, matched_keywords=matched))
        hits.sort(key=lambda item: (-item.score, item.relpath))
        top_hits = [asdict(hit) for hit in hits[:3]]
        top_score = hits[0].score if hits else 0.0
        results.append({
            "id": query["id"],
            "query": query["query"],
            "keywords": keywords,
            "top_score": round(top_score, 3),
            "passed": top_score >= float(query.get("min_score", 0.5)),
            "top_hits": top_hits,
        })
    return results


def discover_side_hustle_indexes(jcoder_root: Path) -> List[str]:
    index_dir = jcoder_root / "data" / "indexes"
    if not index_dir.exists():
        return []
    return sorted(path.name for path in index_dir.glob("*side_hustle*"))


def build_audit_report(
    summary_path: Path,
    source_root: Path,
    dest_root: Path,
    jcoder_root: Path,
) -> Dict[str, object]:
    sync_summary = {}
    if summary_path.exists():
        sync_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    source_index = build_file_index(source_root)
    dest_index = build_file_index(dest_root)
    comparison = compare_file_indexes(source_index, dest_index)
    source_newest = max(
        (record.modified_at_utc for record in source_index.values()),
        default=None,
    )
    dest_newest = max(
        (record.modified_at_utc for record in dest_index.values()),
        default=None,
    )
    report = {
        "generated_at_utc": _utc_now(),
        "sync_summary_generated_at_utc": sync_summary.get("generated_at_utc"),
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "source_newest_utc": source_newest,
        "dest_newest_utc": dest_newest,
        "comparison": comparison,
        "source_lanes": summarize_lanes(source_index),
        "dest_lanes": summarize_lanes(dest_index),
        "query_audit": run_query_audit(dest_root, DEFAULT_QUERIES),
        "side_hustle_index_artifacts": discover_side_hustle_indexes(jcoder_root),
    }
    return report


def render_markdown(report: Dict[str, object]) -> str:
    comparison = report["comparison"]
    query_lines = []
    for result in report["query_audit"]:
        status = "PASS" if result["passed"] else "FAIL"
        hit_text = "none"
        if result["top_hits"]:
            best = result["top_hits"][0]
            hit_text = f"{best['relpath']} ({best['score']:.2f})"
        query_lines.append(
            f"- `{result['id']}` {status} -- score {result['top_score']:.2f} -- {hit_text}",
        )
    lane_lines = [
        f"- `{item['lane']}`: {item['files']} files"
        for item in report["dest_lanes"]
    ]
    index_artifacts = report["side_hustle_index_artifacts"] or []
    if index_artifacts:
        index_lines = [f"- `{name}`" for name in index_artifacts]
    else:
        index_lines = ["- None; mirror exists but no side-hustle index artifacts are present in `data/indexes`."]
    return "\n".join([
        "# Side Hustle Sync Audit",
        "",
        f"- Generated: {report['generated_at_utc']}",
        f"- Last sync summary: {report['sync_summary_generated_at_utc'] or 'missing'}",
        f"- Source root: `{report['source_root']}`",
        f"- Dest root: `{report['dest_root']}`",
        f"- Source newest: `{report['source_newest_utc'] or 'n/a'}`",
        f"- Dest newest: `{report['dest_newest_utc'] or 'n/a'}`",
        "",
        "## Mirror State",
        "",
        f"- Source files: {comparison['source_file_count']}",
        f"- Mirrored files: {comparison['mirrored_file_count']}",
        f"- Destination files: {comparison['dest_file_count']}",
        f"- Missing in JCoder: {len(comparison['missing_in_jcoder'])}",
        f"- Extra in JCoder: {len(comparison['extra_in_jcoder'])}",
        f"- Hash mismatches: {len(comparison['hash_mismatches'])}",
        f"- Needs resync: {comparison['needs_resync']}",
        "",
        "## Lane Coverage",
        "",
        *lane_lines,
        "",
        "## QA Queries",
        "",
        *query_lines,
        "",
        "## Index Artifacts",
        "",
        *index_lines,
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--source-root", default="")
    parser.add_argument("--dest-root", default="")
    parser.add_argument("--jcoder-root", default=str(DEFAULT_JCODER_ROOT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    source_root = Path(args.source_root or summary.get("source_root") or "").resolve()
    dest_root = Path(args.dest_root or summary.get("dest_root") or "").resolve()
    jcoder_root = Path(args.jcoder_root).resolve()
    if not source_root.exists():
        raise SystemExit(f"Source root does not exist: {source_root}")
    if not dest_root.exists():
        raise SystemExit(f"Destination root does not exist: {dest_root}")

    report = build_audit_report(summary_path, source_root, dest_root, jcoder_root)
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")

    comparison = report["comparison"]
    print(f"Audit JSON: {output_json}")
    print(f"Audit MD:   {output_md}")
    print(f"Mirrored:   {comparison['mirrored_file_count']}/{comparison['source_file_count']}")
    print(f"Missing:    {len(comparison['missing_in_jcoder'])}")
    print(f"Mismatched: {len(comparison['hash_mismatches'])}")

    if args.fail_on_drift and comparison["needs_resync"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
