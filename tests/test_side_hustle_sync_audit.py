from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_side_hustle_sync import (
    build_audit_report,
    build_file_index,
    compare_file_indexes,
    render_markdown,
    run_query_audit,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_compare_file_indexes_detects_missing_and_mismatch(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    _write(source / "source_feed" / "lane_01" / "a.md", "hipaa security analysis")
    _write(source / "source_feed" / "lane_01" / "b.md", "attorney privilege ai")
    _write(dest / "source_feed" / "lane_01" / "a.md", "hipaa security analysis changed")
    _write(dest / "source_feed" / "lane_01" / "extra.md", "extra")

    comparison = compare_file_indexes(build_file_index(source), build_file_index(dest))

    assert comparison["source_file_count"] == 2
    assert comparison["dest_file_count"] == 2
    assert comparison["mirrored_file_count"] == 1
    assert comparison["missing_in_jcoder"] == ["source_feed/lane_01/b.md"]
    assert comparison["extra_in_jcoder"] == ["source_feed/lane_01/extra.md"]
    assert comparison["hash_mismatches"] == ["source_feed/lane_01/a.md"]
    assert comparison["needs_resync"] is True


def test_run_query_audit_surfaces_hits_and_gaps(tmp_path):
    root = tmp_path / "mirror"
    _write(root / "source_feed" / "lane_01" / "hipaa.md", "HIPAA security risk analysis checklist")
    _write(root / "source_feed" / "lane_02" / "startup.md", "Colorado LLC annual report filing guide")

    results = run_query_audit(root, [
        {
            "id": "hipaa",
            "query": "hipaa security risk analysis",
            "keywords": ["hipaa", "security", "risk", "analysis"],
            "min_score": 0.5,
        },
        {
            "id": "privacy",
            "query": "attorney client privilege",
            "keywords": ["attorney", "client", "privilege"],
            "min_score": 0.5,
        },
    ])

    hipaa = next(item for item in results if item["id"] == "hipaa")
    privacy = next(item for item in results if item["id"] == "privacy")
    assert hipaa["passed"] is True
    assert hipaa["top_hits"][0]["relpath"].endswith("hipaa.md")
    assert privacy["passed"] is False
    assert privacy["top_hits"] == []


def test_build_audit_report_and_markdown_include_index_state(tmp_path):
    side_hustle = tmp_path / "side_hustle"
    jcoder = tmp_path / "jcoder"
    source = side_hustle / "research" / "staging" / "verified"
    dest = jcoder / "research" / "side_hustle_rag_business" / "clean_source" / "side_hustle_rag_business"
    summary_path = side_hustle / "research" / "download_state" / "jcoder_sync_summary.json"
    _write(source / "source_feed" / "lane_01" / "audit.md", "security audit logging evidence")
    _write(dest / "source_feed" / "lane_01" / "audit.md", "security audit logging evidence")
    _write(jcoder / "data" / "indexes" / "side_hustle_harvest_meta.fts5.db", "")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "generated_at_utc": "2026-04-05T00:00:00+00:00",
        "source_root": str(source),
        "dest_root": str(dest),
    }), encoding="utf-8")

    report = build_audit_report(summary_path, source, dest, jcoder)
    markdown = render_markdown(report)

    assert report["comparison"]["needs_resync"] is False
    assert report["side_hustle_index_artifacts"] == ["side_hustle_harvest_meta.fts5.db"]
    assert "Needs resync: False" in markdown
    assert "`lane_01`: 1 files" in markdown
