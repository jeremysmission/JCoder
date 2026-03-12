from pathlib import Path

from core.weekly_subjects import (
    already_ingested,
    build_memory_payloads,
    file_sha256,
    find_latest_summary,
    parse_summary_document,
    save_state,
    load_state,
)


_SUMMARY = """# JCoder Weekly Subject Summary -- 2026-03-12

Time window: 2026-03-05 through 2026-03-12 America/Denver

## 1. Python and Python Tooling

Best finding: Python 3.15.0 alpha 7 shows the language is still shifting in
developer-visible ways.

Why it matters to JCoder: Python advice should start mentioning these changes now.

Primary source:
- https://blog.python.org/2026/03/python-3150-alpha-7/

## 2. AI Coding Agents, RAG, and Model Tooling

Best finding: The practical frontier is supervised multi-agent work.

Why it matters to JCoder: JCoder should bias toward orchestrated agent loops.

Primary sources:
- https://help.openai.com/en/articles/6825453-chatgpt-release-notes
- https://openai.com/index/introducing-the-codex-app/
"""


def test_parse_summary_document_extracts_sections():
    doc_date, subjects = parse_summary_document(
        _SUMMARY, "WEEKLY_SUBJECT_SUMMARY_2026-03-12.md"
    )

    assert doc_date == "2026-03-12"
    assert len(subjects) == 2
    assert subjects[0].title == "Python and Python Tooling"
    assert subjects[0].best_finding.startswith("Python 3.15.0 alpha 7")
    assert subjects[0].why_it_matters.startswith("Python advice should start")
    assert subjects[0].sources == [
        "https://blog.python.org/2026/03/python-3150-alpha-7/"
    ]
    assert subjects[1].sources == [
        "https://help.openai.com/en/articles/6825453-chatgpt-release-notes",
        "https://openai.com/index/introducing-the-codex-app/",
    ]


def test_find_latest_summary_prefers_latest_filename_date(tmp_path):
    older = tmp_path / "WEEKLY_SUBJECT_SUMMARY_2026-03-05.md"
    newer = tmp_path / "WEEKLY_SUBJECT_SUMMARY_2026-03-12.md"
    older.write_text(_SUMMARY, encoding="utf-8")
    newer.write_text(_SUMMARY, encoding="utf-8")

    assert find_latest_summary(tmp_path) == newer


def test_build_memory_payloads_builds_tags_and_content(tmp_path):
    summary_path = tmp_path / "WEEKLY_SUBJECT_SUMMARY_2026-03-12.md"
    summary_path.write_text(_SUMMARY, encoding="utf-8")

    payloads = build_memory_payloads(summary_path)

    assert len(payloads) == 2
    assert payloads[0]["source_task"] == (
        "weekly-subject-summary:2026-03-12:python_and_python_tooling"
    )
    assert "weekly_subject_summary" in payloads[0]["tags"]
    assert "week_2026-03-12" in payloads[0]["tags"]
    assert "Summary document:" in payloads[0]["content"]


def test_state_round_trip_and_duplicate_detection(tmp_path):
    state_path = tmp_path / "state.json"
    payload_path = tmp_path / "WEEKLY_SUBJECT_SUMMARY_2026-03-12.md"
    payload_path.write_text(_SUMMARY, encoding="utf-8")
    summary_hash = file_sha256(payload_path)

    save_state(
        state_path,
        {
            "runs": [
                {
                    "summary_sha256": summary_hash,
                    "summary_path": str(payload_path),
                }
            ]
        },
    )

    loaded = load_state(state_path)
    assert already_ingested(loaded, summary_hash)
