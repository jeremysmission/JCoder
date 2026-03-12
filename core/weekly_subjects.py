"""Helpers for weekly subject-summary ingestion."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

SUMMARY_GLOB = "WEEKLY_SUBJECT_SUMMARY_*.md"
SUMMARY_FILE_RE = re.compile(r"WEEKLY_SUBJECT_SUMMARY_(\d{4}-\d{2}-\d{2})\.md$")
TITLE_DATE_RE = re.compile(r"--\s*(\d{4}-\d{2}-\d{2})$")
SECTION_RE = re.compile(r"^##\s+\d+\.\s+(.+)$")
_FIELD_LABELS = (
    "Best finding:",
    "Why it matters to JCoder:",
    "Primary source:",
    "Primary sources:",
)


@dataclass(frozen=True)
class SubjectSummary:
    title: str
    best_finding: str
    why_it_matters: str
    sources: List[str]


def find_latest_summary(docs_dir: str | Path) -> Path:
    """Return the newest weekly subject summary by filename date."""
    docs_path = Path(docs_dir)
    candidates = sorted(docs_path.glob(SUMMARY_GLOB))
    if not candidates:
        raise FileNotFoundError(
            f"No weekly subject summaries found in {docs_path}"
        )

    dated: List[Tuple[str, Path]] = []
    undated: List[Path] = []
    for path in candidates:
        match = SUMMARY_FILE_RE.match(path.name)
        if match:
            dated.append((match.group(1), path))
        else:
            undated.append(path)

    if dated:
        return max(dated, key=lambda item: item[0])[1]
    return max(undated, key=lambda path: path.stat().st_mtime)


def parse_summary_document(
    text: str,
    doc_name: str = "",
) -> Tuple[str | None, List[SubjectSummary]]:
    """Parse a weekly summary markdown document into structured sections."""
    lines = text.splitlines()
    doc_date = _extract_doc_date(lines, doc_name)

    sections: List[Tuple[str, List[str]]] = []
    current_title: str | None = None
    current_lines: List[str] = []

    for line in lines:
        match = SECTION_RE.match(line)
        if match:
            if current_title is not None:
                sections.append((current_title, current_lines))
            current_title = match.group(1).strip()
            current_lines = []
            continue
        if current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        sections.append((current_title, current_lines))

    parsed: List[SubjectSummary] = []
    for title, section_lines in sections:
        parsed.append(
            SubjectSummary(
                title=title,
                best_finding=_extract_field(section_lines, "Best finding:"),
                why_it_matters=_extract_field(
                    section_lines, "Why it matters to JCoder:"
                ),
                sources=_extract_sources(section_lines),
            )
        )

    return doc_date, parsed


def build_memory_payloads(summary_path: str | Path) -> List[Dict]:
    """Turn a summary markdown file into AgentMemory ingest payloads."""
    path = Path(summary_path)
    text = path.read_text(encoding="utf-8")
    doc_date, subjects = parse_summary_document(text, doc_name=path.name)
    if not subjects:
        raise ValueError(f"No subject sections found in {path}")

    week_tag = f"week_{doc_date}" if doc_date else "week_unknown"
    payloads: List[Dict] = []
    for subject in subjects:
        slug = slugify(subject.title)
        content = build_memory_content(subject, doc_date, path)
        payloads.append(
            {
                "title": subject.title,
                "source_task": f"weekly-subject-summary:{doc_date or 'unknown'}:{slug}",
                "content": content,
                "tags": [
                    "weekly_subject_summary",
                    "weekly_update",
                    week_tag,
                    slug,
                    "time_sensitive",
                ],
                "confidence": 0.95,
                "tokens_used": estimate_tokens(content),
            }
        )
    return payloads


def build_memory_content(
    subject: SubjectSummary,
    doc_date: str | None,
    summary_path: str | Path,
) -> str:
    """Create the stored memory text for a subject summary."""
    summary_ref = Path(summary_path).as_posix()
    lines = [
        f"Subject: {subject.title}",
        f"Summary date: {doc_date or 'unknown'}",
        "",
        f"Best finding: {subject.best_finding}",
        "",
        f"Why it matters to JCoder: {subject.why_it_matters}",
        "",
        "Primary sources:",
    ]
    for source in subject.sources:
        lines.append(f"- {source}")
    lines.extend(
        [
            "",
            f"Summary document: {summary_ref}",
        ]
    )
    return "\n".join(lines)


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", value) or "subject"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def file_sha256(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def load_state(path: str | Path) -> Dict:
    state_path = Path(path)
    if not state_path.exists():
        return {"runs": []}
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(path: str | Path, state: Dict) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed = dict(state)
    trimmed["runs"] = list(state.get("runs", []))[-52:]
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(trimmed, handle, indent=2)


def already_ingested(state: Dict, summary_hash: str) -> bool:
    return any(
        run.get("summary_sha256") == summary_hash
        for run in state.get("runs", [])
    )


def _extract_doc_date(lines: Sequence[str], doc_name: str) -> str | None:
    for line in lines[:3]:
        title_match = TITLE_DATE_RE.search(line.strip())
        if title_match:
            return title_match.group(1)
    file_match = SUMMARY_FILE_RE.match(doc_name)
    if file_match:
        return file_match.group(1)
    return None


def _extract_field(lines: Sequence[str], label: str) -> str:
    index = _find_label(lines, label)
    if index is None:
        return ""

    stripped = lines[index].strip()
    value = stripped[len(label):].strip()
    parts = [value] if value else []

    index += 1
    while index < len(lines):
        current = lines[index].strip()
        if not current:
            if parts:
                break
            index += 1
            continue
        if SECTION_RE.match(current) or current in _FIELD_LABELS:
            break
        if current.startswith("- "):
            break
        parts.append(current)
        index += 1
    return " ".join(parts).strip()


def _extract_sources(lines: Sequence[str]) -> List[str]:
    labels = ("Primary source:", "Primary sources:")
    index = None
    label = ""
    for candidate in labels:
        index = _find_label(lines, candidate)
        if index is not None:
            label = candidate
            break
    if index is None:
        return []

    stripped = lines[index].strip()
    inline_value = stripped[len(label):].strip()
    sources: List[str] = [inline_value] if inline_value else []

    index += 1
    while index < len(lines):
        current = lines[index].strip()
        if not current:
            if sources:
                break
            index += 1
            continue
        if current.startswith("- "):
            sources.append(current[2:].strip())
            index += 1
            continue
        if SECTION_RE.match(current) or current in _FIELD_LABELS:
            break
        break
    return sources


def _find_label(lines: Sequence[str], label: str) -> int | None:
    for index, line in enumerate(lines):
        if line.strip().startswith(label):
            return index
    return None
