"""
Sanitization pipeline for pre-index cleaning.

Pipeline:
    raw source -> sanitize -> archive clean copy -> return clean run path

This module is intentionally conservative:
- Keeps code blocks and technical explanations.
- Strips common PII markers and noisy metadata.
- Stores clean outputs under a permanent archive.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from langdetect import detect_langs
except Exception:  # pragma: no cover
    detect_langs = None

from ingestion.chunker import LANGUAGE_MAP
from ingestion.parser_registry import DOCUMENT_EXTENSIONS
from ingestion.sanitizer_archives import ArchiveProcessorMixin
from ingestion.sanitizer_code import CodeCommentMixin

# Derive code extension->language mapping from the single source of truth.
# Only entries with a non-None language value are code files (AST-parseable).
CODE_EXT_TO_LANG = {ext: lang for ext, lang in LANGUAGE_MAP.items() if lang is not None}

# All extensions the sanitizer should consider as candidates.
# Union of code + text/config (LANGUAGE_MAP) + document formats (parser registry)
# plus domain-specific formats (.jsonl, .zst) used for SE/Reddit data.
_SUPPORTED_TEXT_EXTS = (
    set(LANGUAGE_MAP.keys()) | set(DOCUMENT_EXTENSIONS) | {".jsonl", ".zst"}
)

SE_LANG_TAGS = {
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "c",
    "c++",
    "c#",
    "ruby",
    "php",
    "kotlin",
    "sql",
    "bash",
    "powershell",
    "html",
    "css",
}

PII_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"https?://\S+"),
    re.compile(r"www\.\S+"),
    re.compile(r"(?<!\w)@\w+"),
]

NAMEISH_PATTERNS = [
    re.compile(r"\bby\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"),
    re.compile(r"\bsigned[, ]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", re.IGNORECASE),
]

BACKTICK_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
INLINE_BACKTICK_RE = re.compile(r"`([^`\n]{3,})`")
CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


@dataclass
class SanitizationStats:
    files_seen: int = 0
    entries_written: int = 0
    code_blocks_kept: int = 0
    pii_replacements: int = 0
    non_english_removed: int = 0
    compressed_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    reddit_format_samples: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class SanitizationConfig:
    enabled: bool = True
    clean_archive_dir: str = str(
        Path(os.environ.get("JCODER_DATA", os.environ.get("JCODER_DATA_DIR", "data"))) / "clean_source"
    )
    langdetect_threshold: float = 0.8


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return s[:120] or "item"


def _extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    for m in BACKTICK_BLOCK_RE.finditer(text):
        lang = (m.group(1) or "").strip().lower()
        blocks.append((lang, m.group(2).strip()))
    for m in CODE_TAG_RE.finditer(text):
        code = html.unescape(m.group(1)).strip()
        if code:
            blocks.append(("", code))
    for m in INLINE_BACKTICK_RE.finditer(text):
        code = m.group(1).strip()
        if "\n" not in code:
            blocks.append(("", code))
    unique = []
    seen = set()
    for lang, code in blocks:
        key = (lang, code)
        if code and key not in seen:
            seen.add(key)
            unique.append((lang, code))
    return unique


def _strip_code_regions(text: str) -> str:
    out = BACKTICK_BLOCK_RE.sub(" ", text)
    out = CODE_TAG_RE.sub(" ", out)
    out = INLINE_BACKTICK_RE.sub(" ", out)
    return out


def _strip_markup(text: str) -> str:
    out = MARKDOWN_LINK_RE.sub(r"\1", text)
    out = TAG_RE.sub(" ", out)
    out = html.unescape(out)
    out = out.replace("&nbsp;", " ")
    out = re.sub(r"[*_~>#-]+", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _strip_pii(text: str, stats: SanitizationStats) -> str:
    out = text
    for pat in PII_PATTERNS:
        out2, n = pat.subn(" ", out)
        if n:
            stats.pii_replacements += n
        out = out2
    for pat in NAMEISH_PATTERNS:
        out2, n = pat.subn(" ", out)
        if n:
            stats.pii_replacements += n
        out = out2
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _is_english_or_unknown(text: str, threshold: float, stats: SanitizationStats) -> bool:
    if not text:
        return False
    if detect_langs is None:
        return True
    try:
        candidates = detect_langs(text[:5000])
    except Exception:
        return True
    if not candidates:
        return True
    top = candidates[0]
    if top.lang == "en":
        return True
    if top.prob >= threshold:
        stats.non_english_removed += 1
        return False
    return True


def _normalize_lang(tag: str) -> str:
    t = (tag or "").strip().lower()
    if t in ("c++", "cpp"):
        return "cpp"
    if t in ("c#", "csharp", "c_sharp"):
        return "csharp"
    if t in ("js",):
        return "javascript"
    if t in ("ts",):
        return "typescript"
    return t


def _infer_lang_from_tags(tags_field: str) -> str:
    tags = re.findall(r"<([^>]+)>", tags_field or "")
    for t in tags:
        norm = _normalize_lang(t)
        if norm in {_normalize_lang(x) for x in SE_LANG_TAGS}:
            return norm
    return "unknown"


class SanitizationPipeline(ArchiveProcessorMixin, CodeCommentMixin):
    def __init__(self, cfg: SanitizationConfig):
        self.cfg = cfg
        self.clean_root = Path(cfg.clean_archive_dir)
        self.logs_root = self.clean_root / "_logs"
        self.runs_root = self.clean_root / "_ingest_runs"
        self._supported_text_exts = _SUPPORTED_TEXT_EXTS
        self.clean_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def run(self, raw_root: str, index_name: str) -> Tuple[str, SanitizationStats, str]:
        """
        Sanitize raw_root into permanent archive + run snapshot.
        Returns (run_dir, stats, log_path).
        """
        stats = SanitizationStats()
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_root / f"{run_id}_{_safe_slug(index_name)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        raw_path = Path(raw_root)
        if not raw_path.exists():
            raise FileNotFoundError(f"Sanitize input path does not exist: {raw_root}")

        for fp in self._iter_candidate_files(raw_path):
            stats.files_seen += 1
            try:
                self._process_file(fp, run_dir, stats)
            except Exception as e:
                stats.errors.append(f"{fp}: {e}")

        log_path = self._write_logs(raw_root, run_dir, stats)
        return str(run_dir), stats, str(log_path)

    def _iter_candidate_files(self, root: Path) -> Iterable[Path]:
        archive_exts = {".7z", ".zip", ".tar", ".gz", ".xz", ".parquet"}
        if root.is_file():
            ext = root.suffix.lower()
            if ext in _SUPPORTED_TEXT_EXTS or ext in archive_exts:
                yield root
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "venv", "__pycache__"}]
            for fn in filenames:
                fp = Path(dirpath) / fn
                ext = fp.suffix.lower()
                if ext in _SUPPORTED_TEXT_EXTS or ext in archive_exts:
                    yield fp

    def _process_file(self, fp: Path, run_dir: Path, stats: SanitizationStats) -> None:
        ext = fp.suffix.lower()
        lower = str(fp).lower()

        if ext == ".7z":
            self._process_7z_archive(fp, run_dir, stats)
            return

        if ext in {".zip", ".tar", ".gz", ".xz"}:
            self._process_standard_archive(fp, run_dir, stats)
            return

        if ext in {".parquet"}:
            stats.compressed_skipped += 1
            stats.skipped_files.append(f"{fp} [unsupported_archive]")
            return

        if "stackexchange" in lower and fp.name.lower() == "posts.xml":
            self._process_stackexchange_posts(fp, run_dir, stats)
            return

        if "reddit" in lower and ext in {".zst", ".json", ".jsonl"}:
            self._process_reddit(fp, run_dir, stats)
            return

        if ext in CODE_EXT_TO_LANG:
            self._process_code_comments(fp, run_dir, stats)
            return

        # Generic text fallback: only keep code blocks + short technical context.
        if ext in {".md", ".txt", ".json", ".jsonl", ".xml"}:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                stats.skipped_files.append(f"{fp} [read_error]")
                return
            blocks = _extract_code_blocks(text)
            if not blocks:
                stats.skipped_files.append(f"{fp} [no_code_blocks]")
                return
            expl = _strip_markup(_strip_code_regions(text))
            expl = _strip_pii(expl, stats)
            if not _is_english_or_unknown(expl, self.cfg.langdetect_threshold, stats):
                expl = ""
            entry = self._build_md_entry(
                title=fp.stem,
                source_path=str(fp),
                source_kind="generic",
                language="unknown",
                explanation=expl[:2000],
                code_blocks=blocks[:20],
                tags=[],
            )
            self._write_entry(entry, "unknown", "generic", fp.stem, run_dir, stats)
            return

    def _process_stackexchange_posts(self, fp: Path, run_dir: Path, stats: SanitizationStats) -> None:
        questions: Dict[int, Dict[str, object]] = {}
        answers: Dict[int, List[Dict[str, object]]] = {}

        for _event, elem in ET.iterparse(str(fp), events=("end",)):
            if elem.tag != "row":
                continue
            attrs = elem.attrib
            post_type = attrs.get("PostTypeId", "")
            try:
                post_id = int(attrs.get("Id", "0"))
            except Exception:
                post_id = 0
            if post_id <= 0:
                elem.clear()
                continue

            if post_type == "1":
                questions[post_id] = {
                    "title": attrs.get("Title", ""),
                    "tags": attrs.get("Tags", ""),
                    "accepted": int(attrs.get("AcceptedAnswerId", "0") or "0"),
                }
            elif post_type == "2":
                try:
                    parent_id = int(attrs.get("ParentId", "0") or "0")
                except Exception:
                    parent_id = 0
                if parent_id > 0:
                    ans = {
                        "id": post_id,
                        "score": int(attrs.get("Score", "0") or "0"),
                        "created": attrs.get("CreationDate", ""),
                        "body": attrs.get("Body", ""),
                    }
                    answers.setdefault(parent_id, []).append(ans)
            elem.clear()

        site_name = fp.parent.name.lower()
        for qid, q in questions.items():
            title = _strip_pii(_strip_markup(str(q.get("title", ""))), stats)
            tags_field = str(q.get("tags", ""))
            lang = _infer_lang_from_tags(tags_field)
            accepted_id = int(q.get("accepted", 0) or 0)
            candidates = answers.get(qid, [])
            if not candidates:
                continue

            normalized = []
            for ans in candidates:
                body = ans["body"]
                blocks = _extract_code_blocks(body)
                if not blocks:
                    continue
                expl = _strip_markup(_strip_code_regions(body))
                expl = _strip_pii(expl, stats)
                if not _is_english_or_unknown(expl, self.cfg.langdetect_threshold, stats):
                    expl = ""
                normalized.append({
                    "id": int(ans["id"]),
                    "score": int(ans["score"]),
                    "created": ans["created"],
                    "blocks": blocks,
                    "expl": expl,
                })
            if not normalized:
                continue

            chosen: List[Dict[str, object]] = []
            # Accepted first when present and contains code.
            if accepted_id:
                acc = next((a for a in normalized if int(a["id"]) == accepted_id), None)
                if acc:
                    chosen.append(acc)
            # Plus top 3 by score desc, created asc (tie-break).
            ranked = sorted(normalized, key=lambda a: (-int(a["score"]), str(a["created"])))
            for a in ranked:
                if len(chosen) >= 3:
                    break
                if all(int(x["id"]) != int(a["id"]) for x in chosen):
                    chosen.append(a)

            all_blocks: List[Tuple[str, str]] = []
            expl_parts: List[str] = []
            for a in chosen:
                all_blocks.extend(list(a["blocks"]))  # type: ignore[arg-type]
                if a["expl"]:
                    expl_parts.append(str(a["expl"]))

            tags = [t for t in re.findall(r"<([^>]+)>", tags_field or "") if t]
            entry = self._build_md_entry(
                title=title or f"Question {qid}",
                source_path=f"{fp}#q{qid}",
                source_kind="stackoverflow",
                language=lang,
                explanation=(" ".join(expl_parts)).strip()[:3000],
                code_blocks=all_blocks[:30],
                tags=tags[:20],
            )
            file_stub = f"{site_name}_q{qid}"
            self._write_entry(entry, lang, "stackoverflow", file_stub, run_dir, stats)

    def _process_reddit(self, fp: Path, run_dir: Path, stats: SanitizationStats) -> None:
        lower = str(fp).lower()
        source = "reddit"
        reader = self._reddit_line_reader(fp, stats)
        if reader is None:
            stats.skipped_files.append(f"{fp} [unsupported_reddit_format]")
            return

        for idx, line in enumerate(reader):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if idx < 5:
                stats.reddit_format_samples.append({
                    "file": str(fp),
                    "keys": sorted(list(obj.keys()))[:20],
                })
            title = str(obj.get("title", "") or "")
            body = str(obj.get("selftext", "") or obj.get("body", "") or "")
            merged = (title + "\n\n" + body).strip()
            blocks = _extract_code_blocks(merged)
            if not blocks:
                continue
            expl = _strip_markup(_strip_code_regions(merged))
            expl = _strip_pii(expl, stats)
            if not _is_english_or_unknown(expl, self.cfg.langdetect_threshold, stats):
                expl = ""

            lang = "unknown"
            for fence_lang, _code in blocks:
                norm = _normalize_lang(fence_lang)
                if norm and norm != "text":
                    lang = norm
                    break
            tags = []
            subr = str(obj.get("subreddit", "") or "")
            if subr:
                tags.append(subr)
            entry = self._build_md_entry(
                title=_strip_pii(title[:200], stats) or f"Reddit item {idx}",
                source_path=f"{fp}#line{idx+1}",
                source_kind=source,
                language=lang,
                explanation=expl[:2000],
                code_blocks=blocks[:20],
                tags=tags,
            )
            self._write_entry(entry, lang, source, f"{fp.stem}_{idx+1}", run_dir, stats)

    # Archive methods: _process_7z_archive, _process_standard_archive,
    # _safe_member_target, _extract_zip/tar_members_safe, _reddit_line_reader
    # → ingestion/sanitizer_archives.py (ArchiveProcessorMixin)

    # Code comment methods: _process_code_comments, _extract_python_docstrings,
    # _extract_python_comments, _extract_generic_comments
    # → ingestion/sanitizer_code.py (CodeCommentMixin)

    def _build_md_entry(
        self,
        title: str,
        source_path: str,
        source_kind: str,
        language: str,
        explanation: str,
        code_blocks: List[Tuple[str, str]],
        tags: List[str],
    ) -> str:
        lines = [
            f"# {title}",
            "",
            f"- source_kind: {source_kind}",
            f"- language: {language}",
            f"- source_path: {source_path}",
            f"- generated_at: {_utc_now_iso()}",
        ]
        if tags:
            lines.append(f"- tags: {', '.join(tags)}")
        lines.append("")
        if explanation:
            lines.extend(["## Technical Explanation", explanation, ""])
        if code_blocks:
            lines.append("## Code")
            for lang, code in code_blocks:
                fence_lang = (lang or language or "").strip()
                lines.extend([f"```{fence_lang}", code, "```", ""])
        return "\n".join(lines).strip() + "\n"

    def _write_entry(
        self,
        content_md: str,
        language: str,
        source: str,
        stem: str,
        run_dir: Path,
        stats: SanitizationStats,
    ) -> None:
        lang = _normalize_lang(language) or "unknown"
        bucket = self.clean_root / lang / source
        run_bucket = run_dir / lang / source
        bucket.mkdir(parents=True, exist_ok=True)
        run_bucket.mkdir(parents=True, exist_ok=True)

        fname = f"{_safe_slug(stem)}.md"
        dest = bucket / fname
        run_dest = run_bucket / fname
        # Append when file exists to avoid collisions during batch processing.
        with open(dest, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n")
            f.write(content_md)
        with open(run_dest, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n")
            f.write(content_md)
        stats.entries_written += 1
        stats.code_blocks_kept += content_md.count("```") // 2

    def _write_logs(self, raw_root: str, run_dir: Path, stats: SanitizationStats) -> Path:
        payload = {
            "generated_at": _utc_now_iso(),
            "raw_root": raw_root,
            "clean_root": str(self.clean_root),
            "run_dir": str(run_dir),
            "stats": {
                "files_seen": stats.files_seen,
                "entries_written": stats.entries_written,
                "code_blocks_kept": stats.code_blocks_kept,
                "pii_replacements": stats.pii_replacements,
                "non_english_removed": stats.non_english_removed,
                "compressed_skipped": stats.compressed_skipped,
                "errors_count": len(stats.errors),
                "skipped_count": len(stats.skipped_files),
            },
            "reddit_format_samples": stats.reddit_format_samples[:20],
            "errors": stats.errors[:5000],
            "skipped_files": stats.skipped_files[:5000],
        }
        out = self.logs_root / f"sanitize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out
