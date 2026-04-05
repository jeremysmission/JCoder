"""Text sanitization helpers extracted from sanitizer.py for size discipline."""

from __future__ import annotations

import html
import re
from typing import Dict, List, Tuple


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

_NORMALIZED_SE_LANG_TAGS = set()


def _extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    for match in BACKTICK_BLOCK_RE.finditer(text):
        lang = (match.group(1) or "").strip().lower()
        blocks.append((lang, match.group(2).strip()))
    for match in CODE_TAG_RE.finditer(text):
        code = html.unescape(match.group(1)).strip()
        if code:
            blocks.append(("", code))
    for match in INLINE_BACKTICK_RE.finditer(text):
        code = match.group(1).strip()
        if "\n" not in code:
            blocks.append(("", code))
    unique: List[Tuple[str, str]] = []
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


def _strip_pii(text: str, stats) -> str:
    out = text
    for pattern in PII_PATTERNS:
        out2, count = pattern.subn(" ", out)
        if count:
            stats.pii_replacements += count
        out = out2
    for pattern in NAMEISH_PATTERNS:
        out2, count = pattern.subn(" ", out)
        if count:
            stats.pii_replacements += count
        out = out2
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _is_english_or_unknown(text: str, threshold: float, stats, detect_langs) -> bool:
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
    text = (tag or "").strip().lower()
    if text in ("c++", "cpp"):
        return "cpp"
    if text in ("c#", "csharp", "c_sharp"):
        return "csharp"
    if text in ("js",):
        return "javascript"
    if text in ("ts",):
        return "typescript"
    return text


def _normalized_se_lang_tags() -> set[str]:
    if not _NORMALIZED_SE_LANG_TAGS:
        _NORMALIZED_SE_LANG_TAGS.update(_normalize_lang(tag) for tag in SE_LANG_TAGS)
    return _NORMALIZED_SE_LANG_TAGS


def _infer_lang_from_tags(tags_field: str) -> str:
    tags = re.findall(r"<([^>]+)>", tags_field or "")
    normalized = _normalized_se_lang_tags()
    for tag in tags:
        norm = _normalize_lang(tag)
        if norm in normalized:
            return norm
    return "unknown"
