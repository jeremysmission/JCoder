"""Helpers extracted from embedding_engine.py to keep runtime modules small."""

from __future__ import annotations

import re
from typing import List, Tuple


_CODE_INDICATORS = re.compile(
    r"""
    (?:^|\s) (?:def|class|import|from|return|raise|async|await) \s
    | (?:^|\s) (?:function|const|let|var|export|require) [\s(]
    | =>
    | [{}]
    | \b\w+\(.*\)
    | ^\s*(?://|/\*|\#!)
    | ^\s*@\w+
    | \b(?:if|else|for|while|switch|try|catch)\s*[\({]
    """,
    re.VERBOSE | re.MULTILINE,
)

_CODE_LINE_THRESHOLD = 0.3


def detect_content_type(text: str) -> str:
    """Return 'code' or 'text' using lightweight syntax heuristics."""
    lines = text.splitlines()
    non_blank = [line for line in lines if line.strip()]
    if not non_blank:
        return "text"
    hits = sum(1 for line in non_blank if _CODE_INDICATORS.search(line))
    ratio = hits / len(non_blank)
    return "code" if ratio >= _CODE_LINE_THRESHOLD else "text"


def pack_token_budget_batches(
    texts: List[str],
    token_budget: int,
    max_batch_size: int,
    min_batch_size: int = 8,
) -> List[Tuple[int, int]]:
    """Pack texts into index ranges that approximately fit a token budget."""
    batches: List[Tuple[int, int]] = []
    index = 0
    total = len(texts)
    while index < total:
        budget_remaining = token_budget
        end = index
        while end < total and (end - index) < max_batch_size:
            estimated_tokens = max(1, len(texts[end]) // 4)
            if budget_remaining - estimated_tokens < 0 and (end - index) >= min_batch_size:
                break
            budget_remaining -= estimated_tokens
            end += 1
        if (end - index) < min_batch_size and end < total:
            end = min(total, index + min_batch_size)
        batches.append((index, end))
        index = end
    return batches
