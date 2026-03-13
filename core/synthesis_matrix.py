"""
Synthesis Matrix Engine
-----------------------
Builds a structured theme-by-source matrix from digested papers.
Identifies agreement, contradiction, and gaps across sources.

Based on: Glass (1976) meta-analysis, Hedges & Olkin (1985)
evidence synthesis, and practitioner source matrix methods.

Replaces narrative "paper A says X, paper B says Y" with a
structured grid that reveals patterns at a glance.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from core.runtime import Runtime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SynthesisCell:
    """One cell in the matrix: what a single source says about a theme."""
    source_title: str
    position: str       # what this source says about this theme
    sentiment: str      # "supports" | "contradicts" | "neutral" | "silent"
    evidence_strength: float  # 0.0-1.0


@dataclass
class ThemeRow:
    """One row in the matrix: a theme and every source's position on it."""
    theme: str
    cells: List[SynthesisCell]
    consensus: str           # "strong_agreement" | "mixed" | "contradicted"
    contradiction_flag: bool


@dataclass
class MatrixReport:
    """Complete synthesis matrix report."""
    query: str
    themes: List[ThemeRow]
    total_sources: int
    strong_agreements: int
    contradictions: int
    gaps: int  # themes where most sources are "silent"


_VALID_SENTIMENTS = {"supports", "contradicts", "neutral", "silent"}


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_THEME_PROMPT = (
    "Given these papers about '{query}', identify {max_themes} distinct "
    "themes that the papers collectively address. Return one theme per "
    "line, nothing else.\n\n"
    "Papers:\n{paper_block}"
)

_CLASSIFY_PROMPT = (
    "For each (theme, paper) pair below, classify the paper's position "
    "on the theme. Reply as a JSON array of objects with keys: "
    "source, theme, sentiment, position, strength.\n\n"
    "sentiment MUST be one of: supports, contradicts, neutral, silent\n"
    "strength MUST be a float between 0.0 and 1.0\n"
    "position is a brief (1-sentence) description of what the paper says.\n\n"
    "Themes:\n{theme_block}\n\n"
    "Papers:\n{paper_block}\n\n"
    "Return ONLY valid JSON, no markdown fences."
)


# ---------------------------------------------------------------------------
# Heuristic helpers (used when no runtime is available)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Split text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z][a-z0-9]{2,}", text.lower())


def _extract_noun_phrases(texts: List[str], max_phrases: int) -> List[str]:
    """
    Simple frequency-based phrase extraction.
    Collects 2-word and single-word tokens, returns most common.
    """
    counts: Counter = Counter()
    for text in texts:
        tokens = _tokenize(text)
        # single tokens
        counts.update(tokens)
        # bigrams
        for i in range(len(tokens) - 1):
            counts[f"{tokens[i]} {tokens[i + 1]}"] += 1

    # filter out stopwords
    stops = {
        "the", "and", "for", "that", "this", "with", "from", "are",
        "was", "were", "has", "have", "been", "not", "but", "can",
        "will", "its", "our", "their", "also", "more", "than",
        "which", "these", "those", "into", "such", "may", "most",
    }
    filtered = [
        (phrase, c) for phrase, c in counts.items()
        if phrase not in stops
        and len(phrase) > 3
        and not all(w in stops for w in phrase.split())
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [phrase for phrase, _ in filtered[:max_phrases]]


# ---------------------------------------------------------------------------
# SynthesisMatrix
# ---------------------------------------------------------------------------

class SynthesisMatrix:
    """
    Builds a theme-by-source matrix from digested papers.

    With a Runtime: uses LLM to extract themes and classify positions.
    Without a Runtime: falls back to keyword heuristics (no LLM needed).
    """

    def __init__(self, runtime: Optional[Runtime] = None):
        self._runtime = runtime

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        digested_papers: List[Dict],
        query: str,
    ) -> MatrixReport:
        """
        Build a full synthesis matrix from digested papers.

        Args:
            digested_papers: list of dicts with at least 'title' and
                either 'key_claims' or 'triage_summary' / 'abstract'.
            query: the research question driving the synthesis.

        Returns:
            MatrixReport with themes, cells, consensus, and counts.
        """
        if not digested_papers:
            return MatrixReport(
                query=query, themes=[], total_sources=0,
                strong_agreements=0, contradictions=0, gaps=0,
            )

        # Step 1: extract themes
        themes = self._extract_themes(digested_papers, query)

        # Step 2: classify positions per source per theme
        rows = self._classify_positions(digested_papers, themes)

        # Step 3: detect contradictions / consensus
        rows = self._detect_contradictions(rows)

        # Step 4: aggregate counts
        strong = sum(1 for r in rows if r.consensus == "strong_agreement")
        contras = sum(1 for r in rows if r.contradiction_flag)
        gap_count = self._count_gaps(rows, len(digested_papers))

        return MatrixReport(
            query=query,
            themes=rows,
            total_sources=len(digested_papers),
            strong_agreements=strong,
            contradictions=contras,
            gaps=gap_count,
        )

    def to_markdown_table(self, report: MatrixReport) -> str:
        """
        Render the matrix as a markdown table.

        Rows = themes, columns = sources.
        Cell content: emoji + brief position.
        """
        if not report.themes:
            return "(empty matrix)"

        # Collect unique source titles in order
        sources = self._ordered_sources(report)

        # Header
        header = "| Theme | " + " | ".join(sources) + " |"
        sep = "|" + "---|" * (len(sources) + 1)

        # Rows
        body_lines = []
        for row in report.themes:
            cell_map = {c.source_title: c for c in row.cells}
            cells = []
            for src in sources:
                cell = cell_map.get(src)
                if cell is None:
                    cells.append("-")
                else:
                    icon = self._sentiment_icon(cell.sentiment)
                    brief = cell.position[:40] if cell.position else ""
                    cells.append(f"{icon} {brief}".strip())
            body_lines.append(f"| {row.theme} | " + " | ".join(cells) + " |")

        # Summary footer
        summary = (
            f"Agreements: {report.strong_agreements} | "
            f"Contradictions: {report.contradictions} | "
            f"Gaps: {report.gaps}"
        )

        return "\n".join([header, sep] + body_lines + ["", summary])

    def to_dict(self, report: MatrixReport) -> Dict[str, Any]:
        """Serialize a MatrixReport to a plain dict for JSON output."""
        return asdict(report)

    # ------------------------------------------------------------------
    # Internal: theme extraction
    # ------------------------------------------------------------------

    def _extract_themes(
        self,
        papers: List[Dict],
        query: str,
        max_themes: int = 8,
    ) -> List[str]:
        """
        Identify key themes across papers.

        LLM path: structured prompt asking for themes.
        Heuristic path: noun-phrase frequency from key_claims/abstracts.
        """
        if self._runtime:
            try:
                return self._extract_themes_llm(papers, query, max_themes)
            except Exception:
                logger.info(
                    "LLM theme extraction failed, falling back to heuristic",
                    exc_info=True,
                )
        return self._extract_themes_heuristic(papers, max_themes)

    def _extract_themes_llm(
        self,
        papers: List[Dict],
        query: str,
        max_themes: int,
    ) -> List[str]:
        paper_block = self._paper_summary_block(papers)
        prompt = _THEME_PROMPT.format(
            query=query, max_themes=max_themes, paper_block=paper_block,
        )
        raw = self._runtime.generate(
            question=prompt,
            context_chunks=[],
            system_prompt="You are a research synthesis assistant.",
            temperature=0.0,
            max_tokens=256,
        )
        themes = [
            line.strip().lstrip("0123456789.-) ")
            for line in raw.strip().splitlines()
            if line.strip()
        ]
        return themes[:max_themes]

    def _extract_themes_heuristic(
        self,
        papers: List[Dict],
        max_themes: int,
    ) -> List[str]:
        """Keyword frequency fallback: collect claims/abstracts, find phrases."""
        texts: List[str] = []
        for paper in papers:
            claims = paper.get("key_claims", [])
            if isinstance(claims, list):
                texts.extend(claims)
            summary = paper.get("triage_summary", "")
            if summary:
                texts.append(summary)
            abstract = paper.get("abstract", "")
            if abstract:
                texts.append(abstract[:200])
        return _extract_noun_phrases(texts, max_themes)

    # ------------------------------------------------------------------
    # Internal: position classification
    # ------------------------------------------------------------------

    def _classify_positions(
        self,
        papers: List[Dict],
        themes: List[str],
    ) -> List[ThemeRow]:
        """Classify each paper's position on each theme."""
        if self._runtime:
            try:
                return self._classify_positions_llm(papers, themes)
            except Exception:
                logger.info(
                    "LLM position classification failed, falling back to heuristic",
                    exc_info=True,
                )
        return self._classify_positions_heuristic(papers, themes)

    def _classify_positions_llm(
        self,
        papers: List[Dict],
        themes: List[str],
    ) -> List[ThemeRow]:
        paper_block = self._paper_summary_block(papers)
        theme_block = "\n".join(f"- {t}" for t in themes)
        prompt = _CLASSIFY_PROMPT.format(
            theme_block=theme_block, paper_block=paper_block,
        )
        raw = self._runtime.generate(
            question=prompt,
            context_chunks=[],
            system_prompt="You are a research synthesis assistant.",
            temperature=0.0,
            max_tokens=2048,
        )
        entries = self._parse_classify_json(raw)
        return self._entries_to_rows(entries, themes)

    def _classify_positions_heuristic(
        self,
        papers: List[Dict],
        themes: List[str],
    ) -> List[ThemeRow]:
        """Keyword overlap: if theme words appear in paper text -> supports."""
        rows: List[ThemeRow] = []
        for theme in themes:
            theme_tokens = set(_tokenize(theme))
            cells: List[SynthesisCell] = []
            for paper in papers:
                title = paper.get("title", "unknown")
                text = self._paper_text(paper).lower()
                text_tokens = set(_tokenize(text))
                overlap = theme_tokens & text_tokens
                if overlap:
                    cells.append(SynthesisCell(
                        source_title=title,
                        position=f"mentions {', '.join(sorted(overlap))}",
                        sentiment="supports",
                        evidence_strength=0.5,
                    ))
                else:
                    cells.append(SynthesisCell(
                        source_title=title,
                        position="",
                        sentiment="silent",
                        evidence_strength=0.0,
                    ))
            rows.append(ThemeRow(
                theme=theme, cells=cells,
                consensus="mixed", contradiction_flag=False,
            ))
        return rows

    # ------------------------------------------------------------------
    # Internal: contradiction / consensus detection
    # ------------------------------------------------------------------

    def _detect_contradictions(self, rows: List[ThemeRow]) -> List[ThemeRow]:
        """
        Evaluate consensus for each theme row.

        Rules:
        - If any cell has "contradicts" -> consensus = "contradicted"
        - If all non-silent cells agree (supports/neutral) -> "strong_agreement"
        - Otherwise -> "mixed"
        """
        for row in rows:
            non_silent = [c for c in row.cells if c.sentiment != "silent"]
            if not non_silent:
                row.consensus = "mixed"
                row.contradiction_flag = False
                continue

            has_contradiction = any(
                c.sentiment == "contradicts" for c in non_silent
            )
            if has_contradiction:
                row.consensus = "contradicted"
                row.contradiction_flag = True
            elif all(c.sentiment in ("supports", "neutral") for c in non_silent):
                row.consensus = "strong_agreement"
                row.contradiction_flag = False
            else:
                row.consensus = "mixed"
                row.contradiction_flag = False

        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_gaps(rows: List[ThemeRow], total_sources: int) -> int:
        """A theme is a 'gap' when more than half the sources are silent."""
        if total_sources == 0:
            return 0
        threshold = total_sources / 2.0
        return sum(
            1 for row in rows
            if sum(1 for c in row.cells if c.sentiment == "silent") > threshold
        )

    @staticmethod
    def _ordered_sources(report: MatrixReport) -> List[str]:
        """Collect unique source titles in first-seen order."""
        seen: Dict[str, None] = {}
        for row in report.themes:
            for cell in row.cells:
                if cell.source_title not in seen:
                    seen[cell.source_title] = None
        return list(seen.keys())

    @staticmethod
    def _sentiment_icon(sentiment: str) -> str:
        return {
            "supports": "[+]",
            "contradicts": "[X]",
            "neutral": "[?]",
            "silent": "[-]",
        }.get(sentiment, "[-]")

    @staticmethod
    def _paper_summary_block(papers: List[Dict]) -> str:
        """Build a compact text block summarizing each paper for prompts."""
        lines = []
        for paper in papers:
            title = paper.get("title", "untitled")
            summary = paper.get("triage_summary", "")
            if not summary:
                summary = paper.get("abstract", "")[:200]
            claims = paper.get("key_claims", [])
            claim_text = "; ".join(claims) if isinstance(claims, list) else ""
            lines.append(f"- {title}: {summary} Claims: {claim_text}")
        return "\n".join(lines)

    @staticmethod
    def _paper_text(paper: Dict) -> str:
        """Concatenate all text fields for keyword matching."""
        parts = [paper.get("title", "")]
        claims = paper.get("key_claims", [])
        if isinstance(claims, list):
            parts.extend(claims)
        parts.append(paper.get("triage_summary", ""))
        parts.append(paper.get("abstract", ""))
        parts.append(paper.get("novel_method", ""))
        return " ".join(parts)

    @staticmethod
    def _parse_classify_json(raw: str) -> List[Dict]:
        """Parse the LLM's JSON array response, tolerating minor formatting."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()
        try:
            entries = json.loads(cleaned)
            if isinstance(entries, list):
                return entries
        except json.JSONDecodeError:
            pass
        return []

    @staticmethod
    def _entries_to_rows(
        entries: List[Dict],
        themes: List[str],
    ) -> List[ThemeRow]:
        """Group parsed LLM entries into ThemeRows."""
        theme_map: Dict[str, List[SynthesisCell]] = {t: [] for t in themes}
        for entry in entries:
            theme = entry.get("theme", "")
            if theme not in theme_map:
                continue
            sentiment = entry.get("sentiment", "neutral")
            if sentiment not in _VALID_SENTIMENTS:
                sentiment = "neutral"
            strength = entry.get("strength", 0.5)
            if not isinstance(strength, (int, float)):
                strength = 0.5
            strength = max(0.0, min(1.0, float(strength)))
            theme_map[theme].append(SynthesisCell(
                source_title=entry.get("source", "unknown"),
                position=entry.get("position", ""),
                sentiment=sentiment,
                evidence_strength=strength,
            ))
        return [
            ThemeRow(theme=t, cells=cells, consensus="mixed",
                     contradiction_flag=False)
            for t, cells in theme_map.items()
        ]
