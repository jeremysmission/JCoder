"""
Rapid Research Digestion Engine
---------------------------------
Automates the entire pipeline from "raw paper" to "actionable code"
in minutes instead of hours. Implements the 3-pass reading method
with LLM acceleration.

Pipeline:
  RAW PAPER -> Triage (30 seconds)
            -> Extract (2 minutes)
            -> Synthesize (1 minute)
            -> Prototype (5 minutes)
            -> Store (instant)

Based on:
- Keshav's "How to Read a Paper" (3-pass method, accelerated)
- Feynman Technique (explain simply to verify understanding)
- Active Recall (test-yourself extraction beats passive reading)
- Research Sprints (intense bursts with immediate application)

The key insight: Most papers have 1-3 actionable ideas buried in
20 pages of related work and formalism. This engine extracts those
ideas and generates working code stubs in minutes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.runtime import Runtime

logger = logging.getLogger(__name__)


@dataclass
class PaperDigest:
    """Complete digest of a research paper."""
    digest_id: str
    title: str
    source_url: str
    # Pass 1: Triage (30 seconds)
    category: str  # "breakthrough" | "incremental" | "survey" | "skip"
    relevance: float  # 0.0-1.0 to our focus areas
    triage_summary: str  # 1-sentence summary
    # Pass 2: Extract (2 minutes)
    key_claims: List[str]  # 3-5 main claims
    novel_method: str  # the core new technique
    results_summary: str  # quantitative results
    limitations: List[str]  # what doesn't work
    code_available: bool
    code_url: str
    # Pass 3: Synthesize (1 minute)
    actionable_ideas: List[str]  # what we can USE
    implementation_sketch: str  # pseudocode / approach
    connections: List[str]  # links to other known work
    # Metadata
    digested_at: float = 0.0
    total_seconds: float = 0.0


@dataclass
class PrototypeStub:
    """Auto-generated code stub from a paper's methodology."""
    paper_title: str
    class_name: str
    description: str
    code: str
    estimated_effort: str  # "trivial" | "afternoon" | "weekend" | "project"
    dependencies: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Triage prompts (Pass 1: 30 seconds)
# ---------------------------------------------------------------------------

_TRIAGE_PROMPT = (
    "You are a research triage system. In ONE sentence, summarize this paper "
    "and classify it.\n\n"
    "Title: {title}\n"
    "Abstract: {abstract}\n\n"
    "Output format (exactly 3 lines):\n"
    "CATEGORY: breakthrough|incremental|survey|skip\n"
    "RELEVANCE: 0.0-1.0 (to code RAG, retrieval, self-learning)\n"
    "SUMMARY: <one sentence>"
)

# ---------------------------------------------------------------------------
# Extraction prompts (Pass 2: 2 minutes)
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = (
    "Extract the key information from this paper in structured format.\n\n"
    "Title: {title}\n"
    "Content: {content}\n\n"
    "Output EXACTLY this format:\n"
    "CLAIMS:\n"
    "- <claim 1>\n"
    "- <claim 2>\n"
    "- <claim 3>\n"
    "METHOD: <the core novel technique in 2-3 sentences>\n"
    "RESULTS: <quantitative results: X% improvement on Y benchmark>\n"
    "LIMITS:\n"
    "- <limitation 1>\n"
    "- <limitation 2>\n"
    "CODE: yes|no\n"
    "CODE_URL: <url or none>"
)

# ---------------------------------------------------------------------------
# Synthesis prompts (Pass 3: 1 minute)
# ---------------------------------------------------------------------------

_SYNTHESIZE_PROMPT = (
    "Given this paper's method, generate actionable implementation ideas "
    "for a LOCAL code RAG system (Python + SQLite + Ollama LLM).\n\n"
    "Paper: {title}\n"
    "Method: {method}\n"
    "Results: {results}\n\n"
    "Output EXACTLY this format:\n"
    "IDEAS:\n"
    "- <actionable idea 1>\n"
    "- <actionable idea 2>\n"
    "- <actionable idea 3>\n"
    "SKETCH: <Python pseudocode for the core algorithm, 10-20 lines>\n"
    "CONNECTS: <what existing techniques this relates to>"
)

# ---------------------------------------------------------------------------
# Prototype generation prompt
# ---------------------------------------------------------------------------

_PROTOTYPE_PROMPT = (
    "Generate a working Python class that implements the core idea from "
    "this research paper. Use only stdlib + sqlite3 + httpx.\n\n"
    "Paper: {title}\n"
    "Core method: {method}\n"
    "Implementation sketch: {sketch}\n\n"
    "Requirements:\n"
    "- Class with __init__, main method, and docstring\n"
    "- SQLite storage if state is needed\n"
    "- Type hints\n"
    "- Under 200 lines\n"
    "- WORKING code, not pseudocode\n\n"
    "Output ONLY the Python code."
)


class RapidDigester:
    """
    3-pass research digestion engine with LLM acceleration.

    Pass 1 (Triage): Is this paper worth reading? (30 seconds)
    Pass 2 (Extract): What are the key ideas? (2 minutes)
    Pass 3 (Synthesize): How do we use this? (1 minute)
    Bonus: Generate working prototype stub (5 minutes)

    Total: ~8 minutes from raw paper to working code stub.
    """

    def __init__(
        self,
        runtime: Runtime,
        db_path: str = "_digests/digest.db",
    ):
        self.runtime = runtime
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS digests (
                    digest_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_url TEXT,
                    category TEXT,
                    relevance REAL,
                    triage_summary TEXT,
                    key_claims_json TEXT,
                    novel_method TEXT,
                    results_summary TEXT,
                    limitations_json TEXT,
                    code_available INTEGER,
                    code_url TEXT,
                    actionable_ideas_json TEXT,
                    implementation_sketch TEXT,
                    connections_json TEXT,
                    prototype_code TEXT,
                    digested_at REAL,
                    total_seconds REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_digest_relevance
                ON digests(relevance DESC)
            """)
            conn.commit()

    def digest(
        self,
        title: str,
        abstract: str,
        full_text: str = "",
        source_url: str = "",
        generate_prototype: bool = True,
    ) -> PaperDigest:
        """
        Full 3-pass digest of a paper.

        Args:
            title: Paper title
            abstract: Paper abstract
            full_text: Full paper text (optional, abstract used if empty)
            source_url: URL of the paper
            generate_prototype: Whether to generate code stub

        Returns:
            PaperDigest with all extracted information
        """
        t0 = time.time()
        content = full_text if full_text else abstract
        digest_id = hashlib.sha256(
            f"{title}:{abstract[:200]}".encode()
        ).hexdigest()[:12]

        # === Pass 1: Triage (30 seconds) ===
        category, relevance, summary = self._triage(title, abstract)

        # Skip papers that aren't relevant
        if category == "skip" or relevance < 0.2:
            return PaperDigest(
                digest_id=digest_id, title=title, source_url=source_url,
                category=category, relevance=relevance,
                triage_summary=summary,
                key_claims=[], novel_method="", results_summary="",
                limitations=[], code_available=False, code_url="",
                actionable_ideas=[], implementation_sketch="",
                connections=[],
                digested_at=time.time(),
                total_seconds=time.time() - t0,
            )

        # === Pass 2: Extract (2 minutes) ===
        extraction = self._extract(title, content)

        # === Pass 3: Synthesize (1 minute) ===
        synthesis = self._synthesize(
            title,
            extraction.get("method", ""),
            extraction.get("results", ""),
        )

        # === Bonus: Prototype Generation ===
        prototype_code = ""
        if generate_prototype and relevance >= 0.5:
            prototype = self.generate_prototype(
                title,
                extraction.get("method", ""),
                synthesis.get("sketch", ""),
            )
            if prototype:
                prototype_code = prototype.code

        total_seconds = time.time() - t0

        digest = PaperDigest(
            digest_id=digest_id,
            title=title,
            source_url=source_url,
            category=category,
            relevance=relevance,
            triage_summary=summary,
            key_claims=extraction.get("claims", []),
            novel_method=extraction.get("method", ""),
            results_summary=extraction.get("results", ""),
            limitations=extraction.get("limits", []),
            code_available=extraction.get("code", False),
            code_url=extraction.get("code_url", ""),
            actionable_ideas=synthesis.get("ideas", []),
            implementation_sketch=synthesis.get("sketch", ""),
            connections=synthesis.get("connects", []),
            digested_at=time.time(),
            total_seconds=total_seconds,
        )

        self._persist(digest, prototype_code)
        return digest

    def batch_triage(
        self,
        papers: List[Dict[str, str]],
        min_relevance: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Quickly triage a batch of papers. Returns only the relevant ones.
        ~30 seconds per paper, parallelizable.

        Args:
            papers: List of {"title": ..., "abstract": ...}
            min_relevance: Minimum relevance threshold
        """
        results = []
        for paper in papers:
            category, relevance, summary = self._triage(
                paper.get("title", ""),
                paper.get("abstract", ""),
            )
            if relevance >= min_relevance:
                results.append({
                    "title": paper.get("title", ""),
                    "category": category,
                    "relevance": relevance,
                    "summary": summary,
                    "url": paper.get("url", ""),
                })

        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results

    def generate_prototype(
        self,
        title: str,
        method: str,
        sketch: str,
    ) -> Optional[PrototypeStub]:
        """Generate a working code prototype from paper methodology."""
        if not method:
            return None

        try:
            prompt = _PROTOTYPE_PROMPT.format(
                title=title, method=method, sketch=sketch,
            )
            code = self.runtime.generate(
                question=prompt,
                context_chunks=[],
                system_prompt="Generate clean, working Python code.",
                temperature=0.2,
                max_tokens=2048,
            )

            # Extract class name
            class_match = re.search(r"class\s+(\w+)", code)
            class_name = class_match.group(1) if class_match else "Prototype"

            # Estimate effort
            lines = len(code.split("\n"))
            if lines < 50:
                effort = "trivial"
            elif lines < 150:
                effort = "afternoon"
            elif lines < 400:
                effort = "weekend"
            else:
                effort = "project"

            # Extract dependencies
            deps = re.findall(r"(?:from|import)\s+(\w+)", code)
            deps = sorted(set(d for d in deps if d not in (
                "os", "sys", "re", "json", "math", "time",
                "typing", "dataclasses", "pathlib", "hashlib",
                "sqlite3", "collections", "functools",
            )))

            return PrototypeStub(
                paper_title=title,
                class_name=class_name,
                description=method[:200],
                code=code,
                estimated_effort=effort,
                dependencies=deps,
            )
        except Exception as exc:
            logger.warning(
                "Failed to generate prototype for '%s': %s",
                title,
                exc,
            )
            return None

    def _triage(
        self, title: str, abstract: str
    ) -> Tuple[str, float, str]:
        """Pass 1: Quick triage."""
        prompt = _TRIAGE_PROMPT.format(title=title, abstract=abstract[:500])
        try:
            raw = self.runtime.generate(
                question=prompt, context_chunks=[],
                system_prompt="You are a research triage system.",
                temperature=0.0, max_tokens=128,
            )
            return self._parse_triage(raw)
        except Exception:
            logger.warning("Failed to triage paper '%s'", title, exc_info=True)
            return "skip", 0.0, "Failed to triage"

    def _extract(self, title: str, content: str) -> Dict[str, Any]:
        """Pass 2: Deep extraction."""
        prompt = _EXTRACT_PROMPT.format(
            title=title, content=content[:3000]
        )
        try:
            raw = self.runtime.generate(
                question=prompt, context_chunks=[],
                system_prompt="Extract structured information from papers.",
                temperature=0.0, max_tokens=512,
            )
            return self._parse_extraction(raw)
        except Exception:
            logger.warning("Failed to extract paper '%s'", title, exc_info=True)
            return {}

    def _synthesize(
        self, title: str, method: str, results: str
    ) -> Dict[str, Any]:
        """Pass 3: Actionable synthesis."""
        prompt = _SYNTHESIZE_PROMPT.format(
            title=title, method=method, results=results,
        )
        try:
            raw = self.runtime.generate(
                question=prompt, context_chunks=[],
                system_prompt="Generate actionable implementation ideas.",
                temperature=0.3, max_tokens=512,
            )
            return self._parse_synthesis(raw)
        except Exception:
            logger.warning("Failed to synthesize paper '%s'", title, exc_info=True)
            return {}

    @staticmethod
    def _parse_triage(raw: str) -> Tuple[str, float, str]:
        """Parse triage response."""
        category = "skip"
        relevance = 0.0
        summary = ""

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("CATEGORY:"):
                cat = line.split(":", 1)[1].strip().lower()
                if cat in ("breakthrough", "incremental", "survey", "skip"):
                    category = cat
            elif line.startswith("RELEVANCE:"):
                try:
                    relevance = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()

        return category, relevance, summary

    @staticmethod
    def _parse_extraction(raw: str) -> Dict[str, Any]:
        """Parse extraction response."""
        result: Dict[str, Any] = {
            "claims": [], "method": "", "results": "",
            "limits": [], "code": False, "code_url": "",
        }

        lines = raw.split("\n")
        current_section = ""

        for line in lines:
            line = line.strip()
            if line.startswith("CLAIMS:"):
                current_section = "claims"
            elif line.startswith("METHOD:"):
                result["method"] = line.split(":", 1)[1].strip()
                current_section = "method"
            elif line.startswith("RESULTS:"):
                result["results"] = line.split(":", 1)[1].strip()
                current_section = ""
            elif line.startswith("LIMITS:"):
                current_section = "limits"
            elif line.startswith("CODE:"):
                result["code"] = "yes" in line.lower()
                current_section = ""
            elif line.startswith("CODE_URL:"):
                url = line.split(":", 1)[1].strip()
                if url and url != "none":
                    result["code_url"] = url
            elif line.startswith("- ") and current_section == "claims":
                result["claims"].append(line[2:].strip())
            elif line.startswith("- ") and current_section == "limits":
                result["limits"].append(line[2:].strip())
            elif current_section == "method" and line:
                result["method"] += " " + line

        return result

    @staticmethod
    def _parse_synthesis(raw: str) -> Dict[str, Any]:
        """Parse synthesis response."""
        result: Dict[str, Any] = {
            "ideas": [], "sketch": "", "connects": [],
        }

        lines = raw.split("\n")
        current_section = ""
        sketch_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("IDEAS:"):
                current_section = "ideas"
            elif stripped.startswith("SKETCH:"):
                current_section = "sketch"
                rest = stripped.split(":", 1)[1].strip()
                if rest:
                    sketch_lines.append(rest)
            elif stripped.startswith("CONNECTS:"):
                current_section = "connects"
                rest = stripped.split(":", 1)[1].strip()
                if rest:
                    result["connects"] = [
                        c.strip() for c in rest.split(",")
                    ]
            elif stripped.startswith("- ") and current_section == "ideas":
                result["ideas"].append(stripped[2:].strip())
            elif current_section == "sketch" and stripped:
                sketch_lines.append(line)  # preserve indentation

        result["sketch"] = "\n".join(sketch_lines)
        return result

    def _persist(self, digest: PaperDigest, prototype_code: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO digests
                    (digest_id, title, source_url, category, relevance,
                     triage_summary, key_claims_json, novel_method,
                     results_summary, limitations_json, code_available,
                     code_url, actionable_ideas_json, implementation_sketch,
                     connections_json, prototype_code, digested_at,
                     total_seconds)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    digest.digest_id, digest.title, digest.source_url,
                    digest.category, digest.relevance, digest.triage_summary,
                    json.dumps(digest.key_claims),
                    digest.novel_method, digest.results_summary,
                    json.dumps(digest.limitations),
                    1 if digest.code_available else 0,
                    digest.code_url,
                    json.dumps(digest.actionable_ideas),
                    digest.implementation_sketch,
                    json.dumps(digest.connections),
                    prototype_code,
                    digest.digested_at, digest.total_seconds,
                ))
                conn.commit()
        except Exception:
            logger.warning(
                "Failed to persist digest %s",
                digest.digest_id,
                exc_info=True,
            )

    def top_actionable(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return most actionable digested papers."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT title, category, relevance, triage_summary, "
                "novel_method, actionable_ideas_json, source_url "
                "FROM digests WHERE category != 'skip' "
                "ORDER BY relevance DESC LIMIT ?",
                (limit,),
            )
            return [
                {
                    "title": r[0], "category": r[1],
                    "relevance": r[2], "summary": r[3],
                    "method": r[4][:200],
                    "ideas": json.loads(r[5]) if r[5] else [],
                    "url": r[6],
                }
                for r in cur.fetchall()
            ]

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM digests"
            ).fetchone()[0]
            by_cat = conn.execute(
                "SELECT category, COUNT(*), AVG(relevance), AVG(total_seconds) "
                "FROM digests GROUP BY category"
            ).fetchall()
            with_code = conn.execute(
                "SELECT COUNT(*) FROM digests WHERE prototype_code IS NOT NULL "
                "AND prototype_code != ''"
            ).fetchone()[0]

        return {
            "total_digested": total,
            "prototypes_generated": with_code,
            "by_category": [
                {
                    "category": r[0], "count": r[1],
                    "avg_relevance": round(r[2], 3),
                    "avg_seconds": round(r[3], 1),
                }
                for r in by_cat
            ],
        }
