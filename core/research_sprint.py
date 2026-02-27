"""
Research Sprint Engine (Focused Burst + Immediate Application)
-----------------------------------------------------------------
Orchestrates a complete research sprint: discover -> triage ->
digest -> prototype -> integrate, all in one automated session.

Based on:
- Research Sprints (Agile Research, 2024): Time-boxed research bursts
- OODA Loop (Boyd): Observe-Orient-Decide-Act for research
- Just-In-Time Learning: Learn what you need exactly when you need it
- Implementation-First Reading: Skip theory, go straight to code

A research sprint is a focused 30-minute session where:
1. DISCOVER: Pull latest papers from configured sources (2 min)
2. TRIAGE: LLM scores and filters for relevance (3 min)
3. DIGEST: Deep-read the top 3-5 papers (10 min)
4. PROTOTYPE: Generate code stubs for best ideas (10 min)
5. INTEGRATE: Wire prototypes into the existing codebase (5 min)

The sprint produces:
- Digest summaries stored in SQLite
- Working code prototypes saved to _prototypes/
- An integration report with next steps
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.runtime import Runtime


@dataclass
class SprintResult:
    """Result of a complete research sprint."""
    sprint_id: str
    started_at: float
    duration_seconds: float
    # Phase results
    papers_discovered: int
    papers_triaged: int
    papers_digested: int
    prototypes_generated: int
    # Quality metrics
    avg_relevance: float
    top_papers: List[Dict[str, str]]
    breakthroughs_found: int
    # Output paths
    digest_path: str = ""
    prototype_paths: List[str] = field(default_factory=list)
    report_path: str = ""


@dataclass
class SprintConfig:
    """Configuration for a research sprint."""
    focus_topics: List[str] = field(default_factory=lambda: [
        "self-learning RAG",
        "retrieval augmented generation",
        "code generation self-improvement",
        "evolutionary prompt optimization",
    ])
    max_papers_to_triage: int = 30
    max_papers_to_digest: int = 5
    min_relevance: float = 0.4
    generate_prototypes: bool = True
    output_dir: str = "_sprints"


class ResearchSprinter:
    """
    Orchestrates focused research sprints.

    Wires together: source discovery -> triage -> digest -> prototype
    into a single automated pipeline.
    """

    def __init__(
        self,
        runtime: Runtime,
        discover_fn: Optional[Callable[[str], List[Dict]]] = None,
        config: Optional[SprintConfig] = None,
    ):
        """
        Args:
            runtime: LLM runtime for digestion and prototyping
            discover_fn: Function(topic) -> list of papers
                         Each paper: {"title": ..., "abstract": ..., "url": ...}
            config: Sprint configuration
        """
        self.runtime = runtime
        self.discover_fn = discover_fn
        self.config = config or SprintConfig()
        self.out_dir = Path(self.config.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Lazy imports to avoid circular deps
        self._digester = None

    def _get_digester(self):
        if self._digester is None:
            from core.rapid_digest import RapidDigester
            self._digester = RapidDigester(
                self.runtime,
                db_path=str(self.out_dir / "sprint_digests.db"),
            )
        return self._digester

    def run_sprint(
        self,
        topics: Optional[List[str]] = None,
        papers: Optional[List[Dict[str, str]]] = None,
    ) -> SprintResult:
        """
        Run a complete research sprint.

        Args:
            topics: Override focus topics
            papers: Pre-fetched papers (skip discovery phase)
        """
        t0 = time.time()
        sprint_id = f"sprint_{int(t0)}"
        sprint_dir = self.out_dir / sprint_id
        sprint_dir.mkdir(parents=True, exist_ok=True)

        topics = topics or self.config.focus_topics

        # === Phase 1: DISCOVER (2 min) ===
        if papers is None:
            papers = self._discover(topics)

        # === Phase 2: TRIAGE (3 min) ===
        triaged = self._triage(papers)

        # === Phase 3: DIGEST (10 min) ===
        digested = self._digest_top(triaged, sprint_dir)

        # === Phase 4: PROTOTYPE (10 min) ===
        prototype_paths = []
        if self.config.generate_prototypes:
            prototype_paths = self._generate_prototypes(digested, sprint_dir)

        # === Phase 5: REPORT ===
        report_path = self._write_report(
            sprint_id, sprint_dir, triaged, digested, prototype_paths, t0
        )

        duration = time.time() - t0
        breakthroughs = sum(
            1 for d in digested if d.get("category") == "breakthrough"
        )

        return SprintResult(
            sprint_id=sprint_id,
            started_at=t0,
            duration_seconds=duration,
            papers_discovered=len(papers or []),
            papers_triaged=len(triaged),
            papers_digested=len(digested),
            prototypes_generated=len(prototype_paths),
            avg_relevance=(
                sum(t.get("relevance", 0) for t in triaged) / len(triaged)
                if triaged else 0.0
            ),
            top_papers=[
                {"title": t.get("title", ""), "relevance": t.get("relevance", 0)}
                for t in triaged[:5]
            ],
            breakthroughs_found=breakthroughs,
            digest_path=str(sprint_dir / "digests.json"),
            prototype_paths=prototype_paths,
            report_path=report_path,
        )

    def _discover(self, topics: List[str]) -> List[Dict[str, str]]:
        """Phase 1: Discover papers across topics."""
        all_papers = []
        seen_titles = set()

        if not self.discover_fn:
            return []

        for topic in topics:
            try:
                papers = self.discover_fn(topic)
                for p in papers:
                    title = p.get("title", "").lower().strip()
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_papers.append(p)
            except Exception:
                pass

        return all_papers[:self.config.max_papers_to_triage]

    def _triage(self, papers: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Phase 2: Quick triage to filter relevant papers."""
        digester = self._get_digester()
        triaged = digester.batch_triage(papers, self.config.min_relevance)
        return triaged

    def _digest_top(
        self,
        triaged: List[Dict[str, Any]],
        sprint_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Phase 3: Deep digest of top papers."""
        digester = self._get_digester()
        digested = []

        for paper in triaged[:self.config.max_papers_to_digest]:
            try:
                digest = digester.digest(
                    title=paper.get("title", ""),
                    abstract=paper.get("summary", ""),
                    source_url=paper.get("url", ""),
                    generate_prototype=False,  # separate phase
                )
                digested.append(asdict(digest))
            except Exception:
                pass

        # Save digests
        digest_path = sprint_dir / "digests.json"
        digest_path.write_text(
            json.dumps(digested, indent=2, default=str),
            encoding="utf-8",
        )

        return digested

    def _generate_prototypes(
        self,
        digested: List[Dict[str, Any]],
        sprint_dir: Path,
    ) -> List[str]:
        """Phase 4: Generate code prototypes for actionable papers."""
        digester = self._get_digester()
        prototype_paths = []
        proto_dir = sprint_dir / "prototypes"
        proto_dir.mkdir(exist_ok=True)

        for d in digested:
            if not d.get("novel_method"):
                continue
            if d.get("relevance", 0) < 0.5:
                continue

            try:
                proto = digester.generate_prototype(
                    title=d.get("title", ""),
                    method=d.get("novel_method", ""),
                    sketch=d.get("implementation_sketch", ""),
                )
                if proto and proto.code:
                    # Save prototype
                    safe_name = "".join(
                        c if c.isalnum() or c == "_" else "_"
                        for c in proto.class_name.lower()
                    )
                    path = proto_dir / f"{safe_name}.py"
                    path.write_text(proto.code, encoding="utf-8")
                    prototype_paths.append(str(path))
            except Exception:
                pass

        return prototype_paths

    def _write_report(
        self,
        sprint_id: str,
        sprint_dir: Path,
        triaged: List[Dict],
        digested: List[Dict],
        prototype_paths: List[str],
        start_time: float,
    ) -> str:
        """Phase 5: Generate sprint report."""
        duration = time.time() - start_time
        report_lines = [
            f"# Research Sprint: {sprint_id}",
            f"Duration: {duration:.0f} seconds",
            "",
            f"## Papers Triaged: {len(triaged)}",
        ]

        for t in triaged[:10]:
            report_lines.append(
                f"- [{t.get('category', '?')}] "
                f"({t.get('relevance', 0):.2f}) "
                f"{t.get('title', 'Unknown')}"
            )

        report_lines.extend(["", f"## Papers Digested: {len(digested)}"])
        for d in digested:
            report_lines.extend([
                f"### {d.get('title', 'Unknown')}",
                f"Category: {d.get('category', '?')} | "
                f"Relevance: {d.get('relevance', 0):.2f}",
                f"Method: {d.get('novel_method', 'N/A')[:200]}",
                "Ideas:",
            ])
            for idea in d.get("actionable_ideas", []):
                report_lines.append(f"  - {idea}")
            report_lines.append("")

        report_lines.extend([
            f"## Prototypes Generated: {len(prototype_paths)}",
        ])
        for p in prototype_paths:
            report_lines.append(f"- {p}")

        report_lines.extend(["", "---",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}"])

        report_path = sprint_dir / "report.md"
        report_path.write_text(
            "\n".join(report_lines), encoding="utf-8"
        )
        return str(report_path)
