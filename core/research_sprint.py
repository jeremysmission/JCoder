"""
Research Sprint Engine (Enhanced 7-Phase Pipeline)
---------------------------------------------------
Orchestrates a complete research sprint with PRISMA tracking,
layered triage, credibility scoring, evidence weighting,
optional bias control, structured synthesis, and reporting.

Enhanced pipeline (based on 10 papers + 10 articles synthesis):
1. DISCOVER:    Pull latest papers from configured sources
2. SCREEN:      Satellite + Drone layered triage (70% eliminated pre-LLM)
3. SCORE:       CRAAP credibility scoring per source
4. DIGEST:      Deep-read top papers via RapidDigester
5. VERIFY:      Optional claim verification + devil's advocate
6. SYNTHESIZE:  Theme x source matrix with contradiction detection
7. REPORT:      PRISMA flow + synthesis matrix + prototypes

Research methodology sources:
- PRISMA (Moher 2009): Systematic review pipeline
- CRAAP (Illinois State): Source credibility framework
- Layered Preview (MIT/Oxford): 60->22 min triage
- Hedges & Olkin (1985): Inverse-variance evidence weighting
- Glass (1976): Meta-analysis and cross-study synthesis
- Stanford SHEG: Lateral reading verification
- ATLAS.ti: Confirmation bias mitigation
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
    # Enhanced pipeline outputs
    prisma_flow: Dict[str, int] = field(default_factory=dict)
    synthesis_markdown: str = ""
    verification_count: int = 0
    balance_ratio: float = -1.0  # -1 = not run


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
    # Enhanced pipeline toggles
    prisma_enabled: bool = True
    credibility_scoring: bool = True
    devils_advocate: bool = False
    claim_verification: bool = False
    satellite_cutoff: float = 0.3
    drone_cutoff: float = 0.5
    max_deep_dive: int = 5
    max_counter_queries: int = 3
    max_verify_claims: int = 5
    synthesis_max_themes: int = 8


class ResearchSprinter:
    """
    Orchestrates focused research sprints with a 7-phase pipeline.

    Wires together: discovery -> layered triage -> credibility scoring ->
    deep digest -> verification -> synthesis -> reporting.
    """

    def __init__(
        self,
        runtime: Runtime,
        discover_fn: Optional[Callable[[str], List[Dict]]] = None,
        fetch_fn: Optional[Callable[[str], List[Dict]]] = None,
        config: Optional[SprintConfig] = None,
    ):
        """
        Args:
            runtime: LLM runtime for digestion, triage, and synthesis
            discover_fn: Function(topic) -> list of paper dicts
            fetch_fn: Function(query) -> list of paper dicts (for verification)
            config: Sprint configuration
        """
        self.runtime = runtime
        self.discover_fn = discover_fn
        self.fetch_fn = fetch_fn
        self.config = config or SprintConfig()
        self.out_dir = Path(self.config.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded components
        self._digester = None
        self._triage_engine = None
        self._scorer = None
        self._weighter = None
        self._synthesizer = None
        self._verifier = None
        self._advocate = None
        self._prisma = None

    # ------------------------------------------------------------------
    # Lazy component initialization
    # ------------------------------------------------------------------

    def _get_digester(self):
        if self._digester is None:
            from core.rapid_digest import RapidDigester
            self._digester = RapidDigester(
                self.runtime,
                db_path=str(self.out_dir / "sprint_digests.db"),
            )
        return self._digester

    def _get_triage(self):
        if self._triage_engine is None:
            from core.layered_triage import LayeredTriage
            self._triage_engine = LayeredTriage(runtime=self.runtime)
        return self._triage_engine

    def _get_scorer(self):
        if self._scorer is None:
            from core.source_scorer import SourceScorer
            self._scorer = SourceScorer(runtime=self.runtime)
        return self._scorer

    def _get_weighter(self):
        if self._weighter is None:
            from core.evidence_weighter import EvidenceWeighter
            self._weighter = EvidenceWeighter()
        return self._weighter

    def _get_synthesizer(self):
        if self._synthesizer is None:
            from core.synthesis_matrix import SynthesisMatrix
            self._synthesizer = SynthesisMatrix(runtime=self.runtime)
        return self._synthesizer

    def _get_verifier(self):
        if self._verifier is None:
            from core.claim_verifier import ClaimVerifier
            self._verifier = ClaimVerifier(
                runtime=self.runtime, fetch_fn=self.fetch_fn,
            )
        return self._verifier

    def _get_advocate(self):
        if self._advocate is None:
            from core.devils_advocate import DevilsAdvocate
            self._advocate = DevilsAdvocate(
                runtime=self.runtime, fetch_fn=self.fetch_fn,
            )
        return self._advocate

    def _get_prisma(self, sprint_dir: Path):
        if self._prisma is None:
            from core.prisma_tracker import PrismaTracker
            self._prisma = PrismaTracker(
                db_path=str(sprint_dir / "prisma.db"),
            )
        return self._prisma

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_sprint(
        self,
        topics: Optional[List[str]] = None,
        papers: Optional[List[Dict[str, str]]] = None,
    ) -> SprintResult:
        """Run the enhanced 7-phase research sprint."""
        t0 = time.time()
        sprint_id = f"sprint_{int(t0)}"
        sprint_dir = self.out_dir / sprint_id
        sprint_dir.mkdir(parents=True, exist_ok=True)

        topics = topics or self.config.focus_topics
        cfg = self.config

        # PRISMA tracker (wraps entire pipeline)
        prisma = None
        if cfg.prisma_enabled:
            prisma = self._get_prisma(sprint_dir)

        # === Phase 1: DISCOVER ===
        if papers is None:
            papers = self._discover(topics)

        # Log all discovered papers to PRISMA
        if prisma:
            from core.layered_triage import _content_hash
            for p in papers:
                prisma.identify(
                    title=p.get("title", ""),
                    source=p.get("source", "unknown"),
                    content_hash=_content_hash(p),
                )

        # === Phase 2: SCREEN (Layered Triage) ===
        triage_results = self._screen(papers, topics, prisma)
        screened_papers = self._triage_to_paper_list(triage_results, papers)

        # === Phase 3: SCORE (CRAAP Credibility) ===
        cred_scores = []
        if cfg.credibility_scoring and screened_papers:
            cred_scores = self._score_credibility(
                screened_papers, " ".join(topics),
            )
            # Log eligible papers to PRISMA
            if prisma:
                from core.layered_triage import _content_hash
                for i, p in enumerate(screened_papers):
                    score = cred_scores[i].composite if i < len(cred_scores) else 0
                    prisma.eligible(
                        _content_hash(p), passed=score >= 0.3,
                        reason=f"credibility={score:.2f}",
                    )

        # === Phase 4: DIGEST ===
        digested = self._digest_top(screened_papers, sprint_dir, prisma)

        # === Phase 5: VERIFY (optional) ===
        verification_count = 0
        balance_ratio = -1.0
        if digested:
            verification_count, balance_ratio = self._verify(
                digested, screened_papers, cfg,
            )

        # === Phase 6: SYNTHESIZE ===
        synthesis_md = ""
        if digested:
            synthesis_md = self._synthesize(
                digested, " ".join(topics), sprint_dir, cfg,
            )

        # === Phase 7: PROTOTYPE + REPORT ===
        prototype_paths = []
        if cfg.generate_prototypes:
            prototype_paths = self._generate_prototypes(digested, sprint_dir)

        report_path = self._write_report(
            sprint_id, sprint_dir, triage_results, digested,
            prototype_paths, synthesis_md, t0, prisma,
        )

        duration = time.time() - t0
        breakthroughs = sum(
            1 for d in digested if d.get("category") == "breakthrough"
        )

        # Build triaged-style list for avg_relevance from triage results
        triaged_scores = [
            r.drone_score if r.drone_pass else r.satellite_score
            for r in triage_results if r.satellite_pass
        ]

        prisma_flow = {}
        if prisma:
            prisma_flow = prisma.flow_counts()
            prisma.close()

        return SprintResult(
            sprint_id=sprint_id,
            started_at=t0,
            duration_seconds=duration,
            papers_discovered=len(papers or []),
            papers_triaged=len([r for r in triage_results if r.satellite_pass]),
            papers_digested=len(digested),
            prototypes_generated=len(prototype_paths),
            avg_relevance=(
                sum(triaged_scores) / len(triaged_scores)
                if triaged_scores else 0.0
            ),
            top_papers=[
                {"title": r.title, "relevance": str(r.drone_score)}
                for r in sorted(triage_results, key=lambda x: x.drone_score, reverse=True)[:5]
            ],
            breakthroughs_found=breakthroughs,
            digest_path=str(sprint_dir / "digests.json"),
            prototype_paths=prototype_paths,
            report_path=report_path,
            prisma_flow=prisma_flow,
            synthesis_markdown=synthesis_md,
            verification_count=verification_count,
            balance_ratio=balance_ratio,
        )

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _discover(self, topics: List[str]) -> List[Dict[str, str]]:
        """Phase 1: Discover papers across topics."""
        all_papers = []
        seen_titles = set()

        if not self.discover_fn:
            return []

        for topic in topics:
            try:
                found = self.discover_fn(topic)
                for p in found:
                    title = p.get("title", "").lower().strip()
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_papers.append(p)
            except Exception:
                pass

        return all_papers[:self.config.max_papers_to_triage]

    def _screen(self, papers, topics, prisma):
        """Phase 2: Layered triage (Satellite -> Drone -> Deep Dive)."""
        triage = self._get_triage()
        query = " ".join(topics) if isinstance(topics, list) else str(topics)
        results = triage.triage_batch(
            papers, query,
            satellite_cutoff=self.config.satellite_cutoff,
            drone_cutoff=self.config.drone_cutoff,
            max_deep_dive=self.config.max_deep_dive,
        )

        # Log screening results to PRISMA
        if prisma:
            for r in results:
                prisma.screen(
                    r.content_hash, passed=r.satellite_pass,
                    reason=f"satellite={r.satellite_score:.2f}",
                )

        return results

    def _triage_to_paper_list(self, triage_results, papers):
        """Convert triage results back to paper dicts for papers that passed."""
        from core.layered_triage import _content_hash
        hash_to_paper = {_content_hash(p): p for p in papers}
        passed_hashes = {r.content_hash for r in triage_results if r.deep_dive}
        return [hash_to_paper[h] for h in passed_hashes if h in hash_to_paper]

    def _score_credibility(self, papers, query):
        """Phase 3: CRAAP credibility scoring."""
        scorer = self._get_scorer()
        return scorer.score_batch(papers, query)

    def _digest_top(self, papers, sprint_dir, prisma):
        """Phase 4: Deep digest of top papers."""
        digester = self._get_digester()
        digested = []

        for paper in papers[:self.config.max_papers_to_digest]:
            try:
                digest = digester.digest(
                    title=paper.get("title", ""),
                    abstract=paper.get("summary", paper.get("abstract", "")),
                    source_url=paper.get("url", ""),
                    generate_prototype=False,
                )
                digested.append(asdict(digest))
                if prisma:
                    from core.layered_triage import _content_hash
                    prisma.include(
                        _content_hash(paper),
                        reason=f"digested, category={digest.category}",
                    )
            except Exception:
                pass

        digest_path = sprint_dir / "digests.json"
        digest_path.write_text(
            json.dumps(digested, indent=2, default=str),
            encoding="utf-8",
        )
        return digested

    def _verify(self, digested, papers, cfg):
        """Phase 5: Optional claim verification + devil's advocate."""
        verification_count = 0
        balance_ratio = -1.0

        # Claim verification
        if cfg.claim_verification and self.fetch_fn:
            verifier = self._get_verifier()
            claims = []
            for d in digested[:cfg.max_verify_claims]:
                for claim in d.get("key_claims", [])[:2]:
                    claims.append(claim)
            if claims:
                results = verifier.verify_batch(claims[:cfg.max_verify_claims])
                verification_count = sum(1 for r in results if r.verified)

        # Devil's advocate
        if cfg.devils_advocate:
            advocate = self._get_advocate()
            # Challenge the top finding
            top = digested[0] if digested else None
            if top and top.get("key_claims"):
                claim = top["key_claims"][0]
                report = advocate.challenge(
                    claim, papers,
                    max_counter_queries=cfg.max_counter_queries,
                )
                balance_ratio = report.balance_ratio

        return verification_count, balance_ratio

    def _synthesize(self, digested, query, sprint_dir, cfg):
        """Phase 6: Build synthesis matrix."""
        synthesizer = self._get_synthesizer()
        try:
            report = synthesizer.build(digested, query)
            md = synthesizer.to_markdown_table(report)
            # Save synthesis
            syn_path = sprint_dir / "synthesis_matrix.md"
            syn_path.write_text(md, encoding="utf-8")
            return md
        except Exception:
            return ""

    def _generate_prototypes(self, digested, sprint_dir):
        """Phase 7a: Generate code prototypes for actionable papers."""
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
        self, sprint_id, sprint_dir, triage_results, digested,
        prototype_paths, synthesis_md, start_time, prisma,
    ):
        """Phase 7b: Generate enhanced sprint report."""
        duration = time.time() - start_time
        lines = [
            f"# Research Sprint: {sprint_id}",
            f"Duration: {duration:.0f} seconds",
            "",
        ]

        # PRISMA flow diagram
        if prisma:
            lines.extend([
                "## PRISMA Pipeline",
                "```",
                prisma.flow_diagram_text(),
                "```",
                "",
            ])

        # Triage results
        passed = [r for r in triage_results if r.satellite_pass]
        lines.append(f"## Screened Papers: {len(passed)} / {len(triage_results)}")
        for r in sorted(passed, key=lambda x: x.drone_score, reverse=True)[:10]:
            summary = f" -- {r.drone_summary}" if r.drone_summary else ""
            lines.append(
                f"- ({r.drone_score:.2f}) {r.title}{summary}"
            )

        # Digested papers
        lines.extend(["", f"## Papers Digested: {len(digested)}"])
        for d in digested:
            lines.extend([
                f"### {d.get('title', 'Unknown')}",
                f"Category: {d.get('category', '?')} | "
                f"Relevance: {d.get('relevance', 0):.2f}",
                f"Method: {d.get('novel_method', 'N/A')[:200]}",
                "Ideas:",
            ])
            for idea in d.get("actionable_ideas", []):
                lines.append(f"  - {idea}")
            lines.append("")

        # Synthesis matrix
        if synthesis_md:
            lines.extend(["## Synthesis Matrix", "", synthesis_md, ""])

        # Prototypes
        lines.append(f"## Prototypes Generated: {len(prototype_paths)}")
        for p in prototype_paths:
            lines.append(f"- {p}")

        lines.extend(["", "---",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}"])

        report_path = sprint_dir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)
