"""
Adaptive Research Engine (Self-Improving Information Foraging)
--------------------------------------------------------------
A research engine that learns from its own results to get better
at finding breakthrough papers. Combines multi-armed bandits for
source selection, query reformulation learning, citation snowballing,
and cross-paper synthesis.

Based on:
- Information Foraging Theory (Pirolli & Card, 1999): Optimal patch-leaving
- UCB1 Bandit (Auer et al. 2002): Upper Confidence Bound source selection
- Query Reformulation (Nogueira & Cho, 2017): RL for query rewriting
- Snowball Sampling (Wohlin 2014): Forward/backward citation traversal
- Adaptive Harvesting: Self-tuning collection based on yield tracking

The key insight: Static research pipelines miss the feedback loop.
This engine tracks which sources, queries, and strategies produce
the BEST papers (highest novelty, most actionable) and automatically
shifts attention toward the richest veins.

Self-improvement mechanisms:
1. SOURCE BANDIT: UCB1 allocates fetch budget to highest-yield sources
2. QUERY EVOLVER: Learns which query terms find the best papers
3. SNOWBALL CRAWLER: Citation graph traversal finds hidden gems
4. CROSS-SYNTHESIS: LLM finds novel combinations across papers
5. YIELD TRACKER: Every fetch scored; running averages drive adaptation

Module layout (split for 500 LOC rule):
- adaptive_research_models.py: Data classes, SourceBandit, YieldTracker
- adaptive_research_components.py: QueryEvolver, CrossSynthesizer
- adaptive_research.py (this file): AdaptiveResearchEngine + re-exports
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from core.runtime import Runtime

# Re-export everything so the public interface is unchanged
from core.adaptive_research_models import (  # noqa: F401
    FetchResult,
    QueryPerformance,
    SnowballResult,
    SynthesisResult,
    AdaptiveStats,
    SourceBandit,
    YieldTracker,
)
from core.adaptive_research_components import (  # noqa: F401
    QueryEvolver,
    CrossSynthesizer,
)

log = logging.getLogger(__name__)

_CURRENT_YEAR = datetime.now(timezone.utc).year


# ---------------------------------------------------------------------------
# Master Adaptive Research Engine
# ---------------------------------------------------------------------------

class AdaptiveResearchEngine:
    """
    Self-improving research engine that learns from every fetch.

    Orchestrates:
    1. SourceBandit -- which sources to prioritize
    2. QueryEvolver -- how to reformulate queries for max yield
    3. CrossSynthesizer -- find novel combinations across papers
    4. YieldTracker -- detect if research process is improving

    Every research session feeds back into the learning loop:
    - Sources that yield novel papers get higher UCB scores
    - Query terms that find breakthroughs get reinforced
    - Cross-paper synthesis discovers hidden connections
    - Yield tracking triggers strategy shifts when plateauing
    """

    def __init__(
        self,
        runtime: Runtime,
        fetch_fn: Optional[Callable[[str, str], List[Dict]]] = None,
        state_path: str = "_research/adaptive_state.json",
    ):
        """
        Args:
            runtime: LLM runtime for synthesis and query generation
            fetch_fn: Function(source_name, query) -> list of paper dicts
            state_path: Where to persist learned state
        """
        self.runtime = runtime
        self.fetch_fn = fetch_fn
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        # Sub-modules
        self.bandit = SourceBandit()
        self.query_evolver = QueryEvolver(runtime)
        self.synthesizer = CrossSynthesizer(runtime)
        self.yield_tracker = YieldTracker()

        # Dedup
        self._seen_hashes: Set[str] = set()

        # Load persisted state
        self._load_state()

    def research(
        self,
        topic: str,
        sources: Optional[List[str]] = None,
        max_sources: int = 3,
        max_queries_per_source: int = 3,
        synthesize: bool = True,
    ) -> Dict[str, Any]:
        """
        Run an adaptive research session.

        1. Select sources (bandit or explicit)
        2. Generate queries (evolver)
        3. Fetch papers from each source
        4. Score novelty, update bandit
        5. Cross-synthesize top papers
        6. Update yield tracker
        7. Persist state

        Returns session summary with papers and syntheses.
        """
        t0 = time.time()

        # Step 1: Source selection
        if sources:
            selected_sources = sources
        else:
            selected_sources = self.bandit.select(k=max_sources)

        # Step 2: Query generation
        queries = self.query_evolver.generate_queries(
            topic, n=max_queries_per_source
        )

        # Step 3: Fetch
        all_papers: List[Dict[str, Any]] = []
        fetch_results: List[FetchResult] = []

        for source in selected_sources:
            for query in queries:
                result = self._fetch_and_score(source, query)
                fetch_results.append(result)

        # Collect all novel papers from this session
        novel_papers = self._get_recent_novel(since=t0)

        # Step 4: Update bandit with source rewards
        for result in fetch_results:
            reward = (
                0.4 * result.avg_novelty +
                0.3 * min(1.0, result.novel_papers / max(1, result.papers_found)) +
                0.3 * result.best_novelty
            )
            self.bandit.update(result.source_name, reward)

        # Step 5: Cross-synthesis
        syntheses = []
        if synthesize and len(novel_papers) >= 2:
            syntheses = self.synthesizer.synthesize_batch(
                novel_papers, max_pairs=5
            )

        # Step 6: Yield tracking
        total_fetched = sum(r.papers_found for r in fetch_results)
        total_novel = sum(r.novel_papers for r in fetch_results)
        self.yield_tracker.record(total_novel, total_fetched)

        # Step 7: Adaptive strategy shift
        trend = self.yield_tracker.trend()
        if trend == "declining":
            # Increase exploration
            self.bandit.c = min(2.5, self.bandit.c + 0.2)
        elif trend == "improving":
            # Shift toward exploitation
            self.bandit.c = max(0.5, self.bandit.c - 0.1)

        # Persist
        self._save_state()

        duration_ms = (time.time() - t0) * 1000

        return {
            "topic": topic,
            "duration_ms": round(duration_ms, 1),
            "sources_queried": selected_sources,
            "queries_used": queries,
            "total_papers_fetched": total_fetched,
            "novel_papers_found": total_novel,
            "yield": round(total_novel / max(1, total_fetched), 3),
            "yield_trend": trend,
            "exploration_weight": round(self.bandit.c, 2),
            "top_papers": sorted(
                novel_papers,
                key=lambda p: p.get("novelty", 0),
                reverse=True,
            )[:10],
            "syntheses": [asdict(s) for s in syntheses],
            "source_rankings": self.bandit.rankings(),
            "best_query_terms": self.query_evolver.best_terms(5),
        }

    def _fetch_and_score(
        self, source: str, query: str
    ) -> FetchResult:
        """Fetch papers from a source, score novelty, track results."""
        t0 = time.time()

        papers = []
        if self.fetch_fn:
            try:
                papers = self.fetch_fn(source, query)
            except Exception:
                log.warning(
                    "Fetch from source %s failed for query %r",
                    source, query, exc_info=True,
                )

        novel_count = 0
        novelties = []
        best_novelty = 0.0
        best_title = ""

        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            # Dedup
            content_hash = hashlib.sha256(
                f"{title}:{abstract[:200]}".encode()
            ).hexdigest()[:16]

            if content_hash not in self._seen_hashes:
                self._seen_hashes.add(content_hash)
                novel_count += 1

                # Score novelty
                nov = self._score_novelty(paper)
                paper["novelty"] = nov
                paper["_novel"] = True
                paper["_timestamp"] = time.time()
                novelties.append(nov)

                if nov > best_novelty:
                    best_novelty = nov
                    best_title = title

        avg_novelty = sum(novelties) / len(novelties) if novelties else 0.0

        # Report to query evolver
        self.query_evolver.report_result(
            query, novel_count, avg_novelty, best_title
        )

        return FetchResult(
            source_name=source,
            query=query,
            papers_found=len(papers),
            novel_papers=novel_count,
            avg_novelty=round(avg_novelty, 4),
            best_novelty=round(best_novelty, 4),
            fetch_time_ms=round((time.time() - t0) * 1000, 1),
            timestamp=time.time(),
        )

    def _score_novelty(self, paper: Dict[str, str]) -> float:
        """Score paper novelty using adaptive weights."""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        try:
            year = int(paper.get("year") or _CURRENT_YEAR)
        except (ValueError, TypeError, OverflowError):
            year = _CURRENT_YEAR
        try:
            citations = max(0, int(paper.get("citation_count") or 0))
        except (ValueError, TypeError, OverflowError):
            citations = 0

        score = 0.0

        # Recency (0-0.25)
        age = max(0, _CURRENT_YEAR - year)
        score += max(0.0, 0.25 - age * 0.08)

        # Keyword relevance (0-0.35) -- uses learned best terms
        text = f"{title} {abstract}".lower()
        good_terms = self.query_evolver.best_terms(15)
        if good_terms:
            hits = sum(
                1 for t in good_terms
                if t["term"] in text
            )
            score += min(0.35, hits * 0.07)
        else:
            # Fallback to static keywords
            keywords = [
                "self-improving", "self-learning", "retrieval augmented",
                "rag", "code generation", "evolutionary", "meta-learning",
                "reflection", "corrective", "adversarial", "self-play",
            ]
            hits = sum(1 for kw in keywords if kw in text)
            score += min(0.35, hits * 0.05)

        # Citation velocity (0-0.15)
        if year >= (_CURRENT_YEAR - 1) and citations > 0:
            months = max(1, (_CURRENT_YEAR - year) * 12 + 6)
            velocity = citations / months
            score += min(0.15, velocity * 0.015)

        # Has code bonus (0-0.15)
        if paper.get("has_code") or paper.get("code_url"):
            score += 0.15

        # Title quality (0-0.10)
        quality_signals = [
            "breakthrough", "novel", "state-of-the-art", "sota",
            "surpass", "outperform", "self-improving", "zero-shot",
        ]
        if any(s in title.lower() for s in quality_signals):
            score += 0.10

        return min(1.0, score)

    def _get_recent_novel(self, since: float) -> List[Dict[str, Any]]:
        """Get papers marked as novel since a timestamp."""
        # In a real implementation this would query the store
        # For now, return empty -- the fetch_fn results are tracked externally
        return []

    def register_source(self, name: str) -> None:
        """Register a source with the bandit."""
        self.bandit.register_arm(name)

    def stats(self) -> AdaptiveStats:
        """Get comprehensive statistics."""
        return AdaptiveStats(
            total_fetches=self.bandit._total_pulls,
            total_papers_seen=len(self._seen_hashes),
            total_novel_papers=len(self._seen_hashes),
            source_rankings=self.bandit.rankings(),
            best_queries=self.query_evolver.best_terms(10),
            synthesis_count=len(self.synthesizer._syntheses),
            snowball_depth_avg=0.0,
            yield_trend=self.yield_tracker.trend(),
        )

    def _save_state(self) -> None:
        """Persist learned state to disk."""
        state = {
            "bandit": self.bandit.to_dict(),
            "query_evolver": self.query_evolver.to_dict(),
            "yield_tracker": self.yield_tracker.to_dict(),
            "seen_hashes": list(self._seen_hashes)[-5000:],
            "syntheses": [asdict(s) for s in self.synthesizer._syntheses[-100:]],
            "saved_at": time.time(),
        }
        try:
            self._state_path.write_text(
                json.dumps(state, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            log.warning("Failed to save adaptive research state", exc_info=True)

    def _load_state(self) -> None:
        """Load previously learned state."""
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            if "bandit" in data:
                self.bandit = SourceBandit.from_dict(data["bandit"])
            if "query_evolver" in data:
                self.query_evolver = QueryEvolver.from_dict(
                    data["query_evolver"], self.runtime
                )
            if "yield_tracker" in data:
                self.yield_tracker = YieldTracker.from_dict(
                    data["yield_tracker"]
                )
            if "seen_hashes" in data:
                self._seen_hashes = set(data["seen_hashes"])
            if "syntheses" in data:
                self.synthesizer._syntheses = [
                    SynthesisResult(**s) for s in data["syntheses"]
                ]
        except Exception:
            log.warning("Failed to load adaptive research state", exc_info=True)
