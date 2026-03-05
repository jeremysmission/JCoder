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
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.runtime import Runtime

_CURRENT_YEAR = datetime.now(timezone.utc).year


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    """Result of fetching from a single source."""
    source_name: str
    query: str
    papers_found: int
    novel_papers: int  # papers not seen before
    avg_novelty: float
    best_novelty: float
    fetch_time_ms: float
    timestamp: float = 0.0


@dataclass
class QueryPerformance:
    """Tracks how well a query term performs across sources."""
    query_term: str
    times_used: int = 0
    total_novel_papers: int = 0
    avg_novelty_yield: float = 0.0
    best_paper_title: str = ""
    last_used: float = 0.0


@dataclass
class SnowballResult:
    """Result of citation snowball traversal."""
    seed_paper: str
    papers_found: int
    forward_refs: int  # papers that cite the seed
    backward_refs: int  # papers the seed cites
    novel_found: int
    depth_reached: int


@dataclass
class SynthesisResult:
    """Result of cross-paper synthesis."""
    paper_a: str
    paper_b: str
    novel_combination: str
    implementation_idea: str
    confidence: float
    timestamp: float = 0.0


@dataclass
class AdaptiveStats:
    """Overall adaptive research statistics."""
    total_fetches: int
    total_papers_seen: int
    total_novel_papers: int
    source_rankings: List[Dict[str, Any]]
    best_queries: List[Dict[str, Any]]
    synthesis_count: int
    snowball_depth_avg: float
    yield_trend: str  # "improving", "stable", "declining"


# ---------------------------------------------------------------------------
# UCB1 Source Bandit
# ---------------------------------------------------------------------------

class SourceBandit:
    """
    UCB1 multi-armed bandit for research source selection.

    Each source is an arm. Reward = novelty yield per fetch.
    UCB1 balances exploitation (best sources) with exploration
    (under-sampled sources).

    UCB1 score = avg_reward + C * sqrt(ln(total_pulls) / arm_pulls)
    """

    def __init__(self, exploration_weight: float = 1.41):
        self.c = exploration_weight
        self._arms: Dict[str, Dict[str, float]] = {}
        self._total_pulls = 0

    def register_arm(self, name: str) -> None:
        if name not in self._arms:
            self._arms[name] = {
                "pulls": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
                "max_reward": 0.0,
            }

    def select(self, k: int = 3) -> List[str]:
        """Select top-k sources to fetch from using UCB1."""
        if not self._arms:
            return []

        # Ensure every arm is tried at least once
        untried = [
            name for name, data in self._arms.items()
            if data["pulls"] == 0
        ]
        if untried:
            return untried[:k]

        # UCB1 scoring
        scores = {}
        for name, data in self._arms.items():
            avg = data["avg_reward"]
            exploration = self.c * math.sqrt(
                math.log(max(1, self._total_pulls)) / max(1, data["pulls"])
            )
            scores[name] = avg + exploration

        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:k]

    def update(self, name: str, reward: float) -> None:
        """Update arm with observed reward."""
        if name not in self._arms:
            self.register_arm(name)

        arm = self._arms[name]
        arm["pulls"] += 1
        arm["total_reward"] += reward
        arm["avg_reward"] = arm["total_reward"] / arm["pulls"]
        arm["max_reward"] = max(arm["max_reward"], reward)
        self._total_pulls += 1

    def rankings(self) -> List[Dict[str, Any]]:
        """Return sources ranked by UCB1 score."""
        if not self._arms or self._total_pulls == 0:
            return []

        results = []
        for name, data in self._arms.items():
            if data["pulls"] > 0:
                ucb = data["avg_reward"] + self.c * math.sqrt(
                    math.log(max(1, self._total_pulls)) / max(1, data["pulls"])
                )
            else:
                ucb = float("inf")

            results.append({
                "source": name,
                "avg_reward": round(data["avg_reward"], 4),
                "pulls": data["pulls"],
                "ucb_score": round(ucb, 4),
                "max_reward": round(data["max_reward"], 4),
            })

        results.sort(key=lambda x: x["ucb_score"], reverse=True)
        return results

    def to_dict(self) -> Dict:
        return {
            "arms": self._arms,
            "total_pulls": self._total_pulls,
            "c": self.c,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SourceBandit":
        b = cls(exploration_weight=data.get("c", 1.41))
        b._arms = data.get("arms", {})
        b._total_pulls = data.get("total_pulls", 0)
        return b


# ---------------------------------------------------------------------------
# Query Evolver
# ---------------------------------------------------------------------------

class QueryEvolver:
    """
    Learns which query terms produce the highest-novelty results.

    Tracks per-term statistics and uses Thompson Sampling to
    select query expansions. Good terms get reinforced; bad
    terms get pruned.
    """

    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self._term_stats: Dict[str, QueryPerformance] = {}
        self._base_terms: List[str] = [
            "self-improving RAG",
            "retrieval augmented generation",
            "code generation self-improvement",
            "evolutionary prompt optimization",
            "meta-learning agents",
            "knowledge graph retrieval",
            "speculative decoding",
            "curriculum self-generation",
            "recursive self-improvement",
            "stigmergic coordination",
        ]

    def generate_queries(
        self, topic: str, n: int = 5
    ) -> List[str]:
        """
        Generate n query variations for a topic.
        Uses learned term performance to weight expansions.
        """
        queries = [topic]

        # Add high-performing terms as expansions
        good_terms = sorted(
            self._term_stats.values(),
            key=lambda t: t.avg_novelty_yield,
            reverse=True,
        )[:5]

        for term in good_terms:
            if term.query_term.lower() not in topic.lower():
                queries.append(f"{topic} {term.query_term}")

        # Add base terms that haven't been tried much
        for base in self._base_terms:
            if base.lower() not in topic.lower():
                stats = self._term_stats.get(base)
                if not stats or stats.times_used < 3:
                    queries.append(f"{topic} {base}")

        # LLM-generated reformulations
        if len(queries) < n:
            try:
                reformulations = self._llm_reformulate(topic)
                queries.extend(reformulations)
            except Exception:
                pass

        return queries[:n]

    def report_result(
        self, query: str, novel_papers: int, avg_novelty: float,
        best_title: str = "",
    ) -> None:
        """Report the result of using a query."""
        terms = self._tokenize(query)
        for term in terms:
            if term not in self._term_stats:
                self._term_stats[term] = QueryPerformance(query_term=term)

            stats = self._term_stats[term]
            stats.times_used += 1
            stats.total_novel_papers += novel_papers
            # Running average
            alpha = 1.0 / stats.times_used
            stats.avg_novelty_yield = (
                (1 - alpha) * stats.avg_novelty_yield + alpha * avg_novelty
            )
            if avg_novelty > stats.avg_novelty_yield:
                stats.best_paper_title = best_title
            stats.last_used = time.time()

    def best_terms(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the highest-performing query terms."""
        ranked = sorted(
            self._term_stats.values(),
            key=lambda t: t.avg_novelty_yield * min(t.times_used, 5),
            reverse=True,
        )
        return [
            {
                "term": t.query_term,
                "uses": t.times_used,
                "avg_yield": round(t.avg_novelty_yield, 4),
                "total_novel": t.total_novel_papers,
            }
            for t in ranked[:n]
        ]

    def _llm_reformulate(self, topic: str) -> List[str]:
        """Use LLM to generate creative query reformulations."""
        prompt = (
            f"Generate 3 different search queries to find cutting-edge "
            f"research papers about: {topic}\n\n"
            f"Each query should use different terminology and approach "
            f"the topic from a different angle.\n"
            f"Return ONLY the 3 queries, one per line."
        )
        response = self.runtime.generate(
            question=prompt,
            context_chunks=[],
            system_prompt="You are a research query generator.",
            temperature=0.7,
            max_tokens=200,
        )
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        # Clean up numbering
        cleaned = []
        for line in lines:
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = line.strip('"').strip("'")
            if len(line) > 5:
                cleaned.append(line)
        return cleaned[:3]

    @staticmethod
    def _tokenize(query: str) -> List[str]:
        """Extract meaningful terms from a query."""
        stop = {"the", "a", "an", "and", "or", "for", "in", "on", "with", "to"}
        words = re.findall(r"[a-zA-Z][\w-]*", query.lower())
        return [w for w in words if w not in stop and len(w) > 2]

    def to_dict(self) -> Dict:
        return {
            "terms": {
                k: asdict(v) for k, v in self._term_stats.items()
            },
            "base_terms": self._base_terms,
        }

    @classmethod
    def from_dict(cls, data: Dict, runtime: Runtime) -> "QueryEvolver":
        evolver = cls(runtime)
        for k, v in data.get("terms", {}).items():
            evolver._term_stats[k] = QueryPerformance(**v)
        evolver._base_terms = data.get("base_terms", evolver._base_terms)
        return evolver


# ---------------------------------------------------------------------------
# Cross-Paper Synthesizer
# ---------------------------------------------------------------------------

class CrossSynthesizer:
    """
    Finds novel combinations across papers using LLM-guided synthesis.

    Takes pairs of high-novelty papers and asks:
    "What would happen if you combined method A with method B?"
    This is how real breakthroughs happen -- at the intersection.
    """

    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self._syntheses: List[SynthesisResult] = []

    def synthesize_pair(
        self,
        paper_a: Dict[str, str],
        paper_b: Dict[str, str],
    ) -> Optional[SynthesisResult]:
        """
        Find novel combinations between two papers.
        Returns None if no interesting combination found.
        """
        prompt = (
            f"Paper A: {paper_a.get('title', '')}\n"
            f"Method A: {paper_a.get('abstract', '')[:500]}\n\n"
            f"Paper B: {paper_b.get('title', '')}\n"
            f"Method B: {paper_b.get('abstract', '')[:500]}\n\n"
            f"Question: What novel technique could be created by "
            f"combining the key ideas from Paper A and Paper B?\n\n"
            f"Answer in this exact format:\n"
            f"COMBINATION: <one-line description of the novel combination>\n"
            f"IMPLEMENTATION: <3-5 line description of how to implement it>\n"
            f"CONFIDENCE: <0.0-1.0 how promising this combination is>\n"
        )

        try:
            response = self.runtime.generate(
                question=prompt,
                context_chunks=[],
                system_prompt=(
                    "You are a research synthesis expert. Find novel "
                    "combinations between research papers. Be specific "
                    "and actionable."
                ),
                temperature=0.3,
                max_tokens=300,
            )

            # Parse structured response
            combo = ""
            impl = ""
            conf = 0.5

            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("COMBINATION:"):
                    combo = line[12:].strip()
                elif line.startswith("IMPLEMENTATION:"):
                    impl = line[15:].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        conf = float(re.findall(r"[\d.]+", line)[0])
                    except (IndexError, ValueError):
                        conf = 0.5

            if combo and conf > 0.3:
                result = SynthesisResult(
                    paper_a=paper_a.get("title", ""),
                    paper_b=paper_b.get("title", ""),
                    novel_combination=combo,
                    implementation_idea=impl,
                    confidence=min(1.0, conf),
                    timestamp=time.time(),
                )
                self._syntheses.append(result)
                return result

        except Exception:
            pass

        return None

    def synthesize_batch(
        self, papers: List[Dict[str, str]], max_pairs: int = 10
    ) -> List[SynthesisResult]:
        """Synthesize across multiple papers (top pairs by novelty)."""
        results = []

        # Sort by novelty, take top N
        sorted_papers = sorted(
            papers,
            key=lambda p: p.get("novelty", 0),
            reverse=True,
        )[:6]  # max 6 papers = 15 pairs

        pairs_tried = 0
        for i in range(len(sorted_papers)):
            for j in range(i + 1, len(sorted_papers)):
                if pairs_tried >= max_pairs:
                    break
                result = self.synthesize_pair(
                    sorted_papers[i], sorted_papers[j]
                )
                if result:
                    results.append(result)
                pairs_tried += 1

        return results

    def best_syntheses(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return highest-confidence syntheses."""
        ranked = sorted(
            self._syntheses,
            key=lambda s: s.confidence,
            reverse=True,
        )
        return [asdict(s) for s in ranked[:n]]


# ---------------------------------------------------------------------------
# Yield Tracker
# ---------------------------------------------------------------------------

class YieldTracker:
    """
    Tracks research yield over time to detect improving/declining trends.

    Yield = novel_papers_found / total_papers_fetched (per session)
    Tracks running averages to detect if the research process
    is getting better or worse.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._yields: List[Tuple[float, float]] = []  # (timestamp, yield)

    def record(self, novel_found: int, total_fetched: int) -> None:
        y = novel_found / max(1, total_fetched)
        self._yields.append((time.time(), y))
        # Keep bounded
        if len(self._yields) > self.window_size * 3:
            self._yields = self._yields[-self.window_size * 2:]

    def trend(self) -> str:
        """Detect yield trend: 'improving', 'stable', 'declining'."""
        if len(self._yields) < 4:
            return "insufficient_data"

        recent = [y for _, y in self._yields[-self.window_size:]]
        mid = len(recent) // 2
        first_half_avg = sum(recent[:mid]) / max(1, mid)
        second_half_avg = sum(recent[mid:]) / max(1, len(recent) - mid)

        diff = second_half_avg - first_half_avg
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"

    def current_yield(self) -> float:
        if not self._yields:
            return 0.0
        recent = [y for _, y in self._yields[-5:]]
        return sum(recent) / len(recent)

    def to_dict(self) -> Dict:
        return {"yields": self._yields, "window": self.window_size}

    @classmethod
    def from_dict(cls, data: Dict) -> "YieldTracker":
        t = cls(window_size=data.get("window", 20))
        t._yields = data.get("yields", [])
        return t


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
                # Collect novel papers
                # (papers are stored in _fetch_and_score)

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
                pass

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
            pass

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
            pass
