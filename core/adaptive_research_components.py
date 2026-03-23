"""
Adaptive Research -- LLM-Dependent Components
----------------------------------------------
QueryEvolver and CrossSynthesizer: components that require an LLM
runtime for query reformulation and cross-paper synthesis.
Split from adaptive_research.py for the 500 LOC-per-module rule.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from core.runtime import Runtime
from core.adaptive_research_models import QueryPerformance, SynthesisResult

log = logging.getLogger(__name__)


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
                log.debug("LLM query reformulation failed", exc_info=True)

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
            log.warning("Cross-paper synthesis failed", exc_info=True)

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
