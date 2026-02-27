"""
Corrective Retrieval (CRAG Pattern)
------------------------------------
Wraps the standard retrieval pipeline with self-assessment and
corrective strategies. Based on the CRAG (Corrective RAG) architecture.

Flow:
1. Standard retrieval
2. Reflection: score relevance of retrieved chunks
3. If confidence HIGH (>= threshold) -> use as-is
4. If confidence LOW (< threshold) -> trigger corrective actions:
   a. Query reformulation (rephrase + retry)
   b. Decomposition (split complex query into sub-queries)
   c. Keyword fallback (strip to key terms, search again)
5. Merge and deduplicate results
6. Log telemetry for feedback loop

This is the self-healing layer that catches retrieval failures
before they reach the LLM. No external dependencies needed.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Dict, List, Optional, Tuple

from core.retrieval_engine import RetrievalEngine
from core.reflection import ReflectionEngine


class CorrectiveRetriever:
    """
    Wraps RetrievalEngine with confidence-gated corrective strategies.
    Falls through to standard retrieval when reflection is unavailable.
    """

    def __init__(
        self,
        retriever: RetrievalEngine,
        reflection: Optional[ReflectionEngine] = None,
        confidence_threshold: float = 0.5,
        max_reformulations: int = 2,
    ):
        self.retriever = retriever
        self.reflection = reflection
        self.confidence_threshold = confidence_threshold
        self.max_reformulations = max_reformulations

    def retrieve(self, query: str) -> Tuple[List[Dict], Dict]:
        """
        Corrective retrieval pipeline.

        Returns:
            (chunks, metadata) where metadata contains:
            - strategy: which strategy produced the final result
            - confidence: reflection score (if available)
            - attempts: number of retrieval attempts
            - reformulations: list of reformulated queries tried
        """
        meta = {
            "strategy": "standard",
            "confidence": 1.0,
            "attempts": 1,
            "reformulations": [],
        }

        # Step 1: Standard retrieval
        chunks = self.retriever.retrieve(query)
        if not chunks:
            # Nothing found -- try corrective immediately
            return self._corrective_fallback(query, meta)

        # Step 2: Confidence check (skip if no reflection engine)
        if not self.reflection:
            return chunks, meta

        rel_score = self.reflection.score_relevance(query, chunks)
        meta["confidence"] = rel_score

        if rel_score >= self.confidence_threshold:
            meta["strategy"] = "standard_confident"
            return chunks, meta

        # Step 3: Corrective strategies
        return self._corrective_fallback(query, meta, initial_chunks=chunks)

    def _corrective_fallback(self, query: str, meta: Dict,
                              initial_chunks: Optional[List[Dict]] = None
                              ) -> Tuple[List[Dict], Dict]:
        """Try reformulation strategies to improve retrieval."""
        all_chunks = list(initial_chunks or [])
        seen_ids = {c.get("id", "") for c in all_chunks}

        strategies = [
            ("keyword_extract", self._keyword_query),
            ("decompose", self._decompose_query),
        ]

        for name, reformulate_fn in strategies:
            if meta["attempts"] > self.max_reformulations + 1:
                break

            alt_queries = reformulate_fn(query)
            for alt_q in alt_queries:
                meta["attempts"] += 1
                meta["reformulations"].append(alt_q)

                new_chunks = self.retriever.retrieve(alt_q)
                for c in new_chunks:
                    cid = c.get("id", "")
                    if cid and cid not in seen_ids:
                        all_chunks.append(c)
                        seen_ids.add(cid)

        if len(all_chunks) > len(initial_chunks or []):
            meta["strategy"] = "corrective_merged"
        else:
            meta["strategy"] = "corrective_no_improvement"

        # Re-check confidence on merged set
        if self.reflection and all_chunks:
            meta["confidence"] = self.reflection.score_relevance(
                query, all_chunks[:10])

        return all_chunks, meta

    @staticmethod
    def _keyword_query(query: str) -> List[str]:
        """
        Extract key programming terms from the query.
        Strips filler words, keeps identifiers and technical terms.
        """
        stop = {
            "what", "how", "does", "the", "this", "that", "is", "are",
            "was", "were", "do", "can", "could", "would", "should",
            "where", "when", "why", "which", "who", "a", "an", "in",
            "on", "at", "to", "for", "of", "with", "from", "by", "and",
            "or", "not", "it", "its", "my", "your", "me", "i", "we",
            "they", "he", "she", "about", "into", "be", "been", "being",
            "have", "has", "had", "will", "shall", "may", "might",
        }
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*", query)
        keywords = [w for w in words if w.lower() not in stop and len(w) > 1]
        if not keywords:
            return []
        return [" ".join(keywords)]

    @staticmethod
    def _decompose_query(query: str) -> List[str]:
        """
        Split a complex query into sub-queries at conjunctions and commas.
        For multi-part questions like "How does X work and where is Y defined?"
        """
        parts = re.split(r"\band\b|\bor\b|,|\?", query, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]
        # Only decompose if we got multiple meaningful parts
        if len(parts) >= 2:
            return parts[:3]
        return []
