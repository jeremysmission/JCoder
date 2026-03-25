"""
Retrieval fusion methods for combining dense and sparse search results.
----------------------------------------------------------------------
Extracted from index_engine.py to maintain the 500-line module limit.

Two methods available:
  - RRF (Reciprocal Rank Fusion): rank-based, ignores score magnitudes.
    Best for zero-shot/no-training-data scenarios.
  - DBSF (Distribution-Based Score Fusion): normalizes scores using
    mean +/- 3*std, uses actual score values. Better when score
    magnitudes are meaningful.

Both return a dict mapping chunk index -> fused score.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

# Optional FlashRank reranker (lightweight, no server required)
try:
    from flashrank import Ranker, RerankRequest

    _FLASHRANK_RANKER = None  # lazy init

    def _get_flashrank():
        global _FLASHRANK_RANKER
        if _FLASHRANK_RANKER is None:
            _FLASHRANK_RANKER = Ranker()
            log.info("FlashRank reranker loaded")
        return _FLASHRANK_RANKER

    HAS_FLASHRANK = True
except ImportError:
    HAS_FLASHRANK = False
    Ranker = None
    RerankRequest = None

    def _get_flashrank():
        return None


def rrf_fusion(
    vector_results: List[Tuple[int, float]],
    keyword_results: List[Tuple[int, float]],
    rrf_k: int = 60,
) -> Dict[int, float]:
    """Reciprocal Rank Fusion: rank-based, ignores score magnitudes."""
    scores: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(vector_results):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (idx, _) in enumerate(keyword_results):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


def dbsf_fusion(
    vector_results: List[Tuple[int, float]],
    keyword_results: List[Tuple[int, float]],
) -> Dict[int, float]:
    """Distribution-Based Score Fusion: normalizes using mean +/- 3*std.

    Unlike RRF, DBSF uses actual score magnitudes. Each retriever's
    scores are normalized to [0, 1] using the distribution within that
    retriever's results (mean +/- 3 standard deviations as bounds).
    Normalized scores are then summed across retrievers.

    Ref: LlamaIndex QueryFusionRetriever, Qdrant hybrid queries.
    """

    def _normalize(results: List[Tuple[int, float]]) -> Dict[int, float]:
        if not results:
            return {}
        scores = [s for _, s in results]
        mean = sum(scores) / len(scores)
        if len(scores) > 1:
            var = sum((s - mean) ** 2 for s in scores) / len(scores)
            std = var ** 0.5
        else:
            std = 0.0
        lo = mean - 3 * std
        span = 6 * std if std > 0 else 1.0
        return {idx: max(0.0, min(1.0, (s - lo) / span)) for idx, s in results}

    v_norm = _normalize(vector_results)
    k_norm = _normalize(keyword_results)

    fused: Dict[int, float] = {}
    for idx, score in v_norm.items():
        fused[idx] = fused.get(idx, 0.0) + score
    for idx, score in k_norm.items():
        fused[idx] = fused.get(idx, 0.0) + score
    return fused


def rerank_candidates(
    query_text: str,
    candidates: List[Tuple[float, dict]],
    k: int,
) -> List[Tuple[float, dict]]:
    """Rerank candidates using FlashRank if available.

    Returns top-k candidates reranked by cross-encoder score.
    Falls back to original order if FlashRank unavailable.
    """
    if not HAS_FLASHRANK or len(candidates) <= 1:
        return candidates[:k]

    try:
        ranker = _get_flashrank()
        passages = [
            {"id": i, "text": m.get("text", m.get("search_content", ""))[:1000]}
            for i, (_, m) in enumerate(candidates)
        ]
        rerank_req = RerankRequest(query=query_text, passages=passages)
        reranked = ranker.rerank(rerank_req)
        results = []
        for item in reranked[:k]:
            orig_idx = int(item["id"])
            results.append((float(item["score"]), candidates[orig_idx][1]))
        return results
    except Exception as exc:
        log.warning("FlashRank reranking failed, using original order: %s", exc)
        return candidates[:k]
