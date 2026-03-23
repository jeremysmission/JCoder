"""
Retrieval Quality Evaluation (R14)
-----------------------------------
Measures Recall@5, MRR@10, and symbol hit rate against the golden
question set using the FTS5-based retrieval pipeline.

No LLM/Ollama needed -- this evaluates retrieval quality only.

Usage:
    cd C:\\Users\\jerem\\JCoder
    python scripts/run_retrieval_eval.py [--top-k 10] [--min-quality 0]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.index_engine import IndexEngine
from core.config import StorageConfig


def load_golden_set(path: str = "") -> List[Dict]:
    if not path:
        path = os.path.join(_ROOT, "evaluation", "golden_questions_v1.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return re.sub(r"[_\-./\\:]", " ", text).lower()


def evaluate_query(
    index: IndexEngine,
    item: Dict,
    top_k: int = 10,
    min_quality: int = 0,
) -> Dict:
    """Evaluate a single golden question against the index.

    Returns dict with: question, file_hit, symbol_hits, rank, mrr.
    """
    question = item["question"]
    expected_file = item.get("expected_file_contains", "")
    expected_symbols = item.get("expected_symbols", [])

    # Try hybrid search, fall back to FTS5 direct
    results: List[Tuple[float, Dict]] = []
    if hasattr(index, "search_fts5_direct"):
        results = index.search_fts5_direct(question, top_k, min_quality=min_quality)
    if not results:
        kw_results = index.search_keywords(question, top_k)
        for idx, score in kw_results:
            if 0 <= idx < len(index.metadata):
                results.append((score, index.metadata[idx]))

    # Score: file hit
    file_hit = False
    file_rank = 0
    for rank, (score, meta) in enumerate(results, 1):
        source = meta.get("source_path", meta.get("source", ""))
        if expected_file and expected_file.replace("\\", "/") in source.replace("\\", "/"):
            file_hit = True
            file_rank = rank
            break

    # Score: symbol hits
    all_content = " ".join(
        _normalize(meta.get("content", ""))
        for _, meta in results
    )
    symbol_hits = []
    for sym in expected_symbols:
        sym_norm = _normalize(sym)
        hit = sym_norm in all_content
        symbol_hits.append({"symbol": sym, "hit": hit})

    symbols_found = sum(1 for s in symbol_hits if s["hit"])
    symbols_total = len(expected_symbols)
    symbol_rate = symbols_found / symbols_total if symbols_total > 0 else 1.0

    # MRR: 1/rank of first relevant file
    mrr = 1.0 / file_rank if file_hit else 0.0

    return {
        "id": item.get("id", ""),
        "question": question,
        "file_hit": file_hit,
        "file_rank": file_rank,
        "symbol_hits": symbol_hits,
        "symbol_rate": symbol_rate,
        "mrr": mrr,
        "results_count": len(results),
    }


def run_eval(
    golden_path: str = "",
    top_k: int = 10,
    min_quality: int = 0,
    index_dir: str = "",
) -> Dict:
    """Run full retrieval eval. Returns summary dict."""
    golden = load_golden_set(golden_path)
    if not index_dir:
        index_dir = os.environ.get("JCODER_DATA", r"D:\JCoder_Data")

    # Try to load repo index (the one that indexes JCoder's own code)
    config = StorageConfig(index_dir=os.path.join(index_dir, "indexes"))
    index = IndexEngine(config)

    # Try loading agent_memory or repos index
    loaded = False
    for idx_name in ["agent_memory", "repos", "default"]:
        try:
            index.load(idx_name)
            if index.count > 0 or hasattr(index, "_fts_conn"):
                loaded = True
                break
        except Exception:
            continue

    if not loaded:
        # Use FTS5 direct mode with any available .fts5.db
        fts_dir = Path(config.index_dir)
        fts_files = sorted(fts_dir.glob("*.fts5.db")) if fts_dir.exists() else []
        if not fts_files:
            return {"error": "No indexes found", "golden_count": len(golden)}

    results = []
    t0 = time.monotonic()
    for item in golden:
        try:
            result = evaluate_query(index, item, top_k=top_k, min_quality=min_quality)
            results.append(result)
        except Exception as exc:
            results.append({
                "id": item.get("id", ""),
                "question": item["question"],
                "error": str(exc),
                "file_hit": False,
                "mrr": 0.0,
                "symbol_rate": 0.0,
            })
    elapsed = time.monotonic() - t0

    # Aggregate metrics
    file_hits = sum(1 for r in results if r.get("file_hit"))
    avg_mrr = sum(r.get("mrr", 0) for r in results) / len(results) if results else 0
    avg_symbol = sum(r.get("symbol_rate", 0) for r in results) / len(results) if results else 0
    recall_at_5 = sum(
        1 for r in results
        if r.get("file_hit") and r.get("file_rank", 99) <= 5
    ) / len(results) if results else 0

    summary = {
        "timestamp": time.time(),
        "golden_count": len(golden),
        "evaluated": len(results),
        "top_k": top_k,
        "min_quality": min_quality,
        "elapsed_s": round(elapsed, 2),
        "metrics": {
            "file_hit_rate": round(file_hits / len(results), 3) if results else 0,
            "recall_at_5": round(recall_at_5, 3),
            "mrr_at_10": round(avg_mrr, 3),
            "avg_symbol_rate": round(avg_symbol, 3),
        },
        "per_question": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="JCoder Retrieval Quality Eval")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval")
    parser.add_argument("--min-quality", type=int, default=0, help="Min quality score filter")
    parser.add_argument("--golden", default="", help="Path to golden questions JSON")
    parser.add_argument("--output", default="", help="Output JSON path")
    args = parser.parse_args()

    print("\nJCoder Retrieval Quality Evaluation")
    print("=" * 50)

    summary = run_eval(
        golden_path=args.golden,
        top_k=args.top_k,
        min_quality=args.min_quality,
    )

    if "error" in summary:
        print(f"[ERROR] {summary['error']}")
        return

    m = summary["metrics"]
    print(f"Questions: {summary['evaluated']}/{summary['golden_count']}")
    print(f"Top-K: {summary['top_k']}, Min Quality: {summary['min_quality']}")
    print(f"Time: {summary['elapsed_s']}s")
    print()
    print(f"  File Hit Rate:    {m['file_hit_rate']:.1%}")
    print(f"  Recall@5:         {m['recall_at_5']:.1%}")
    print(f"  MRR@10:           {m['mrr_at_10']:.3f}")
    print(f"  Avg Symbol Rate:  {m['avg_symbol_rate']:.1%}")
    print()

    # Show misses
    misses = [r for r in summary["per_question"] if not r.get("file_hit")]
    if misses:
        print(f"Misses ({len(misses)}):")
        for r in misses[:10]:
            print(f"  [{r.get('id','')}] {r['question'][:60]}")

    # Save results
    output = args.output or os.path.join(
        _ROOT, "evaluation", "retrieval_eval_results.json")
    Path(output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
