"""
Autonomous Self-Improvement Engine
====================================
This is the AGI core: a system that evaluates itself, diagnoses its
own weaknesses, generates targeted improvements, and measures the delta.

Unlike a simple eval loop, this engine:
1. Analyzes failure PATTERNS, not just scores
2. Generates HYPOTHESES about why failures occur
3. Tests hypotheses by varying retrieval strategies
4. Stores successful strategies in experience replay
5. Evolves its own retrieval queries based on what worked

The key insight: improvement comes from understanding the
STRUCTURE of failures, not just counting them.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


class FailureAnalyzer:
    """Analyzes patterns in eval failures to diagnose root causes."""

    def __init__(self, eval_results: List[Dict[str, Any]]):
        self.results = eval_results
        self.failures = [r for r in eval_results if r.get("score", 0) < 0.5]
        self.successes = [r for r in eval_results if r.get("score", 0) >= 0.5]

    def diagnose(self) -> Dict[str, Any]:
        """Produce a structured diagnosis of failure patterns."""
        diagnosis = {
            "timestamp": _timestamp(),
            "total": len(self.results),
            "failures": len(self.failures),
            "successes": len(self.successes),
            "success_rate": len(self.successes) / max(len(self.results), 1),
        }

        # Pattern 1: Category-level weaknesses
        cat_failures = Counter(r["category"] for r in self.failures)
        cat_totals = Counter(r["category"] for r in self.results)
        diagnosis["weak_categories"] = {
            cat: {
                "failure_rate": count / cat_totals[cat],
                "failures": count,
                "total": cat_totals[cat],
            }
            for cat, count in cat_failures.most_common()
            if count / cat_totals[cat] > 0.5  # >50% failure rate
        }

        # Pattern 2: Retrieval failures (context_chunks == 0)
        no_context = [r for r in self.failures if r.get("context_chunks", 0) == 0]
        diagnosis["retrieval_failures"] = {
            "count": len(no_context),
            "pct_of_failures": len(no_context) / max(len(self.failures), 1),
            "categories": dict(Counter(r["category"] for r in no_context)),
        }

        # Pattern 3: Keyword vs LLM judge disagreement
        disagreements = []
        for r in self.results:
            kw = r.get("keyword_score", 0)
            llm = r.get("llm_judge_score", 0)
            if abs(kw - llm) > 0.4:
                disagreements.append({
                    "question_id": r["question_id"],
                    "category": r["category"],
                    "keyword_score": kw,
                    "llm_judge_score": llm,
                    "gap": abs(kw - llm),
                })
        diagnosis["scorer_disagreements"] = {
            "count": len(disagreements),
            "worst": sorted(disagreements, key=lambda x: -x["gap"])[:5],
        }

        # Pattern 4: Consistent failures (same question always fails)
        diagnosis["hypothesis"] = self._generate_hypotheses()

        return diagnosis

    def _generate_hypotheses(self) -> List[str]:
        """Generate hypotheses about WHY failures occur."""
        hypotheses = []

        # Check if failures correlate with question length
        fail_lens = [len(r.get("question_id", "")) for r in self.failures]
        success_lens = [len(r.get("question_id", "")) for r in self.successes]

        # Check retrieval failure pattern
        no_ctx = sum(1 for r in self.failures if r.get("context_chunks", 0) == 0)
        if no_ctx > len(self.failures) * 0.3:
            hypotheses.append(
                f"Retrieval gap: {no_ctx}/{len(self.failures)} failures have 0 "
                f"context chunks. FTS5 keyword matching may not cover these "
                f"question types. Consider expanding FAISS semantic search."
            )

        # Check category concentration
        cat_counts = Counter(r["category"] for r in self.failures)
        top_cat, top_count = cat_counts.most_common(1)[0] if cat_counts else ("", 0)
        if top_count > len(self.failures) * 0.2:
            hypotheses.append(
                f"Category blind spot: '{top_cat}' accounts for "
                f"{top_count}/{len(self.failures)} failures "
                f"({top_count/len(self.failures):.0%}). "
                f"May need domain-specific indexing or retrieval tuning."
            )

        # Check keyword vs LLM judge pattern
        kw_better = sum(
            1 for r in self.results
            if r.get("keyword_score", 0) > r.get("llm_judge_score", 0) + 0.2
        )
        llm_better = sum(
            1 for r in self.results
            if r.get("llm_judge_score", 0) > r.get("keyword_score", 0) + 0.2
        )
        if llm_better > kw_better * 2:
            hypotheses.append(
                f"Scoring undercount: LLM judge scores higher than keyword "
                f"overlap in {llm_better} cases. The keyword scorer may be "
                f"missing relevant context that the LLM can recognize."
            )

        if not hypotheses:
            hypotheses.append("No clear failure pattern identified.")

        return hypotheses


class StrategyGenerator:
    """Generates improvement strategies based on failure diagnosis."""

    def __init__(self, diagnosis: Dict[str, Any], index_dir: str):
        self.diagnosis = diagnosis
        self.index_dir = index_dir

    def generate_strategies(self) -> List[Dict[str, Any]]:
        """Generate concrete improvement strategies."""
        strategies = []

        # Strategy 1: Domain-specific index queries for weak categories
        for cat, info in self.diagnosis.get("weak_categories", {}).items():
            strategies.append({
                "type": "targeted_retrieval",
                "category": cat,
                "action": f"Generate {min(20, info['failures'] * 3)} targeted "
                         f"FTS5 queries for '{cat}' domain",
                "priority": info["failure_rate"],
                "expected_impact": f"Reduce {cat} failure rate from "
                                  f"{info['failure_rate']:.0%}",
            })

        # Strategy 2: Expand FAISS coverage for retrieval failures
        ret_fail = self.diagnosis.get("retrieval_failures", {})
        if ret_fail.get("pct_of_failures", 0) > 0.3:
            strategies.append({
                "type": "index_expansion",
                "action": "Rebuild FAISS indexes with higher chunk coverage "
                         "for databases with <10% vector coverage",
                "priority": ret_fail["pct_of_failures"],
                "expected_impact": f"Fix {ret_fail['count']} retrieval failures",
            })

        # Strategy 3: Improve scoring for disagreement cases
        disagreements = self.diagnosis.get("scorer_disagreements", {})
        if disagreements.get("count", 0) > 10:
            strategies.append({
                "type": "scoring_calibration",
                "action": "Increase LLM judge weight from 60% to 70% "
                         "(LLM consistently finds relevant context that "
                         "keyword scorer misses)",
                "priority": 0.5,
                "expected_impact": f"Resolve {disagreements['count']} "
                                  f"scorer disagreements",
            })

        return sorted(strategies, key=lambda x: -x["priority"])


def run_autonomous_improvement(
    eval_set_path: str = None,
    index_dir: str = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    """Run one complete autonomous improvement cycle.

    Returns a report with diagnosis, strategies, and measured delta.
    """
    eval_set_path = eval_set_path or str(_ROOT / "evaluation" / "agent_eval_set_200.json")
    index_dir = index_dir or str(_ROOT / "data" / "indexes")
    output_dir = output_dir or str(
        _ROOT / "logs" / "autonomous_improvement"
        / f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AUTONOMOUS SELF-IMPROVEMENT ENGINE")
    print(f"Eval set: {eval_set_path}")
    print(f"Index dir: {index_dir}")
    print("=" * 60)

    # Phase 1: Evaluate current performance
    print("\n[Phase 1] Evaluating current performance...")
    from scripts.learning_cycle import run_baseline_eval
    baseline_path = str(Path(output_dir) / "baseline.json")
    baseline = run_baseline_eval(eval_set_path, index_dir, baseline_path)

    # Phase 2: Diagnose failures
    print("\n[Phase 2] Analyzing failure patterns...")
    results = baseline.get("results", [])
    analyzer = FailureAnalyzer(results)
    diagnosis = analyzer.diagnose()

    print(f"  Success rate: {diagnosis['success_rate']:.1%}")
    print(f"  Weak categories: {list(diagnosis['weak_categories'].keys())}")
    print(f"  Retrieval failures: {diagnosis['retrieval_failures']['count']}")
    print(f"  Hypotheses:")
    for h in diagnosis["hypothesis"]:
        print(f"    - {h}")

    # Phase 3: Generate strategies
    print("\n[Phase 3] Generating improvement strategies...")
    generator = StrategyGenerator(diagnosis, index_dir)
    strategies = generator.generate_strategies()
    for s in strategies:
        print(f"  [{s['type']}] {s['action']}")

    # Phase 4: Execute targeted study (learn from weak areas)
    print("\n[Phase 4] Targeted study on weak categories...")
    from scripts.learning_cycle import generate_study_queries, run_study_engine
    if results:
        study_queries = generate_study_queries(
            baseline, eval_set_path, n_queries=30,
        )
        study_path = str(Path(output_dir) / "study_queries.json")
        with open(study_path, "w", encoding="utf-8") as f:
            json.dump(study_queries, f, indent=2)

        study_result = run_study_engine(study_queries, index_dir)
        with open(str(Path(output_dir) / "study_result.json"), "w", encoding="utf-8") as f:
            json.dump(study_result, f, indent=2)

    # Phase 5: Re-evaluate
    print("\n[Phase 5] Re-evaluating after study...")
    reeval_path = str(Path(output_dir) / "reeval.json")
    reeval = run_baseline_eval(eval_set_path, index_dir, reeval_path)

    # Phase 6: Measure delta
    print("\n[Phase 6] Measuring improvement delta...")
    baseline_score = baseline.get("overall_score", 0)
    reeval_score = reeval.get("overall_score", 0)
    delta = reeval_score - baseline_score

    report = {
        "timestamp": _timestamp(),
        "baseline_score": baseline_score,
        "reeval_score": reeval_score,
        "delta": delta,
        "improved": delta > 0,
        "diagnosis": diagnosis,
        "strategies": strategies,
        "category_deltas": {},
    }

    # Per-category deltas
    base_cats = baseline.get("category_scores", {})
    re_cats = reeval.get("category_scores", {})
    for cat in set(base_cats) | set(re_cats):
        b = base_cats.get(cat, 0)
        r = re_cats.get(cat, 0)
        report["category_deltas"][cat] = {
            "baseline": b,
            "reeval": r,
            "delta": r - b,
        }

    # Save report
    report_path = str(Path(output_dir) / "improvement_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULT: {baseline_score:.1%} -> {reeval_score:.1%} (delta: {delta:+.1%})")
    print(f"Improved: {'YES' if delta > 0 else 'NO'}")
    print(f"Report: {report_path}")
    print(f"{'=' * 60}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Self-Improvement Engine")
    parser.add_argument("--eval-set", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_autonomous_improvement(args.eval_set, args.index_dir, args.output_dir)
