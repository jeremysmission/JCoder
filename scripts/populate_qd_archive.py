"""
Populate the QD Archive with initial solutions from eval results.

Reads evaluation results and creates QDSolution entries for each,
building diversity metrics from category + score + answer characteristics.

Usage:
  python scripts/populate_qd_archive.py                    # populate from eval results
  python scripts/populate_qd_archive.py --dry-run          # preview only
  python scripts/populate_qd_archive.py --results PATH     # custom results file
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.quality_diversity import (
    QualityDiversityArchive,
    QDSolution,
    compute_behavior,
    niche_key,
)

log = logging.getLogger("populate_qd")

# Map eval categories to answer types for behavior computation
CATEGORY_TO_TYPE = {
    "python": "explain",
    "javascript": "explain",
    "algorithms": "reasoning",
    "debugging": "debug",
    "systems": "design",
    "shell": "lookup",
    "security": "reasoning",
    "database": "explain",
    "devops": "design",
    "testing": "explain",
    "architecture": "design",
    "networking": "explain",
    "general": "explain",
}


def _estimate_complexity(result: dict) -> float:
    """Estimate query complexity from eval result characteristics."""
    score = result.get("score", 0.5)
    elapsed = result.get("elapsed_s", 1.0)
    tokens = result.get("tokens_used", 500)

    # Higher elapsed time and token usage suggest higher complexity
    time_signal = min(1.0, elapsed / 30.0)
    token_signal = min(1.0, tokens / 2000.0)
    # Lower scores suggest harder questions
    difficulty_signal = 1.0 - score

    return 0.4 * difficulty_signal + 0.3 * time_signal + 0.3 * token_signal


def _build_config(result: dict) -> dict:
    """Extract a config snapshot from an eval result."""
    return {
        "category": result.get("category", "unknown"),
        "strategy": "standard",
        "question_id": result.get("question_id", ""),
    }


def populate_from_eval(
    results_path: str = "evaluation/results_local/eval_results.json",
    archive_path: str = "_qd_archive/agent_archive.db",
    dry_run: bool = False,
) -> dict:
    """
    Load eval results and populate the QD archive.

    Returns summary dict.
    """
    t0 = time.time()
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results_loaded": 0,
        "solutions_accepted": 0,
        "solutions_rejected": 0,
        "niches_filled": 0,
        "dry_run": dry_run,
    }

    # Load eval results
    rp = Path(results_path)
    if not rp.exists():
        log.error("Results file not found: %s", results_path)
        return summary

    with open(rp, encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list):
        log.error("Expected list of results, got %s", type(results).__name__)
        return summary

    summary["results_loaded"] = len(results)
    log.info("Loaded %d eval results from %s", len(results), results_path)

    if dry_run:
        # Preview: compute behaviors and show distribution
        niches_seen = set()
        for r in results:
            category = r.get("category", "general")
            answer_type = CATEGORY_TO_TYPE.get(category, "explain")
            complexity = _estimate_complexity(r)
            confidence = r.get("score", 0.5)

            behavior = compute_behavior(
                query_complexity=complexity,
                answer_type=answer_type,
                retrieval_confidence=confidence,
            )
            key = niche_key(behavior)
            niches_seen.add(key)

        summary["niches_filled"] = len(niches_seen)
        log.info("[DRY RUN] Would fill %d unique niches from %d results",
                 len(niches_seen), len(results))
        return summary

    # Build and populate archive
    archive = QualityDiversityArchive(db_path=archive_path, resolution=4)
    accepted = 0
    rejected = 0

    for r in results:
        category = r.get("category", "general")
        answer_type = CATEGORY_TO_TYPE.get(category, "explain")
        complexity = _estimate_complexity(r)
        score = r.get("score", 0.0)

        behavior = compute_behavior(
            query_complexity=complexity,
            answer_type=answer_type,
            retrieval_confidence=score,
        )

        solution = QDSolution(
            config=_build_config(r),
            fitness=score,
            behavior=behavior,
        )

        if archive.add(solution):
            accepted += 1
        else:
            rejected += 1

    summary["solutions_accepted"] = accepted
    summary["solutions_rejected"] = rejected

    coverage = archive.coverage()
    summary["niches_filled"] = coverage.get("filled", 0)
    summary["coverage_pct"] = coverage.get("coverage_pct", 0.0)
    summary["avg_fitness"] = coverage.get("avg_fitness", 0.0)
    summary["elapsed_s"] = round(time.time() - t0, 2)

    log.info("Archive populated: %d accepted, %d rejected, %d niches filled (%.1f%%)",
             accepted, rejected, coverage.get("filled", 0),
             coverage.get("coverage_pct", 0.0))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Populate QD archive from eval results")
    parser.add_argument("--results", default="evaluation/results_local/eval_results.json",
                        help="Path to eval results JSON")
    parser.add_argument("--archive", default="_qd_archive/agent_archive.db",
                        help="Path to QD archive DB")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    summary = populate_from_eval(
        results_path=args.results,
        archive_path=args.archive,
        dry_run=args.dry_run,
    )

    print("\n=== QD Archive Population Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
