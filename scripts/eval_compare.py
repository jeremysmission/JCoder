"""Compare two eval result sets and generate an A/B report.

Usage:
    cd D:\\JCoder
    python scripts/eval_compare.py \\
        --baseline evaluation/results_local/eval_results.json \\
        --candidate evaluation/results_api_gpt-4_1-mini/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _load_results(path: str) -> dict[str, dict]:
    """Load eval results keyed by question_id."""
    with open(path, "r", encoding="utf-8") as f:
        return {r["question_id"]: r for r in json.load(f)}


def _cat_stats(results: dict[str, dict]) -> dict[str, dict]:
    """Compute per-category stats."""
    cats: dict[str, list[float]] = defaultdict(list)
    for r in results.values():
        cats[r["category"]].append(r["score"])
    return {
        cat: {
            "count": len(scores),
            "avg": sum(scores) / len(scores),
            "pass_rate": sum(1 for s in scores if s >= 0.5) / len(scores),
            "min": min(scores),
        }
        for cat, scores in sorted(cats.items())
    }


def compare(baseline_path: str, candidate_path: str, output_path: str = "") -> str:
    """Generate markdown comparison report."""
    base = _load_results(baseline_path)
    cand = _load_results(candidate_path)

    # Overall stats
    b_scores = [r["score"] for r in base.values()]
    c_scores = [r["score"] for r in cand.values()]

    b_pass = sum(1 for s in b_scores if s >= 0.5)
    c_pass = sum(1 for s in c_scores if s >= 0.5)

    b_avg = sum(b_scores) / len(b_scores) if b_scores else 0
    c_avg = sum(c_scores) / len(c_scores) if c_scores else 0

    b_latency = [r["elapsed_s"] for r in base.values() if r["elapsed_s"] > 0]
    c_latency = [r["elapsed_s"] for r in cand.values() if r["elapsed_s"] > 0]

    b_cat = _cat_stats(base)
    c_cat = _cat_stats(cand)

    # Per-question deltas
    common_ids = sorted(set(base.keys()) & set(cand.keys()))
    deltas = []
    for qid in common_ids:
        delta = cand[qid]["score"] - base[qid]["score"]
        deltas.append((qid, base[qid]["score"], cand[qid]["score"], delta, base[qid]["category"]))

    # Sort by delta (biggest improvements first, then biggest regressions)
    improved = sorted([d for d in deltas if d[3] > 0.01], key=lambda x: -x[3])
    regressed = sorted([d for d in deltas if d[3] < -0.01], key=lambda x: x[3])
    unchanged = [d for d in deltas if abs(d[3]) <= 0.01]

    # Category gap analysis
    cat_gaps = []
    all_cats = sorted(set(b_cat.keys()) | set(c_cat.keys()))
    for cat in all_cats:
        b_info = b_cat.get(cat, {"avg": 0, "pass_rate": 0, "count": 0})
        c_info = c_cat.get(cat, {"avg": 0, "pass_rate": 0, "count": 0})
        gap = c_info["avg"] - b_info["avg"]
        cat_gaps.append((cat, b_info, c_info, gap))
    cat_gaps.sort(key=lambda x: -x[3])  # biggest improvement first

    # Build report
    lines = [
        "# Eval A/B Comparison Report", "",
        f"Baseline: `{Path(baseline_path).parent.name}` ({len(base)} questions)",
        f"Candidate: `{Path(candidate_path).parent.name}` ({len(cand)} questions)",
        "",
        "## Overall", "",
        "| Metric | Baseline | Candidate | Delta |",
        "|--------|----------|-----------|-------|",
        f"| Questions | {len(base)} | {len(cand)} | -- |",
        f"| Pass rate | {b_pass}/{len(base)} ({b_pass/len(base):.1%}) "
        f"| {c_pass}/{len(cand)} ({c_pass/len(cand):.1%}) "
        f"| {'+' if c_pass >= b_pass else ''}{c_pass - b_pass} |",
        f"| Avg score | {b_avg:.4f} | {c_avg:.4f} "
        f"| {'+' if c_avg >= b_avg else ''}{c_avg - b_avg:.4f} |",
    ]
    if b_latency and c_latency:
        b_lat = sum(b_latency) / len(b_latency)
        c_lat = sum(c_latency) / len(c_latency)
        lines.append(
            f"| Avg latency | {b_lat:.1f}s | {c_lat:.1f}s "
            f"| {'+' if c_lat >= b_lat else ''}{c_lat - b_lat:.1f}s |"
        )

    # Per-category comparison
    lines += ["", "## Per Category", "",
              "| Category | Baseline Avg | Candidate Avg | Delta | Baseline Pass | Candidate Pass |",
              "|----------|-------------|---------------|-------|--------------|----------------|"]
    for cat, b_info, c_info, gap in cat_gaps:
        sign = "+" if gap >= 0 else ""
        lines.append(
            f"| {cat} | {b_info['avg']:.3f} | {c_info['avg']:.3f} | "
            f"{sign}{gap:.3f} | {b_info['pass_rate']:.0%} | {c_info['pass_rate']:.0%} |"
        )

    # Weakest categories (where distillation should focus)
    lines += ["", "## Distillation Targets (Weakest in Baseline)", ""]
    weak_cats = sorted(cat_gaps, key=lambda x: x[1]["avg"])[:5]
    for rank, (cat, b_info, c_info, gap) in enumerate(weak_cats, 1):
        lines.append(f"{rank}. **{cat}** -- baseline avg={b_info['avg']:.3f}, "
                     f"candidate avg={c_info['avg']:.3f}, gap={gap:+.3f}")

    # Improved questions
    if improved:
        lines += ["", f"## Improved Questions ({len(improved)})", ""]
        for qid, b_score, c_score, delta, cat in improved[:15]:
            lines.append(f"- **{qid}** ({cat}): {b_score:.2f} -> {c_score:.2f} ({delta:+.2f})")
        if len(improved) > 15:
            lines.append(f"- ... and {len(improved) - 15} more")

    # Regressed questions
    if regressed:
        lines += ["", f"## Regressed Questions ({len(regressed)})", ""]
        for qid, b_score, c_score, delta, cat in regressed[:10]:
            lines.append(f"- **{qid}** ({cat}): {b_score:.2f} -> {c_score:.2f} ({delta:+.2f})")

    # Summary
    lines += [
        "", "## Summary", "",
        f"- **{len(improved)}** questions improved",
        f"- **{len(regressed)}** questions regressed",
        f"- **{len(unchanged)}** questions unchanged",
        f"- Net score delta: {c_avg - b_avg:+.4f}",
        "",
    ]

    text = "\n".join(lines) + "\n"

    if not output_path:
        output_path = str(Path(candidate_path).parent / "comparison_report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] Comparison report: {output_path}")

    return text


def main():
    parser = argparse.ArgumentParser(description="Compare two eval result sets")
    parser.add_argument("--baseline", required=True, help="Baseline eval_results.json")
    parser.add_argument("--candidate", required=True, help="Candidate eval_results.json")
    parser.add_argument("--output", default="", help="Output report path")
    args = parser.parse_args()

    text = compare(args.baseline, args.candidate, args.output)

    # Print summary to console
    for line in text.split("\n"):
        if line.startswith("| ") or line.startswith("#") or line.startswith("-"):
            print(line)


if __name__ == "__main__":
    main()
