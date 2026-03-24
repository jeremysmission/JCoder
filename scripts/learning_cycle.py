"""
Automated Learning Cycle (Sprint 13)
-------------------------------------
Runs a complete learn-evaluate loop:

Phase 1: Baseline eval (record scores per category)
Phase 2: Generate study queries from weak categories
Phase 3: Run study engine on each query (generates telemetry, experience, memory)
Phase 4: Close feedback loop (distill weak topics)
Phase 5: Re-eval (record new scores)
Phase 6: Compare and report delta

Usage:
    python scripts/learning_cycle.py --eval-set evaluation/agent_eval_set_200.json
    python scripts/learning_cycle.py --eval-set evaluation/agent_eval_set_200.json --phases 1,2,3
    python scripts/learning_cycle.py --report-only --cycle-dir logs/learning_cycles/cycle_001
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

def _fix_stdout():
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Phase 1: Baseline eval
# ---------------------------------------------------------------------------

def run_baseline_eval(
    eval_set_path: str,
    index_dir: str,
    output_path: str,
) -> Dict[str, Any]:
    """Run eval and record scores per category."""
    print(f"[Phase 1] Baseline eval: {eval_set_path}")
    t0 = time.time()

    with open(eval_set_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Simple FTS5-based eval (no live LLM needed)
    results = []
    category_scores: Dict[str, List[float]] = {}

    for q in questions:
        qid = q.get("id", "")
        category = q.get("category", "general")
        question_text = q.get("question", "")

        # Retrieve context
        context = _hybrid_retrieve(question_text, index_dir)
        has_context = len(context) > 0

        # Score: keyword overlap + optional LLM judge
        expected = q.get("expected_keywords", [])
        kw_score = _keyword_score(context, expected)

        # LLM-as-judge for context relevancy (RAGAS-style)
        llm_score = _llm_judge_score(question_text, context, expected)
        # Combined: 40% keyword overlap + 60% LLM judge (if available)
        score = 0.4 * kw_score + 0.6 * llm_score if llm_score > 0 else kw_score

        results.append({
            "question_id": qid,
            "category": category,
            "score": score,
            "keyword_score": kw_score,
            "llm_judge_score": llm_score,
            "has_context": has_context,
            "context_chunks": len(context),
        })

        category_scores.setdefault(category, []).append(score)

    # Compute category averages
    category_avgs = {
        cat: sum(scores) / len(scores)
        for cat, scores in category_scores.items()
    }

    overall = sum(r["score"] for r in results) / len(results) if results else 0.0

    report = {
        "timestamp": _timestamp(),
        "eval_set": eval_set_path,
        "total_questions": len(questions),
        "overall_score": round(overall, 4),
        "category_scores": {k: round(v, 4) for k, v in category_avgs.items()},
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  [OK] Overall: {overall:.2%} | Categories: {len(category_avgs)}")
    for cat, avg in sorted(category_avgs.items(), key=lambda x: x[1]):
        print(f"    {cat}: {avg:.2%}")

    return report


# ---------------------------------------------------------------------------
# Phase 2: Generate study queries
# ---------------------------------------------------------------------------

def generate_study_queries(
    baseline_report: Dict[str, Any],
    eval_set_path: str,
    n_queries: int = 50,
    weakness_threshold: float = 0.5,
) -> List[Dict[str, str]]:
    """Generate study queries targeting weak categories."""
    print(f"[Phase 2] Generating {n_queries} study queries from weak categories")

    with open(eval_set_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Find weak categories
    cat_scores = baseline_report.get("category_scores", {})
    weak_cats = [
        cat for cat, score in cat_scores.items()
        if score < weakness_threshold
    ]

    if not weak_cats:
        # Fall back to bottom 3
        sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1])
        weak_cats = [cat for cat, _ in sorted_cats[:3]]

    print(f"  Weak categories: {weak_cats}")

    result_scores = {
        r["question_id"]: r["score"]
        for r in baseline_report.get("results", [])
    }

    # Select questions from weak categories
    weak_questions = [
        q for q in questions
        if q.get("category", "") in weak_cats
    ]
    weak_questions.sort(
        key=lambda q: result_scores.get(q.get("id", ""), 0.0),
        reverse=True,
    )

    # Also add lowest-scoring questions regardless of category
    all_by_score = sorted(
        questions,
        key=lambda q: result_scores.get(q.get("id", ""), 1.0),
        reverse=True,
    )

    study = []
    seen = set()
    for q in weak_questions + all_by_score:
        qid = q.get("id", "")
        if qid in seen:
            continue
        seen.add(qid)
        study.append({
            "id": qid,
            "question": q.get("question", ""),
            "category": q.get("category", ""),
            "baseline_score": result_scores.get(qid, 0.0),
        })
        if len(study) >= n_queries:
            break

    study.sort(key=lambda q: q["baseline_score"], reverse=True)

    print(f"  [OK] Generated {len(study)} study queries")
    return study


# ---------------------------------------------------------------------------
# Phase 3: Run study engine
# ---------------------------------------------------------------------------

def run_study_engine(
    study_queries: List[Dict[str, str]],
    index_dir: str,
    telemetry_db: str = "_telemetry/agent_events.db",
    experience_db: str = "_experience/agent_replay.db",
) -> Dict[str, Any]:
    """Run each study query through retrieval and log telemetry."""
    print(f"[Phase 3] Running study engine on {len(study_queries)} queries")
    t0 = time.time()

    studied = 0
    errors = 0

    for i, sq in enumerate(study_queries, 1):
        question = sq.get("question", "")
        try:
            context = _fts5_retrieve(question, index_dir)
            # Log to telemetry (if db exists)
            _log_study_event(
                telemetry_db, sq.get("id", ""),
                question, len(context),
            )
            studied += 1
        except Exception as exc:
            errors += 1
            if i <= 3:
                print(f"  [WARN] Query {i}: {exc}")

    elapsed = time.time() - t0
    print(f"  [OK] Studied {studied}/{len(study_queries)} queries in {elapsed:.1f}s")

    return {
        "studied": studied,
        "errors": errors,
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Phase 5: Re-eval
# ---------------------------------------------------------------------------

def run_reeval(
    eval_set_path: str,
    index_dir: str,
    output_path: str,
) -> Dict[str, Any]:
    """Run post-learning evaluation."""
    print(f"[Phase 5] Re-evaluation")
    return run_baseline_eval(eval_set_path, index_dir, output_path)


# ---------------------------------------------------------------------------
# Phase 6: Compare and report
# ---------------------------------------------------------------------------

def compare_and_report(
    baseline: Dict[str, Any],
    reeval: Dict[str, Any],
    cycle_dir: str,
) -> Dict[str, Any]:
    """Compare baseline vs re-eval scores and generate report."""
    print(f"[Phase 6] Comparing baseline vs re-eval")

    base_cats = baseline.get("category_scores", {})
    re_cats = reeval.get("category_scores", {})

    deltas = {}
    for cat in set(list(base_cats.keys()) + list(re_cats.keys())):
        base_score = base_cats.get(cat, 0.0)
        re_score = re_cats.get(cat, 0.0)
        delta = re_score - base_score
        deltas[cat] = {
            "baseline": round(base_score, 4),
            "reeval": round(re_score, 4),
            "delta": round(delta, 4),
            "improved": delta > 0,
        }

    overall_delta = reeval.get("overall_score", 0) - baseline.get("overall_score", 0)
    improved_cats = sum(1 for d in deltas.values() if d["improved"])
    regressed_cats = sum(1 for d in deltas.values() if d["delta"] < -0.01)

    report = {
        "timestamp": _timestamp(),
        "overall_baseline": baseline.get("overall_score", 0),
        "overall_reeval": reeval.get("overall_score", 0),
        "overall_delta": round(overall_delta, 4),
        "categories_improved": improved_cats,
        "categories_regressed": regressed_cats,
        "category_deltas": deltas,
    }

    report_path = Path(cycle_dir) / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  Overall: {baseline.get('overall_score', 0):.2%} -> "
          f"{reeval.get('overall_score', 0):.2%} (delta: {overall_delta:+.2%})")
    print(f"  Categories improved: {improved_cats} | Regressed: {regressed_cats}")

    for cat, d in sorted(deltas.items(), key=lambda x: x[1]["delta"], reverse=True):
        marker = "+" if d["improved"] else ("-" if d["delta"] < 0 else "=")
        print(f"    [{marker}] {cat}: {d['baseline']:.2%} -> {d['reeval']:.2%} ({d['delta']:+.2%})")

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hybrid_retrieve(question: str, index_dir: str, top_k: int = 5) -> List[str]:
    """Hybrid FAISS+FTS5 retrieval with RRF fusion."""
    fts5_results = _fts5_retrieve(question, index_dir, top_k=top_k * 2)

    # Try FAISS semantic search
    faiss_results = []
    try:
        from core.config import load_config
        from core.embedding_engine import EmbeddingEngine
        import faiss as _faiss
        import numpy as np

        cfg = load_config()
        engine = EmbeddingEngine(cfg.embedder, timeout=15)
        query_vec = engine.embed_single(question)

        idx_path = Path(index_dir)
        for faiss_file in idx_path.glob("*.faiss"):
            try:
                index = _faiss.read_index(str(faiss_file))
                if index.ntotal == 0:
                    continue
                meta_file = faiss_file.with_suffix(".faiss.meta.json")
                if not meta_file.exists():
                    continue
                import json as _json
                with open(meta_file, "r", encoding="utf-8") as mf:
                    meta = _json.load(mf)
                D, I = index.search(query_vec.reshape(1, -1), min(5, index.ntotal))
                for idx_val, score in zip(I[0], D[0]):
                    if 0 <= idx_val < len(meta):
                        text = meta[idx_val].get("text", "")[:500]
                        if text:
                            faiss_results.append((float(score), text))
            except Exception:
                continue
        engine.close()
    except Exception:
        pass  # Fall back to FTS5-only

    # RRF fusion: merge FTS5 and FAISS results
    rrf_k = 60
    scored: Dict[str, float] = {}
    for rank, text in enumerate(fts5_results):
        scored[text] = scored.get(text, 0) + 1.0 / (rrf_k + rank + 1)
    for rank, (_, text) in enumerate(sorted(faiss_results, key=lambda x: -x[0])):
        scored[text] = scored.get(text, 0) + 1.0 / (rrf_k + rank + 1)

    if scored:
        fused = sorted(scored.items(), key=lambda x: -x[1])
        return [text for text, _ in fused[:top_k]]
    return fts5_results[:top_k]


def _fts5_retrieve(question: str, index_dir: str, top_k: int = 5) -> List[str]:
    """Quick FTS5 search across indexes."""
    results = []
    idx_path = Path(index_dir)
    if not idx_path.exists():
        return results

    clean_q = " ".join(
        w for w in question.split()
        if w.isalnum() or w.replace("_", "").isalnum()
    )
    if not clean_q:
        return results

    for entry in idx_path.iterdir():
        if not entry.name.endswith(".fts5.db"):
            continue
        try:
            conn = sqlite3.connect(str(entry))
            rows = conn.execute(
                "SELECT search_content FROM chunks WHERE chunks MATCH ? LIMIT ?",
                (clean_q, 3),
            ).fetchall()
            conn.close()
            for row in rows:
                results.append(row[0][:500])
        except Exception:
            continue

    return results[:top_k]


def _keyword_score(contexts: List[str], expected: List[str]) -> float:
    """Score based on keyword overlap."""
    if not expected:
        return 0.5 if contexts else 0.0

    combined = " ".join(contexts).lower()
    hits = sum(1 for kw in expected if kw.lower() in combined)
    return hits / len(expected)


def _llm_judge_score(
    question: str,
    contexts: List[str],
    expected_keywords: List[str],
    model: str = "phi4:14b-q4_K_M",
    endpoint: str = "http://localhost:11434/api/generate",
) -> float:
    """Use LLM-as-judge to score retrieval quality (RAGAS-style).

    Scores context_relevancy: does the retrieved context contain
    information needed to answer the question?
    Returns a float 0.0-1.0.
    """
    if not contexts:
        return 0.0

    context_text = "\n---\n".join(c[:300] for c in contexts[:5])
    prompt = (
        "You are a retrieval quality judge. Given a question and retrieved "
        "context chunks, rate how relevant the context is for answering the "
        "question.\n\n"
        f"Question: {question}\n\n"
        f"Expected concepts: {', '.join(expected_keywords)}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        "Rate the context relevancy on a scale of 0 to 10, where:\n"
        "0 = completely irrelevant\n"
        "5 = partially relevant (some useful info)\n"
        "10 = highly relevant (contains key concepts needed)\n\n"
        "Respond with ONLY a single integer 0-10, nothing else."
    )

    try:
        import httpx

        r = httpx.post(
            endpoint,
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"num_predict": 5, "temperature": 0.0}},
            timeout=30,
        )
        answer = r.json().get("response", "").strip()
        # Extract first integer from response
        for token in answer.split():
            token_clean = token.strip(".,;:!")
            if token_clean.isdigit():
                score = int(token_clean)
                return min(max(score / 10.0, 0.0), 1.0)
        return 0.0
    except Exception:
        return 0.0


def _log_study_event(db_path: str, qid: str, question: str, chunks: int) -> None:
    """Log a study event to telemetry."""
    db = Path(db_path)
    if not db.parent.exists():
        return
    try:
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS study_events (
                query_id TEXT, question TEXT, chunks_found INTEGER,
                timestamp REAL
            )
        """)
        conn.execute(
            "INSERT INTO study_events VALUES (?, ?, ?, ?)",
            (qid, question[:500], chunks, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _load_distillation_config() -> dict:
    """Load distillation settings from config/agent.yaml."""
    cfg_path = _ROOT / "config" / "agent.yaml"
    if not cfg_path.exists():
        return {"enabled": False}
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("agent", {}).get("distillation", {"enabled": False})
    except ImportError:
        pass
    # Fallback: regex parse
    text = cfg_path.read_text(encoding="utf-8")
    import re
    section = re.search(r"distillation:\s*\n((?:\s+\w+:.*\n)+)", text)
    if not section:
        return {"enabled": False}
    cfg: dict = {}
    for line in section.group(1).splitlines():
        m = re.match(r"\s+(\w+):\s*(.*)", line)
        if m:
            k, v = m.group(1), m.group(2).strip().strip('"').strip("'")
            if v.lower() == "true":
                cfg[k] = True
            elif v.lower() == "false":
                cfg[k] = False
            else:
                try:
                    cfg[k] = float(v) if "." in v else int(v)
                except ValueError:
                    cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Automated Learning Cycle")
    parser.add_argument("--eval-set", default=str(_ROOT / "evaluation" / "agent_eval_set_200.json"),
                        help="Eval question set")
    parser.add_argument("--index-dir", default=str(_ROOT / "data" / "indexes"),
                        help="FTS5 index directory")
    parser.add_argument("--cycle-dir", default=None,
                        help="Output directory for this cycle (auto-generated if omitted)")
    parser.add_argument("--phases", default="1,2,3,4,5,6",
                        help="Comma-separated phase numbers to run (default: 1,2,3,4,5,6)")
    parser.add_argument("--study-queries", type=int, default=50,
                        help="Number of study queries to generate")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate comparison report from existing cycle data")
    args = parser.parse_args()

    phases = set(int(p) for p in args.phases.split(","))

    # Set up cycle directory
    if args.cycle_dir:
        cycle_dir = args.cycle_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cycle_dir = str(_ROOT / "logs" / "learning_cycles" / f"cycle_{stamp}")
    Path(cycle_dir).mkdir(parents=True, exist_ok=True)

    baseline_path = Path(cycle_dir) / "baseline_eval.json"
    reeval_path = Path(cycle_dir) / "reeval.json"
    study_path = Path(cycle_dir) / "study_queries.json"

    print(f"{'=' * 60}")
    print(f"Learning Cycle: {cycle_dir}")
    print(f"Phases: {sorted(phases)}")
    print(f"{'=' * 60}\n")

    baseline = None
    reeval = None

    # Phase 1: Baseline
    if 1 in phases:
        baseline = run_baseline_eval(args.eval_set, args.index_dir, str(baseline_path))
        print()

    # Load baseline if not just run
    if baseline is None and baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)

    # Phase 2: Generate study queries
    if 2 in phases and baseline:
        study_queries = generate_study_queries(
            baseline, args.eval_set, n_queries=args.study_queries,
        )
        with open(study_path, "w", encoding="utf-8") as f:
            json.dump(study_queries, f, indent=2)
        print()

    # Phase 3: Run study engine
    if 3 in phases and study_path.exists():
        with open(study_path, "r", encoding="utf-8") as f:
            study_queries = json.load(f)
        study_result = run_study_engine(study_queries, args.index_dir)
        with open(Path(cycle_dir) / "study_result.json", "w", encoding="utf-8") as f:
            json.dump(study_result, f, indent=2)
        print()

    # Phase 4: Close feedback loop via distillation
    if 4 in phases:
        print("[Phase 4] Distill weak topics via strong model")
        distill_cfg = _load_distillation_config()
        if not distill_cfg.get("enabled", False):
            print("  Distillation disabled in config/agent.yaml — skipping")
        elif baseline is None:
            print("  No baseline results — skipping distillation")
        else:
            try:
                from scripts.distill_weak_topics import run_distillation
                distill_result = run_distillation(
                    eval_results=baseline.get("results", []),
                    eval_set_path=args.eval_set,
                    index_dir=args.index_dir,
                    model=distill_cfg.get("model", "gpt-5"),
                    top=distill_cfg.get("top", 20),
                    budget_usd=distill_cfg.get("budget_usd", 2.0),
                    resume=distill_cfg.get("resume", True),
                )
                with open(Path(cycle_dir) / "distillation_result.json", "w", encoding="utf-8") as f:
                    json.dump(distill_result, f, indent=2)
                print(f"  Distilled {distill_result.get('distilled', 0)} topics, "
                      f"cost: ${distill_result.get('total_cost', 0):.2f}")
            except ImportError:
                print("  distill_weak_topics.py not available — skipping")
            except Exception as exc:
                print(f"  Distillation error: {exc}")
        print()

    # Phase 5: Re-eval
    if 5 in phases:
        reeval = run_reeval(args.eval_set, args.index_dir, str(reeval_path))
        print()

    # Load reeval if not just run
    if reeval is None and reeval_path.exists():
        with open(reeval_path, "r", encoding="utf-8") as f:
            reeval = json.load(f)

    # Phase 6: Compare
    if 6 in phases and baseline and reeval:
        compare_and_report(baseline, reeval, cycle_dir)
        print()

    print(f"{'=' * 60}")
    print(f"Cycle complete: {cycle_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _fix_stdout()
    main()
