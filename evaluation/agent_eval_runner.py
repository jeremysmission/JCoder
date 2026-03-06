"""Agent evaluation runner for JCoder.

Scores agent answers against expected criteria using deterministic
regex-based keyword matching. No LLM judge -- fully reproducible.
Works without a live agent (validation mode).
Results saved incrementally for crash-safety and resumability.
"""
import ast
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvalResult:
    question_id: str
    category: str
    score: float          # 0.0 to 1.0
    subscores: Dict[str, float]
    answer: str
    elapsed_s: float
    tokens_used: int
    passed: bool          # score >= 0.5


class AgentEvalRunner:
    """Run eval questions through an agent and score the answers."""

    def __init__(self, eval_set_path: str, agent: Any = None,
                 output_dir: str = "evaluation/results"):
        with open(eval_set_path, "r", encoding="utf-8") as f:
            self.questions: List[Dict] = json.load(f)
        self.question_map: Dict[str, Dict] = {q["id"]: q for q in self.questions}
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results_path = self.output_dir / "eval_results.json"
        self._completed: Dict[str, Dict] = self._load_completed()

    def _load_completed(self) -> Dict[str, Dict]:
        if self._results_path.exists():
            with open(self._results_path, "r", encoding="utf-8") as f:
                return {r["question_id"]: r for r in json.load(f)}
        return {}

    def _save_results(self, results: List[EvalResult]) -> None:
        tmp = self._results_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        tmp.replace(self._results_path)

    @staticmethod
    def _has_code_block(text: str) -> bool:
        return "```" in text

    @staticmethod
    def _keyword_ratio(text: str, keywords: List[str]) -> float:
        if not keywords:
            return 1.0
        low = text.lower()
        return sum(1 for kw in keywords if kw.lower() in low) / len(keywords)

    @staticmethod
    def _import_ratio(text: str, imports: List[str]) -> float:
        if not imports:
            return 1.0
        low = text.lower()
        return sum(1 for imp in imports if imp.lower() in low) / len(imports)

    @staticmethod
    def _extract_python(text: str) -> str:
        m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _is_valid_python(code: str) -> bool:
        if not code:
            return False
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _cites_source(text: str) -> bool:
        return bool(re.search(r"https?://", text) or
                     re.search(r"[a-zA-Z0-9_/\\]+\.\w{1,5}", text))

    def score_answer(self, question: Dict, answer: str) -> Dict[str, float]:
        """Score an answer against expected criteria. Returns subscores + weighted_total."""
        weights = question.get("scoring", {})
        keywords = question.get("expected_keywords", [])
        imports = question.get("expected_imports", [])
        code = self._extract_python(answer)
        has_block = self._has_code_block(answer)

        sub: Dict[str, float] = {
            "has_code": 1.0 if has_block else 0.0,
            "has_correct_api": self._keyword_ratio(answer, keywords),
            "has_imports": self._import_ratio(answer, imports),
            "is_runnable": (1.0 if self._is_valid_python(code) else 0.0) if code
                           else (1.0 if has_block else 0.0),
            "cites_source": 1.0 if self._cites_source(answer) else 0.0,
        }
        wsum = sum(weights.get(k, 0) for k in sub)
        total = sum(sub[k] * weights.get(k, 0) for k in sub)
        sub["weighted_total"] = round(total / wsum, 4) if wsum > 0 else 0.0
        return sub

    # -- Execution --------------------------------------------------------

    def run_single(self, question_id: str) -> EvalResult:
        """Run a single eval question through the agent."""
        q = self.question_map.get(question_id)
        if q is None:
            raise KeyError(f"Unknown question ID: {question_id}")
        if self.agent is None:
            return EvalResult(question_id=question_id, category=q["category"],
                              score=0.0, subscores={}, answer="[no agent configured]",
                              elapsed_s=0.0, tokens_used=0, passed=False)
        t0 = time.time()
        result = self.agent.run(q["question"])
        elapsed = time.time() - t0
        answer_text = result.answer if hasattr(result, "answer") else str(result)
        tokens = result.tokens_used if hasattr(result, "tokens_used") else 0
        sub = self.score_answer(q, answer_text)
        score = sub.get("weighted_total", 0.0)
        return EvalResult(question_id=question_id, category=q["category"],
                          score=round(score, 4), subscores=sub, answer=answer_text,
                          elapsed_s=round(elapsed, 3), tokens_used=tokens,
                          passed=score >= 0.5)

    def run_all(self, categories: Optional[List[str]] = None,
                max_questions: int = 0, resume: bool = True) -> List[EvalResult]:
        """Run all (or filtered) questions. Saves incrementally."""
        qs = self.questions
        if categories:
            cats = {c.lower() for c in categories}
            qs = [q for q in qs if q["category"].lower() in cats]
        if max_questions > 0:
            qs = qs[:max_questions]
        results: List[EvalResult] = []
        total = len(qs)
        for idx, q in enumerate(qs, 1):
            qid = q["id"]
            if resume and qid in self._completed:
                prev = self._completed[qid]
                results.append(EvalResult(**prev))
                print(f"  [{idx}/{total}] {qid}: RESUMED (score={prev['score']:.2f})")
                continue
            print(f"  [{idx}/{total}] {qid}: running...", end="", flush=True)
            er = self.run_single(qid)
            results.append(er)
            self._completed[qid] = asdict(er)
            print(f" {'PASS' if er.passed else 'FAIL'} (score={er.score:.2f}, {er.elapsed_s:.1f}s)")
            self._save_results(results)
        return results

    # -- Reporting --------------------------------------------------------

    @staticmethod
    def summary(results: List[EvalResult]) -> Dict:
        """Overall + per-category statistics."""
        if not results:
            return {"total": 0}
        n = len(results)
        passed = sum(1 for r in results if r.passed)
        scores = [r.score for r in results]
        latencies = [r.elapsed_s for r in results if r.elapsed_s > 0]
        tokens = [r.tokens_used for r in results if r.tokens_used > 0]
        cats: Dict[str, List[float]] = {}
        for r in results:
            cats.setdefault(r.category, []).append(r.score)
        per_cat = {
            c: {"count": len(s), "avg_score": round(sum(s)/len(s), 4),
                 "pass_rate": round(sum(1 for v in s if v >= 0.5)/len(s), 4)}
            for c, s in sorted(cats.items())
        }
        worst = sorted(results, key=lambda r: r.score)[:5]
        return {
            "total": n, "passed": passed,
            "pass_rate": round(passed / n, 4),
            "avg_score": round(sum(scores) / n, 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "per_category": per_cat,
            "worst_5": [{"id": r.question_id, "score": r.score} for r in worst],
            "avg_latency_s": round(sum(latencies)/len(latencies), 3) if latencies else 0.0,
            "avg_tokens": round(sum(tokens)/len(tokens)) if tokens else 0,
        }

    def report(self, results: List[EvalResult], output_path: str = "") -> str:
        """Write a markdown report and return the text."""
        s = self.summary(results)
        if not output_path:
            output_path = str(self.output_dir / "eval_report.md")
        lines = [
            "# Agent Evaluation Report", "",
            "## Overall", "",
            "| Metric | Value |", "|--------|-------|",
            f"| Total questions | {s['total']} |",
            f"| Passed | {s.get('passed', 0)} |",
            f"| Pass rate | {s.get('pass_rate', 0):.1%} |",
            f"| Avg score | {s.get('avg_score', 0):.4f} |",
            f"| Min score | {s.get('min_score', 0):.4f} |",
            f"| Max score | {s.get('max_score', 0):.4f} |",
            f"| Avg latency | {s.get('avg_latency_s', 0):.3f}s |",
            f"| Avg tokens | {s.get('avg_tokens', 0)} |",
            "", "## Per Category", "",
            "| Category | Count | Avg Score | Pass Rate |",
            "|----------|-------|-----------|-----------|",
        ]
        for cat, info in s.get("per_category", {}).items():
            lines.append(f"| {cat} | {info['count']} | {info['avg_score']:.4f} | {info['pass_rate']:.1%} |")
        lines += ["", "## Worst 5 Questions", ""]
        for w in s.get("worst_5", []):
            lines.append(f"- **{w['id']}**: {w['score']:.4f}")
        failed = [r for r in results if not r.passed]
        if failed:
            lines += ["", "## Failed Question Details", ""]
            for r in sorted(failed, key=lambda x: x.score):
                q = self.question_map.get(r.question_id, {})
                preview = r.answer[:300] + "..." if len(r.answer) > 300 else r.answer
                lines += [
                    f"### {r.question_id} (score: {r.score:.4f})", "",
                    f"**Question:** {q.get('question', 'N/A')}", "",
                    f"**Subscores:** {r.subscores}", "",
                    f"**Answer preview:** {preview}", "",
                ]
        text = "\n".join(lines) + "\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] Report written to {output_path}")
        return text

    # -- Validation (no agent needed) -------------------------------------

    def validate_eval_set(self) -> List[str]:
        """Check the eval set for structural issues. Empty list = all good."""
        issues: List[str] = []
        seen: set = set()
        required = {"id", "category", "question", "expected_keywords", "scoring"}
        for idx, q in enumerate(self.questions):
            for fld in required:
                if fld not in q:
                    issues.append(f"index {idx}: missing '{fld}'")
            qid = q.get("id", f"MISSING_{idx}")
            if qid in seen:
                issues.append(f"index {idx}: duplicate id '{qid}'")
            seen.add(qid)
            wt = sum(q.get("scoring", {}).values())
            if q.get("scoring") and abs(wt - 1.0) > 0.01:
                issues.append(f"{qid}: scoring weights sum to {wt:.2f}, expected 1.0")
            if not q.get("expected_keywords"):
                issues.append(f"{qid}: no expected_keywords")
        return issues


# -- CLI entry point ------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="JCoder Agent Eval Runner")
    ap.add_argument("--eval-set", default=str(Path(__file__).parent / "agent_eval_set.json"))
    ap.add_argument("--output-dir", default="evaluation/results")
    ap.add_argument("--validate-only", action="store_true", help="Just validate the eval set")
    ap.add_argument("--category", action="append", help="Filter by category (repeatable)")
    ap.add_argument("--max", type=int, default=0, help="Max questions to run")
    ap.add_argument("--question", type=str, default="", help="Run single question by ID")
    ap.add_argument("--no-resume", action="store_true", help="Discard previous results")
    args = ap.parse_args()
    runner = AgentEvalRunner(args.eval_set, agent=None, output_dir=args.output_dir)
    if args.validate_only:
        issues = runner.validate_eval_set()
        if issues:
            print(f"[WARN] {len(issues)} issue(s):")
            for i in issues:
                print(f"  - {i}")
        else:
            print(f"[OK] Eval set valid: {len(runner.questions)} questions, no issues")
        return
    if args.question:
        print(json.dumps(asdict(runner.run_single(args.question)), indent=2))
        return
    print(f"Running eval set: {len(runner.questions)} questions")
    if runner.agent is None:
        print("[WARN] No agent configured -- scores will be 0.0")
    results = runner.run_all(categories=args.category, max_questions=args.max,
                             resume=not args.no_resume)
    runner.report(results)


if __name__ == "__main__":
    main()
