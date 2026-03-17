"""
Adversarial Self-Play Training Runner (Sprint 11)
--------------------------------------------------
Runs adversarial self-play games to discover system weaknesses:
  Game 1: Hardness escalation (progressively harder code questions)
  Game 2: Trick question detection (unanswerable/trap questions)
  Game 3: Ambiguity challenge (require clarification)

Failures feed back into experience replay + prompt evolution.

Usage:
    python scripts/adversarial_training.py                       # defaults
    python scripts/adversarial_training.py --rounds 10           # more rounds per game
    python scripts/adversarial_training.py --dry-run              # preview config
    python scripts/adversarial_training.py --game hardness        # single game only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _fix_stdout():
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_code_chunks(
    index_dir: str = "data",
    max_chunks: int = 100,
) -> List[Dict[str, str]]:
    """Load code chunks from FTS5 indexes for adversarial question generation."""
    import sqlite3
    chunks: List[Dict[str, str]] = []
    data_dir = Path(index_dir)

    if not data_dir.exists():
        return []

    for db_file in sorted(data_dir.glob("*.fts5.db"))[:5]:
        try:
            conn = sqlite3.connect(str(db_file))
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            for table in tables:
                if table.endswith("_content") or table.startswith("sqlite_"):
                    continue
                try:
                    cur = conn.execute(
                        f"SELECT content FROM [{table}] LIMIT ?",
                        (max_chunks // max(1, len(tables)),),
                    )
                    for row in cur.fetchall():
                        if row[0] and len(row[0]) > 50:
                            chunks.append({
                                "content": row[0][:1500],
                                "source_path": f"{db_file.name}/{table}",
                            })
                except Exception:
                    continue
            conn.close()
        except Exception:
            continue

    return chunks[:max_chunks]


def run_adversarial_training(
    rounds_per_game: int = 5,
    difficulty_start: float = 0.3,
    game_filter: Optional[str] = None,
    output_dir: str = "logs/adversarial_training",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run one adversarial self-play session."""
    from core.runtime import Runtime
    from core.config import load_config
    from core.adversarial_self_play import AdversarialSelfPlay

    config = load_config()
    chunks = _load_code_chunks()

    if not chunks:
        # Fallback: generate synthetic code chunks
        chunks = [
            {
                "content": (
                    "def process_items(items: list) -> list:\n"
                    "    result = []\n"
                    "    for item in items:\n"
                    "        if item.is_valid():\n"
                    "            result.append(item.transform())\n"
                    "    return result\n"
                ),
                "source_path": "core/processor.py",
            },
            {
                "content": (
                    "class DatabaseConnection:\n"
                    "    def __init__(self, url: str):\n"
                    "        self._url = url\n"
                    "        self._conn = None\n"
                    "    def connect(self):\n"
                    "        import sqlite3\n"
                    "        self._conn = sqlite3.connect(self._url)\n"
                    "    def close(self):\n"
                    "        if self._conn:\n"
                    "            self._conn.close()\n"
                ),
                "source_path": "core/database.py",
            },
        ]

    print(f"[OK] Loaded {len(chunks)} code chunks for adversarial generation")
    print(f"[OK] Rounds per game: {rounds_per_game}")
    print(f"[OK] Difficulty start: {difficulty_start}")
    if game_filter:
        print(f"[OK] Game filter: {game_filter}")

    if dry_run:
        print("[OK] Dry run -- exiting before self-play.")
        return {"dry_run": True, "chunks": len(chunks)}

    runtime = Runtime(config.model, timeout=120)

    # Build answer function using the runtime as a simple RAG proxy
    def answer_fn(question: str) -> str:
        return runtime.generate(
            question=question,
            context_chunks=[c["content"][:500] for c in chunks[:5]],
            temperature=0.1,
            max_tokens=256,
        )

    self_play = AdversarialSelfPlay(
        runtime=runtime,
        answer_fn=answer_fn,
        db_path="_self_play/games.db",
    )

    print(f"[OK] Starting adversarial session at {_timestamp()}")
    t0 = time.time()

    result = self_play.play_session(
        code_chunks=chunks,
        rounds_per_game=rounds_per_game,
        difficulty_start=difficulty_start,
    )

    elapsed = time.time() - t0

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": _timestamp(),
        "total_challenges": result.total_challenges,
        "correct": result.correct,
        "accuracy": round(result.accuracy, 3),
        "weakness_report": result.weakness_report,
        "hardest_failures": result.hardest_failures,
        "failed_rounds": result.failed_rounds,
        "elapsed_s": round(elapsed, 1),
    }
    report_path = out_dir / f"adversarial_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Feed failures into experience replay if available
    failure_count = _feed_failures_to_experience(result.hardest_failures)

    print(f"[OK] Adversarial session complete in {elapsed:.1f}s")
    print(f"[OK] Challenges: {result.total_challenges}, "
          f"Correct: {result.correct}, "
          f"Accuracy: {result.accuracy:.1%}")
    if result.weakness_report:
        print("[OK] Weakness report:")
        for mode, count in result.weakness_report.items():
            print(f"  {mode}: {count}")
    if result.hardest_failures:
        print(f"[OK] Top {len(result.hardest_failures)} hardest failures saved")
    if failure_count:
        print(f"[OK] Fed {failure_count} failures to experience replay")
    print(f"[OK] Report: {report_path}")

    # Historical weakness analysis
    analysis = self_play.weakness_analysis()
    if analysis.get("weaknesses"):
        print("[OK] Historical weaknesses:")
        for w in analysis["weaknesses"][:5]:
            print(f"  {w['game']}/{w['failure_mode']}: "
                  f"{w['count']} occurrences (avg conf: {w['avg_confidence']:.2f})")

    self_play.close()
    return report


def _feed_failures_to_experience(
    failures: List[Dict[str, str]],
) -> int:
    """Store adversarial failures in experience replay for future retrieval."""
    try:
        from core.experience_replay import ExperienceStore
        store = ExperienceStore()
        count = 0
        for f in failures:
            cid = f.get("challenge_id", "")
            snippet = f.get("answer_snippet", "")
            mode = f.get("failure_mode", "unknown")
            if cid and snippet:
                store.store(
                    exp_id=f"adversarial_{cid}",
                    query=f"[ADVERSARIAL:{mode}] {snippet[:100]}",
                    answer=snippet,
                    source_files=[],
                    confidence=0.1,
                )
                count += 1
        return count
    except Exception:
        return 0


def main():
    _fix_stdout()
    parser = argparse.ArgumentParser(description="Adversarial self-play training")
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="Rounds per game type (total challenges = rounds * 3)")
    parser.add_argument(
        "--difficulty", type=float, default=0.3,
        help="Starting difficulty (0.0-1.0)")
    parser.add_argument(
        "--game", default=None, choices=["hardness", "trick", "ambiguity"],
        help="Run only one game type")
    parser.add_argument(
        "--output-dir", default="logs/adversarial_training",
        help="Output directory for reports")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview config without running")
    args = parser.parse_args()

    run_adversarial_training(
        rounds_per_game=args.rounds,
        difficulty_start=args.difficulty,
        game_filter=args.game,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
