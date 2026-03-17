"""Ingest the latest reviewed weekly subject summary into agent_memory."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory import AgentMemory
from core.weekly_subjects import (
    already_ingested,
    build_memory_payloads,
    file_sha256,
    find_latest_summary,
    load_state,
    save_state,
)

STATE_PATH = REPO_ROOT / "logs" / "weekly_subject_updates" / "state.json"
RUN_LOG_DIR = REPO_ROOT / "logs" / "weekly_subject_updates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest the latest reviewed weekly subject summary into agent memory."
    )
    parser.add_argument(
        "--summary",
        help="Specific summary markdown file to ingest.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Resolve and ingest the newest summary file in docs/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ingest even if the same summary hash was recorded before.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without writing memory or state.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = resolve_summary_path(args)
    payloads = build_memory_payloads(summary_path)
    summary_hash = file_sha256(summary_path)
    state = load_state(STATE_PATH)

    if already_ingested(state, summary_hash) and not args.force:
        print(f"[SKIP] Summary already ingested: {summary_path}")
        return 0

    if args.dry_run:
        print(f"[DRY RUN] {summary_path}")
        for payload in payloads:
            print(f"  - {payload['title']} ({', '.join(payload['tags'])})")
        return 0

    memory_config = load_memory_config(REPO_ROOT / "config" / "memory.yaml")
    ingested = ingest_payloads(payloads, memory_config)

    run_record = {
        "ran_at": datetime.now().astimezone().isoformat(),
        "summary_path": str(summary_path),
        "summary_sha256": summary_hash,
        "subjects": [payload["title"] for payload in payloads],
        "entry_ids": [item["entry_id"] for item in ingested],
    }
    state.setdefault("runs", []).append(run_record)
    save_state(STATE_PATH, state)
    write_run_log(run_record)

    print(
        f"[OK] Ingested {len(ingested)} weekly subject summaries from {summary_path.name}"
    )
    for item in ingested:
        print(f"  - {item['title']} -> {item['entry_id']}")
    return 0


def resolve_summary_path(args: argparse.Namespace) -> Path:
    if args.summary:
        return Path(args.summary).resolve()
    docs_dir = REPO_ROOT / "docs"
    return find_latest_summary(docs_dir)


def load_memory_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    memory_cfg = raw.get("memory", {})
    return {
        "index_name": memory_cfg.get("index_name", "agent_memory"),
        "index_dir": _resolve_repo_path(memory_cfg.get("index_dir", "data/indexes")),
        "knowledge_dir": _resolve_repo_path(
            memory_cfg.get("knowledge_dir", "data/agent_knowledge")
        ),
        "dimension": int(memory_cfg.get("dimension", 768)),
    }


def ingest_payloads(payloads: list[dict], memory_config: dict) -> list[dict]:
    memory = AgentMemory(
        embedding_engine=None,
        index_dir=str(memory_config["index_dir"]),
        index_name=memory_config["index_name"],
        dimension=memory_config["dimension"],
        knowledge_dir=str(memory_config["knowledge_dir"]),
    )
    try:
        results = []
        for payload in payloads:
            entry = memory.ingest(
                content=payload["content"],
                source_task=payload["source_task"],
                tags=payload["tags"],
                confidence=payload["confidence"],
                tokens_used=payload["tokens_used"],
            )
            results.append({"title": payload["title"], "entry_id": entry.id})
        return results
    finally:
        memory.close()


def write_run_log(run_record: dict) -> None:
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RUN_LOG_DIR / f"weekly_subject_update_{stamp}.json"
    path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")


def optimize_fts5_indexes(index_dir: Path | str, max_indexes: int = 10) -> int:
    """Run FTS5 OPTIMIZE on a bounded number of local index databases."""
    root = Path(index_dir)
    if not root.exists() or not root.is_dir():
        return 0

    optimized = 0
    for db_path in sorted(root.glob("*.fts5.db"))[:max_indexes]:
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("INSERT INTO chunks(chunks) VALUES ('optimize')")
                conn.commit()
                optimized += 1
            finally:
                conn.close()
        except sqlite3.Error:
            continue
    return optimized


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
