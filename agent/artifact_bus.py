"""
Artifact Bus -- shared FTS5 knowledge store for cross-agent handoff
--------------------------------------------------------------------
Agents publish artifacts (code, research findings, test results)
and other agents can query them by task_id or content search.

Extracted from agent/multi_agent.py to keep modules under 500 LOC.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.sqlite_owner import SQLiteConnectionOwner

if TYPE_CHECKING:
    from agent.multi_agent import SubTask

log = logging.getLogger(__name__)


_ARTIFACT_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS task_log (
    task_id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    result_summary TEXT DEFAULT '',
    tokens_used INTEGER DEFAULT 0,
    iterations INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    completed_at REAL DEFAULT 0
);
"""


class ArtifactBus:
    """Shared knowledge store for cross-agent artifact handoff.

    Agents publish artifacts (code, research findings, test results)
    and other agents can query them by task_id or content search.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = SQLiteConnectionOwner(self._db_path)
        conn = self._owner.connect()
        conn.executescript(_ARTIFACT_SCHEMA)
        conn.commit()

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._owner.connect()

    def publish(
        self,
        task_id: str,
        parent_id: str,
        artifact_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Publish an artifact to the bus. Returns artifact_id."""
        aid = f"art_{uuid.uuid4().hex[:12]}"
        conn = self._conn
        conn.execute(
            "INSERT INTO artifacts "
            "(artifact_id, task_id, parent_id, artifact_type, content, "
            "metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                aid, task_id, parent_id, artifact_type, content,
                json.dumps(metadata or {}, default=str), time.time(),
            ),
        )
        conn.commit()
        log.debug("Published artifact %s (type=%s, task=%s)", aid, artifact_type, task_id)
        return aid

    def get_artifacts(
        self,
        parent_id: str,
        task_id: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve artifacts for a coordination run."""
        query = "SELECT * FROM artifacts WHERE parent_id=?"
        params: list = [parent_id]
        if task_id:
            query += " AND task_id=?"
            params.append(task_id)
        if artifact_type:
            query += " AND artifact_type=?"
            params.append(artifact_type)
        query += " ORDER BY created_at LIMIT 1000"

        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "artifact_id": r[0], "task_id": r[1], "parent_id": r[2],
                "artifact_type": r[3], "content": r[4],
                "metadata": json.loads(r[5] or "{}"), "created_at": r[6],
            }
            for r in rows
        ]

    def search_content(self, parent_id: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search artifact content by substring match."""
        rows = self._conn.execute(
            "SELECT * FROM artifacts WHERE parent_id=? AND content LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (parent_id, f"%{query}%", limit),
        ).fetchall()
        return [
            {
                "artifact_id": r[0], "task_id": r[1], "parent_id": r[2],
                "artifact_type": r[3], "content": r[4],
                "metadata": json.loads(r[5] or "{}"), "created_at": r[6],
            }
            for r in rows
        ]

    def log_task(self, subtask: SubTask) -> None:
        """Record subtask outcome in the task log."""
        conn = self._conn
        conn.execute(
            "INSERT OR REPLACE INTO task_log "
            "(task_id, parent_id, task_type, description, status, "
            "result_summary, tokens_used, iterations, created_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                subtask.task_id, subtask.parent_id, subtask.task_type.value,
                subtask.description, subtask.status.value,
                subtask.result_summary, subtask.tokens_used,
                subtask.iterations, subtask.created_at, subtask.completed_at,
            ),
        )
        conn.commit()

    def get_task_log(self, parent_id: str) -> List[Dict[str, Any]]:
        """Get all logged subtasks for a coordination run."""
        rows = self._conn.execute(
            "SELECT * FROM task_log WHERE parent_id=? ORDER BY created_at LIMIT 500",
            (parent_id,),
        ).fetchall()
        return [
            {
                "task_id": r[0], "parent_id": r[1], "task_type": r[2],
                "description": r[3], "status": r[4], "result_summary": r[5],
                "tokens_used": r[6], "iterations": r[7],
                "created_at": r[8], "completed_at": r[9],
            }
            for r in rows
        ]

    def close(self) -> None:
        self._owner.close()
