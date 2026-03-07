import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class ProceduralExperience:
    state_hash: str
    action: str
    outcome: str
    success: bool
    metadata: Dict[str, object] = field(default_factory=dict)


class ProceduralMemory:
    """Stores action/outcome pairs keyed by state hash for PRAXIS-style recall."""

    def __init__(self, db_path: str = "_procedural_memory/memory.db"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._path))

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    state_hash TEXT,
                    action TEXT,
                    outcome TEXT,
                    success INTEGER,
                    metadata_json TEXT,
                    timestamp REAL DEFAULT (strftime('%s','now')),
                    PRIMARY KEY (state_hash, action)
                )
                """
            )
            conn.commit()

    def store(self, experience: ProceduralExperience) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiences
                (state_hash, action, outcome, success, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experience.state_hash,
                    experience.action,
                    experience.outcome,
                    1 if experience.success else 0,
                    json.dumps(experience.metadata),
                ),
            )
            conn.commit()

    def recall(self, state_hash: str, max_results: int = 5) -> List[ProceduralExperience]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT action, outcome, success, metadata_json
                FROM experiences
                WHERE state_hash = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (state_hash, max_results),
            )
            rows = cur.fetchall()
        results: List[ProceduralExperience] = []
        for row in rows:
            metadata = json.loads(row[3]) if row[3] else {}
            results.append(
                ProceduralExperience(
                    state_hash=state_hash,
                    action=row[0],
                    outcome=row[1],
                    success=bool(row[2]),
                    metadata=metadata,
                )
            )
        return results
