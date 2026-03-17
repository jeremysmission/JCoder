"""
PRISMA Pipeline Tracker
-----------------------
Tracks papers through the systematic review pipeline:
Identification -> Screening -> Eligibility -> Inclusion

Based on: Moher et al. (2009) PRISMA Statement (60K+ citations).
Every paper is logged at each stage with a reason for advancement
or exclusion, enabling transparent and reproducible research.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PrismaRecord:
    """A single PRISMA pipeline event."""
    content_hash: str
    title: str
    stage: str  # "identified" | "screened" | "eligible" | "included" | "excluded"
    reason: str
    timestamp: float
    source: str


# Valid stage progression order (excluding "excluded" which is terminal)
_PIPELINE_STAGES = ("identified", "screened", "eligible", "included")
_ALL_STAGES = ("identified", "screened", "eligible", "included", "excluded")


class PrismaTracker:
    """PRISMA-compliant pipeline tracker with SQLite backend."""

    _initialized_paths: set[str] = set()
    _initialized_lock = threading.Lock()

    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_key = str(self._db_path.resolve())
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            timeout=30,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        # Buffered inserts dramatically reduce commit overhead under stress.
        self._pending_rows: List[Tuple[str, str, str, str, float, str]] = []
        # Defer flushes so hot-path logging avoids per-row commit overhead.
        self._flush_every = 20000
        self._closed = False
        self._ensure_db_ready()
        # In-memory title cache to avoid repeated lookups
        self._title_cache: Dict[str, Tuple[str, str]] = {}

    def _ensure_db_ready(self) -> None:
        """Initialize schema once per database path in the current process."""
        with self._initialized_lock:
            if self._db_key in self._initialized_paths:
                return
            self._init_db()
            self._initialized_paths.add(self._db_key)

    def _init_db(self) -> None:
        for attempt in range(20):
            try:
                with self._lock:
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("""
                        CREATE TABLE IF NOT EXISTS prisma_log (
                            content_hash TEXT,
                            title TEXT,
                            stage TEXT,
                            reason TEXT,
                            timestamp REAL,
                            source TEXT
                        )
                    """)
                    self._conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_prisma_hash
                        ON prisma_log(content_hash)
                    """)
                    self._conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_prisma_stage
                        ON prisma_log(stage)
                    """)
                    self._conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < 19:
                    time.sleep(0.02 * (attempt + 1))
                    continue
                raise

    def _flush_locked(self) -> None:
        """Flush buffered rows to SQLite. Caller must hold self._lock."""
        if not self._pending_rows or self._conn is None:
            return

        rows = list(self._pending_rows)
        for attempt in range(40):
            try:
                self._conn.executemany(
                    "INSERT INTO prisma_log "
                    "(content_hash, title, stage, reason, timestamp, source) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                self._conn.commit()
                self._pending_rows.clear()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < 39:
                    self._conn.rollback()
                    time.sleep(0.01 * (attempt + 1))
                    continue
                raise

    def _log(self, content_hash: str, title: str, stage: str,
             reason: str, source: str) -> None:
        """Log an event row and flush when the batch threshold is reached."""
        with self._lock:
            if self._closed:
                raise RuntimeError("PrismaTracker is closed")
            self._pending_rows.append(
                (content_hash, title, stage, reason, time.time(), source)
            )
            if len(self._pending_rows) >= self._flush_every:
                self._flush_locked()

    def _lookup_title(self, content_hash: str) -> Tuple[str, str]:
        """Return (title, source) for a known hash."""
        with self._lock:
            if content_hash in self._title_cache:
                return self._title_cache[content_hash]
            cur = self._conn.execute(
                "SELECT title, source FROM prisma_log "
                "WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            )
            row = cur.fetchone()
            if row is None:
                return ("", "")
            result = (row[0], row[1])
            self._title_cache[content_hash] = result
            return result

    # ------------------------------------------------------------------
    # Pipeline stage methods
    # ------------------------------------------------------------------

    def identify(self, title: str, source: str, content_hash: str) -> None:
        """Log a paper entering the pipeline."""
        with self._lock:
            self._title_cache[content_hash] = (title, source)
        self._log(content_hash, title, "identified", "initial discovery", source)

    def screen(self, content_hash: str, passed: bool, reason: str) -> None:
        """Screen a paper. If failed, automatically excludes."""
        title, source = self._lookup_title(content_hash)
        if passed:
            self._log(content_hash, title, "screened", reason, source)
        else:
            self.exclude(content_hash, "screened", reason)

    def eligible(self, content_hash: str, passed: bool, reason: str) -> None:
        """Check eligibility. If failed, automatically excludes."""
        title, source = self._lookup_title(content_hash)
        if passed:
            self._log(content_hash, title, "eligible", reason, source)
        else:
            self.exclude(content_hash, "eligible", reason)

    def include(self, content_hash: str, reason: str) -> None:
        """Mark a paper as included in the final review."""
        title, source = self._lookup_title(content_hash)
        self._log(content_hash, title, "included", reason, source)

    def exclude(self, content_hash: str, stage: str, reason: str) -> None:
        """Exclude a paper. The reason field records the stage of exclusion."""
        title, source = self._lookup_title(content_hash)
        self._log(content_hash, title, "excluded", f"{stage}: {reason}", source)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def flow_counts(self) -> Dict[str, int]:
        """Return counts per stage."""
        with self._lock:
            self._flush_locked()
            cur = self._conn.execute(
                "SELECT stage, COUNT(*) FROM prisma_log GROUP BY stage"
            )
            counts = {row[0]: row[1] for row in cur.fetchall()}
        return {stage: counts.get(stage, 0) for stage in _ALL_STAGES}

    def exclusion_reasons(self, stage: str) -> List[Tuple[str, int]]:
        """Grouped counts of exclusion reasons that mention a given stage."""
        with self._lock:
            self._flush_locked()
            cur = self._conn.execute(
                "SELECT reason, COUNT(*) FROM prisma_log "
                "WHERE stage = 'excluded' AND reason LIKE ? "
                "GROUP BY reason ORDER BY COUNT(*) DESC",
                (f"{stage}:%",),
            )
            return [(row[0], row[1]) for row in cur.fetchall()]

    def flow_diagram_text(self) -> str:
        """ASCII art PRISMA flow diagram with counts."""
        counts = self.flow_counts()

        with self._lock:
            self._flush_locked()
            cur = self._conn.execute(
                "SELECT reason, COUNT(*) FROM prisma_log "
                "WHERE stage = 'excluded' GROUP BY reason"
            )
            excl_rows = {row[0]: row[1] for row in cur.fetchall()}

        def _excl_at(stage: str) -> int:
            return sum(v for k, v in excl_rows.items() if k.startswith(f"{stage}:"))

        lines = [
            "PRISMA Flow Diagram",
            "===================",
            f"Identified:  {counts['identified']}",
            "    |",
            f"Screened:    {counts['screened']}  (excluded: {_excl_at('screened')})",
            "    |",
            f"Eligible:    {counts['eligible']}  (excluded: {_excl_at('eligible')})",
            "    |",
            f"Included:    {counts['included']}  (excluded: {_excl_at('included')})",
        ]
        return "\n".join(lines)

    def close(self) -> None:
        """Close the persistent database connection."""
        with self._lock:
            if self._conn:
                self._flush_locked()
                self._conn.close()
                self._conn = None
            self._closed = True

