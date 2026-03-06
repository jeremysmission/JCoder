"""
Session Persistence
-------------------
Save and resume agent conversation history across restarts.

Each session is stored as a single JSON file containing metadata
and the full OpenAI-format message history. Atomic writes (tmp +
rename) prevent corruption if the process crashes mid-save.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Lightweight metadata about a stored session."""

    session_id: str
    task: str
    created_at: str  # ISO-8601
    updated_at: str  # ISO-8601
    status: str  # "active", "completed", "failed"
    iterations: int
    total_tokens: int
    message_count: int


class SessionStore:
    """Persist and resume agent conversation sessions.

    Parameters
    ----------
    store_dir : str
        Directory where ``{session_id}.json`` files are written.
        Created on first use if it does not exist.
    """

    def __init__(self, store_dir: str = "data/agent_sessions"):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- Public API --------------------------------------------------------

    def save(
        self,
        session_id: str,
        task: str,
        history: List[Dict[str, Any]],
        status: str = "active",
        iterations: int = 0,
        tokens: int = 0,
    ) -> None:
        """Save (or overwrite) a session to disk.

        Uses atomic write: data goes to a ``.tmp`` file first, then is
        renamed over the target so a crash never leaves a half-written file.
        """
        now = _now_iso()
        target = self._path(session_id)

        # Preserve original created_at when updating an existing session
        created_at = now
        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
                created_at = existing.get("created_at", now)
            except (json.JSONDecodeError, OSError):
                pass

        payload = {
            "session_id": session_id,
            "task": task,
            "created_at": created_at,
            "updated_at": now,
            "status": status,
            "iterations": iterations,
            "total_tokens": tokens,
            "message_count": len(history),
            "history": history,
        }

        tmp = target.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, default=str),
                           encoding="utf-8")
            tmp.replace(target)
        except OSError:
            log.exception("Failed to save session %s", session_id)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    def load(self, session_id: str) -> Dict[str, Any]:
        """Load a full session (metadata + history) by ID.

        Returns
        -------
        dict
            Keys: session_id, task, history, status, iterations,
            total_tokens, created_at, updated_at, message_count.

        Raises
        ------
        FileNotFoundError
            If no session with that ID exists.
        """
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"No session found: {session_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return data

    def resume_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Load only the message history for resuming an agent run."""
        return self.load(session_id)["history"]

    def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[SessionInfo]:
        """List recent sessions sorted by *updated_at* descending.

        Parameters
        ----------
        status : str, optional
            If given, only sessions with this status are returned.
        limit : int
            Maximum number of results.
        """
        infos: List[SessionInfo] = []
        for fp in self._dir.glob("*.json"):
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                info = _info_from_dict(raw)
            except (json.JSONDecodeError, KeyError, OSError):
                continue
            if status and info.status != status:
                continue
            infos.append(info)

        infos.sort(key=lambda s: s.updated_at, reverse=True)
        return infos[:limit]

    def delete(self, session_id: str) -> bool:
        """Delete a session file. Returns *True* if it existed."""
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def cleanup(self, max_age_days: int = 30) -> int:
        """Delete sessions older than *max_age_days*. Returns count deleted."""
        cutoff = datetime.now(timezone.utc).timestamp() - (
            max_age_days * 86400
        )
        deleted = 0
        for fp in self._dir.glob("*.json"):
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                updated = datetime.fromisoformat(raw["updated_at"]).timestamp()
            except (json.JSONDecodeError, KeyError, ValueError, OSError):
                continue
            if updated < cutoff:
                fp.unlink(missing_ok=True)
                deleted += 1
        return deleted

    def search(self, query: str) -> List[SessionInfo]:
        """Substring search across session task descriptions."""
        query_lower = query.lower()
        results: List[SessionInfo] = []
        for fp in self._dir.glob("*.json"):
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                if query_lower in raw.get("task", "").lower():
                    results.append(_info_from_dict(raw))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        results.sort(key=lambda s: s.updated_at, reverse=True)
        return results

    # -- Helpers -----------------------------------------------------------

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _info_from_dict(raw: Dict[str, Any]) -> SessionInfo:
    return SessionInfo(
        session_id=raw["session_id"],
        task=raw["task"],
        created_at=raw["created_at"],
        updated_at=raw["updated_at"],
        status=raw["status"],
        iterations=raw.get("iterations", 0),
        total_tokens=raw.get("total_tokens", 0),
        message_count=raw.get("message_count", 0),
    )
