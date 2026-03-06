"""Structured JSONL logger for agent actions -- thread-safe, crash-safe, rotated."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AgentLogEntry:
    """Single structured log record."""
    timestamp: str
    session_id: str
    event_type: str
    data: Dict[str, Any]
    elapsed_s: float = 0.0
    tokens: int = 0


class AgentLogger:
    """Structured JSON-lines logger for agent actions."""

    def __init__(
        self,
        log_dir: str = "logs/agent",
        max_file_size_mb: int = 50,
    ):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_file_size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._current_fh: Optional[Any] = None
        self._current_path: Optional[Path] = None

    def log(
        self,
        session_id: str,
        event_type: str,
        data: Dict[str, Any],
        elapsed_s: float = 0.0,
        tokens: int = 0,
    ) -> None:
        """Write a single log entry. Thread-safe, flushed immediately."""
        entry = AgentLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            event_type=event_type,
            data=data,
            elapsed_s=round(elapsed_s, 4),
            tokens=tokens,
        )
        line = json.dumps(asdict(entry), default=str) + "\n"

        with self._lock:
            fh = self._get_file_handle()
            fh.write(line)
            fh.flush()


    def log_task_start(self, session_id: str, task: str) -> None:
        self.log(session_id, "task_start", {"task": task})

    def log_tool_call(self, session_id: str, tool_name: str,
                      tool_args: Dict[str, Any], iteration: int) -> None:
        self.log(session_id, "tool_call", {
            "tool": tool_name,
            "args": _truncate(json.dumps(tool_args, default=str), 500),
            "iteration": iteration,
        })

    def log_tool_result(self, session_id: str, tool_name: str,
                        success: bool, output_preview: str,
                        elapsed_s: float) -> None:
        self.log(session_id, "tool_result", {
            "tool": tool_name, "success": success,
            "output": _truncate(output_preview, 500),
        }, elapsed_s=elapsed_s)

    def log_llm_call(self, session_id: str, model: str,
                     input_tokens: int, output_tokens: int,
                     elapsed_s: float) -> None:
        self.log(session_id, "llm_call", {
            "model": model, "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }, elapsed_s=elapsed_s, tokens=input_tokens + output_tokens)

    def log_task_complete(self, session_id: str, success: bool,
                          summary: str, total_tokens: int,
                          iterations: int) -> None:
        self.log(session_id, "task_complete", {
            "success": success, "summary": _truncate(summary, 500),
            "iterations": iterations,
        }, tokens=total_tokens)

    def log_error(self, session_id: str, error: str,
                  context: str = "") -> None:
        self.log(session_id, "error", {
            "error": _truncate(error, 500),
            "context": _truncate(context, 500) if context else "",
        })


    def query(self, session_id: str = "", event_type: str = "",
              since: str = "", limit: int = 100) -> List[AgentLogEntry]:
        """Read and filter log entries from JSONL files."""
        results: List[AgentLogEntry] = []
        for path in sorted(self._log_dir.glob("agent_*.jsonl")):
            if len(results) >= limit:
                break
            for raw in _read_jsonl(path):
                if session_id and raw.get("session_id") != session_id:
                    continue
                if event_type and raw.get("event_type") != event_type:
                    continue
                if since and raw.get("timestamp", "") < since:
                    continue
                results.append(AgentLogEntry(
                    timestamp=raw.get("timestamp", ""),
                    session_id=raw.get("session_id", ""),
                    event_type=raw.get("event_type", ""),
                    data=raw.get("data", {}),
                    elapsed_s=raw.get("elapsed_s", 0.0),
                    tokens=raw.get("tokens", 0),
                ))
                if len(results) >= limit:
                    break
        return results

    def session_summary(self, session_id: str) -> Dict[str, Any]:
        """Summarize a session: tools called, tokens, duration, success."""
        entries = self.query(session_id=session_id, limit=10_000)
        if not entries:
            return {"session_id": session_id, "found": False}

        tool_calls = [e for e in entries if e.event_type == "tool_call"]
        llm_calls = [e for e in entries if e.event_type == "llm_call"]
        completes = [e for e in entries if e.event_type == "task_complete"]
        errors = [e for e in entries if e.event_type == "error"]

        total_tokens = sum(e.tokens for e in entries)
        total_elapsed = sum(e.elapsed_s for e in entries)
        success = completes[-1].data.get("success", False) if completes else False

        return {
            "session_id": session_id,
            "found": True,
            "tool_calls": len(tool_calls),
            "llm_calls": len(llm_calls),
            "errors": len(errors),
            "total_tokens": total_tokens,
            "total_elapsed_s": round(total_elapsed, 2),
            "success": success,
        }

    def daily_stats(self, date: str = "") -> Dict[str, Any]:
        """Daily aggregate stats. date format: YYYY-MM-DD (default: today)."""
        if not date:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_prefix = date.replace("-", "")
        target = self._log_dir / f"agent_{day_prefix}.jsonl"

        sessions: set[str] = set()
        total_tokens = 0
        tool_calls = 0
        completions = 0
        successes = 0

        if target.exists():
            for raw in _read_jsonl(target):
                sessions.add(raw.get("session_id", ""))
                total_tokens += raw.get("tokens", 0)
                if raw.get("event_type") == "tool_call":
                    tool_calls += 1
                if raw.get("event_type") == "task_complete":
                    completions += 1
                    if raw.get("data", {}).get("success"):
                        successes += 1

        return {
            "date": date,
            "sessions": len(sessions),
            "total_tokens": total_tokens,
            "tool_calls": tool_calls,
            "completions": completions,
            "success_rate": round(successes / completions, 3) if completions else 0.0,
        }


    def _get_file_handle(self):
        """Return (or rotate to) the correct daily log file. Caller holds lock."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self._log_dir / f"agent_{today}.jsonl"

        needs_new = (
            self._current_fh is None
            or self._current_path != path
            or (path.exists() and path.stat().st_size >= self._max_bytes)
        )

        if needs_new:
            if self._current_fh is not None:
                self._current_fh.close()
            # Handle rotation: add suffix if file exceeds max size
            if path.exists() and path.stat().st_size >= self._max_bytes:
                n = 1
                while True:
                    rotated = self._log_dir / f"agent_{today}.{n}.jsonl"
                    if not rotated.exists():
                        path = rotated
                        break
                    n += 1
            self._current_fh = open(path, "a", encoding="utf-8")
            self._current_path = path

        return self._current_fh

    def close(self) -> None:
        """Flush and close the current file handle."""
        with self._lock:
            if self._current_fh is not None:
                self._current_fh.close()
                self._current_fh = None



def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines."""
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return records
