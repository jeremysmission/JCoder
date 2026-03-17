"""
Multi-Agent Coordination (Sprint 15)
--------------------------------------
Spawn, dispatch, and coordinate multiple JCoder agents working
on decomposed subtasks of a larger goal.

Architecture:
  Coordinator  -- decomposes a task into subtasks via LLM
  AgentPool    -- manages a pool of Agent instances with shared resources
  SubTask      -- typed work unit (research / implement / verify)
  TaskLedger   -- SQLite audit trail for all subtask outcomes
  ArtifactBus  -- shared FTS5 memory for cross-agent knowledge handoff

Gate: 3 subagents complete a coordinated task
      (research + implement + verify).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.sqlite_owner import SQLiteConnectionOwner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class SubTaskType(str, Enum):
    RESEARCH = "research"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    GENERIC = "generic"


class SubTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTask:
    """A unit of work dispatched to a subagent."""
    task_id: str
    parent_id: str
    task_type: SubTaskType
    description: str
    depends_on: List[str] = field(default_factory=list)
    status: SubTaskStatus = SubTaskStatus.PENDING
    assigned_agent: str = ""
    result_summary: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    tokens_used: int = 0
    iterations: int = 0

    @property
    def is_done(self) -> bool:
        return self.status in (SubTaskStatus.COMPLETED, SubTaskStatus.FAILED)


@dataclass
class CoordinationResult:
    """Outcome of a full multi-agent coordination run."""
    parent_id: str
    success: bool
    summary: str
    subtasks: List[SubTask] = field(default_factory=list)
    total_tokens: int = 0
    total_elapsed_s: float = 0.0
    agents_used: int = 0


# ---------------------------------------------------------------------------
# Artifact Bus -- shared FTS5 knowledge store for cross-agent handoff
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Task Decomposer -- breaks a high-level task into typed subtasks
# ---------------------------------------------------------------------------

def decompose_task(
    task: str,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> List[SubTask]:
    """Decompose a task into research/implement/verify subtasks.

    If llm_fn is provided, uses the LLM to intelligently decompose.
    Otherwise falls back to a deterministic 3-phase decomposition.
    """
    parent_id = f"coord_{uuid.uuid4().hex[:12]}"

    if llm_fn:
        return _llm_decompose(task, parent_id, llm_fn)

    return _default_decompose(task, parent_id)


def _default_decompose(task: str, parent_id: str) -> List[SubTask]:
    """Deterministic 3-phase decomposition: research -> implement -> verify."""
    research = SubTask(
        task_id=f"{parent_id}_research",
        parent_id=parent_id,
        task_type=SubTaskType.RESEARCH,
        description=(
            f"Research the following task. Search the codebase and knowledge base "
            f"for relevant context, patterns, and dependencies. Produce a summary "
            f"of findings and a recommended approach.\n\nTask: {task}"
        ),
    )
    implement = SubTask(
        task_id=f"{parent_id}_implement",
        parent_id=parent_id,
        task_type=SubTaskType.IMPLEMENT,
        depends_on=[research.task_id],
        description=(
            f"Implement the following task based on research findings from the "
            f"research phase. Write clean, tested code. Publish any created or "
            f"modified files as artifacts.\n\nTask: {task}"
        ),
    )
    verify = SubTask(
        task_id=f"{parent_id}_verify",
        parent_id=parent_id,
        task_type=SubTaskType.VERIFY,
        depends_on=[implement.task_id],
        description=(
            f"Verify the implementation from the implement phase. Run tests, "
            f"check for edge cases, review code quality. Report pass/fail "
            f"with specific findings.\n\nTask: {task}"
        ),
    )
    return [research, implement, verify]


def _llm_decompose(
    task: str, parent_id: str, llm_fn: Callable[[str], str],
) -> List[SubTask]:
    """Use LLM to decompose task into subtasks."""
    prompt = (
        "Decompose the following task into subtasks. Each subtask must have:\n"
        "- type: one of 'research', 'implement', 'verify', 'generic'\n"
        "- description: what the subagent should do\n"
        "- depends_on: list of subtask indices (0-based) that must complete first\n\n"
        "Return ONLY valid JSON: [{\"type\": \"...\", \"description\": \"...\", "
        "\"depends_on\": [...]}]\n\n"
        f"Task: {task}"
    )

    try:
        raw = llm_fn(prompt)
        # Extract JSON from response (handle markdown code blocks)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        items = json.loads(raw.strip())
    except (json.JSONDecodeError, IndexError, KeyError):
        log.warning("LLM decomposition failed, using default 3-phase")
        return _default_decompose(task, parent_id)

    subtasks = []
    for i, item in enumerate(items):
        st = SubTask(
            task_id=f"{parent_id}_sub{i}",
            parent_id=parent_id,
            task_type=SubTaskType(item.get("type", "generic")),
            description=item.get("description", f"Subtask {i}"),
            depends_on=[
                f"{parent_id}_sub{d}"
                for d in item.get("depends_on", [])
                if isinstance(d, int) and 0 <= d < i
            ],
        )
        subtasks.append(st)

    if not subtasks:
        return _default_decompose(task, parent_id)

    return subtasks


# ---------------------------------------------------------------------------
# Agent Pool -- manages concurrent agent instances
# ---------------------------------------------------------------------------

class AgentPool:
    """Pool of agent instances for parallel subtask execution.

    Each agent runs in its own thread. The pool provides:
    - Spawn/retire lifecycle
    - Shared artifact bus for knowledge handoff
    - Dependency-aware scheduling
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any],
        max_agents: int = 3,
        artifact_bus: Optional[ArtifactBus] = None,
    ):
        self._factory = agent_factory
        self._max_agents = max_agents
        self._bus = artifact_bus
        self._executor = ThreadPoolExecutor(
            max_workers=max_agents,
            thread_name_prefix="jcoder-subagent",
        )
        self._active: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, subtask: SubTask) -> Future:
        """Submit a subtask for execution. Returns a Future."""
        future = self._executor.submit(self._run_subtask, subtask)
        with self._lock:
            self._active[subtask.task_id] = future
        return future

    def _run_subtask(self, subtask: SubTask) -> SubTask:
        """Execute a subtask with a fresh agent instance."""
        subtask.status = SubTaskStatus.RUNNING
        subtask.assigned_agent = threading.current_thread().name

        log.info(
            "Subagent %s starting: [%s] %s",
            subtask.assigned_agent, subtask.task_type.value,
            subtask.description[:80],
        )

        # Inject context from dependency artifacts
        context = self._gather_dependency_context(subtask)
        full_description = subtask.description
        if context:
            full_description = (
                f"## Context from prior phases\n\n{context}\n\n"
                f"## Your task\n\n{subtask.description}"
            )

        agent = self._factory()
        try:
            result = agent.run(full_description)
            subtask.result_summary = result.summary
            subtask.tokens_used = result.tokens
            subtask.iterations = result.iterations
            subtask.status = (
                SubTaskStatus.COMPLETED if result.success
                else SubTaskStatus.FAILED
            )
        except Exception as exc:
            log.error("Subagent %s failed: %s", subtask.task_id, exc)
            subtask.status = SubTaskStatus.FAILED
            subtask.result_summary = f"Agent error: {exc}"

        subtask.completed_at = time.time()

        # Release the slot as soon as agent execution finishes so active_count
        # reflects live agent work rather than bookkeeping tail work.
        with self._lock:
            self._active.pop(subtask.task_id, None)

        # Publish result as artifact
        if self._bus:
            self._bus.publish(
                task_id=subtask.task_id,
                parent_id=subtask.parent_id,
                artifact_type=f"{subtask.task_type.value}_result",
                content=subtask.result_summary,
                metadata={
                    "status": subtask.status.value,
                    "tokens": subtask.tokens_used,
                    "iterations": subtask.iterations,
                },
            )
            self._bus.log_task(subtask)

        log.info(
            "Subagent %s finished: %s (%d tokens, %d iterations)",
            subtask.task_id, subtask.status.value,
            subtask.tokens_used, subtask.iterations,
        )
        return subtask

    def _gather_dependency_context(self, subtask: SubTask) -> str:
        """Collect artifacts from completed dependencies."""
        if not self._bus or not subtask.depends_on:
            return ""

        parts = []
        for dep_id in subtask.depends_on:
            artifacts = self._bus.get_artifacts(
                parent_id=subtask.parent_id,
                task_id=dep_id,
            )
            for art in artifacts:
                parts.append(
                    f"### {art['artifact_type']} (from {dep_id})\n\n"
                    f"{art['content'][:4000]}"
                )

        return "\n\n".join(parts)

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the thread pool."""
        self._executor.shutdown(wait=wait)

    @property
    def active_count(self) -> int:
        with self._lock:
            done_task_ids = [
                task_id
                for task_id, future in self._active.items()
                if future.done()
            ]
            for task_id in done_task_ids:
                self._active.pop(task_id, None)
            return len(self._active)


# ---------------------------------------------------------------------------
# Coordinator -- orchestrates the full multi-agent workflow
# ---------------------------------------------------------------------------

class Coordinator:
    """Orchestrate a multi-agent workflow: decompose -> dispatch -> collect.

    Usage:
        coordinator = Coordinator(agent_factory=my_factory)
        result = coordinator.run("Build a sorting algorithm benchmark")
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any],
        max_agents: int = 3,
        bus_db_path: str | Path = "_coordination/artifact_bus.db",
        llm_decompose_fn: Optional[Callable[[str], str]] = None,
    ):
        self._factory = agent_factory
        self._max_agents = max_agents
        self._bus = ArtifactBus(bus_db_path)
        self._pool = AgentPool(
            agent_factory=agent_factory,
            max_agents=max_agents,
            artifact_bus=self._bus,
        )
        self._llm_decompose = llm_decompose_fn

    def run(self, task: str) -> CoordinationResult:
        """Decompose task, dispatch to subagents, collect results."""
        t_start = time.monotonic()

        # Phase 1: Decompose
        subtasks = decompose_task(task, self._llm_decompose)
        parent_id = subtasks[0].parent_id if subtasks else f"coord_{uuid.uuid4().hex[:12]}"

        log.info(
            "Coordinator decomposed task into %d subtasks (parent=%s)",
            len(subtasks), parent_id,
        )

        # Phase 2: Dispatch with dependency ordering
        completed: Dict[str, SubTask] = {}
        futures: Dict[str, Future] = {}

        for subtask in subtasks:
            # Wait for dependencies
            for dep_id in subtask.depends_on:
                if dep_id in futures:
                    dep_result = futures[dep_id].result(timeout=600)
                    completed[dep_id] = dep_result

            # Submit this subtask
            future = self._pool.submit(subtask)
            futures[subtask.task_id] = future

        # Phase 3: Collect all results
        for task_id, future in futures.items():
            if task_id not in completed:
                try:
                    completed[task_id] = future.result(timeout=600)
                except Exception as exc:
                    log.error("Subtask %s failed to collect: %s", task_id, exc)
                    # Find the subtask and mark failed
                    for st in subtasks:
                        if st.task_id == task_id and not st.is_done:
                            st.status = SubTaskStatus.FAILED
                            st.result_summary = f"Collection error: {exc}"
                            completed[task_id] = st

        # Phase 4: Build result
        all_done = [completed.get(st.task_id, st) for st in subtasks]
        all_success = all(st.status == SubTaskStatus.COMPLETED for st in all_done)
        total_tokens = sum(st.tokens_used for st in all_done)
        agents_used = len({st.assigned_agent for st in all_done if st.assigned_agent})

        summary_parts = []
        for st in all_done:
            summary_parts.append(
                f"[{st.task_type.value}] {st.status.value}: "
                f"{st.result_summary[:200]}"
            )

        result = CoordinationResult(
            parent_id=parent_id,
            success=all_success,
            summary="\n".join(summary_parts),
            subtasks=all_done,
            total_tokens=total_tokens,
            total_elapsed_s=time.monotonic() - t_start,
            agents_used=agents_used,
        )

        log.info(
            "Coordination complete: success=%s, subtasks=%d, tokens=%d, elapsed=%.1fs",
            result.success, len(all_done), total_tokens, result.total_elapsed_s,
        )
        return result

    def close(self) -> None:
        """Clean up resources."""
        self._pool.shutdown(wait=False)
        self._bus.close()
