"""
Goal Queue & Self-Study Engine
-------------------------------
Enables autonomous learning: the agent queues goals, decomposes them
into subtasks via the LLM, and works through them without hand-holding.

Goals persist to JSON so they survive restarts.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

log = logging.getLogger(__name__)

# -- Status constants ------------------------------------------------------

PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
FAILED = "failed"


# -- Data model ------------------------------------------------------------

@dataclass
class Goal:
    """A single learning or work goal."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""
    status: str = PENDING
    priority: int = 5
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    completed_at: Optional[str] = None
    result_summary: str = ""
    subtasks: List[str] = field(default_factory=list)
    tokens_used: int = 0
    parent_id: Optional[str] = None


# -- Persistent queue ------------------------------------------------------

class GoalQueue:
    """
    Priority queue of goals backed by a JSON file.

    Parameters
    ----------
    persist_path : str
        Path to the JSON file. Created automatically if missing.
    """

    def __init__(self, persist_path: str = "data/agent_goals.json"):
        self._path = Path(persist_path)
        self._goals: List[Goal] = []
        self._load()

    def add(self, title: str, description: str, priority: int = 5,
            parent_id: Optional[str] = None) -> Goal:
        """Create a goal, persist, and return it."""
        goal = Goal(
            title=title,
            description=description,
            priority=priority,
            parent_id=parent_id,
        )
        self._goals.append(goal)
        self._persist()
        log.info("Goal added: [%s] %s (pri=%d)", goal.id, title, priority)
        return goal

    def next(self) -> Optional[Goal]:
        """Return the highest-priority pending goal (lowest number wins)."""
        pending = [g for g in self._goals if g.status == PENDING]
        if not pending:
            return None
        return min(pending, key=lambda g: g.priority)

    def get(self, goal_id: str) -> Optional[Goal]:
        """Look up a goal by ID."""
        for g in self._goals:
            if g.id == goal_id:
                return g
        return None

    def complete(self, goal_id: str, summary: str, tokens_used: int = 0):
        """Mark a goal completed with a summary."""
        goal = self.get(goal_id)
        if goal is None:
            log.warning("complete() called on unknown goal %s", goal_id)
            return
        goal.status = COMPLETED
        goal.result_summary = summary
        goal.tokens_used = tokens_used
        goal.completed_at = datetime.now(timezone.utc).isoformat()
        self._persist()
        log.info("Goal completed: [%s] %s", goal_id, summary[:80])

    def fail(self, goal_id: str, reason: str):
        """Mark a goal failed."""
        goal = self.get(goal_id)
        if goal is None:
            log.warning("fail() called on unknown goal %s", goal_id)
            return
        goal.status = FAILED
        goal.result_summary = reason
        goal.completed_at = datetime.now(timezone.utc).isoformat()
        self._persist()
        log.info("Goal failed: [%s] %s", goal_id, reason[:80])

    def list(self, status: Optional[str] = None) -> List[Goal]:
        """Return goals, optionally filtered by status."""
        if status is None:
            return list(self._goals)
        return [g for g in self._goals if g.status == status]

    def _persist(self):
        """Write all goals to the JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(g) for g in self._goals]
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self):
        """Read goals from the JSON file if it exists."""
        if not self._path.exists():
            self._goals = []
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._goals = [Goal(**entry) for entry in raw]
            log.info("Loaded %d goals from %s", len(self._goals), self._path)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            log.error("Corrupt goal file %s: %s", self._path, exc)
            self._goals = []


# -- Self-study engine -----------------------------------------------------

_DECOMPOSE_PROMPT = """\
Break the following goal into 3-6 concrete, actionable subtasks that a \
coding agent can execute sequentially. Each subtask should be a single \
clear instruction. Return ONLY a JSON array of strings, nothing else.

Goal: {title}
Description: {description}"""

_STUDY_SUBTASKS = [
    "Search the codebase for files related to: {topic}",
    "Read the most relevant files and understand the key concepts",
    "Summarize what was learned in 3-5 bullet points",
    "Write a knowledge note to data/agent_knowledge/{slug}.md",
]


class StudyEngine:
    """
    Autonomous study loop: decompose goals, run them through the agent.

    Parameters
    ----------
    agent : Any
        An Agent instance (agent.core.Agent) with a .run(task) method.
    goals : GoalQueue
        The persistent goal queue.
    llm_backend : Any
        An LLMBackend used for goal decomposition (can differ from agent's).
    """

    def __init__(self, agent: Any, goals: GoalQueue, llm_backend: Any):
        self._agent = agent
        self._goals = goals
        self._llm = llm_backend

    def decompose(self, goal: Goal) -> List[str]:
        """Use the LLM to split a goal into subtasks. Returns descriptions."""
        prompt = _DECOMPOSE_PROMPT.format(
            title=goal.title, description=goal.description,
        )
        resp = self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        try:
            subtask_texts = json.loads(resp.content)
            if not isinstance(subtask_texts, list):
                raise TypeError("LLM did not return a JSON array")
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("Decompose parse failed (%s), using raw split", exc)
            subtask_texts = [
                line.strip().lstrip("0123456789.-) ")
                for line in resp.content.splitlines()
                if line.strip()
            ]

        # Create child goals in the queue
        created = []
        for i, desc in enumerate(subtask_texts, 1):
            child = self._goals.add(
                title=f"{goal.title} -- step {i}",
                description=desc,
                priority=goal.priority,
                parent_id=goal.id,
            )
            created.append(desc)
            goal.subtasks.append(child.id)

        self._goals._persist()
        log.info("Decomposed [%s] into %d subtasks", goal.id, len(created))
        return created

    def study(self, topic: str):
        """Queue a structured self-study session on a topic."""
        slug = topic.lower().replace(" ", "_")[:40]
        parent = self._goals.add(
            title=f"Learn about {topic}",
            description=f"Self-study session: understand {topic} thoroughly.",
            priority=3,
        )
        for template in _STUDY_SUBTASKS:
            desc = template.format(topic=topic, slug=slug)
            child = self._goals.add(
                title=f"Study {topic} -- {desc[:50]}",
                description=desc,
                priority=3,
                parent_id=parent.id,
            )
            parent.subtasks.append(child.id)
        self._goals._persist()
        log.info("Queued study session: %s (%d subtasks)",
                 topic, len(parent.subtasks))

    def run_next(self) -> Optional[Any]:
        """Pick the next goal, decompose if needed, run the agent on it."""
        goal = self._goals.next()
        if goal is None:
            log.info("Goal queue empty")
            return None

        goal.status = IN_PROGRESS
        self._goals._persist()

        # Auto-decompose top-level goals that have no subtasks
        if not goal.subtasks and goal.parent_id is None:
            self.decompose(goal)
            # After decomposition, the parent is done -- subtasks carry the work
            self._goals.complete(
                goal.id,
                summary=f"Decomposed into {len(goal.subtasks)} subtasks",
            )
            # Now run the first subtask
            return self.run_next()

        # Execute via the agent
        log.info("Running goal [%s]: %s", goal.id, goal.title)
        try:
            result = self._agent.run(goal.description)
            tokens = result.total_input_tokens + result.total_output_tokens
            if result.success:
                self._goals.complete(goal.id, result.summary, tokens)
            else:
                self._goals.fail(goal.id, result.summary)
            return result
        except Exception as exc:
            self._goals.fail(goal.id, str(exc))
            log.error("Goal [%s] raised: %s", goal.id, exc)
            return None

    def run_loop(self, max_goals: int = 10,
                 on_complete: Optional[Callable] = None):
        """
        Keep running goals until the queue is empty or max_goals is hit.

        Parameters
        ----------
        max_goals : int
            Safety cap on how many goals to process in one loop.
        on_complete : callable, optional
            Called as on_complete(goal, result) after each goal finishes.
        """
        completed = 0
        while completed < max_goals:
            goal = self._goals.next()
            if goal is None:
                log.info("Study loop done: queue empty after %d goals",
                         completed)
                break

            result = self.run_next()
            completed += 1

            if on_complete is not None:
                finished = self._goals.get(goal.id)
                on_complete(finished, result)

        log.info("Study loop finished: %d goals processed", completed)
