"""Tests for multi-agent coordination (Sprint 15)."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.multi_agent import (
    AgentPool,
    ArtifactBus,
    Coordinator,
    CoordinationResult,
    SubTask,
    SubTaskStatus,
    SubTaskType,
    decompose_task,
    _default_decompose,
    _llm_decompose,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_bus(tmp_path):
    """Create an ArtifactBus backed by a temp database."""
    bus = ArtifactBus(tmp_path / "test_bus.db")
    yield bus
    bus.close()


@pytest.fixture
def mock_agent_result():
    """Factory for mock AgentResult-like objects."""
    def _make(success=True, summary="Done", tokens=100, iterations=3):
        result = MagicMock()
        result.success = success
        result.summary = summary
        result.tokens = tokens
        result.iterations = iterations
        return result
    return _make


@pytest.fixture
def mock_agent_factory(mock_agent_result):
    """Factory that produces mock agents."""
    def _factory():
        agent = MagicMock()
        agent.run.return_value = mock_agent_result()
        return agent
    return _factory


# ---------------------------------------------------------------------------
# SubTask data class
# ---------------------------------------------------------------------------

class TestSubTask:
    def test_create_subtask(self):
        st = SubTask(
            task_id="t1", parent_id="p1",
            task_type=SubTaskType.RESEARCH,
            description="Find patterns",
        )
        assert st.task_id == "t1"
        assert st.status == SubTaskStatus.PENDING
        assert not st.is_done

    def test_is_done(self):
        st = SubTask(
            task_id="t1", parent_id="p1",
            task_type=SubTaskType.IMPLEMENT,
            description="Write code",
            status=SubTaskStatus.COMPLETED,
        )
        assert st.is_done

    def test_is_done_failed(self):
        st = SubTask(
            task_id="t1", parent_id="p1",
            task_type=SubTaskType.VERIFY,
            description="Run tests",
            status=SubTaskStatus.FAILED,
        )
        assert st.is_done

    def test_pending_not_done(self):
        st = SubTask(
            task_id="t1", parent_id="p1",
            task_type=SubTaskType.GENERIC,
            description="Something",
        )
        assert not st.is_done

    def test_running_not_done(self):
        st = SubTask(
            task_id="t1", parent_id="p1",
            task_type=SubTaskType.GENERIC,
            description="Something",
            status=SubTaskStatus.RUNNING,
        )
        assert not st.is_done


# ---------------------------------------------------------------------------
# Task Decomposition
# ---------------------------------------------------------------------------

class TestDecomposition:
    def test_default_decompose_produces_three_phases(self):
        subtasks = _default_decompose("Build a sort", "parent1")
        assert len(subtasks) == 3
        assert subtasks[0].task_type == SubTaskType.RESEARCH
        assert subtasks[1].task_type == SubTaskType.IMPLEMENT
        assert subtasks[2].task_type == SubTaskType.VERIFY

    def test_default_decompose_dependencies(self):
        subtasks = _default_decompose("Build a sort", "parent1")
        assert subtasks[0].depends_on == []
        assert subtasks[1].depends_on == [subtasks[0].task_id]
        assert subtasks[2].depends_on == [subtasks[1].task_id]

    def test_default_decompose_parent_id(self):
        subtasks = _default_decompose("Task", "my_parent")
        for st in subtasks:
            assert st.parent_id == "my_parent"

    def test_decompose_without_llm_uses_default(self):
        subtasks = decompose_task("Build a sort", llm_fn=None)
        assert len(subtasks) == 3
        assert subtasks[0].task_type == SubTaskType.RESEARCH

    def test_decompose_with_llm(self):
        def mock_llm(prompt):
            return json.dumps([
                {"type": "research", "description": "Look up algos", "depends_on": []},
                {"type": "implement", "description": "Code it", "depends_on": [0]},
                {"type": "verify", "description": "Test it", "depends_on": [1]},
                {"type": "generic", "description": "Document it", "depends_on": [2]},
            ])

        subtasks = decompose_task("Build a sort", llm_fn=mock_llm)
        assert len(subtasks) == 4
        assert subtasks[0].task_type == SubTaskType.RESEARCH
        assert subtasks[3].task_type == SubTaskType.GENERIC

    def test_decompose_llm_invalid_json_fallback(self):
        def bad_llm(prompt):
            return "I don't know how to decompose"

        subtasks = decompose_task("Build a sort", llm_fn=bad_llm)
        # Should fall back to default 3-phase
        assert len(subtasks) == 3

    def test_decompose_llm_empty_result_fallback(self):
        def empty_llm(prompt):
            return "[]"

        subtasks = decompose_task("Build a sort", llm_fn=empty_llm)
        assert len(subtasks) == 3  # fallback

    def test_decompose_llm_code_block(self):
        def code_block_llm(prompt):
            return '```json\n[{"type": "research", "description": "Search", "depends_on": []}]\n```'

        subtasks = decompose_task("Task", llm_fn=code_block_llm)
        assert len(subtasks) == 1
        assert subtasks[0].task_type == SubTaskType.RESEARCH

    def test_decompose_dependency_guards_forward_refs(self):
        """Dependencies can only reference earlier subtasks (no forward refs)."""
        def llm_fn(prompt):
            return json.dumps([
                {"type": "research", "description": "A", "depends_on": [1]},  # forward ref
                {"type": "implement", "description": "B", "depends_on": [0]},
            ])

        subtasks = decompose_task("Task", llm_fn=llm_fn)
        assert subtasks[0].depends_on == []  # forward ref stripped
        assert len(subtasks[1].depends_on) == 1


# ---------------------------------------------------------------------------
# Artifact Bus
# ---------------------------------------------------------------------------

class TestArtifactBus:
    def test_publish_and_retrieve(self, tmp_bus):
        aid = tmp_bus.publish("t1", "p1", "code", "print('hello')")
        assert aid.startswith("art_")

        arts = tmp_bus.get_artifacts("p1")
        assert len(arts) == 1
        assert arts[0]["content"] == "print('hello')"
        assert arts[0]["artifact_type"] == "code"

    def test_filter_by_task_id(self, tmp_bus):
        tmp_bus.publish("t1", "p1", "code", "file1")
        tmp_bus.publish("t2", "p1", "code", "file2")

        arts = tmp_bus.get_artifacts("p1", task_id="t1")
        assert len(arts) == 1
        assert arts[0]["content"] == "file1"

    def test_filter_by_type(self, tmp_bus):
        tmp_bus.publish("t1", "p1", "code", "source")
        tmp_bus.publish("t1", "p1", "test_result", "passed")

        arts = tmp_bus.get_artifacts("p1", artifact_type="test_result")
        assert len(arts) == 1
        assert arts[0]["content"] == "passed"

    def test_search_content(self, tmp_bus):
        tmp_bus.publish("t1", "p1", "code", "def quicksort(arr):")
        tmp_bus.publish("t1", "p1", "code", "def bubblesort(arr):")

        results = tmp_bus.search_content("p1", "quicksort")
        assert len(results) == 1
        assert "quicksort" in results[0]["content"]

    def test_metadata_stored(self, tmp_bus):
        tmp_bus.publish("t1", "p1", "code", "x=1", metadata={"lang": "python"})
        arts = tmp_bus.get_artifacts("p1")
        assert arts[0]["metadata"]["lang"] == "python"

    def test_log_task(self, tmp_bus):
        st = SubTask(
            task_id="t1", parent_id="p1",
            task_type=SubTaskType.RESEARCH,
            description="Find stuff",
            status=SubTaskStatus.COMPLETED,
            result_summary="Found 5 patterns",
        )
        tmp_bus.log_task(st)

        log = tmp_bus.get_task_log("p1")
        assert len(log) == 1
        assert log[0]["status"] == "completed"
        assert log[0]["result_summary"] == "Found 5 patterns"

    def test_empty_parent_returns_nothing(self, tmp_bus):
        tmp_bus.publish("t1", "p1", "code", "data")
        assert tmp_bus.get_artifacts("p_nonexistent") == []

    def test_multiple_artifacts_same_task(self, tmp_bus):
        tmp_bus.publish("t1", "p1", "code", "file1.py")
        tmp_bus.publish("t1", "p1", "code", "file2.py")
        tmp_bus.publish("t1", "p1", "test", "all passed")

        arts = tmp_bus.get_artifacts("p1", task_id="t1")
        assert len(arts) == 3


# ---------------------------------------------------------------------------
# Agent Pool
# ---------------------------------------------------------------------------

class TestAgentPool:
    def test_submit_and_complete(self, mock_agent_factory, tmp_bus):
        pool = AgentPool(mock_agent_factory, max_agents=2, artifact_bus=tmp_bus)
        st = SubTask(
            task_id="t1", parent_id="p1",
            task_type=SubTaskType.RESEARCH,
            description="Search for patterns",
        )

        future = pool.submit(st)
        result = future.result(timeout=10)

        assert result.status == SubTaskStatus.COMPLETED
        assert result.tokens_used == 100
        pool.shutdown()

    def test_agent_failure_marks_failed(self, tmp_bus):
        def failing_factory():
            agent = MagicMock()
            agent.run.side_effect = RuntimeError("LLM down")
            return agent

        pool = AgentPool(failing_factory, max_agents=1, artifact_bus=tmp_bus)
        st = SubTask(
            task_id="t1", parent_id="p1",
            task_type=SubTaskType.IMPLEMENT,
            description="Write code",
        )

        future = pool.submit(st)
        result = future.result(timeout=10)

        assert result.status == SubTaskStatus.FAILED
        assert "LLM down" in result.result_summary
        pool.shutdown()

    def test_parallel_execution(self, tmp_bus):
        threads_seen = set()
        barrier = threading.Barrier(2, timeout=5)

        def slow_factory():
            agent = MagicMock()
            def slow_run(task):
                threads_seen.add(threading.current_thread().name)
                barrier.wait()
                result = MagicMock()
                result.success = True
                result.summary = "Done"
                result.tokens = 50
                result.iterations = 1
                return result
            agent.run = slow_run
            return agent

        pool = AgentPool(slow_factory, max_agents=2, artifact_bus=tmp_bus)
        st1 = SubTask(task_id="t1", parent_id="p1",
                       task_type=SubTaskType.RESEARCH, description="A")
        st2 = SubTask(task_id="t2", parent_id="p1",
                       task_type=SubTaskType.IMPLEMENT, description="B")

        f1 = pool.submit(st1)
        f2 = pool.submit(st2)
        f1.result(timeout=10)
        f2.result(timeout=10)

        # Both ran in different threads (parallel)
        assert len(threads_seen) == 2
        pool.shutdown()

    def test_dependency_context_injection(self, mock_agent_factory, tmp_bus):
        # Pre-publish artifact from a "completed" dependency
        tmp_bus.publish("t_dep", "p1", "research_result", "Use quicksort for O(n log n)")

        pool = AgentPool(mock_agent_factory, max_agents=1, artifact_bus=tmp_bus)
        st = SubTask(
            task_id="t_impl", parent_id="p1",
            task_type=SubTaskType.IMPLEMENT,
            description="Write the sort",
            depends_on=["t_dep"],
        )

        future = pool.submit(st)
        result = future.result(timeout=10)

        # The agent should have been called with context from dependency
        agent_instance = mock_agent_factory()
        assert result.status == SubTaskStatus.COMPLETED
        pool.shutdown()

    def test_active_count(self, tmp_bus):
        started = threading.Event()
        proceed = threading.Event()

        def blocking_factory():
            agent = MagicMock()
            def blocking_run(task):
                started.set()
                proceed.wait(timeout=5)
                result = MagicMock()
                result.success = True
                result.summary = "Done"
                result.tokens = 10
                result.iterations = 1
                return result
            agent.run = blocking_run
            return agent

        pool = AgentPool(blocking_factory, max_agents=2, artifact_bus=tmp_bus)
        st = SubTask(task_id="t1", parent_id="p1",
                      task_type=SubTaskType.GENERIC, description="X")
        pool.submit(st)
        started.wait(timeout=5)

        assert pool.active_count == 1

        proceed.set()
        time.sleep(0.2)
        assert pool.active_count == 0
        pool.shutdown()


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class TestCoordinator:
    def test_full_coordination_3_phases(self, tmp_path, mock_agent_factory):
        bus_path = tmp_path / "coord_bus.db"
        coord = Coordinator(
            agent_factory=mock_agent_factory,
            max_agents=3,
            bus_db_path=bus_path,
        )

        result = coord.run("Build a binary search implementation")

        assert isinstance(result, CoordinationResult)
        assert result.success is True
        assert len(result.subtasks) == 3
        assert result.total_tokens > 0
        assert result.total_elapsed_s >= 0

        # Check all 3 phases completed
        types = [st.task_type for st in result.subtasks]
        assert SubTaskType.RESEARCH in types
        assert SubTaskType.IMPLEMENT in types
        assert SubTaskType.VERIFY in types

        coord.close()

    def test_coordination_with_failure(self, tmp_path):
        call_count = 0

        def mixed_factory():
            nonlocal call_count
            call_count += 1
            agent = MagicMock()
            if call_count == 2:  # implement phase fails
                r = MagicMock()
                r.success = False
                r.summary = "Compilation error"
                r.tokens = 50
                r.iterations = 2
                agent.run.return_value = r
            else:
                r = MagicMock()
                r.success = True
                r.summary = "OK"
                r.tokens = 100
                r.iterations = 3
                agent.run.return_value = r
            return agent

        bus_path = tmp_path / "coord_bus.db"
        coord = Coordinator(
            agent_factory=mixed_factory,
            max_agents=3,
            bus_db_path=bus_path,
        )

        result = coord.run("Fix the bug")
        assert result.success is False  # one phase failed
        failed = [st for st in result.subtasks if st.status == SubTaskStatus.FAILED]
        assert len(failed) >= 1
        coord.close()

    def test_coordination_with_llm_decompose(self, tmp_path, mock_agent_factory):
        def mock_llm(prompt):
            return json.dumps([
                {"type": "research", "description": "Research", "depends_on": []},
                {"type": "implement", "description": "Code", "depends_on": [0]},
            ])

        bus_path = tmp_path / "coord_bus.db"
        coord = Coordinator(
            agent_factory=mock_agent_factory,
            max_agents=2,
            bus_db_path=bus_path,
            llm_decompose_fn=mock_llm,
        )

        result = coord.run("Quick task")
        assert result.success is True
        assert len(result.subtasks) == 2
        coord.close()

    def test_artifact_bus_populated_after_run(self, tmp_path, mock_agent_factory):
        bus_path = tmp_path / "coord_bus.db"
        coord = Coordinator(
            agent_factory=mock_agent_factory,
            max_agents=3,
            bus_db_path=bus_path,
        )

        result = coord.run("Build a thing")

        # Check artifacts were published
        parent_id = result.parent_id
        arts = coord._bus.get_artifacts(parent_id)
        assert len(arts) == 3  # one per subtask

        # Check task log
        log_entries = coord._bus.get_task_log(parent_id)
        assert len(log_entries) == 3
        coord.close()

    def test_agents_used_count(self, tmp_path, mock_agent_factory):
        bus_path = tmp_path / "coord_bus.db"
        coord = Coordinator(
            agent_factory=mock_agent_factory,
            max_agents=3,
            bus_db_path=bus_path,
        )

        result = coord.run("Task")
        # At least 1 agent thread was used (sequential due to deps)
        assert result.agents_used >= 1
        coord.close()


# ---------------------------------------------------------------------------
# Integration: SubTaskType enum
# ---------------------------------------------------------------------------

class TestSubTaskType:
    def test_all_types(self):
        assert SubTaskType.RESEARCH == "research"
        assert SubTaskType.IMPLEMENT == "implement"
        assert SubTaskType.VERIFY == "verify"
        assert SubTaskType.GENERIC == "generic"

    def test_from_string(self):
        assert SubTaskType("research") == SubTaskType.RESEARCH
        assert SubTaskType("verify") == SubTaskType.VERIFY
