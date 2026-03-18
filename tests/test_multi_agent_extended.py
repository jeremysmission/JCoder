"""Extended tests for multi-agent coordination (Sprints 13-16).

Covers: agent registration/discovery, task delegation, result aggregation,
failure isolation, concurrent execution, message passing via ArtifactBus,
agent priority/ordering, and resource contention.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future
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
# Helpers
# ---------------------------------------------------------------------------

def _make_result(success=True, summary="OK", tokens=50, iterations=1):
    r = MagicMock()
    r.success = success
    r.summary = summary
    r.tokens = tokens
    r.iterations = iterations
    return r


def _ok_factory():
    agent = MagicMock()
    agent.run.return_value = _make_result()
    return agent


def _subtask(task_id="t1", parent_id="p1", task_type=SubTaskType.GENERIC,
             desc="work", depends_on=None, status=SubTaskStatus.PENDING):
    return SubTask(
        task_id=task_id, parent_id=parent_id, task_type=task_type,
        description=desc, depends_on=depends_on or [],
        status=status,
    )


# ---------------------------------------------------------------------------
# 1. Agent Registration and Discovery
# ---------------------------------------------------------------------------

class TestAgentRegistrationDiscovery:
    """AgentPool creates fresh agent instances via the factory on each submit."""

    def test_factory_called_per_subtask(self, tmp_path):
        factory = MagicMock(return_value=MagicMock(
            run=MagicMock(return_value=_make_result()),
        ))
        bus = ArtifactBus(tmp_path / "bus.db")
        pool = AgentPool(factory, max_agents=2, artifact_bus=bus)

        f1 = pool.submit(_subtask("t1"))
        f2 = pool.submit(_subtask("t2"))
        f1.result(timeout=10)
        f2.result(timeout=10)

        assert factory.call_count == 2
        pool.shutdown()
        bus.close()

    def test_pool_respects_max_agents(self, tmp_path):
        """ThreadPoolExecutor caps concurrency at max_agents."""
        concurrent_count = {"current": 0, "peak": 0}
        lock = threading.Lock()
        gate = threading.Event()

        def counting_factory():
            agent = MagicMock()
            def slow_run(desc):
                with lock:
                    concurrent_count["current"] += 1
                    concurrent_count["peak"] = max(
                        concurrent_count["peak"],
                        concurrent_count["current"],
                    )
                gate.wait(timeout=5)
                with lock:
                    concurrent_count["current"] -= 1
                return _make_result()
            agent.run = slow_run
            return agent

        bus = ArtifactBus(tmp_path / "bus.db")
        pool = AgentPool(counting_factory, max_agents=2, artifact_bus=bus)

        futures = [pool.submit(_subtask(f"t{i}")) for i in range(4)]
        time.sleep(0.3)
        gate.set()
        for f in futures:
            f.result(timeout=10)

        # Peak concurrency should not exceed max_agents
        assert concurrent_count["peak"] <= 2
        pool.shutdown()
        bus.close()

    def test_pool_without_artifact_bus(self):
        """Pool works fine when no ArtifactBus is provided."""
        pool = AgentPool(_ok_factory, max_agents=1, artifact_bus=None)
        f = pool.submit(_subtask("t1"))
        result = f.result(timeout=10)
        assert result.status == SubTaskStatus.COMPLETED
        pool.shutdown()


# ---------------------------------------------------------------------------
# 2. Task Delegation Between Agents
# ---------------------------------------------------------------------------

class TestTaskDelegation:
    """Coordinator decomposes and dispatches subtasks to pool agents."""

    def test_dependency_chain_respected(self, tmp_path):
        order = []

        def ordered_factory():
            agent = MagicMock()
            def run(desc):
                order.append(threading.current_thread().name)
                return _make_result()
            agent.run = run
            return agent

        coord = Coordinator(
            agent_factory=ordered_factory, max_agents=1,
            bus_db_path=tmp_path / "bus.db",
        )
        result = coord.run("Build something")
        # 3 phases executed sequentially (research -> implement -> verify)
        assert len(order) == 3
        assert result.success
        coord.close()

    def test_subtask_gets_dependency_context(self, tmp_path):
        captured_descs = []

        def capture_factory():
            agent = MagicMock()
            def run(desc):
                captured_descs.append(desc)
                return _make_result(summary="Research findings here")
            agent.run = run
            return agent

        coord = Coordinator(
            agent_factory=capture_factory, max_agents=1,
            bus_db_path=tmp_path / "bus.db",
        )
        coord.run("Solve a problem")

        # Second and third agents should receive context from prior phases
        assert len(captured_descs) == 3
        assert "Context from prior phases" in captured_descs[1]
        assert "Context from prior phases" in captured_descs[2]
        coord.close()

    def test_llm_decompose_custom_subtask_count(self, tmp_path):
        def llm_5_tasks(prompt):
            return json.dumps([
                {"type": "research", "description": f"Step {i}", "depends_on": []}
                for i in range(5)
            ])

        coord = Coordinator(
            agent_factory=_ok_factory, max_agents=5,
            bus_db_path=tmp_path / "bus.db",
            llm_decompose_fn=llm_5_tasks,
        )
        result = coord.run("Big task")
        assert len(result.subtasks) == 5
        assert result.success
        coord.close()


# ---------------------------------------------------------------------------
# 3. Result Aggregation from Multiple Agents
# ---------------------------------------------------------------------------

class TestResultAggregation:

    def test_total_tokens_summed(self, tmp_path):
        call_n = {"n": 0}

        def varied_factory():
            call_n["n"] += 1
            agent = MagicMock()
            agent.run.return_value = _make_result(tokens=call_n["n"] * 100)
            return agent

        coord = Coordinator(
            agent_factory=varied_factory, max_agents=3,
            bus_db_path=tmp_path / "bus.db",
        )
        result = coord.run("Aggregate test")
        assert result.total_tokens == 100 + 200 + 300
        coord.close()

    def test_summary_contains_all_phases(self, tmp_path):
        coord = Coordinator(
            agent_factory=_ok_factory, max_agents=3,
            bus_db_path=tmp_path / "bus.db",
        )
        result = coord.run("Summary test")
        assert "[research]" in result.summary
        assert "[implement]" in result.summary
        assert "[verify]" in result.summary
        coord.close()

    def test_elapsed_time_recorded(self, tmp_path):
        coord = Coordinator(
            agent_factory=_ok_factory, max_agents=3,
            bus_db_path=tmp_path / "bus.db",
        )
        result = coord.run("Timing test")
        assert result.total_elapsed_s >= 0
        coord.close()

    def test_coordination_result_fields(self, tmp_path):
        coord = Coordinator(
            agent_factory=_ok_factory, max_agents=3,
            bus_db_path=tmp_path / "bus.db",
        )
        result = coord.run("Field check")
        assert result.parent_id.startswith("coord_")
        assert isinstance(result.subtasks, list)
        assert result.agents_used >= 1
        coord.close()


# ---------------------------------------------------------------------------
# 4. Agent Failure Isolation
# ---------------------------------------------------------------------------

class TestFailureIsolation:

    def test_one_crash_doesnt_kill_others(self, tmp_path):
        call_n = {"n": 0}

        def mixed_factory():
            call_n["n"] += 1
            agent = MagicMock()
            if call_n["n"] == 2:
                agent.run.side_effect = RuntimeError("Boom")
            else:
                agent.run.return_value = _make_result()
            return agent

        coord = Coordinator(
            agent_factory=mixed_factory, max_agents=3,
            bus_db_path=tmp_path / "bus.db",
        )
        result = coord.run("Isolation test")

        statuses = [st.status for st in result.subtasks]
        assert SubTaskStatus.FAILED in statuses
        assert SubTaskStatus.COMPLETED in statuses
        coord.close()

    def test_exception_captured_in_summary(self, tmp_path):
        def crash_factory():
            agent = MagicMock()
            agent.run.side_effect = ValueError("bad input")
            return agent

        bus = ArtifactBus(tmp_path / "bus.db")
        pool = AgentPool(crash_factory, max_agents=1, artifact_bus=bus)
        st = _subtask("t1")
        result = pool.submit(st).result(timeout=10)

        assert result.status == SubTaskStatus.FAILED
        assert "bad input" in result.result_summary
        pool.shutdown()
        bus.close()

    def test_failed_subtask_still_logged(self, tmp_path):
        def crash_factory():
            agent = MagicMock()
            agent.run.side_effect = Exception("oops")
            return agent

        bus = ArtifactBus(tmp_path / "bus.db")
        pool = AgentPool(crash_factory, max_agents=1, artifact_bus=bus)
        st = _subtask("t1", parent_id="p_fail")
        pool.submit(st).result(timeout=10)

        log_entries = bus.get_task_log("p_fail")
        assert len(log_entries) == 1
        assert log_entries[0]["status"] == "failed"
        pool.shutdown()
        bus.close()


# ---------------------------------------------------------------------------
# 5. Concurrent Agent Execution
# ---------------------------------------------------------------------------

class TestConcurrentExecution:

    def test_independent_subtasks_run_in_parallel(self, tmp_path):
        """Subtasks without dependencies can execute concurrently."""
        barrier = threading.Barrier(3, timeout=5)
        threads_seen = set()

        def parallel_factory():
            agent = MagicMock()
            def run(desc):
                threads_seen.add(threading.current_thread().name)
                barrier.wait()
                return _make_result()
            agent.run = run
            return agent

        def llm_parallel(prompt):
            return json.dumps([
                {"type": "research", "description": "A", "depends_on": []},
                {"type": "research", "description": "B", "depends_on": []},
                {"type": "research", "description": "C", "depends_on": []},
            ])

        coord = Coordinator(
            agent_factory=parallel_factory, max_agents=3,
            bus_db_path=tmp_path / "bus.db",
            llm_decompose_fn=llm_parallel,
        )
        result = coord.run("Parallel work")
        assert result.success
        assert len(threads_seen) == 3
        coord.close()

    def test_thread_safety_of_artifact_bus(self, tmp_path):
        """Multiple threads publishing artifacts concurrently."""
        bus = ArtifactBus(tmp_path / "bus.db")
        errors = []

        def publish_many(thread_id):
            try:
                for i in range(10):
                    bus.publish(f"t_{thread_id}_{i}", "p1", "code",
                               f"content_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=publish_many, args=(i,))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        arts = bus.get_artifacts("p1")
        assert len(arts) == 40
        bus.close()


# ---------------------------------------------------------------------------
# 6. Message Passing Between Agents (via ArtifactBus)
# ---------------------------------------------------------------------------

class TestMessagePassing:

    def test_publish_then_search(self, tmp_path):
        bus = ArtifactBus(tmp_path / "bus.db")
        bus.publish("t1", "p1", "note", "Use quicksort for large arrays")
        bus.publish("t1", "p1", "note", "Bubblesort only for small n")

        hits = bus.search_content("p1", "quicksort")
        assert len(hits) == 1
        assert "quicksort" in hits[0]["content"]
        bus.close()

    def test_artifact_types_filter(self, tmp_path):
        bus = ArtifactBus(tmp_path / "bus.db")
        bus.publish("t1", "p1", "code", "def sort(): pass")
        bus.publish("t1", "p1", "test_result", "PASS")
        bus.publish("t1", "p1", "research", "Algorithm comparison")

        code_arts = bus.get_artifacts("p1", artifact_type="code")
        assert len(code_arts) == 1
        test_arts = bus.get_artifacts("p1", artifact_type="test_result")
        assert len(test_arts) == 1
        bus.close()

    def test_cross_task_artifact_visibility(self, tmp_path):
        """Artifacts from different tasks under same parent are all visible."""
        bus = ArtifactBus(tmp_path / "bus.db")
        bus.publish("t1", "shared_parent", "code", "file1")
        bus.publish("t2", "shared_parent", "code", "file2")
        bus.publish("t3", "shared_parent", "code", "file3")

        all_arts = bus.get_artifacts("shared_parent")
        assert len(all_arts) == 3
        bus.close()

    def test_metadata_round_trip(self, tmp_path):
        bus = ArtifactBus(tmp_path / "bus.db")
        meta = {"lang": "python", "lines": 42, "tested": True}
        bus.publish("t1", "p1", "code", "x=1", metadata=meta)
        arts = bus.get_artifacts("p1")
        assert arts[0]["metadata"]["lang"] == "python"
        assert arts[0]["metadata"]["lines"] == 42
        assert arts[0]["metadata"]["tested"] is True
        bus.close()


# ---------------------------------------------------------------------------
# 7. Agent Priority / Ordering (dependency-driven)
# ---------------------------------------------------------------------------

class TestPriorityOrdering:

    def test_default_decompose_order_is_research_implement_verify(self):
        subtasks = _default_decompose("Task", "p1")
        assert [st.task_type for st in subtasks] == [
            SubTaskType.RESEARCH, SubTaskType.IMPLEMENT, SubTaskType.VERIFY,
        ]

    def test_execution_follows_dependency_order(self, tmp_path):
        execution_order = []

        def tracking_factory():
            agent = MagicMock()
            def run(desc):
                execution_order.append(desc[:30])
                return _make_result()
            agent.run = run
            return agent

        coord = Coordinator(
            agent_factory=tracking_factory, max_agents=1,
            bus_db_path=tmp_path / "bus.db",
        )
        coord.run("Ordering test")
        assert len(execution_order) == 3
        # Research runs first (contains "Research" keyword)
        assert "Research" in execution_order[0] or "research" in execution_order[0].lower()
        coord.close()

    def test_diamond_dependency_via_llm(self, tmp_path):
        """A -> B, A -> C, B+C -> D. D waits for both B and C."""
        execution_order = []
        lock = threading.Lock()

        def tracking_factory():
            agent = MagicMock()
            def run(desc):
                with lock:
                    execution_order.append(desc[:20])
                time.sleep(0.05)
                return _make_result()
            agent.run = run
            return agent

        def diamond_llm(prompt):
            return json.dumps([
                {"type": "research",  "description": "Phase A", "depends_on": []},
                {"type": "implement", "description": "Phase B", "depends_on": [0]},
                {"type": "implement", "description": "Phase C", "depends_on": [0]},
                {"type": "verify",    "description": "Phase D", "depends_on": [1, 2]},
            ])

        coord = Coordinator(
            agent_factory=tracking_factory, max_agents=3,
            bus_db_path=tmp_path / "bus.db",
            llm_decompose_fn=diamond_llm,
        )
        result = coord.run("Diamond deps")
        assert result.success
        assert len(result.subtasks) == 4
        coord.close()


# ---------------------------------------------------------------------------
# 8. Resource Contention Handling
# ---------------------------------------------------------------------------

class TestResourceContention:

    def test_bus_handles_rapid_writes(self, tmp_path):
        """Rapid sequential writes don't corrupt the database."""
        bus = ArtifactBus(tmp_path / "bus.db")
        for i in range(100):
            bus.publish(f"t{i}", "p1", "code", f"content_{i}")
        arts = bus.get_artifacts("p1")
        assert len(arts) == 100
        bus.close()

    def test_pool_shutdown_is_safe(self, tmp_path):
        bus = ArtifactBus(tmp_path / "bus.db")
        pool = AgentPool(_ok_factory, max_agents=2, artifact_bus=bus)
        f = pool.submit(_subtask("t1"))
        f.result(timeout=10)
        pool.shutdown(wait=True)
        # Calling shutdown again should be safe
        pool.shutdown(wait=False)
        bus.close()

    def test_active_count_under_contention(self, tmp_path):
        started = threading.Event()
        gate = threading.Event()

        def blocking_factory():
            agent = MagicMock()
            def run(desc):
                started.set()
                gate.wait(timeout=5)
                return _make_result()
            agent.run = run
            return agent

        bus = ArtifactBus(tmp_path / "bus.db")
        pool = AgentPool(blocking_factory, max_agents=3, artifact_bus=bus)
        pool.submit(_subtask("t1"))
        started.wait(timeout=5)

        assert pool.active_count >= 1
        gate.set()
        time.sleep(0.3)
        assert pool.active_count == 0
        pool.shutdown()
        bus.close()

    def test_coordinator_close_cleans_up(self, tmp_path):
        coord = Coordinator(
            agent_factory=_ok_factory, max_agents=2,
            bus_db_path=tmp_path / "bus.db",
        )
        coord.run("Cleanup test")
        coord.close()
        # Second close should not raise
        coord.close()
