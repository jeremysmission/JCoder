"""Tests for session persistence (SessionStore) and structured logging (AgentLogger)."""

import json
import threading
import time

import pytest

from agent.session import SessionStore, SessionInfo
from agent.logger import AgentLogger, AgentLogEntry


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class TestSessionStore:

    @pytest.fixture()
    def store(self, tmp_path):
        return SessionStore(store_dir=str(tmp_path / "sessions"))

    def test_save_and_load(self, store):
        history = [{"role": "user", "content": "hello"}]
        store.save("s1", "Fix bug", history, status="active", iterations=2, tokens=100)
        data = store.load("s1")
        assert data["session_id"] == "s1"
        assert data["task"] == "Fix bug"
        assert data["status"] == "active"
        assert data["iterations"] == 2
        assert data["total_tokens"] == 100
        assert data["message_count"] == 1
        assert data["history"] == history
        assert data["created_at"]
        assert data["updated_at"]

    def test_resume_history(self, store):
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        store.save("s2", "task", msgs)
        assert store.resume_history("s2") == msgs

    def test_list_sessions(self, store):
        for i in range(3):
            store.save(f"s{i}", f"task {i}", [], iterations=i)
            time.sleep(0.01)  # ensure distinct updated_at
        infos = store.list_sessions()
        assert len(infos) == 3
        # newest first
        assert infos[0].session_id == "s2"
        assert infos[-1].session_id == "s0"

    def test_list_sessions_filter(self, store):
        store.save("a1", "active task", [], status="active")
        store.save("c1", "done task", [], status="completed")
        store.save("c2", "another done", [], status="completed")
        active = store.list_sessions(status="active")
        completed = store.list_sessions(status="completed")
        assert len(active) == 1
        assert active[0].session_id == "a1"
        assert len(completed) == 2

    def test_delete(self, store):
        store.save("d1", "task", [])
        assert store.delete("d1") is True
        assert store.delete("d1") is False
        with pytest.raises(FileNotFoundError):
            store.load("d1")

    def test_cleanup_old(self, store):
        store.save("old1", "task1", [])
        store.save("old2", "task2", [])
        time.sleep(0.05)  # ensure timestamps are strictly before cutoff
        deleted = store.cleanup(max_age_days=0)
        assert deleted == 2
        assert store.list_sessions() == []

    def test_search(self, store):
        store.save("s1", "Fix database migration", [])
        store.save("s2", "Add REST endpoint", [])
        store.save("s3", "Fix database index", [])
        results = store.search("database")
        assert len(results) == 2
        ids = {r.session_id for r in results}
        assert ids == {"s1", "s3"}

    def test_atomic_write(self, store, tmp_path):
        store.save("aw", "task", [{"role": "user", "content": "x"}])
        tmp_files = list((tmp_path / "sessions").glob("*.tmp"))
        assert tmp_files == [], ".tmp file should not linger after save"

    def test_load_missing(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent_session_id")

    def test_update_preserves_created_at(self, store):
        store.save("up", "task v1", [{"role": "user", "content": "a"}])
        original = store.load("up")["created_at"]
        time.sleep(0.01)
        store.save("up", "task v2", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
        updated = store.load("up")
        assert updated["created_at"] == original
        assert updated["updated_at"] >= original
        assert updated["task"] == "task v2"
        assert updated["message_count"] == 2


# ---------------------------------------------------------------------------
# AgentLogger
# ---------------------------------------------------------------------------

class TestAgentLogger:

    @pytest.fixture()
    def logger(self, tmp_path):
        lg = AgentLogger(log_dir=str(tmp_path / "logs"), max_file_size_mb=50)
        yield lg
        lg.close()

    def test_log_and_query(self, logger):
        logger.log("sess1", "custom", {"key": "val"})
        logger.log("sess1", "custom", {"key": "val2"})
        entries = logger.query(session_id="sess1")
        assert len(entries) == 2
        assert entries[0].data["key"] == "val"

    def test_log_task_start(self, logger):
        logger.log_task_start("s1", "Refactor parser")
        entries = logger.query(session_id="s1")
        assert len(entries) == 1
        e = entries[0]
        assert e.event_type == "task_start"
        assert e.data["task"] == "Refactor parser"
        assert e.session_id == "s1"
        assert e.timestamp  # non-empty ISO string

    def test_log_tool_call(self, logger):
        logger.log_tool_call("s1", "read_file", {"path": "/a.py"}, iteration=3)
        entries = logger.query(session_id="s1", event_type="tool_call")
        assert len(entries) == 1
        d = entries[0].data
        assert d["tool"] == "read_file"
        assert d["iteration"] == 3
        assert "/a.py" in d["args"]

    def test_log_tool_result(self, logger):
        long_output = "x" * 1000
        logger.log_tool_result("s1", "read_file", True, long_output, elapsed_s=0.5)
        entries = logger.query(session_id="s1", event_type="tool_result")
        assert len(entries) == 1
        d = entries[0].data
        assert d["success"] is True
        assert len(d["output"]) <= 500
        assert d["output"].endswith("...")

    def test_log_llm_call(self, logger):
        logger.log_llm_call("s1", "phi4:14b", 1200, 350, elapsed_s=2.1)
        entries = logger.query(session_id="s1", event_type="llm_call")
        assert len(entries) == 1
        e = entries[0]
        assert e.data["model"] == "phi4:14b"
        assert e.data["input_tokens"] == 1200
        assert e.data["output_tokens"] == 350
        assert e.tokens == 1550
        assert e.elapsed_s == 2.1

    def test_log_task_complete(self, logger):
        logger.log_task_complete("s1", True, "All done", total_tokens=5000, iterations=4)
        entries = logger.query(session_id="s1", event_type="task_complete")
        assert len(entries) == 1
        e = entries[0]
        assert e.data["success"] is True
        assert e.data["summary"] == "All done"
        assert e.data["iterations"] == 4
        assert e.tokens == 5000

    def test_log_error(self, logger):
        logger.log_error("s1", "KeyError: 'foo'", context="parsing config")
        entries = logger.query(session_id="s1", event_type="error")
        assert len(entries) == 1
        d = entries[0].data
        assert d["error"] == "KeyError: 'foo'"
        assert d["context"] == "parsing config"

    def test_query_by_session(self, logger):
        logger.log("sess_a", "ev", {"x": 1})
        logger.log("sess_b", "ev", {"x": 2})
        logger.log("sess_a", "ev", {"x": 3})
        a = logger.query(session_id="sess_a")
        b = logger.query(session_id="sess_b")
        assert len(a) == 2
        assert len(b) == 1

    def test_query_by_event_type(self, logger):
        logger.log("s1", "tool_call", {"t": 1})
        logger.log("s1", "llm_call", {"t": 2})
        logger.log("s1", "tool_call", {"t": 3})
        tools = logger.query(event_type="tool_call")
        assert len(tools) == 2
        llms = logger.query(event_type="llm_call")
        assert len(llms) == 1

    def test_session_summary(self, logger):
        sid = "sum1"
        logger.log_task_start(sid, "do stuff")
        logger.log_tool_call(sid, "grep", {"q": "x"}, iteration=1)
        logger.log_tool_result(sid, "grep", True, "found", elapsed_s=0.2)
        logger.log_llm_call(sid, "phi4:14b", 500, 200, elapsed_s=1.5)
        logger.log_tool_call(sid, "edit", {"f": "a.py"}, iteration=2)
        logger.log_tool_result(sid, "edit", True, "ok", elapsed_s=0.1)
        logger.log_task_complete(sid, True, "finished", total_tokens=700, iterations=2)

        summary = logger.session_summary(sid)
        assert summary["found"] is True
        assert summary["tool_calls"] == 2
        assert summary["llm_calls"] == 1
        assert summary["errors"] == 0
        assert summary["total_tokens"] == 700 + (500 + 200)  # task_complete + llm_call
        assert summary["success"] is True

    def test_daily_stats(self, logger):
        logger.log_task_start("s1", "task")
        logger.log_tool_call("s1", "read", {}, iteration=1)
        logger.log_tool_call("s1", "edit", {}, iteration=2)
        logger.log_task_complete("s1", True, "done", total_tokens=400, iterations=2)
        logger.log_task_start("s2", "task2")

        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stats = logger.daily_stats(today)
        assert stats["date"] == today
        assert stats["sessions"] == 2
        assert stats["tool_calls"] == 2
        assert stats["completions"] == 1
        assert stats["success_rate"] == 1.0

    def test_file_rotation(self, tmp_path):
        lg = AgentLogger(log_dir=str(tmp_path / "rot"), max_file_size_mb=0.001)
        for i in range(200):
            lg.log("s1", "bulk", {"i": i, "pad": "x" * 50})
        lg.close()
        log_files = list((tmp_path / "rot").glob("agent_*.jsonl"))
        assert len(log_files) >= 2, "Expected rotation to create multiple files"

    def test_thread_safety(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path / "threads"), max_file_size_mb=50)
        errors = []

        def log_many(thread_id):
            try:
                for i in range(50):
                    logger.log(f"session_{thread_id}", "test", {"i": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=log_many, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        entries = logger.query(limit=500)
        assert len(entries) == 250  # 5 threads * 50 entries
        logger.close()
