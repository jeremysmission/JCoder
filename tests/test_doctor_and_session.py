"""Tests for R15: Doctor command enhancements + session robustness."""

from __future__ import annotations

import json
import time
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from cli.doctor_cmd import (
    _CheckResult,
    _check_python,
    _check_packages,
    _check_configs,
    _check_data_dirs,
    _check_disk_space,
    _check_fts5_indexes,
)
from agent.session import SessionStore


# ---------------------------------------------------------------------------
# Doctor _CheckResult
# ---------------------------------------------------------------------------

class TestCheckResult:

    def test_ok_increments_pass_count(self):
        cr = _CheckResult()
        cr.ok("test", "detail")
        assert cr.pass_count == 1
        assert cr.fail_count == 0

    def test_warn_increments_warn_count(self):
        cr = _CheckResult()
        cr.warn("test", "detail")
        assert cr.warn_count == 1

    def test_fail_increments_fail_count(self):
        cr = _CheckResult()
        cr.fail("test", "detail")
        assert cr.fail_count == 1


# ---------------------------------------------------------------------------
# Doctor checks (isolated)
# ---------------------------------------------------------------------------

class TestDoctorChecks:

    def test_check_python_ok(self):
        cr = _CheckResult()
        _check_python(cr)
        assert cr.pass_count == 1

    def test_check_packages(self):
        cr = _CheckResult()
        _check_packages(cr)
        # Should at least run without error
        assert cr.pass_count + cr.warn_count >= 1

    def test_check_configs(self):
        cr = _CheckResult()
        _check_configs(cr)
        # agent.yaml + memory.yaml should exist in repo
        assert cr.pass_count >= 1

    def test_check_disk_space(self):
        cr = _CheckResult()
        _check_disk_space(cr)
        assert cr.pass_count + cr.warn_count >= 1

    def test_check_data_dirs(self):
        cr = _CheckResult()
        _check_data_dirs(cr)
        assert len(cr.rows) >= 1


# ---------------------------------------------------------------------------
# Session robustness (R15.3)
# ---------------------------------------------------------------------------

class TestSessionRobustness:

    def test_load_corrupted_session_raises(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        # Write corrupted JSON
        session_path = tmp_path / "sessions" / "corrupt.json"
        session_path.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            store.load("corrupt")

    def test_load_missing_session_raises_filenotfounderror(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_resume_stale_session_logs_warning(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        # Save a session with old timestamp
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        session_path = tmp_path / "sessions" / "old.json"
        data = {
            "session_id": "old",
            "task": "old task",
            "created_at": old_time,
            "updated_at": old_time,
            "status": "active",
            "iterations": 0,
            "total_tokens": 0,
            "message_count": 1,
            "history": [{"role": "user", "content": "hello"}],
        }
        session_path.write_text(json.dumps(data), encoding="utf-8")

        with patch("agent.session.log") as mock_log:
            history = store.resume_history("old")
            assert len(history) == 1
            mock_log.warning.assert_called_once()
            assert "hours old" in str(mock_log.warning.call_args)

    def test_resume_fresh_session_no_warning(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        store.save("fresh", "task", [{"role": "user", "content": "hi"}])

        with patch("agent.session.log") as mock_log:
            history = store.resume_history("fresh")
            assert len(history) == 1
            mock_log.warning.assert_not_called()

    def test_save_and_load_roundtrip(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        store.save("rt", "test task", [{"role": "user", "content": "q"}],
                   status="active", iterations=5, tokens=1000)
        data = store.load("rt")
        assert data["task"] == "test task"
        assert data["iterations"] == 5
        assert data["total_tokens"] == 1000
        assert len(data["history"]) == 1

    def test_cleanup_respects_grace_period(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        store.save("a", task="t", history=[])
        store.save("b", task="t", history=[])
        deleted = store.cleanup(max_age_days=0)
        assert deleted == 0  # Grace period keeps fresh sessions
