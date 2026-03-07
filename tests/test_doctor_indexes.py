"""
Tests for cli/doctor_cmd._check_fts5_indexes
---------------------------------------------
Verifies that doctor checks both memory_index_dir and federated_data_dir
from config/memory.yaml rather than only the hardcoded repo-relative path.
"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from cli.doctor_cmd import _CheckResult, _check_fts5_indexes


def _create_fts5_db(path: Path) -> None:
    """Create a minimal valid .fts5.db file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
        "USING fts5(search_content, source_path, chunk_id)"
    )
    conn.close()


class TestCheckFts5Indexes:

    def test_finds_indexes_in_federated_data_dir(self, tmp_path):
        """Doctor should find indexes at the configured federated_data_dir."""
        fed_dir = tmp_path / "external_indexes"
        _create_fts5_db(fed_dir / "se_codereview.fts5.db")
        _create_fts5_db(fed_dir / "csn_python.fts5.db")

        mem_yaml = {
            "memory": {"index_dir": str(tmp_path / "mem_indexes")},
            "federated_search": {"data_dir": str(fed_dir)},
        }

        cr = _CheckResult()
        with patch("cli.doctor_cmd._load_memory_yaml", return_value=mem_yaml):
            _check_fts5_indexes(cr)

        tags = [tag for tag, _, _ in cr.rows]
        details = [detail for _, _, detail in cr.rows]
        # Should have an OK for the federated dir
        assert "OK" in tags
        ok_details = [d for t, _, d in cr.rows if t == "OK"]
        assert any("2 in" in d for d in ok_details)

    def test_warns_when_no_indexes_anywhere(self, tmp_path):
        """Doctor should warn when both configured dirs are empty."""
        mem_yaml = {
            "memory": {"index_dir": str(tmp_path / "empty_mem")},
            "federated_search": {"data_dir": str(tmp_path / "empty_fed")},
        }

        cr = _CheckResult()
        with patch("cli.doctor_cmd._load_memory_yaml", return_value=mem_yaml):
            _check_fts5_indexes(cr)

        tags = [tag for tag, _, _ in cr.rows]
        assert all(t == "WARN" for t in tags)

    def test_checks_both_dirs_independently(self, tmp_path):
        """When indexes exist in both dirs, both are reported."""
        mem_dir = tmp_path / "mem_indexes"
        fed_dir = tmp_path / "fed_indexes"
        _create_fts5_db(mem_dir / "agent_memory.fts5.db")
        _create_fts5_db(fed_dir / "csn_python.fts5.db")

        mem_yaml = {
            "memory": {"index_dir": str(mem_dir)},
            "federated_search": {"data_dir": str(fed_dir)},
        }

        cr = _CheckResult()
        with patch("cli.doctor_cmd._load_memory_yaml", return_value=mem_yaml):
            _check_fts5_indexes(cr)

        ok_rows = [(n, d) for t, n, d in cr.rows if t == "OK"]
        assert len(ok_rows) == 2
        labels = [n for n, _ in ok_rows]
        assert any("memory_index_dir" in l for l in labels)
        assert any("federated_data_dir" in l for l in labels)

    def test_deduplicates_same_dir(self, tmp_path):
        """If memory_index_dir == federated_data_dir, only scan once."""
        shared_dir = tmp_path / "shared"
        _create_fts5_db(shared_dir / "test.fts5.db")

        mem_yaml = {
            "memory": {"index_dir": str(shared_dir)},
            "federated_search": {"data_dir": str(shared_dir)},
        }

        cr = _CheckResult()
        with patch("cli.doctor_cmd._load_memory_yaml", return_value=mem_yaml):
            _check_fts5_indexes(cr)

        ok_rows = [r for r in cr.rows if r[0] == "OK"]
        # Should only report the directory once
        assert len(ok_rows) == 1
