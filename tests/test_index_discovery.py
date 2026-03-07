"""
Tests for core.index_discovery -- FTS5 index scanning and federated config loading.
"""

import os
import sqlite3
import textwrap

import pytest
import yaml

from core.index_discovery import (
    _DEFAULT_WEIGHT,
    build_federated_from_config,
    discover_fts5_indexes,
    load_federated_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_fts5_db(path: str) -> None:
    """Create a minimal valid FTS5 database at *path*."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE VIRTUAL TABLE chunks "
        "USING fts5(search_content, source_path, chunk_id)"
    )
    conn.execute(
        "INSERT INTO chunks(search_content, source_path, chunk_id) "
        "VALUES ('hello world', 'test.py', 'chunk_001')"
    )
    conn.commit()
    conn.close()


def _write_memory_yaml(path: str, content: dict) -> None:
    """Write a memory.yaml file."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(content, fh)


# ---------------------------------------------------------------------------
# discover_fts5_indexes
# ---------------------------------------------------------------------------

class TestDiscoverFts5Indexes:
    """Tests for the directory scanner."""

    def test_finds_fts5_files(self, tmp_path):
        """Discovery returns entries for each *.fts5.db file."""
        _create_fts5_db(str(tmp_path / "alpha.fts5.db"))
        _create_fts5_db(str(tmp_path / "beta.fts5.db"))

        found = discover_fts5_indexes(str(tmp_path))

        names = [f["name"] for f in found]
        assert "alpha" in names
        assert "beta" in names
        assert len(found) == 2

    def test_skips_non_fts5_files(self, tmp_path):
        """Non-FTS5 files are ignored."""
        _create_fts5_db(str(tmp_path / "real.fts5.db"))
        (tmp_path / "fake.db").write_text("not fts5")
        (tmp_path / "readme.txt").write_text("notes")
        (tmp_path / "data.faiss").write_bytes(b"\x00" * 16)

        found = discover_fts5_indexes(str(tmp_path))
        assert len(found) == 1
        assert found[0]["name"] == "real"

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        found = discover_fts5_indexes(str(tmp_path))
        assert found == []

    def test_missing_directory_raises(self, tmp_path):
        """Non-existent directory raises FileNotFoundError."""
        bogus = str(tmp_path / "does_not_exist")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            discover_fts5_indexes(bogus)

    def test_returns_absolute_paths(self, tmp_path):
        """Returned paths are absolute."""
        _create_fts5_db(str(tmp_path / "idx.fts5.db"))
        found = discover_fts5_indexes(str(tmp_path))
        assert os.path.isabs(found[0]["path"])

    def test_size_mb_populated(self, tmp_path):
        """Each entry has a non-negative size_mb value."""
        _create_fts5_db(str(tmp_path / "tiny.fts5.db"))
        found = discover_fts5_indexes(str(tmp_path))
        assert found[0]["size_mb"] >= 0

    def test_skips_directories_with_fts5_suffix(self, tmp_path):
        """A directory ending in .fts5.db is not treated as an index."""
        (tmp_path / "tricky.fts5.db").mkdir()
        found = discover_fts5_indexes(str(tmp_path))
        assert found == []


# ---------------------------------------------------------------------------
# load_federated_config
# ---------------------------------------------------------------------------

class TestLoadFederatedConfig:
    """Tests for YAML config loading."""

    def test_loads_full_config(self, tmp_path):
        """Config with all fields loads correctly."""
        data = {
            "federated_search": {
                "rrf_k": 42,
                "index_dir": "custom/path",
                "indexes": {
                    "alpha": {"weight": 2.0},
                    "beta": {"weight": 0.5},
                },
            }
        }
        yaml_path = str(tmp_path / "memory.yaml")
        _write_memory_yaml(yaml_path, data)

        cfg = load_federated_config(yaml_path)
        assert cfg["rrf_k"] == 42
        assert cfg["index_dir"] == "custom/path"
        assert cfg["indexes"]["alpha"]["weight"] == 2.0
        assert cfg["indexes"]["beta"]["weight"] == 0.5

    def test_defaults_when_section_missing(self, tmp_path):
        """Missing federated_search section yields sensible defaults."""
        yaml_path = str(tmp_path / "memory.yaml")
        _write_memory_yaml(yaml_path, {"memory": {"index_name": "test"}})

        cfg = load_federated_config(yaml_path)
        assert cfg["rrf_k"] == 60
        assert cfg["index_dir"] == "data/indexes"
        assert cfg["indexes"] == {}

    def test_missing_file_raises(self, tmp_path):
        """Non-existent YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_federated_config(str(tmp_path / "nope.yaml"))

    def test_bare_weight_value(self, tmp_path):
        """Index entry with a bare numeric value (not a dict) is treated as weight."""
        data = {
            "federated_search": {
                "indexes": {
                    "bare_idx": 1.7,
                },
            }
        }
        yaml_path = str(tmp_path / "memory.yaml")
        _write_memory_yaml(yaml_path, data)

        cfg = load_federated_config(yaml_path)
        assert cfg["indexes"]["bare_idx"]["weight"] == 1.7


# ---------------------------------------------------------------------------
# build_federated_from_config
# ---------------------------------------------------------------------------

class TestBuildFederatedFromConfig:
    """Tests for the end-to-end builder."""

    def test_builds_with_matching_indexes(self, tmp_path):
        """Builder loads indexes that match config entries."""
        _create_fts5_db(str(tmp_path / "alpha.fts5.db"))
        _create_fts5_db(str(tmp_path / "beta.fts5.db"))

        cfg = {
            "rrf_k": 60,
            "index_dir": str(tmp_path),
            "indexes": {
                "alpha": {"weight": 1.5},
                "beta": {"weight": 0.8},
            },
        }
        fed = build_federated_from_config(cfg, str(tmp_path))
        info = fed.list_indexes()
        names = {i["name"] for i in info}

        assert names == {"alpha", "beta"}
        weights = {i["name"]: i["weight"] for i in info}
        assert weights["alpha"] == 1.5
        assert weights["beta"] == 0.8

    def test_unknown_index_gets_default_weight(self, tmp_path):
        """Index on disk but NOT in config gets default weight 1.0."""
        _create_fts5_db(str(tmp_path / "surprise.fts5.db"))

        cfg = {
            "rrf_k": 60,
            "indexes": {},
        }
        fed = build_federated_from_config(cfg, str(tmp_path))
        info = fed.list_indexes()

        assert len(info) == 1
        assert info[0]["name"] == "surprise"
        assert info[0]["weight"] == _DEFAULT_WEIGHT

    def test_missing_index_dir_returns_empty(self, tmp_path):
        """Non-existent index_dir produces empty federation (no crash)."""
        cfg = {"rrf_k": 60, "indexes": {"phantom": {"weight": 1.0}}}
        fed = build_federated_from_config(cfg, str(tmp_path / "nonexistent"))

        assert fed.list_indexes() == []

    def test_configured_but_absent_index_logged(self, tmp_path, caplog):
        """Indexes in config but missing from disk produce a warning."""
        import logging
        with caplog.at_level(logging.WARNING, logger="core.index_discovery"):
            cfg = {
                "rrf_k": 60,
                "indexes": {"ghost": {"weight": 2.0}},
            }
            build_federated_from_config(cfg, str(tmp_path))

        assert any("ghost" in r.message and "not found" in r.message for r in caplog.records)

    def test_corrupt_db_skipped(self, tmp_path):
        """A corrupt .fts5.db file is skipped gracefully."""
        bad = tmp_path / "corrupt.fts5.db"
        bad.write_text("this is not sqlite")

        cfg = {"rrf_k": 60, "indexes": {}}
        fed = build_federated_from_config(cfg, str(tmp_path))
        assert fed.list_indexes() == []
