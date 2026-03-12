from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from core.config import StorageConfig
from core.index_engine import IndexEngine


class _ExplodingConnection:
    def __init__(self, real_conn: sqlite3.Connection) -> None:
        self._real = real_conn

    def __getattr__(self, name):
        return getattr(self._real, name)

    def executemany(self, sql, params):
        raise sqlite3.OperationalError("simulated insert failure")


def test_build_fts5_rolls_back_on_insert_failure(tmp_path):
    storage = StorageConfig(data_dir=str(tmp_path), index_dir=str(tmp_path / "idx"))
    engine = IndexEngine(dimension=4, storage=storage, sparse_only=True)
    engine.metadata = [{
        "id": "old-1",
        "content": "existing searchable content",
        "source_path": "old.txt",
    }]
    engine._db_path = str(tmp_path / "idx" / "test.fts5.db")
    engine._build_fts5()

    engine.metadata = [{
        "id": "new-1",
        "content": "new content that should never commit",
        "source_path": "new.txt",
    }]
    real_connect = sqlite3.connect

    with patch(
        "core.index_engine.sqlite3.connect",
        side_effect=lambda *args, **kwargs: _ExplodingConnection(
            real_connect(*args, **kwargs)
        ),
    ):
        with pytest.raises(sqlite3.OperationalError, match="simulated insert failure"):
            engine._build_fts5()

    with sqlite3.connect(engine._db_path) as conn:
        rows = conn.execute(
            "SELECT chunk_id, source_path FROM chunks ORDER BY rowid"
        ).fetchall()

    assert rows == [("old-1", "old.txt")]
