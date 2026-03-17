"""
Tests for storage pruning and append-only semantics.

Tests prune_old() methods added to TelemetryStore, ProceduralMemory,
and the INSERT OR IGNORE fix for ExperimentLedger.
"""

from __future__ import annotations

import pytest

from core.telemetry import TelemetryStore, QueryEvent
from core.procedural_memory import ProceduralMemory, ProceduralExperience
from core.ledger import ExperimentLedger


# ---------------------------------------------------------------------------
# TelemetryStore pruning
# ---------------------------------------------------------------------------

class TestTelemetryPruning:

    def test_prune_old_removes_oldest(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "tel.db"))
        for i in range(20):
            store.log(QueryEvent(
                query_id=f"q{i:03d}",
                query_text=f"query {i}",
                timestamp=1000.0 + i,
            ))
        deleted = store.prune_old(keep=10)
        assert deleted == 10
        remaining = store.recent(limit=100)
        assert len(remaining) == 10
        ids = {e.query_id for e in remaining}
        assert "q019" in ids
        assert "q000" not in ids

    def test_prune_old_noop_when_under_limit(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "tel.db"))
        for i in range(5):
            store.log(QueryEvent(
                query_id=f"q{i}",
                query_text=f"query {i}",
                timestamp=1000.0 + i,
            ))
        deleted = store.prune_old(keep=100)
        assert deleted == 0


# ---------------------------------------------------------------------------
# ProceduralMemory pruning
# ---------------------------------------------------------------------------

class TestProceduralPruning:

    def test_prune_removes_oldest(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.db"))
        for i in range(20):
            pm.store(ProceduralExperience(
                state_hash=f"state_{i:03d}",
                action=f"action_{i}",
                outcome=f"outcome_{i}",
                success=True,
            ))
        deleted = pm.prune_old(keep=10)
        assert deleted == 10

    def test_prune_noop_under_limit(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.db"))
        pm.store(ProceduralExperience(
            state_hash="s1", action="a1", outcome="o1", success=True,
        ))
        deleted = pm.prune_old(keep=100)
        assert deleted == 0


# ---------------------------------------------------------------------------
# ExperimentLedger -- true append-only
# ---------------------------------------------------------------------------

class TestLedgerAppendOnly:

    def test_duplicate_run_id_does_not_overwrite(self, tmp_path):
        ledger = ExperimentLedger(str(tmp_path / "ledger.db"))
        ledger.write_run(
            run_id="run_001",
            label="first",
            config_fingerprint="fp1",
            metrics={"score": 0.5},
        )
        ledger.write_run(
            run_id="run_001",
            label="second",
            config_fingerprint="fp2",
            metrics={"score": 0.9},
        )
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].label == "first"

    def test_different_run_ids_both_stored(self, tmp_path):
        ledger = ExperimentLedger(str(tmp_path / "ledger.db"))
        ledger.write_run(
            run_id="run_001", label="a",
            config_fingerprint="fp1", metrics={"x": 1},
        )
        ledger.write_run(
            run_id="run_002", label="b",
            config_fingerprint="fp2", metrics={"x": 2},
        )
        runs = ledger.list_runs()
        assert len(runs) == 2
