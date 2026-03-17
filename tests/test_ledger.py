"""Tests for core.ledger -- immutable experiment audit trail."""

import pytest

from core.ledger import ExperimentLedger, RunRecord


@pytest.fixture
def ledger(tmp_path):
    return ExperimentLedger(str(tmp_path / "ledger.db"))


class TestWriteRun:

    def test_write_and_list(self, ledger):
        rec = ledger.write_run(
            run_id="run_001",
            label="baseline",
            config_fingerprint="abc123",
            metrics={"score": 0.85, "latency": 1.2},
        )
        assert isinstance(rec, RunRecord)
        assert rec.run_id == "run_001"
        assert rec.label == "baseline"
        assert rec.metrics_json["score"] == 0.85

    def test_list_runs_newest_first(self, ledger):
        for i in range(5):
            ledger.write_run(
                run_id=f"run_{i:03d}",
                label=f"run {i}",
                config_fingerprint=f"fp_{i}",
                metrics={"i": i},
                created_ts=1000.0 + i,
            )
        runs = ledger.list_runs(limit=3)
        assert len(runs) == 3
        assert runs[0].run_id == "run_004"  # newest
        assert runs[2].run_id == "run_002"

    def test_list_runs_limit(self, ledger):
        for i in range(10):
            ledger.write_run(
                run_id=f"r{i}", label="x",
                config_fingerprint="fp", metrics={},
            )
        assert len(ledger.list_runs(limit=5)) == 5
        assert len(ledger.list_runs(limit=50)) == 10

    def test_git_commit_stored(self, ledger):
        ledger.write_run(
            run_id="r1", label="with-commit",
            config_fingerprint="fp",
            metrics={"x": 1},
            git_commit="abc123def",
        )
        runs = ledger.list_runs()
        assert runs[0].git_commit == "abc123def"

    def test_git_commit_optional(self, ledger):
        ledger.write_run(
            run_id="r1", label="no-commit",
            config_fingerprint="fp",
            metrics={},
        )
        runs = ledger.list_runs()
        assert runs[0].git_commit is None


class TestAppendOnly:

    def test_duplicate_run_id_ignored(self, ledger):
        ledger.write_run(
            run_id="dup", label="first",
            config_fingerprint="fp1", metrics={"v": 1},
        )
        ledger.write_run(
            run_id="dup", label="second",
            config_fingerprint="fp2", metrics={"v": 2},
        )
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].label == "first"  # original preserved


class TestRunRecord:

    def test_frozen(self):
        rec = RunRecord(
            run_id="r1", created_ts=1000.0,
            label="test", config_fingerprint="fp",
            git_commit=None, metrics_json={"x": 1},
        )
        with pytest.raises(AttributeError):
            rec.label = "changed"  # type: ignore[misc]

    def test_metrics_accessible(self):
        rec = RunRecord(
            run_id="r1", created_ts=1000.0,
            label="test", config_fingerprint="fp",
            git_commit=None, metrics_json={"score": 0.9, "pass_rate": 0.95},
        )
        assert rec.metrics_json["score"] == 0.9
        assert rec.metrics_json["pass_rate"] == 0.95


class TestPersistence:

    def test_survives_reopen(self, tmp_path):
        db_path = str(tmp_path / "persist.db")
        ledger1 = ExperimentLedger(db_path)
        ledger1.write_run(
            run_id="r1", label="persistent",
            config_fingerprint="fp", metrics={"x": 42},
        )
        # Open a new instance pointing to same DB
        ledger2 = ExperimentLedger(db_path)
        runs = ledger2.list_runs()
        assert len(runs) == 1
        assert runs[0].metrics_json["x"] == 42

    def test_creates_parent_dirs(self, tmp_path):
        db_path = str(tmp_path / "deep" / "nested" / "ledger.db")
        ledger = ExperimentLedger(db_path)
        ledger.write_run(
            run_id="r1", label="deep",
            config_fingerprint="fp", metrics={},
        )
        assert len(ledger.list_runs()) == 1
