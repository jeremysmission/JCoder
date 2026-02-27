"""Eval guard must refuse tampered benchmarks and detect unlisted files."""

import json
import os
import tempfile

import pytest

from core.eval_guard import (
    compute_file_hash,
    generate_hashes,
    save_hashes,
    verify_hashes,
)


@pytest.fixture
def eval_env(tmp_path):
    """Create a temporary eval directory with benchmark + hash manifest."""
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    benchmark = {"id": "001", "question": "test?", "expected_keyword": "test"}
    bench_path = eval_dir / "test_bench.json"
    bench_path.write_text(json.dumps([benchmark]), encoding="utf-8")

    hash_path = str(config_dir / "benchmark_hashes.json")
    save_hashes(str(eval_dir), hash_path=hash_path)

    return str(eval_dir), hash_path


def test_verify_passes_on_clean(eval_env):
    eval_dir, hash_path = eval_env
    result = verify_hashes(eval_dir, hash_path=hash_path)
    assert result is None


def test_verify_refuses_tampered_file(eval_env):
    eval_dir, hash_path = eval_env

    # Tamper the benchmark
    bench_path = os.path.join(eval_dir, "test_bench.json")
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump([{"id": "TAMPERED"}], f)

    result = verify_hashes(eval_dir, hash_path=hash_path)
    assert result is not None
    assert "tampered" in result.lower()


def test_verify_refuses_missing_file(eval_env):
    eval_dir, hash_path = eval_env

    os.remove(os.path.join(eval_dir, "test_bench.json"))

    result = verify_hashes(eval_dir, hash_path=hash_path)
    assert result is not None
    assert "Missing" in result


def test_verify_refuses_missing_manifest(tmp_path):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()

    result = verify_hashes(str(eval_dir), hash_path=str(tmp_path / "nonexistent.json"))
    assert result is not None
    assert "not found" in result.lower()


def test_verify_detects_unlisted_file(eval_env):
    eval_dir, hash_path = eval_env

    # Sneak in a new benchmark file
    sneaky = os.path.join(eval_dir, "sneaky_bench.json")
    with open(sneaky, "w", encoding="utf-8") as f:
        json.dump([{"id": "SNEAKY"}], f)

    result = verify_hashes(eval_dir, hash_path=hash_path)
    assert result is not None
    assert "Unlisted" in result


def test_hash_is_deterministic():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write('{"test": true}')
        path = f.name

    try:
        h1 = compute_file_hash(path)
        h2 = compute_file_hash(path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex
    finally:
        os.unlink(path)
