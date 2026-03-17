"""Portability checks for script-level data and archive roots."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _reload_script(name: str, monkeypatch):
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    module = importlib.import_module(name)
    return importlib.reload(module)


def test_data_status_uses_env_data_root(monkeypatch, tmp_path):
    data_root = tmp_path / "portable_data"
    data_root.mkdir()
    monkeypatch.setenv("JCODER_DATA", str(data_root))

    mod = _reload_script("data_status", monkeypatch)
    assert mod._data_root() == data_root
    assert mod._disk_usage_target() == data_root


def test_overnight_download_uses_env_data_root(monkeypatch, tmp_path):
    data_root = tmp_path / "portable_data"
    data_root.mkdir()
    monkeypatch.setenv("JCODER_DATA", str(data_root))

    mod = _reload_script("overnight_download", monkeypatch)
    assert mod.DATA_ROOT == data_root
    assert mod._disk_usage_target() == data_root


def test_build_fts5_indexes_uses_jcoder_data_dir(monkeypatch, tmp_path):
    data_root = tmp_path / "portable_data"
    data_root.mkdir()
    monkeypatch.delenv("JCODER_DATA", raising=False)
    monkeypatch.setenv("JCODER_DATA_DIR", str(data_root))

    mod = _reload_script("build_fts5_indexes", monkeypatch)
    assert mod.DATA_ROOT == data_root
    assert mod.CLEAN_DIR == data_root / "clean_source"
    assert mod.INDEX_DIR == data_root / "indexes"


def test_build_fts5_indexes_falls_back_to_repo_local_data(monkeypatch):
    monkeypatch.delenv("JCODER_DATA", raising=False)
    monkeypatch.delenv("JCODER_DATA_DIR", raising=False)

    mod = _reload_script("build_fts5_indexes", monkeypatch)
    expected = SCRIPTS_DIR.parent / "data"
    assert mod.DATA_ROOT == expected
    assert mod.CLEAN_DIR == expected / "clean_source"
    assert mod.INDEX_DIR == expected / "indexes"


def test_build_se_indexes_archive_dirs_follow_env(monkeypatch, tmp_path):
    root_a = tmp_path / "archives_a"
    root_b = tmp_path / "archives_b"
    monkeypatch.setenv("JCODER_SE_ARCHIVE_ROOTS", os.pathsep.join([str(root_a), str(root_b)]))

    mod = _reload_script("build_se_indexes", monkeypatch)
    assert mod.ARCHIVE_DIRS == [root_a, root_b]


def test_validate_and_parallel_sanitize_follow_env_roots(monkeypatch, tmp_path):
    data_root = tmp_path / "portable_data"
    se_root = tmp_path / "portable_archives"
    monkeypatch.setenv("JCODER_DATA", str(data_root))
    monkeypatch.setenv("JCODER_SE_ROOT", str(se_root))

    validate_mod = _reload_script("validate_se_downloads", monkeypatch)
    parallel_mod = _reload_script("parallel_sanitize_se", monkeypatch)

    assert validate_mod._data_root() == data_root
    assert validate_mod.DEFAULT_SE_ROOT == se_root
    assert validate_mod.INTEGRITY_LOG == (
        data_root / "clean_source" / "_logs" / "stackexchange_archive_integrity_20260301.json"
    )
    assert parallel_mod._data_root() == data_root
    assert parallel_mod.DEFAULT_SE_ROOT == se_root
    assert parallel_mod.CLEAN_ROOT == data_root / "clean_source"
