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
    assert mod.DATA_ROOT == data_root


def test_overnight_download_uses_hardcoded_data_root(monkeypatch, tmp_path):
    """overnight_download.py uses a hardcoded DATA_ROOT (D:\\JCoder_Data).

    This test verifies the module-level constant exists and is a Path.
    """
    mod = _reload_script("overnight_download", monkeypatch)
    assert isinstance(mod.DATA_ROOT, Path)


def test_build_fts5_indexes_uses_jcoder_data_env(monkeypatch, tmp_path):
    """build_fts5_indexes reads JCODER_DATA env var for DATA_ROOT."""
    data_root = tmp_path / "portable_data"
    data_root.mkdir()
    monkeypatch.setenv("JCODER_DATA", str(data_root))

    mod = _reload_script("build_fts5_indexes", monkeypatch)
    assert mod.DATA_ROOT == data_root
    assert mod.CLEAN_DIR == data_root / "clean_source"
    assert mod.INDEX_DIR == data_root / "indexes"


def test_build_fts5_indexes_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("JCODER_DATA", raising=False)

    mod = _reload_script("build_fts5_indexes", monkeypatch)
    # Falls back to project_root/data when JCODER_DATA not set
    assert mod.DATA_ROOT.name == "data"
    assert mod.CLEAN_DIR == mod.DATA_ROOT / "clean_source"
    assert mod.INDEX_DIR == mod.DATA_ROOT / "indexes"


def test_build_se_indexes_uses_jcoder_data_env(monkeypatch, tmp_path):
    """build_se_indexes reads JCODER_DATA for DATA_ROOT and INDEX_DIR."""
    data_root = tmp_path / "portable_data"
    data_root.mkdir()
    monkeypatch.setenv("JCODER_DATA", str(data_root))

    mod = _reload_script("build_se_indexes", monkeypatch)
    assert mod.DATA_ROOT == data_root
    assert mod.INDEX_DIR == data_root / "indexes"


def test_validate_se_downloads_has_default_root(monkeypatch):
    """validate_se_downloads exposes DEFAULT_SE_ROOT and INTEGRITY_LOG constants."""
    mod = _reload_script("validate_se_downloads", monkeypatch)
    assert isinstance(mod.DEFAULT_SE_ROOT, Path)
    assert isinstance(mod.INTEGRITY_LOG, Path)
