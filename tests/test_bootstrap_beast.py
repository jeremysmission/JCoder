"""Tests for the BEAST bootstrap script."""
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _import_bootstrap():
    import importlib
    import scripts.bootstrap_beast as mod
    importlib.reload(mod)
    return mod


# ---- GPU detection ----

def test_detect_gpu_success():
    mod = _import_bootstrap()
    csv = "0, NVIDIA GeForce RTX 3090, 24576, 20000, 535.129\n1, NVIDIA GeForce RTX 3090, 24576, 22000, 535.129"
    with patch.object(mod, "_run", return_value=(0, csv, "")):
        info = mod.detect_gpu()
    assert info["available"] is True
    assert len(info["gpus"]) == 2
    assert info["total_vram_gb"] == pytest.approx(48.0, abs=0.1)
    assert info["gpus"][0]["name"] == "NVIDIA GeForce RTX 3090"


def test_detect_gpu_no_nvidia():
    mod = _import_bootstrap()
    with patch.object(mod, "_run", return_value=(-1, "", "Command not found")):
        info = mod.detect_gpu()
    assert info["available"] is False
    assert info["gpus"] == []


# ---- RAM detection ----

def test_detect_ram_psutil():
    mod = _import_bootstrap()
    mock_mem = MagicMock()
    mock_mem.total = 128 * 1024 ** 3
    with patch("psutil.virtual_memory", return_value=mock_mem):
        ram = mod.detect_ram()
    assert ram == pytest.approx(128.0, abs=0.1)


# ---- Ollama ----

def test_check_ollama_running():
    mod = _import_bootstrap()
    with patch.object(mod, "_run", return_value=(0, "NAME\nmodel1\n", "")):
        assert mod.check_ollama() is True


def test_check_ollama_not_running():
    mod = _import_bootstrap()
    with patch.object(mod, "_run", return_value=(-1, "", "not found")):
        assert mod.check_ollama() is False


def test_list_installed_models():
    mod = _import_bootstrap()
    stdout = "NAME                    ID           SIZE    MODIFIED\ndevstral:24b            abc123      14 GB   2 days ago\nphi4:14b-q4_K_M         def456      9 GB    5 days ago\n"
    with patch.object(mod, "_run", return_value=(0, stdout, "")):
        models = mod.list_installed_models()
    assert "devstral:24b" in models
    assert "phi4:14b-q4_K_M" in models


def test_pull_model_success(capsys):
    mod = _import_bootstrap()
    with patch.object(mod, "_run", return_value=(0, "pulling...\nsuccess", "")):
        ok = mod.pull_model("devstral:24b")
    assert ok is True
    assert "[OK]" in capsys.readouterr().out


def test_pull_model_failure(capsys):
    mod = _import_bootstrap()
    with patch.object(mod, "_run", return_value=(1, "", "connection refused")):
        ok = mod.pull_model("devstral:24b")
    assert ok is False
    assert "[FAIL]" in capsys.readouterr().out


def test_validate_model_success():
    mod = _import_bootstrap()
    with patch.object(mod, "_run", return_value=(0, "print('hello')", "")):
        ok, latency = mod.validate_model("devstral:24b")
    assert ok is True
    assert latency >= 0


def test_validate_model_failure():
    mod = _import_bootstrap()
    with patch.object(mod, "_run", return_value=(1, "", "error")):
        ok, latency = mod.validate_model("devstral:24b")
    assert ok is False


# ---- FAISS GPU ----

def test_check_faiss_gpu_available():
    mod = _import_bootstrap()
    with patch.object(mod, "_run", return_value=(0, "faiss_gpus=2\n", "")):
        assert mod.check_faiss_gpu() is True


def test_check_faiss_gpu_none():
    mod = _import_bootstrap()
    with patch.object(mod, "_run", return_value=(0, "faiss_gpus=0\n", "")):
        assert mod.check_faiss_gpu() is False


# ---- Full bootstrap (mocked) ----

def test_bootstrap_check_only(capsys):
    mod = _import_bootstrap()
    gpu_info = {"available": True, "gpus": [{"index": 0, "name": "RTX 3090",
                "total_mb": 24576, "free_mb": 20000, "driver": "535"}],
                "total_vram_gb": 24.0}
    with patch.object(mod, "detect_gpu", return_value=gpu_info), \
         patch.object(mod, "detect_ram", return_value=128.0), \
         patch.object(mod, "detect_disk_free", return_value=500.0), \
         patch.object(mod, "check_ollama", return_value=True), \
         patch.object(mod, "list_installed_models", return_value=["devstral:24b"]), \
         patch.object(mod, "check_faiss_gpu", return_value=True):
        ok = mod.bootstrap(check_only=True)
    assert ok is True
    out = capsys.readouterr().out
    assert "BEAST Hardware Bootstrap" in out
    assert "Check-only" in out


def test_bootstrap_no_gpu(capsys):
    mod = _import_bootstrap()
    gpu_info = {"available": False, "gpus": [], "total_vram_gb": 0.0}
    with patch.object(mod, "detect_gpu", return_value=gpu_info), \
         patch.object(mod, "detect_ram", return_value=16.0), \
         patch.object(mod, "detect_disk_free", return_value=100.0), \
         patch.object(mod, "check_ollama", return_value=False), \
         patch.object(mod, "check_faiss_gpu", return_value=False):
        ok = mod.bootstrap(check_only=True)
    assert ok is False
