"""Tests for the measure CLI command -- schema correctness and graceful degradation."""

import json
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from core.config import JCoderConfig, StorageConfig


REQUIRED_SYSTEM_KEYS = {
    "torch_version": str,
    "torch_cuda_compiled": str,
    "cuda_available": bool,
    "cuda_device_count": int,
    "gpus": list,
    "driver_version": str,
    "cuda_runtime_version": str,
}

REQUIRED_ENDPOINT_KEYS = {
    "llm_models_ok": bool,
    "embed_models_ok": bool,
    "rerank_models_ok": bool,
    "llm_response_ms": (float, int, type(None)),
    "embed_response_ms": (float, int, type(None)),
    "rerank_response_ms": (float, int, type(None)),
}

REQUIRED_PERFORMANCE_KEYS = [
    "tok_per_s_single",
    "p95_latency_ms_4_parallel",
    "embed_items_per_s_batch16",
    "embed_items_per_s_batch32",
    "rerank_pairs_per_s_n50",
    "drift_tok_per_s_delta_10min",
]


def _fake_torch():
    """Build a fake torch module with cuda unavailable."""
    mod = types.ModuleType("torch")
    mod.__version__ = "2.10.0+cpu"
    mod.version = types.SimpleNamespace(cuda=None)
    cuda_ns = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    mod.cuda = cuda_ns
    return mod


def _run_measure(tmp_path):
    """Invoke measure with all external deps mocked. Returns (CliRunner result, json Path)."""
    data_dir = str(tmp_path / "data")
    metrics_json = tmp_path / "data" / "metrics" / "measurements.json"

    fake_torch = _fake_torch()

    cfg = JCoderConfig(
        storage=StorageConfig(
            data_dir=data_dir,
            index_dir=str(tmp_path / "data" / "indexes"),
        ),
    )

    with patch.dict("sys.modules", {"torch": fake_torch}), \
         patch("subprocess.run", side_effect=FileNotFoundError), \
         patch("httpx.Client") as mock_client_cls, \
         patch("cli.commands.load_config", return_value=cfg):

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = ConnectionError("unreachable")
        mock_client_cls.return_value = mock_client

        from cli.commands import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["measure"])

    return result, metrics_json


def test_measure_exits_zero(tmp_path):
    result, _ = _run_measure(tmp_path)
    assert result.exit_code == 0, f"exit_code={result.exit_code}\n{result.output}"


def test_measure_writes_json(tmp_path):
    _, json_path = _run_measure(tmp_path)
    assert json_path.exists(), f"Expected JSON at {json_path}"


def test_measure_json_has_required_top_keys(tmp_path):
    _, json_path = _run_measure(tmp_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "system" in data
    assert "endpoints" in data
    assert "performance" in data


def test_measure_system_keys_and_types(tmp_path):
    _, json_path = _run_measure(tmp_path)
    system = json.loads(json_path.read_text(encoding="utf-8"))["system"]
    for key, expected_type in REQUIRED_SYSTEM_KEYS.items():
        assert key in system, f"Missing system key: {key}"
        assert isinstance(system[key], expected_type), (
            f"system[{key}] is {type(system[key]).__name__}, expected {expected_type.__name__}"
        )


def test_measure_endpoint_keys_and_types(tmp_path):
    _, json_path = _run_measure(tmp_path)
    endpoints = json.loads(json_path.read_text(encoding="utf-8"))["endpoints"]
    for key, expected_type in REQUIRED_ENDPOINT_KEYS.items():
        assert key in endpoints, f"Missing endpoints key: {key}"
        assert isinstance(endpoints[key], expected_type), (
            f"endpoints[{key}] is {type(endpoints[key]).__name__}, expected {expected_type}"
        )


def test_measure_performance_keys_null(tmp_path):
    _, json_path = _run_measure(tmp_path)
    perf = json.loads(json_path.read_text(encoding="utf-8"))["performance"]
    for key in REQUIRED_PERFORMANCE_KEYS:
        assert key in perf, f"Missing performance key: {key}"
        assert perf[key] is None, f"performance[{key}] should be null, got {perf[key]}"


def test_measure_no_cuda_values(tmp_path):
    _, json_path = _run_measure(tmp_path)
    system = json.loads(json_path.read_text(encoding="utf-8"))["system"]
    assert system["cuda_available"] is False
    assert system["cuda_device_count"] == 0
    assert system["gpus"] == []


def test_measure_unreachable_endpoints(tmp_path):
    _, json_path = _run_measure(tmp_path)
    endpoints = json.loads(json_path.read_text(encoding="utf-8"))["endpoints"]
    assert endpoints["llm_models_ok"] is False
    assert endpoints["embed_models_ok"] is False
    assert endpoints["rerank_models_ok"] is False
    assert endpoints["llm_response_ms"] is None
    assert endpoints["embed_response_ms"] is None
    assert endpoints["rerank_response_ms"] is None
