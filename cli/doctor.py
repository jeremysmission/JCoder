"""Environment verification -- checks GPU, vLLM, FAISS, tree-sitter, disk."""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import httpx

from core.config import JCoderConfig
from ingestion.chunker import CHUNKER_VERSION, LANGUAGE_MAP


def _check_endpoint(name: str, url: str, timeout: int = 5) -> Tuple[str, bool, str]:
    """Ping a vLLM endpoint. Tries /v1/models first, falls back to /models."""
    client = httpx.Client(timeout=timeout)
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        # vLLM OpenAI-compatible path
        for path in ("/v1/models", "/models"):
            try:
                response = client.get(f"{base}{path}")
                if response.status_code == 200:
                    models = response.json().get("data", [])
                    model_ids = [m.get("id", "?") for m in models]
                    return name, True, f"Online (via {path}), serving: {', '.join(model_ids)}"
            except httpx.ConnectError:
                return name, False, "Connection refused -- server not running"
            except (httpx.HTTPStatusError, KeyError, ValueError):
                continue
        return name, False, "Reachable but /v1/models returned non-200"
    except httpx.ConnectError:
        return name, False, "Connection refused -- server not running"
    except httpx.HTTPError as e:
        return name, False, str(e)
    finally:
        client.close()


def _check_gpu() -> List[dict]:
    """Query nvidia-smi for per-GPU memory stats."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "total_mb": int(parts[2]),
                        "used_mb": int(parts[3]),
                        "free_mb": int(parts[4]),
                    })
    except FileNotFoundError:
        pass
    except (subprocess.SubprocessError, OSError):
        pass

    # Fallback: try pynvml
    if not gpus:
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpus.append({
                    "index": i,
                    "name": name,
                    "total_mb": mem.total // (1024 * 1024),
                    "used_mb": mem.used // (1024 * 1024),
                    "free_mb": mem.free // (1024 * 1024),
                })
            pynvml.nvmlShutdown()
        except (ImportError, OSError):
            pass

    return gpus


def _check_faiss() -> Tuple[bool, str]:
    """Check FAISS installation and GPU support."""
    try:
        import faiss
        if hasattr(faiss, "get_num_gpus"):
            ngpu = faiss.get_num_gpus()
            if ngpu > 0:
                return True, f"faiss-gpu, {ngpu} GPU(s)"
            return False, "faiss-cpu only (0 GPUs detected)"
        return False, "faiss-cpu (no GPU support compiled)"
    except ImportError:
        return False, "faiss not installed"


def run_doctor(config: JCoderConfig):
    """Run all health checks and print results."""
    print("JCoder Doctor")
    print(f"Python {sys.version}")
    print()

    # --- GPU ---
    print("=== GPU ===")
    gpus = _check_gpu()
    if gpus:
        total_vram = 0
        total_free = 0
        for g in gpus:
            total_vram += g["total_mb"]
            total_free += g["free_mb"]
            print(f"[OK] GPU {g['index']}: {g['name']}  "
                  f"Total: {g['total_mb']} MB  "
                  f"Used: {g['used_mb']} MB  "
                  f"Free: {g['free_mb']} MB")
        print(f"     Total VRAM: {total_vram} MB  Free: {total_free} MB  "
              f"Safety margin: {config.policies.gpu_memory_safety_margin_mb} MB")
    else:
        print("[WARN] No GPUs detected (nvidia-smi and pynvml both failed)")

    faiss_ok, faiss_msg = _check_faiss()
    print(f"{'[OK]' if faiss_ok else '[WARN]'} FAISS: {faiss_msg}")
    print()

    # --- vLLM Endpoints ---
    print("=== vLLM Services ===")
    endpoints = [
        ("LLM", config.llm.endpoint),
        ("Embedder", config.embedder.endpoint),
        ("Reranker", config.reranker.endpoint),
    ]
    all_ok = True
    for name, url in endpoints:
        label, ok, msg = _check_endpoint(name, url)
        print(f"{'[OK]' if ok else '[FAIL]'} {label}: {msg}")
        if not ok:
            all_ok = False
    print()

    # --- Disk ---
    print("=== Storage ===")
    data_dir = config.storage.data_dir
    check_path = data_dir if Path(data_dir).exists() else "."
    total, used, free = shutil.disk_usage(check_path)
    print(f"[OK] Disk free: {free // (2**30)} GB (at {check_path})")
    print()

    # --- tree-sitter ---
    print("=== tree-sitter ===")
    installed = []
    missing = []
    for ext, lang in sorted(LANGUAGE_MAP.items()):
        try:
            importlib.import_module(f"tree_sitter_{lang}")
            installed.append(lang)
        except ImportError:
            missing.append(lang)
    print(f"[OK] Installed: {len(installed)}/{len(LANGUAGE_MAP)}")
    if installed:
        print(f"     Ready: {', '.join(installed)}")
    if missing:
        print(f"     Missing: {', '.join(missing)}")
    print()

    # --- Policies ---
    print("=== Policies ===")
    p = config.policies
    print(f"     max_parallel_requests: {p.max_parallel_requests}")
    print(f"     max_context_tokens: {p.max_context_tokens}")
    print(f"     max_chunks_retrieved: {p.max_chunks_retrieved}")
    print(f"     embed_batch_size: {p.embed_batch_size}")
    print(f"     gpu_memory_utilization: {p.gpu_memory_utilization}")
    print(f"     benchmark_hash_verify: {p.benchmark_hash_verify}")
    print()

    if all_ok:
        print("[OK] All systems ready.")
    else:
        print("[WARN] Some services are not running. Start vLLM servers first.")
