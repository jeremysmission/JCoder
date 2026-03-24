"""
BEAST Hardware Bootstrap Script
===============================
One-command setup for the BEAST machine (128 GB RAM, 48 GB VRAM).
Pulls models, validates GPU, runs initial eval smoke test.

Usage:
    python scripts/bootstrap_beast.py              # full setup
    python scripts/bootstrap_beast.py --check-only # just validate
    python scripts/bootstrap_beast.py --skip-eval  # skip eval smoke test
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("bootstrap_beast")

try:
    import psutil as _psutil
except ImportError:
    _psutil = types.ModuleType("psutil")

    def _missing_virtual_memory():
        raise ImportError("psutil is not installed")

    _psutil.virtual_memory = _missing_virtual_memory
    sys.modules.setdefault("psutil", _psutil)

# Models to pull on BEAST (all approved -- no restrictions for JCoder)
REQUIRED_MODELS: List[Dict[str, str]] = [
    {
        "name": "phi4:14b-q4_K_M",
        "description": "Primary code/reasoning model (14B, Q4)",
        "min_vram_gb": 9,
    },
    {
        "name": "nomic-embed-text",
        "description": "Code/text embedder (768-dim, Apache 2.0)",
        "min_vram_gb": 1,
    },
]

OPTIONAL_MODELS: List[Dict[str, str]] = [
    {
        "name": "devstral:24b",
        "description": "Fallback code model (24B, Mistral)",
        "min_vram_gb": 14,
    },
    {
        "name": "mistral-nemo:12b",
        "description": "General-purpose fallback (12B, Mistral)",
        "min_vram_gb": 7,
    },
]

# Minimum hardware thresholds
MIN_RAM_GB = 64
MIN_VRAM_GB = 24
MIN_DISK_FREE_GB = 50


def _run(cmd: List[str], timeout: int = 300) -> Tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


# ---- GPU Detection ----

def detect_gpu() -> Dict:
    """Detect NVIDIA GPU using nvidia-smi."""
    info = {"available": False, "gpus": [], "total_vram_gb": 0.0}
    rc, stdout, stderr = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0:
        log.warning("nvidia-smi failed: %s", stderr.strip())
        return info
    for line in stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpu = {
            "index": int(parts[0]),
            "name": parts[1],
            "total_mb": float(parts[2]),
            "free_mb": float(parts[3]),
            "driver": parts[4],
        }
        info["gpus"].append(gpu)
        info["total_vram_gb"] += gpu["total_mb"] / 1024
    info["available"] = len(info["gpus"]) > 0
    return info


def detect_ram() -> float:
    """Return total system RAM in GB."""
    try:
        return _psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    # Fallback: read from /proc or wmic
    rc, stdout, _ = _run(["wmic", "computersystem", "get", "TotalPhysicalMemory"])
    if rc == 0:
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if line.isdigit():
                return int(line) / (1024 ** 3)
    return 0.0


def detect_disk_free(path: str = ".") -> float:
    """Return free disk space in GB."""
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


# ---- Ollama ----

def check_ollama() -> bool:
    """Check if Ollama is running and responsive."""
    rc, stdout, _ = _run(["ollama", "list"])
    return rc == 0


def list_installed_models() -> List[str]:
    """Return list of installed Ollama model names."""
    rc, stdout, _ = _run(["ollama", "list"])
    if rc != 0:
        return []
    models = []
    for line in stdout.strip().split("\n")[1:]:  # skip header
        if line.strip():
            models.append(line.split()[0])
    return models


def pull_model(name: str) -> bool:
    """Pull a model via Ollama. Returns True on success."""
    print(f"  Pulling {name}...", flush=True)
    rc, stdout, stderr = _run(["ollama", "pull", name], timeout=1800)
    if rc == 0:
        print(f"  [OK] {name} pulled successfully")
        return True
    print(f"  [FAIL] {name}: {stderr.strip()[:200]}")
    return False


def validate_model(name: str) -> Tuple[bool, float]:
    """Generate a test response to validate model loads. Returns (ok, latency_s)."""
    t0 = time.time()
    rc, stdout, stderr = _run(
        ["ollama", "run", name, "Write a Python hello world in 1 line"],
        timeout=120,
    )
    elapsed = time.time() - t0
    if rc == 0 and len(stdout.strip()) > 5:
        return True, elapsed
    return False, elapsed


# ---- FAISS GPU ----

def check_faiss_gpu() -> bool:
    """Check if faiss-gpu is importable and can see GPUs."""
    try:
        code = (
            "import faiss; "
            "n = faiss.get_num_gpus(); "
            "print(f'faiss_gpus={n}')"
        )
        rc, stdout, _ = _run([sys.executable, "-c", code])
        if rc == 0 and "faiss_gpus=" in stdout:
            n = int(stdout.split("=")[1].strip())
            return n > 0
    except Exception:
        pass
    return False


# ---- Eval Smoke Test ----

def run_eval_smoke(max_questions: int = 10) -> Optional[Dict]:
    """Run a quick eval smoke test (first N questions)."""
    eval_set = Path(__file__).parent.parent / "evaluation" / "agent_eval_set_200.json"
    if not eval_set.exists():
        log.warning("Eval set not found: %s", eval_set)
        return None
    try:
        rc, stdout, stderr = _run([
            sys.executable, "-m", "scripts.run_eval_local",
            "--max", str(max_questions),
            "--eval-set", str(eval_set),
        ], timeout=600)
        if rc == 0:
            # Try to parse results
            results_path = Path("evaluation/results/eval_results.json")
            if results_path.exists():
                with open(results_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                passed = sum(1 for r in results if r.get("passed"))
                return {
                    "total": len(results),
                    "passed": passed,
                    "pass_rate": passed / len(results) if results else 0,
                }
        log.warning("Eval run failed: %s", stderr[:300])
    except Exception as exc:
        log.warning("Eval smoke test error: %s", exc)
    return None


# ---- Main ----

def bootstrap(check_only: bool = False, skip_eval: bool = False) -> bool:
    """Run the full BEAST bootstrap sequence. Returns True if ready."""
    print("=" * 60)
    print("  BEAST Hardware Bootstrap")
    print("=" * 60)
    print()
    ok = True

    # 1. Hardware detection
    print("[1/6] Detecting hardware...")
    gpu_info = detect_gpu()
    ram_gb = detect_ram()
    disk_gb = detect_disk_free()

    print(f"  RAM:  {ram_gb:.1f} GB (min: {MIN_RAM_GB} GB)")
    print(f"  VRAM: {gpu_info['total_vram_gb']:.1f} GB (min: {MIN_VRAM_GB} GB)")
    print(f"  Disk: {disk_gb:.1f} GB free (min: {MIN_DISK_FREE_GB} GB)")

    if gpu_info["available"]:
        for g in gpu_info["gpus"]:
            print(f"  GPU {g['index']}: {g['name']} ({g['total_mb']:.0f} MB, "
                  f"free {g['free_mb']:.0f} MB, driver {g['driver']})")
    else:
        print("  [WARN] No NVIDIA GPU detected")
        ok = False

    if ram_gb < MIN_RAM_GB:
        print(f"  [WARN] RAM below minimum ({ram_gb:.0f} < {MIN_RAM_GB})")
    if gpu_info["total_vram_gb"] < MIN_VRAM_GB:
        print(f"  [WARN] VRAM below minimum ({gpu_info['total_vram_gb']:.0f} < {MIN_VRAM_GB})")
    if disk_gb < MIN_DISK_FREE_GB:
        print(f"  [WARN] Disk space below minimum ({disk_gb:.0f} < {MIN_DISK_FREE_GB})")
        ok = False
    print()

    # 2. Ollama
    print("[2/6] Checking Ollama...")
    ollama_ok = check_ollama()
    if ollama_ok:
        installed = list_installed_models()
        print(f"  [OK] Ollama running, {len(installed)} model(s) installed")
        if installed:
            for m in installed[:10]:
                print(f"    - {m}")
    else:
        print("  [FAIL] Ollama not running or not installed")
        print("  Install: https://ollama.com/download")
        ok = False
    print()

    # 3. FAISS GPU
    print("[3/6] Checking FAISS GPU...")
    faiss_gpu = check_faiss_gpu()
    if faiss_gpu:
        print("  [OK] faiss-gpu available with GPU support")
    else:
        print("  [WARN] faiss-gpu not available (will use CPU fallback)")
        print("  Install: pip install faiss-gpu-cu12")
    print()

    if check_only:
        print("Check-only mode -- skipping model pulls and eval.")
        return ok

    # 4. Pull models
    print("[4/6] Pulling required models...")
    if not ollama_ok:
        print("  [SKIP] Ollama not available")
    else:
        installed = list_installed_models()
        for model in REQUIRED_MODELS:
            name = model["name"]
            if any(name in m for m in installed):
                print(f"  [OK] {name} already installed")
            else:
                pulled = pull_model(name)
                if not pulled:
                    ok = False

        print()
        print("  Optional models:")
        for model in OPTIONAL_MODELS:
            name = model["name"]
            if any(name in m for m in installed):
                print(f"  [OK] {name} already installed")
            else:
                pull_model(name)  # optional, don't fail on these
    print()

    # 5. Validate models
    print("[5/6] Validating model loading...")
    if not ollama_ok:
        print("  [SKIP] Ollama not available")
    else:
        for model in REQUIRED_MODELS:
            name = model["name"]
            loaded, latency = validate_model(name)
            if loaded:
                print(f"  [OK] {name}: loaded in {latency:.1f}s")
            else:
                print(f"  [FAIL] {name}: failed to load ({latency:.1f}s)")
                ok = False
    print()

    # 6. Eval smoke test
    if skip_eval:
        print("[6/6] Eval smoke test: SKIPPED")
    else:
        print("[6/6] Running eval smoke test (10 questions)...")
        result = run_eval_smoke(10)
        if result:
            rate = result["pass_rate"]
            status = "[OK]" if rate >= 0.5 else "[WARN]"
            print(f"  {status} {result['passed']}/{result['total']} passed "
                  f"({rate:.0%})")
        else:
            print("  [SKIP] Could not run eval (model or eval set unavailable)")
    print()

    # Summary
    print("=" * 60)
    if ok:
        print("  BEAST is READY")
        print()
        print("  Next steps:")
        print("  1. Run full eval: python scripts/run_eval_local.py --max 200")
        print("  2. Build indexes: python scripts/build_fts5_indexes.py")
        print("  3. Run learning: python scripts/learning_cycle.py")
        print("  4. Close loop:   python scripts/close_feedback_loop.py")
    else:
        print("  BEAST setup INCOMPLETE -- fix warnings above")
    print("=" * 60)
    return ok


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="BEAST Hardware Bootstrap")
    ap.add_argument("--check-only", action="store_true",
                    help="Only check hardware, don't pull models")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Skip the eval smoke test")
    args = ap.parse_args()
    success = bootstrap(check_only=args.check_only, skip_eval=args.skip_eval)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
