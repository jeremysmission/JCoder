"""Click CLI health-check commands for verifying JCoder environment."""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import click
from rich.console import Console
from rich.table import Table

console = Console()

# Project root: two levels up from cli/doctor_cmd.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories that must exist for normal operation
_REQUIRED_DIRS = [
    "data/indexes",
    "data/agent_knowledge",
    "data/agent_sessions",
    "logs/agent",
]

# Packages to probe (import_name, pip_name)
_REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    ("yaml", "pyyaml"),
    ("click", "click"),
    ("rich", "rich"),
    ("httpx", "httpx"),
    ("numpy", "numpy"),
    ("faiss", "faiss-cpu"),
]


class _CheckResult:
    """Accumulator for check outcomes."""

    def __init__(self):
        self.rows: List[Tuple[str, str, str]] = []  # (tag, check, detail)

    def ok(self, check: str, detail: str = ""):
        self.rows.append(("OK", check, detail))

    def warn(self, check: str, detail: str = ""):
        self.rows.append(("WARN", check, detail))

    def fail(self, check: str, detail: str = ""):
        self.rows.append(("FAIL", check, detail))

    @property
    def pass_count(self) -> int:
        return sum(1 for t, _, _ in self.rows if t == "OK")

    @property
    def warn_count(self) -> int:
        return sum(1 for t, _, _ in self.rows if t == "WARN")

    @property
    def fail_count(self) -> int:
        return sum(1 for t, _, _ in self.rows if t == "FAIL")

    def print_live(self, tag: str, check: str, detail: str):
        """Print a single result line immediately."""
        colour = {"OK": "bold green", "WARN": "bold yellow", "FAIL": "bold red"}[tag]
        prefix = f"[{colour}][{tag}][/{colour}]"
        msg = f" {check}" + (f" -- {detail}" if detail else "")
        console.print(f"{prefix}{msg}")


# -- Individual checks -------------------------------------------------------

def _check_python(cr: _CheckResult) -> None:
    vi = sys.version_info
    version_str = f"{vi.major}.{vi.minor}.{vi.micro}"
    if (vi.major, vi.minor) >= (3, 10):
        cr.ok("Python version", version_str)
        cr.print_live("OK", "Python version", version_str)
    else:
        cr.fail("Python version", f"{version_str} (need 3.10+)")
        cr.print_live("FAIL", "Python version", f"{version_str} (need 3.10+)")


def _check_packages(cr: _CheckResult) -> None:
    missing = []
    for imp_name, pip_name in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(imp_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        cr.ok("Required packages", f"all {len(_REQUIRED_PACKAGES)} found")
        cr.print_live("OK", "Required packages", f"all {len(_REQUIRED_PACKAGES)} found")
    else:
        detail = f"missing: {', '.join(missing)}"
        cr.warn("Required packages", detail)
        cr.print_live("WARN", "Required packages", detail)


def _check_configs(cr: _CheckResult) -> None:
    configs = ["config/agent.yaml", "config/memory.yaml"]
    for cfg_rel in configs:
        cfg_path = _PROJECT_ROOT / cfg_rel
        if cfg_path.is_file():
            cr.ok(f"Config {cfg_rel}", str(cfg_path))
            cr.print_live("OK", f"Config {cfg_rel}", str(cfg_path))
        else:
            cr.fail(f"Config {cfg_rel}", "not found")
            cr.print_live("FAIL", f"Config {cfg_rel}", "not found")


def _load_agent_yaml() -> dict:
    """Load config/agent.yaml and return as dict, or empty dict on failure."""
    cfg_path = _PROJECT_ROOT / "config" / "agent.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _check_ollama(cr: _CheckResult) -> List[str]:
    """Check Ollama server. Returns list of pulled model names."""
    pulled_models: List[str] = []
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            pulled_models = [m.get("name", "?") for m in models]
            detail = f"{len(pulled_models)} model(s): {', '.join(pulled_models[:8])}"
            if len(pulled_models) > 8:
                detail += f" (+{len(pulled_models) - 8} more)"
            cr.ok("Ollama server", detail)
            cr.print_live("OK", "Ollama server", detail)
        else:
            cr.warn("Ollama server", f"responded with HTTP {resp.status_code}")
            cr.print_live("WARN", "Ollama server", f"responded with HTTP {resp.status_code}")
    except ImportError:
        cr.warn("Ollama server", "httpx not installed, cannot probe")
        cr.print_live("WARN", "Ollama server", "httpx not installed, cannot probe")
    except Exception:
        cr.fail("Ollama server", "not reachable at localhost:11434")
        cr.print_live("FAIL", "Ollama server", "not reachable at localhost:11434")
    return pulled_models


def _check_required_models(cr: _CheckResult, pulled_models: List[str]) -> None:
    agent_cfg = _load_agent_yaml()
    agent_section = agent_cfg.get("agent", {})

    configured_model = (
        agent_section.get("ollama_model", "")
        or agent_section.get("model", "")
    )

    if not configured_model:
        cr.warn("Required models", "no model configured in agent.yaml")
        cr.print_live("WARN", "Required models", "no model configured in agent.yaml")
        return

    if not pulled_models:
        cr.warn("Required models", f"cannot verify '{configured_model}' (Ollama offline)")
        cr.print_live("WARN", "Required models",
                      f"cannot verify '{configured_model}' (Ollama offline)")
        return

    # Ollama model names may include tags; match base name too
    normalised = [m.split(":")[0] for m in pulled_models]
    cfg_base = configured_model.split(":")[0]
    if configured_model in pulled_models or cfg_base in normalised:
        cr.ok("Required models", f"'{configured_model}' is pulled")
        cr.print_live("OK", "Required models", f"'{configured_model}' is pulled")
    else:
        cr.fail("Required models",
                f"'{configured_model}' not found in Ollama (pull it first)")
        cr.print_live("FAIL", "Required models",
                      f"'{configured_model}' not found in Ollama (pull it first)")

    # Check embedder model (R12.4: nomic-embed-code)
    try:
        import yaml
        models_path = _PROJECT_ROOT / "config" / "models.yaml"
        if models_path.is_file():
            models_cfg = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
            embed_name = models_cfg.get("embedder", {}).get("name", "")
            if embed_name:
                embed_base = embed_name.split(":")[0]
                if embed_name in pulled_models or embed_base in normalised:
                    cr.ok("Embedder model", f"'{embed_name}' is pulled")
                    cr.print_live("OK", "Embedder model", f"'{embed_name}' is pulled")
                else:
                    cr.warn("Embedder model",
                            f"'{embed_name}' not pulled (run: ollama pull {embed_name})")
                    cr.print_live("WARN", "Embedder model",
                                  f"'{embed_name}' not pulled (run: ollama pull {embed_name})")
    except Exception:
        pass  # Non-critical


def _check_data_dirs(cr: _CheckResult) -> None:
    for rel in _REQUIRED_DIRS:
        dir_path = _PROJECT_ROOT / rel
        if dir_path.is_dir():
            cr.ok(f"Directory {rel}", str(dir_path))
            cr.print_live("OK", f"Directory {rel}", str(dir_path))
        else:
            cr.warn(f"Directory {rel}", "missing (use 'doctor fix' to create)")
            cr.print_live("WARN", f"Directory {rel}",
                          "missing (use 'doctor fix' to create)")


def _check_disk_space(cr: _CheckResult) -> None:
    try:
        total, _used, free = shutil.disk_usage(str(_PROJECT_ROOT))
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        detail = f"{free_gb:.1f} GB free of {total_gb:.1f} GB on {_PROJECT_ROOT.anchor}"
        if free_gb < 5.0:
            cr.warn("Disk space", detail)
            cr.print_live("WARN", "Disk space", detail)
        else:
            cr.ok("Disk space", detail)
            cr.print_live("OK", "Disk space", detail)
    except OSError as exc:
        cr.warn("Disk space", str(exc))
        cr.print_live("WARN", "Disk space", str(exc))


def _check_gpu(cr: _CheckResult) -> None:
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
            free_mb = mem.free // (1024 * 1024)
            total_mb = mem.total // (1024 * 1024)
            detail = f"{name} -- {free_mb} MB free / {total_mb} MB total"
            cr.ok(f"GPU {i}", detail)
            cr.print_live("OK", f"GPU {i}", detail)
        pynvml.nvmlShutdown()
    except ImportError:
        cr.warn("GPU/VRAM", "pynvml not installed (pip install pynvml)")
        cr.print_live("WARN", "GPU/VRAM", "pynvml not installed (pip install pynvml)")
    except Exception as exc:
        cr.warn("GPU/VRAM", f"pynvml error: {exc}")
        cr.print_live("WARN", "GPU/VRAM", f"pynvml error: {exc}")


def _load_memory_yaml() -> dict:
    """Load config/memory.yaml and return as dict, or empty dict on failure."""
    cfg_path = _PROJECT_ROOT / "config" / "memory.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _check_fts5_indexes(cr: _CheckResult) -> None:
    mem_yaml = _load_memory_yaml()
    memory_cfg = mem_yaml.get("memory", {})
    fed_cfg = mem_yaml.get("federated_search", {})

    # Determine index directories: repo-local memory dir + configured federated dir
    memory_index_dir = memory_cfg.get("index_dir", "data/indexes")
    fed_data_dir = fed_cfg.get("data_dir", "")

    scan_dirs: List[Tuple[str, Path]] = []
    # Resolve memory index dir (may be relative to project root)
    mem_path = Path(memory_index_dir)
    if not mem_path.is_absolute():
        mem_path = _PROJECT_ROOT / mem_path
    scan_dirs.append(("memory_index_dir", mem_path))

    # Federated data dir (often an absolute external path)
    if fed_data_dir:
        fed_path = Path(fed_data_dir)
        if not fed_path.is_absolute():
            fed_path = _PROJECT_ROOT / fed_path
        if fed_path != mem_path:  # avoid duplicate scan
            scan_dirs.append(("federated_data_dir", fed_path))

    all_fts: List[str] = []
    for label, idx_dir in scan_dirs:
        if not idx_dir.is_dir():
            cr.warn(f"FTS5 indexes ({label})", f"{idx_dir} not found")
            cr.print_live("WARN", f"FTS5 indexes ({label})", f"{idx_dir} not found")
            continue
        fts_files = list(idx_dir.glob("*.fts5.db"))
        if fts_files:
            names = [f.name for f in fts_files]
            all_fts.extend(names)
            detail = f"{len(fts_files)} in {idx_dir}"
            if len(names) <= 6:
                detail += f": {', '.join(names)}"
            else:
                detail += f": {', '.join(names[:6])} (+{len(names)-6} more)"
            cr.ok(f"FTS5 indexes ({label})", detail)
            cr.print_live("OK", f"FTS5 indexes ({label})", detail)
        else:
            cr.warn(f"FTS5 indexes ({label})", f"none found in {idx_dir}")
            cr.print_live("WARN", f"FTS5 indexes ({label})", f"none found in {idx_dir}")

    if not all_fts:
        cr.warn("FTS5 indexes", "no FTS5 indexes found in any configured directory")
        cr.print_live("WARN", "FTS5 indexes",
                      "no FTS5 indexes found in any configured directory")


def _check_hybrid_pipeline(cr: _CheckResult) -> None:
    """Verify the hybrid search pipeline can load and search."""
    try:
        from core.config import load_config
        from core.embedding_engine import EmbeddingEngine

        cfg = load_config()
        engine = EmbeddingEngine(cfg.embedder, timeout=10)
        vec = engine.embed_single("test query")
        engine.close()
        dim = vec.shape[0]
        cr.ok("Hybrid pipeline", f"embed OK (dim={dim}), model={cfg.embedder.name}")
        cr.print_live("OK", "Hybrid pipeline", f"embed OK (dim={dim})")
    except Exception as e:
        cr.warn("Hybrid pipeline", f"embed failed: {e}")
        cr.print_live("WARN", "Hybrid pipeline", str(e))


def _check_agent_package(cr: _CheckResult) -> None:
    errors = []
    for name in ("Agent", "create_backend", "ToolRegistry"):
        try:
            mod = importlib.import_module("agent")
            if not hasattr(mod, name):
                errors.append(f"{name} not in agent module")
        except ImportError:
            errors.append(f"agent module not importable")
            break
    if not errors:
        cr.ok("Agent package", "Agent, create_backend, ToolRegistry all importable")
        cr.print_live("OK", "Agent package",
                      "Agent, create_backend, ToolRegistry all importable")
    else:
        detail = "; ".join(errors)
        cr.warn("Agent package", detail)
        cr.print_live("WARN", "Agent package", detail)


# -- CLI commands -------------------------------------------------------------

@click.group("doctor")
def doctor_cmd():
    """Environment health checks for JCoder."""


@doctor_cmd.command("check")
def check():
    """Run all environment health checks."""
    cr = _CheckResult()

    console.print("[bold]JCoder Doctor[/bold]\n")

    console.print("[bold underline]1. Python version[/bold underline]")
    _check_python(cr)

    console.print("\n[bold underline]2. Required packages[/bold underline]")
    _check_packages(cr)

    console.print("\n[bold underline]3. Config files[/bold underline]")
    _check_configs(cr)

    console.print("\n[bold underline]4. Ollama server[/bold underline]")
    pulled = _check_ollama(cr)

    console.print("\n[bold underline]5. Required models[/bold underline]")
    _check_required_models(cr, pulled)

    console.print("\n[bold underline]6. Data directories[/bold underline]")
    _check_data_dirs(cr)

    console.print("\n[bold underline]7. Disk space[/bold underline]")
    _check_disk_space(cr)

    console.print("\n[bold underline]8. GPU / VRAM[/bold underline]")
    _check_gpu(cr)

    console.print("\n[bold underline]9. FTS5 indexes[/bold underline]")
    _check_fts5_indexes(cr)

    console.print("\n[bold underline]10. Hybrid search pipeline[/bold underline]")
    _check_hybrid_pipeline(cr)

    console.print("\n[bold underline]11. Agent package[/bold underline]")
    _check_agent_package(cr)

    console.print("\n[bold underline]12. FlashRank reranker[/bold underline]")
    try:
        from core.fusion import HAS_FLASHRANK
        if HAS_FLASHRANK:
            cr.ok("FlashRank reranker", "available (TinyBERT-L-2)")
        else:
            cr.warn("FlashRank reranker", "not installed (pip install flashrank)")
    except Exception as e:
        cr.warn("FlashRank reranker", f"import error: {e}")

    # -- Summary table --
    console.print()
    table = Table(title="Doctor Summary")
    table.add_column("Check", style="bold", min_width=25)
    table.add_column("Result", min_width=8)
    table.add_column("Detail")

    for tag, name, detail in cr.rows:
        colour = {"OK": "green", "WARN": "yellow", "FAIL": "red"}[tag]
        table.add_row(name, f"[{colour}][{tag}][/{colour}]", detail)

    console.print(table)

    console.print(
        f"\n[bold green]{cr.pass_count} passed[/bold green]  "
        f"[bold yellow]{cr.warn_count} warnings[/bold yellow]  "
        f"[bold red]{cr.fail_count} failures[/bold red]"
    )

    if cr.fail_count:
        raise SystemExit(1)


@doctor_cmd.command("fix")
def fix():
    """Auto-create missing data directories."""
    console.print("[bold]JCoder Doctor -- Fix[/bold]\n")
    created = 0
    for rel in _REQUIRED_DIRS:
        dir_path = _PROJECT_ROOT / rel
        if dir_path.is_dir():
            console.print(f"[bold green][OK][/bold green] {rel} already exists")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[bold green][OK][/bold green] Created {rel}")
            created += 1

    if created:
        console.print(f"\n[bold]Created {created} directory(ies).[/bold]")
    else:
        console.print("\n[bold]All directories already exist. Nothing to do.[/bold]")


# -----------------------------------------------------------------------
# Wire into main CLI by adding to the bottom of cli/commands.py:
#
#   from cli.doctor_cmd import doctor_cmd      # noqa: E402
#   cli.add_command(doctor_cmd)
# -----------------------------------------------------------------------
