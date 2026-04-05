"""
YAML Loading and Config Resolution Helpers
-------------------------------------------
Extracted from config_loader.py to stay under 500 LOC per module.

Contains:
  _load_yaml()             -- safe YAML file loader
  _find_config_dir()       -- locate config directory
  _resolve_backend_defaults() -- backend/model/endpoint defaults
  _discover_fts5_indexes() -- scan directory for FTS5 databases
  _load_fts5_index()       -- create sparse IndexEngine from FTS5 db
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import yaml

if TYPE_CHECKING:
    from core.index_engine import IndexEngine

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load a YAML file, return empty dict if missing."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_config_dir(config_dir: Optional[str] = None) -> Path:
    """Locate the config directory."""
    if config_dir:
        return Path(config_dir)
    env = os.environ.get("JCODER_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "config"


def _resolve_backend_defaults(agent_raw: dict) -> tuple:
    """Resolve model/endpoint/api_key_env from backend type and YAML fields.

    Returns (backend, model, endpoint, api_key_env).
    """
    backend = agent_raw.get("backend", "openai")
    model = agent_raw.get("model", "")
    endpoint = agent_raw.get("endpoint", "")

    # Auto-fill from backend-specific defaults when not explicitly set
    if backend == "ollama" or (backend == "openai" and not endpoint and not model):
        model = model or agent_raw.get("ollama_model", "phi4:14b-q4_K_M")
        endpoint = endpoint or agent_raw.get("ollama_endpoint", "http://localhost:11434/v1")
        api_key_env = ""
    elif backend == "anthropic":
        model = model or agent_raw.get("api_model", "claude-sonnet-4-20250514")
        endpoint = ""  # Anthropic backend uses hardcoded URL
        api_key_env = "ANTHROPIC_API_KEY"
    else:
        # openai with explicit endpoint (OpenRouter, Azure, vLLM)
        model = model or agent_raw.get("api_model", "")
        endpoint = endpoint or agent_raw.get("api_endpoint", "")
        api_key_env = "OPENROUTER_API_KEY" if "openrouter" in endpoint.lower() else "OPENAI_API_KEY"

    return backend, model, endpoint, api_key_env


# ---------------------------------------------------------------------------
# FTS5 index discovery and loading
# ---------------------------------------------------------------------------

def _discover_fts5_indexes(index_dir: str) -> Dict[str, str]:
    """Scan a directory for *.fts5.db files and return {name: path}.

    Names are derived by stripping the .fts5.db suffix from the filename.
    Only files that actually contain a ``chunks`` FTS5 table are included.
    """
    import sqlite3 as _sqlite3

    found: Dict[str, str] = {}
    if not index_dir or not os.path.isdir(index_dir):
        return found

    for entry in os.scandir(index_dir):
        if not entry.name.endswith(".fts5.db") or not entry.is_file():
            continue
        name = entry.name[: -len(".fts5.db")]
        # Quick validation: does it have a chunks table?
        try:
            conn = _sqlite3.connect(entry.path)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            conn.close()
            if "chunks" in tables:
                found[name] = entry.path
        except _sqlite3.Error:
            continue
    return found


def _load_fts5_index(db_path: str, name: str) -> "IndexEngine":
    """Create a sparse-only IndexEngine pointing at an existing FTS5 database.

    Uses lazy loading: metadata stays empty, chunk count comes from a fast
    COUNT query, and search_fts5_direct() fetches content on demand.
    This avoids loading millions of rows into RAM for large indexes.
    """
    import json as _json
    import sqlite3 as _sqlite3
    from core.config import StorageConfig
    from core.index_engine import IndexEngine

    index_dir = os.path.dirname(db_path)
    storage = StorageConfig(data_dir=index_dir, index_dir=index_dir)
    eng = IndexEngine(dimension=768, storage=storage, sparse_only=True)
    eng._db_path = db_path
    # Keep federated indexes lazy so IndexEngine opens a worker-safe FTS5
    # connection with check_same_thread=False on first search.
    eng._fts_conn = None

    # Load .meta.json only for small indexes (<1 MB).
    # Large .meta.json files (e.g. 332K entries) eat hundreds of MB of RAM.
    _META_SIZE_LIMIT = 1 * 1024 * 1024  # 1 MB
    meta_path = os.path.join(index_dir, name + ".meta.json")
    if os.path.isfile(meta_path):
        try:
            if os.path.getsize(meta_path) <= _META_SIZE_LIMIT:
                with open(meta_path, "r", encoding="utf-8") as f:
                    eng.metadata = _json.load(f)
                return eng
        except (OSError, ValueError):
            pass  # fall through to lazy mode

    # Lazy mode: skip counting (FTS5 COUNT(*) scans all rows).
    # FederatedSearch uses search_fts5_direct() for these indexes.
    # File size is a fast proxy: ~80 KB per chunk in typical FTS5.
    try:
        file_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        eng._fts5_chunk_count = max(1, int(file_size_mb * 12.5))  # estimate
    except OSError:
        eng._fts5_chunk_count = 1  # at least 1 -- we know chunks table exists
    eng.metadata = []

    return eng
