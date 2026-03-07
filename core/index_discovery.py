"""
Index Discovery
----------------
Scans a directory for FTS5 indexes and builds a FederatedSearch instance
from config/memory.yaml settings.

Non-programmer explanation:
JCoder keeps several keyword-search databases (one per corpus like
Stack Overflow, Python docs, etc.). This module finds them on disk,
reads the config that says how important each one is, and wires
them all into the federated search engine so a single query hits
every corpus at once.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import StorageConfig
from .federated_search import FederatedSearch
from .index_engine import IndexEngine

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHT = 1.0
_FTS5_SUFFIX = ".fts5.db"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_fts5_indexes(index_dir: str) -> List[Dict[str, Any]]:
    """Scan *index_dir* for ``*.fts5.db`` files.

    Returns a list of dicts, each with:
        name     -- stem with the ``.fts5`` suffix stripped
        path     -- absolute path to the database
        size_mb  -- file size in megabytes (rounded to 2 dp)

    Raises FileNotFoundError if *index_dir* does not exist.
    """
    dirpath = Path(index_dir)
    if not dirpath.exists():
        raise FileNotFoundError(f"Index directory does not exist: {dirpath}")

    results: List[Dict[str, Any]] = []
    for p in sorted(dirpath.iterdir()):
        if not p.is_file():
            continue
        if not p.name.endswith(_FTS5_SUFFIX):
            continue
        # Strip .fts5.db to get the logical name
        name = p.name[: -len(_FTS5_SUFFIX)]
        size_mb = round(p.stat().st_size / (1024 * 1024), 2)
        results.append({
            "name": name,
            "path": str(p.resolve()),
            "size_mb": size_mb,
        })
        logger.info("Discovered FTS5 index: %s (%.2f MB) at %s", name, size_mb, p)

    logger.info("Discovery complete: %d FTS5 index(es) found in %s", len(results), dirpath)
    return results


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_federated_config(memory_yaml_path: str) -> Dict[str, Any]:
    """Read the ``federated_search`` section from *memory_yaml_path*.

    Returns a dict with:
        rrf_k      -- RRF constant (default 60)
        index_dir  -- where to scan for FTS5 databases
        indexes    -- {name: {weight: float}} mapping

    Raises FileNotFoundError if the YAML file is missing.
    """
    filepath = Path(memory_yaml_path)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    section = raw.get("federated_search", {})
    rrf_k = section.get("rrf_k", 60)
    index_dir = section.get("index_dir", "data/indexes")

    indexes: Dict[str, Dict[str, Any]] = {}
    for name, props in section.get("indexes", {}).items():
        if isinstance(props, dict):
            indexes[name] = {"weight": props.get("weight", _DEFAULT_WEIGHT)}
        else:
            # Bare value treated as weight
            indexes[name] = {"weight": float(props) if props else _DEFAULT_WEIGHT}

    logger.info(
        "Loaded federated config: rrf_k=%d, index_dir=%s, %d configured index(es)",
        rrf_k, index_dir, len(indexes),
    )
    return {"rrf_k": rrf_k, "index_dir": index_dir, "indexes": indexes}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _open_fts5_index(
    db_path: str,
    name: str,
    dimension: int = 768,
) -> Optional[IndexEngine]:
    """Open an FTS5 database as a sparse-only IndexEngine.

    Returns None if the database cannot be opened (logs the reason).
    """
    p = Path(db_path)
    if not p.exists():
        logger.warning("FTS5 database missing, skipping: %s", db_path)
        return None

    # Validate the file is a readable SQLite database
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1 FROM chunks LIMIT 1")
        conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
        logger.warning("FTS5 database unreadable (%s), skipping: %s", exc, db_path)
        return None

    storage = StorageConfig(index_dir=str(p.parent))
    engine = IndexEngine(
        dimension=dimension,
        storage=storage,
        sparse_only=True,
    )
    engine._db_path = db_path
    engine._fts_conn = sqlite3.connect(db_path)

    # Load metadata from companion .meta.json if it exists
    meta_path = p.with_suffix("").with_suffix(".meta.json")
    if meta_path.exists():
        import json
        with open(meta_path, "r", encoding="utf-8") as fh:
            engine.metadata = json.load(fh)

    return engine


def build_federated_from_config(
    config_dict: Dict[str, Any],
    index_dir: str,
) -> FederatedSearch:
    """Create a FederatedSearch wired to every discovered FTS5 index.

    *config_dict* is the output of :func:`load_federated_config`.
    *index_dir* is the directory to scan (overrides config_dict['index_dir']
    if provided).

    Indexes listed in config get their configured weight.
    Indexes found on disk but NOT in config get default weight 1.0.
    Indexes in config but NOT on disk are logged and skipped.
    """
    rrf_k = config_dict.get("rrf_k", 60)
    configured_indexes = config_dict.get("indexes", {})

    fed = FederatedSearch(embedding_engine=None, rrf_k=rrf_k)

    # Discover what is on disk
    dir_path = Path(index_dir)
    if not dir_path.exists():
        logger.warning(
            "Index directory does not exist: %s -- returning empty federation",
            index_dir,
        )
        return fed

    discovered = discover_fts5_indexes(index_dir)
    discovered_names = {d["name"] for d in discovered}

    # Load each discovered index
    loaded = 0
    for info in discovered:
        name = info["name"]
        db_path = info["path"]

        engine = _open_fts5_index(db_path, name)
        if engine is None:
            continue

        weight = configured_indexes.get(name, {}).get("weight", _DEFAULT_WEIGHT)
        if name not in configured_indexes:
            logger.info(
                "Index '%s' not in config -- using default weight %.1f",
                name, _DEFAULT_WEIGHT,
            )
        else:
            logger.info(
                "Index '%s' loaded with configured weight %.1f",
                name, weight,
            )

        fed.add_index(name, engine, weight=weight)
        loaded += 1

    # Warn about configured indexes that were not found on disk
    for name in configured_indexes:
        if name not in discovered_names:
            logger.warning(
                "Configured index '%s' not found in %s -- skipped",
                name, index_dir,
            )

    logger.info(
        "Federated search ready: %d index(es) loaded, rrf_k=%d",
        loaded, rrf_k,
    )
    return fed
