"""Portable path helpers for config and script defaults."""

from __future__ import annotations

import os
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _env_value(name: str) -> str | None:
    if name == "JCODER_DATA":
        names = ("JCODER_DATA", "JCODER_DATA_DIR")
    elif name == "JCODER_DATA_DIR":
        names = ("JCODER_DATA_DIR", "JCODER_DATA")
    else:
        names = (name,)
    for candidate in names:
        value = os.environ.get(candidate)
        if value:
            return value
    return None


def get_data_root(default_root: str | Path = "data") -> Path:
    """Return the configured JCoder data root with env fallback."""
    value = _env_value("JCODER_DATA")
    if value:
        return Path(value)
    return Path(default_root)


def expand_path_config(value: str, default_data_root: str | Path = "data") -> str:
    """Expand `${VAR}` and `${VAR:-fallback}` placeholders in config paths."""
    if not value:
        return value

    default_root = str(default_data_root)

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        fallback = match.group(2)
        env_value = _env_value(name)
        if env_value:
            return env_value
        if fallback is not None:
            return fallback
        if name in ("JCODER_DATA", "JCODER_DATA_DIR"):
            return default_root
        return match.group(0)

    expanded = _PLACEHOLDER_RE.sub(_replace, value)
    expanded = os.path.expandvars(expanded)
    expanded = os.path.expanduser(expanded)
    return expanded


def resolve_repo_path(
    value: str | Path,
    *,
    project_root: str | Path | None = None,
    default_data_root: str | Path | None = None,
) -> Path:
    """Resolve a config path relative to the project root when needed."""
    root = Path(project_root) if project_root is not None else _PROJECT_ROOT
    root = root.resolve()
    data_root = Path(default_data_root) if default_data_root is not None else root / "data"
    if not data_root.is_absolute():
        data_root = (root / data_root).resolve()

    expanded = expand_path_config(str(value), default_data_root=data_root)
    path = Path(expanded)
    if os.path.isabs(expanded):
        return path

    # Config values rooted at "data/" should follow the configured data root,
    # not the repo root, so JCODER_DATA / JCODER_DATA_DIR can relocate them.
    if path.parts and path.parts[0] == "data":
        suffix = Path(*path.parts[1:]) if len(path.parts) > 1 else Path()
        return (data_root / suffix).resolve()

    return (root / path).resolve()
