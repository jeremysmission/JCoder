from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path


class DownloadStaging:
    """Manage incoming, final, and quarantine directories for downloads."""

    def __init__(self, base_path: str, final_root: str | None = None) -> None:
        self.base = Path(base_path)
        self.incoming = self.base / "incoming"
        self.verified = Path(final_root) if final_root else self.base / "verified"
        self.quarantine = self.base / "quarantine"
        self._lock = threading.Lock()
        for directory in (self.incoming, self.verified, self.quarantine):
            directory.mkdir(parents=True, exist_ok=True)

    def incoming_path(self, relative_path: str) -> Path:
        path = self.incoming / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.with_suffix(path.suffix + ".part")

    def verified_path(self, relative_path: str) -> Path:
        path = self.verified / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def promote(self, temp_path: Path, relative_path: str) -> Path:
        final_path = self.verified_path(relative_path)
        with self._lock:
            if final_path.exists():
                final_path.unlink()
            try:
                os.replace(str(temp_path), str(final_path))
            except OSError:
                shutil.move(str(temp_path), str(final_path))
        return final_path

    def quarantine_file(self, path: Path, relative_path: str, reason: str = "") -> Path:
        target = self.quarantine / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            stem = target.stem
            suffix = target.suffix
            counter = 1
            while target.exists():
                target = target.parent / f"{stem}_{counter}{suffix}"
                counter += 1
            try:
                os.replace(str(path), str(target))
            except OSError:
                shutil.move(str(path), str(target))
        if reason:
            target.with_suffix(target.suffix + ".reason").write_text(
                reason,
                encoding="utf-8",
            )
        return target

    def cleanup_incoming(self) -> int:
        removed = 0
        for path in self.incoming.rglob("*.part"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed
