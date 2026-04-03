"""Archive handling for the sanitization pipeline.

Extracted from sanitizer.py to keep module under 500 LOC limit.
Contains: 7z, zip, tar, and zstd archive processing methods.
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

try:
    import pyzstd as zstd
except Exception:  # pragma: no cover
    zstd = None

try:
    import py7zr
except Exception:  # pragma: no cover
    py7zr = None

if TYPE_CHECKING:
    from ingestion.sanitizer import SanitizationStats

MAGIC_7Z = bytes.fromhex("377ABCAF271C")
MAGIC_ZST = bytes.fromhex("28B52FFD")


class ArchiveProcessorMixin:
    """Mixin providing archive processing methods for SanitizationPipeline."""

    def _process_7z_archive(self, fp: Path, run_dir: Path, stats: SanitizationStats) -> None:
        if py7zr is None:
            stats.compressed_skipped += 1
            stats.skipped_files.append(f"{fp} [missing_py7zr]")
            return
        try:
            with open(fp, "rb") as f:
                head = f.read(6)
            if not head.startswith(MAGIC_7Z):
                stats.compressed_skipped += 1
                stats.skipped_files.append(f"{fp} [invalid_7z_magic:{head.hex()}]")
                return
        except Exception:
            stats.compressed_skipped += 1
            stats.skipped_files.append(f"{fp} [read_error]")
            return

        lower = str(fp).lower()
        if ("stackexchange" not in lower and "stackoverflow" not in lower
                and "serverfault" not in lower and "superuser" not in lower):
            stats.compressed_skipped += 1
            stats.skipped_files.append(f"{fp} [archive_not_targeted]")
            return

        tmp_dir = Path(tempfile.mkdtemp(prefix="jcoder_7z_", dir=str(self.clean_root)))
        try:
            with py7zr.SevenZipFile(fp, mode="r") as zf:
                names = zf.getnames()
                post_targets = [n for n in names if n.lower().endswith("posts.xml")]
                if not post_targets:
                    stats.compressed_skipped += 1
                    stats.skipped_files.append(f"{fp} [posts_xml_not_found]")
                    return
                zf.extract(path=tmp_dir, targets=post_targets)
            for rel in post_targets:
                extracted = tmp_dir / rel
                if extracted.exists():
                    self._process_stackexchange_posts(extracted, run_dir, stats)
        except Exception as e:
            stats.compressed_skipped += 1
            stats.errors.append(f"{fp} [7z_error] {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _process_standard_archive(self, fp: Path, run_dir: Path, stats: SanitizationStats) -> None:
        ext = fp.suffix.lower()
        tmp_dir = Path(tempfile.mkdtemp(prefix="jcoder_arc_", dir=str(self.clean_root)))
        try:
            if ext == ".zip":
                with zipfile.ZipFile(fp, "r") as zf:
                    names = zf.namelist()
                    wanted = [n for n in names if n.lower().endswith("posts.xml")]
                    if not wanted:
                        wanted = [n for n in names if Path(n).suffix.lower() in self._supported_text_exts]
                    if not wanted:
                        stats.compressed_skipped += 1
                        stats.skipped_files.append(f"{fp} [no_supported_members]")
                        return
                    self._extract_zip_members_safe(zf, wanted, tmp_dir, stats, str(fp))
            elif ext in {".tar", ".gz", ".xz"}:
                try:
                    with tarfile.open(fp, "r:*") as tf:
                        members = tf.getmembers()
                        wanted = [m for m in members if m.isfile() and (m.name.lower().endswith("posts.xml") or Path(m.name).suffix.lower() in self._supported_text_exts)]
                        if not wanted:
                            stats.compressed_skipped += 1
                            stats.skipped_files.append(f"{fp} [no_supported_members]")
                            return
                        self._extract_tar_members_safe(tf, wanted, tmp_dir, stats, str(fp))
                except tarfile.TarError:
                    stats.compressed_skipped += 1
                    stats.skipped_files.append(f"{fp} [non_tar_stream_unsupported]")
                    return
            else:
                stats.compressed_skipped += 1
                stats.skipped_files.append(f"{fp} [unsupported_archive]")
                return

            for extracted in self._iter_candidate_files(tmp_dir):
                if extracted.suffix.lower() in {".7z", ".zip", ".tar", ".gz", ".xz"}:
                    continue
                self._process_file(extracted, run_dir, stats)
        except Exception as e:
            stats.compressed_skipped += 1
            stats.errors.append(f"{fp} [archive_error] {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _safe_member_target(self, root: Path, member_name: str) -> Optional[Path]:
        normalized = (member_name or "").replace("\\", "/")
        if not normalized:
            return None
        if normalized.startswith("/") or normalized.startswith("../"):
            return None
        if "/../" in normalized or normalized == "..":
            return None
        root_resolved = root.resolve()
        target = (root / Path(normalized)).resolve()
        if target == root_resolved:
            return None
        if root_resolved not in target.parents:
            return None
        return target

    def _extract_zip_members_safe(
        self,
        zf: zipfile.ZipFile,
        names: List[str],
        dest_dir: Path,
        stats: SanitizationStats,
        archive_label: str,
    ) -> None:
        blocked = 0
        extracted = 0
        for name in names:
            target = self._safe_member_target(dest_dir, name)
            if target is None:
                blocked += 1
                continue
            try:
                info = zf.getinfo(name)
            except KeyError:
                continue
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
        if blocked:
            stats.errors.append(f"{archive_label} [zip_path_traversal_blocked:{blocked}]")
        if extracted == 0:
            stats.compressed_skipped += 1
            stats.skipped_files.append(f"{archive_label} [no_safe_supported_members]")

    def _extract_tar_members_safe(
        self,
        tf: tarfile.TarFile,
        members: List,
        dest_dir: Path,
        stats: SanitizationStats,
        archive_label: str,
    ) -> None:
        blocked = 0
        extracted = 0
        for member in members:
            target = self._safe_member_target(dest_dir, member.name)
            if target is None:
                blocked += 1
                continue
            if not member.isfile():
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
        if blocked:
            stats.errors.append(f"{archive_label} [tar_path_traversal_blocked:{blocked}]")
        if extracted == 0:
            stats.compressed_skipped += 1
            stats.skipped_files.append(f"{archive_label} [no_safe_supported_members]")

    def _reddit_line_reader(self, fp: Path, stats: SanitizationStats):
        ext = fp.suffix.lower()
        if ext in {".json", ".jsonl"}:
            def _iter_plain():
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        yield line
            return _iter_plain()
        if ext == ".zst":
            if zstd is None:
                stats.skipped_files.append(f"{fp} [missing_zstandard]")
                return None
            try:
                with open(fp, "rb") as f:
                    head = f.read(4)
                if not head.startswith(MAGIC_ZST):
                    stats.skipped_files.append(f"{fp} [invalid_zst_magic:{head.hex()}]")
                    return None
            except Exception:
                stats.skipped_files.append(f"{fp} [read_error]")
                return None

            def _iter_zst():
                try:
                    with zstd.open(fp, "rt", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            yield line
                except Exception as e:
                    stats.skipped_files.append(f"{fp} [zst_read_error] {e}")
                    return
            try:
                return _iter_zst()
            except Exception:
                return None
        return None
