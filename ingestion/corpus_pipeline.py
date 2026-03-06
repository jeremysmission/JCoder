"""
Corpus Pipeline -- batch ingestion for SO, CodeSearchNet, docs, and raw code
into separate FAISS + FTS5 hybrid indexes.  Supports resume-from-checkpoint
so 900K+ file runs survive interruption.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from core.config import StorageConfig
from core.embedding_engine import EmbeddingEngine
from core.index_engine import IndexEngine
from ingestion.chunker import Chunker, LANGUAGE_MAP
from ingestion.dedup import MinHashDedup
from ingestion.pii_scanner import PIIScanner, sanitize_for_ingest


@dataclass
class IngestStats:
    """Outcome of a single ingestion run."""
    source: str
    files_processed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    elapsed_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        r = self.files_processed / self.elapsed_s if self.elapsed_s > 0 else 0
        return (f"[{self.source}] {self.files_processed} files, "
                f"{self.chunks_created} chunks, {self.chunks_embedded} embedded, "
                f"{self.files_skipped} skipped, {len(self.errors)} errors, "
                f"{self.elapsed_s:.1f}s ({r:.0f} files/s)")


_META_RE = re.compile(r"^- (\w[\w_]*?):\s*(.+)$", re.MULTILINE)
_CODE_EXTENSIONS: Dict[str, str] = {
    ext: lang for ext, lang in LANGUAGE_MAP.items() if lang is not None
}


def _parse_entries(text: str) -> List[Dict]:
    """Parse cleaned markdown (SO / CSN / GitHub). Multiple entries per file
    separated by '---'.  Returns list of dicts with title/tags/explanation/code."""
    entries = []
    for block in re.split(r"^\s*---\s*$", text, flags=re.MULTILINE):
        block = block.strip()
        if not block:
            continue

        entry: Dict = {}
        title_m = re.search(r"^#\s+(.+)$", block, re.MULTILINE)
        if title_m:
            entry["title"] = title_m.group(1).strip()

        for m in _META_RE.finditer(block):
            entry[m.group(1)] = m.group(2).strip()

        for sec in re.split(r"^##\s+", block, flags=re.MULTILINE)[1:]:
            heading, _, body = sec.partition("\n")
            h = heading.strip().lower()
            if h in ("technical explanation", "documentation"):
                entry["explanation"] = body.strip()
            elif h == "code":
                entry["code"] = body.strip()

        if entry.get("title") or entry.get("explanation") or entry.get("code"):
            entries.append(entry)
    return entries


def _split_by_headings(text: str, max_chars: int = 4000) -> List[str]:
    """Split markdown at ## heading boundaries, sub-splitting oversized sections."""
    chunks = []
    for sec in re.split(r"(?=^##\s)", text, flags=re.MULTILINE):
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= max_chars:
            chunks.append(sec)
        else:
            start = 0
            while start < len(sec):
                end = min(start + max_chars, len(sec))
                if end < len(sec):
                    nl = sec.rfind("\n", start, end)
                    if nl > start:
                        end = nl + 1
                chunk = sec[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = end
    return chunks


class CorpusPipeline:
    """Batch ingestion pipeline for code knowledge corpora."""

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        storage_config: Optional[StorageConfig] = None,
        chunker: Optional[Chunker] = None,
        batch_size: int = 64,
        dimension: int = 768,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        pii_scanner: Optional[PIIScanner] = None,
        dedup: Optional[MinHashDedup] = None,
    ):
        self._embedder = embedding_engine
        self._storage = storage_config or StorageConfig()
        self._chunker = chunker or Chunker(max_chars=4000)
        self._batch_size = batch_size
        self._dimension = dimension
        self._progress = progress_callback
        self._pii = pii_scanner
        self._dedup = dedup
        self._checkpoint_dir = os.path.join(self._storage.data_dir, "checkpoints")

    def ingest_stackoverflow(
        self, source_dir: str, index_name: str = "stackoverflow",
        max_files: int = 0, resume: bool = True,
    ) -> IngestStats:
        """Ingest Stack Overflow / StackExchange markdown files."""
        return self._ingest_loop(
            source_dir, index_name, {".md"}, max_files, resume,
            self._chunks_from_qa, source_kind="stackoverflow",
        )

    def ingest_codesearchnet(
        self, source_dir: str, index_name: str = "codesearchnet",
        languages: Optional[List[str]] = None,
        max_files: int = 0, resume: bool = True,
    ) -> IngestStats:
        """Ingest CodeSearchNet markdown files (function + docstring pairs)."""
        return self._ingest_loop(
            source_dir, index_name, {".md"}, max_files, resume,
            self._chunks_from_qa, source_kind="codesearchnet",
            lang_filter=set(languages) if languages else None,
        )

    def ingest_markdown_docs(
        self, source_dir: str, index_name: str = "docs",
        max_files: int = 0, resume: bool = True,
    ) -> IngestStats:
        """Ingest documentation markdown, chunked by heading sections."""
        return self._ingest_loop(
            source_dir, index_name, {".md"}, max_files, resume,
            self._chunks_from_doc_md, source_kind="docs",
        )

    def ingest_code_files(
        self, source_dir: str, index_name: str = "code",
        languages: Optional[List[str]] = None,
        max_files: int = 0, resume: bool = True,
    ) -> IngestStats:
        """Ingest raw source code using AST-aware chunking."""
        if languages:
            exts = {e for e, l in _CODE_EXTENSIONS.items() if l in set(languages)}
        else:
            exts = set(_CODE_EXTENSIONS.keys())
        return self._ingest_loop(
            source_dir, index_name, exts, max_files, resume,
            self._chunks_from_code, source_kind="code",
        )

    def _chunks_from_qa(
        self, fpath: str, raw: str, *, source_kind: str = "",
        lang_filter: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """Produce chunks from parsed Q&A / CSN markdown."""
        chunks = []
        for entry in _parse_entries(raw):
            lang = entry.get("language", "")
            if lang_filter and lang not in lang_filter:
                continue
            parts = [v for k in ("title", "explanation", "code")
                      if (v := entry.get(k))]
            content = "\n\n".join(parts)
            if not content.strip():
                continue
            for text in self._sub_chunk(content, fpath):
                meta = self._chunker._make_chunk(
                    text, fpath,
                    title=entry.get("title", ""),
                    tags=entry.get("tags", ""),
                    language=lang,
                    repo=entry.get("repo", ""),
                    source_kind=source_kind,
                )
                chunks.append(meta)
        return chunks

    def _chunks_from_doc_md(
        self, fpath: str, raw: str, **_kw,
    ) -> List[Dict]:
        """Produce chunks from documentation markdown split by headings."""
        chunks = []
        for text in _split_by_headings(raw, self._chunker.max_chars):
            if not text.strip():
                continue
            chunks.append(self._chunker._make_chunk(text, fpath, source_kind="docs"))
        return chunks

    def _chunks_from_code(
        self, fpath: str, _raw: str, **_kw,
    ) -> List[Dict]:
        """Produce chunks from a source code file via the AST chunker."""
        ext = os.path.splitext(fpath)[1].lower()
        lang = _CODE_EXTENSIONS.get(ext, "")
        file_chunks = self._chunker.chunk_file(fpath)
        for c in file_chunks:
            c["language"] = lang
            c["source_kind"] = "code"
            if self._pii and c.get("content"):
                scan_result = self._pii.scan(c["content"])
                if scan_result.findings:
                    print(f"[WARN] PII scanner: {len(scan_result.findings)} "
                          f"finding(s) redacted in chunk from {fpath}")
                c["content"] = scan_result.clean_text
        return file_chunks

    def _ingest_loop(
        self,
        source_dir: str,
        index_name: str,
        extensions: Set[str],
        max_files: int,
        resume: bool,
        producer: Callable,
        **producer_kw,
    ) -> IngestStats:
        stats = IngestStats(source=index_name)
        t0 = time.monotonic()

        processed = self._load_checkpoint(index_name) if resume else set()
        index = self._make_index()
        pending_chunks: List[Dict] = []
        pending_texts: List[str] = []

        for fpath in self._walk_files(source_dir, extensions):
            if max_files and stats.files_processed >= max_files:
                break
            if fpath in processed:
                stats.files_skipped += 1
                continue

            try:
                # Code producer reads file itself; others need raw text
                if producer is self._chunks_from_code:
                    file_chunks = producer(fpath, "", **producer_kw)
                else:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        raw = f.read()
                    if self._pii:
                        scan_result = self._pii.scan(raw)
                        if scan_result.findings:
                            print(f"[WARN] PII scanner: {len(scan_result.findings)} "
                                  f"finding(s) redacted in {fpath}")
                        raw = scan_result.clean_text
                    file_chunks = producer(fpath, raw, **producer_kw)

                for chunk in file_chunks:
                    if self._dedup and not self._dedup.add(
                        chunk["content"], doc_id=chunk.get("source", "")
                    ):
                        continue
                    pending_chunks.append(chunk)
                    pending_texts.append(chunk["content"])
                    stats.chunks_created += 1

                stats.files_processed += 1
                processed.add(fpath)
            except Exception as exc:
                stats.errors.append(f"{fpath}: {exc}")
                stats.files_skipped += 1

            if len(pending_texts) >= self._batch_size:
                self._flush_batch(index, pending_texts, pending_chunks, stats)
                pending_texts.clear()
                pending_chunks.clear()

            total = stats.files_processed + stats.files_skipped
            if total % 1000 == 0 and total > 0:
                self._report(stats, total, t0)
                if resume:
                    self._save_checkpoint(index_name, processed)

        # Final flush
        if pending_texts:
            self._flush_batch(index, pending_texts, pending_chunks, stats)

        try:
            index.save(index_name)
        except (AttributeError, OSError) as exc:
            # FAISS not installed -- save FTS5 manually
            print(f"[WARN] FAISS save skipped ({exc}); building FTS5 only.")
            idx_path = os.path.join(self._storage.index_dir, index_name)
            os.makedirs(self._storage.index_dir, exist_ok=True)
            import json as _json
            with open(idx_path + ".meta.json", "w", encoding="utf-8") as mf:
                _json.dump(index.metadata, mf)
            index._db_path = idx_path + ".fts5.db"
            index._build_fts5()
        if resume:
            self._save_checkpoint(index_name, processed)
        index.close()

        if self._dedup:
            self._dedup.save()

        stats.elapsed_s = time.monotonic() - t0
        print(f"[OK] {stats.summary()}")
        return stats

    def _make_index(self) -> IndexEngine:
        return IndexEngine(
            dimension=self._dimension,
            storage=self._storage,
            sparse_only=(self._embedder is None),
        )

    @staticmethod
    def _walk_files(root: str, extensions: Set[str]) -> List[str]:
        """Collect files matching extensions, sorted for determinism."""
        result = []
        for dirpath, _dirs, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in extensions:
                    result.append(os.path.join(dirpath, fn))
        result.sort()
        return result

    def _sub_chunk(self, content: str, fpath: str) -> List[str]:
        """Split content if it exceeds the chunker limit."""
        if len(content) <= self._chunker.max_chars:
            return [content]
        return [c["content"] for c in self._chunker._chunk_by_chars(content, fpath)]

    def _embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        if self._embedder is None:
            return None
        try:
            return self._embedder.embed(texts)
        except Exception as exc:
            print(f"[WARN] Embedding batch failed ({len(texts)} texts): {exc}")
            return None

    def _flush_batch(
        self, index: IndexEngine, texts: List[str],
        chunks: List[Dict], stats: IngestStats,
    ):
        vectors = self._embed_batch(texts)
        if vectors is not None:
            index.add_vectors(vectors, chunks)
            stats.chunks_embedded += len(texts)
        else:
            index.metadata.extend(chunks)

    def _report(self, stats: IngestStats, total: int, t0: float):
        elapsed = time.monotonic() - t0
        rate = total / elapsed if elapsed > 0 else 0
        msg = (f"[OK] {stats.source}: {stats.files_processed} processed, "
               f"{stats.chunks_created} chunks, {stats.files_skipped} skipped, "
               f"{rate:.0f} files/s")
        print(msg)
        if self._progress:
            self._progress(total, msg)

    def _save_checkpoint(self, index_name: str, processed: Set[str]):
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        path = os.path.join(self._checkpoint_dir, f"{index_name}_checkpoint.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(processed), f)

    def _load_checkpoint(self, index_name: str) -> Set[str]:
        path = os.path.join(self._checkpoint_dir, f"{index_name}_checkpoint.json")
        if not os.path.exists(path):
            return set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Corrupt checkpoint {path}: {exc}")
            return set()
