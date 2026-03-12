"""Download and ingest new datasets identified 2026-03-10.

Datasets (ordered by size, smallest first):
  1. cruxeval-org/cruxeval -- 800 Python functions (self-assessment)
  2. bigcode/bigcodebench -- 1,140 tasks (API usage patterns)
  3. newfacade/LeetCodeDataset -- 2,870 problems (algorithm practice)
  4. Rtian/DebugBench -- 4,253 instances (bug explanations)
  5. open-r1/codeforces -- 10K problems (editorials + algorithm reasoning)
  6. ajibawa-2023/Software-Architecture -- 450K entries (system design)
  7. nvidia/OpenCodeReasoning-2 -- 2.5M (reasoning traces + critiques)

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\download_new_datasets_2026_03_10.py
    .venv\\Scripts\\python scripts\\download_new_datasets_2026_03_10.py --only debugbench
    .venv\\Scripts\\python scripts\\download_new_datasets_2026_03_10.py --only cruxeval,bigcodebench,leetcode
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

import httpx

from core.download_manager import DownloadManager, fetch_huggingface_parquet_urls

DATA_ROOT = Path(os.environ.get("JCODER_DATA", r"D:\JCoder_Data"))
INDEX_DIR = DATA_ROOT / "indexes"
DOWNLOAD_DIR = DATA_ROOT / "downloads"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 5000
MAX_CHARS = 4000

_NORMALIZE_RE = re.compile(r"[_\-./\\:]")
_CAMEL_RE1 = re.compile(r"([a-z])([A-Z])")
_CAMEL_RE2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_DOWNLOADER: DownloadManager | None = None


def _get_downloader() -> DownloadManager:
    global _DOWNLOADER
    if _DOWNLOADER is None:
        _DOWNLOADER = DownloadManager(DOWNLOAD_DIR, read_timeout_s=600.0)
    return _DOWNLOADER


def _close_downloader() -> None:
    global _DOWNLOADER
    if _DOWNLOADER is not None:
        _DOWNLOADER.close()
        _DOWNLOADER = None


def _normalize(text: str) -> str:
    out = _NORMALIZE_RE.sub(" ", text)
    out = _CAMEL_RE1.sub(r"\1 \2", out)
    out = _CAMEL_RE2.sub(r"\1 \2", out)
    return out.lower()


def _download_parquet(url: str, local_path: Path) -> bool:
    relative_path = local_path.relative_to(DOWNLOAD_DIR)
    result = _get_downloader().download_file(
        url, relative_path, min_existing_bytes=1000, chunk_size=256 * 1024,
    )
    if result.ok:
        return True
    print(f"    [WARN] Download failed: {result.error}")
    return False


def _get_parquet_urls(dataset_id: str, config: str = "default",
                      split: str = "train") -> list:
    return fetch_huggingface_parquet_urls(
        _get_downloader(), dataset_id, config=config, split=split,
    )


def _read_parquet_table(path: Path):
    import pyarrow.parquet as pq
    return pq.read_table(str(path))


class FTS5Builder:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
            "USING fts5(search_content, source_path, chunk_id)"
        )
        self._batch = []
        self.total_chunks = 0
        self.total_entries = 0

    def add(self, text: str, source_id: str):
        if not text or len(text.strip()) < 20:
            return
        self.total_entries += 1
        pos = 0
        cidx = 0
        while pos < len(text):
            end = min(pos + MAX_CHARS, len(text))
            if end < len(text):
                nl = text.rfind("\n", pos, end)
                if nl > pos:
                    end = nl + 1
            chunk = text[pos:end]
            if chunk.strip():
                cid = hashlib.sha256(f"{source_id}:{cidx}".encode()).hexdigest()
                self._batch.append((_normalize(chunk), source_id, cid))
                cidx += 1
            pos = end
        if len(self._batch) >= BATCH_SIZE:
            self._flush()

    def _flush(self):
        if self._batch:
            self.conn.executemany(
                "INSERT INTO chunks(search_content, source_path, chunk_id) "
                "VALUES (?, ?, ?)", self._batch)
            self.conn.commit()
            self.total_chunks += len(self._batch)
            self._batch = []

    def finish(self):
        self._flush()
        self.conn.close()
        size_mb = self.db_path.stat().st_size / 1e6
        return self.total_entries, self.total_chunks, size_mb


def _generic_download_and_index(
    dataset_id: str,
    db_name: str,
    cache_subdir: str,
    prefix: str,
    text_builder_fn,
    config: str = "default",
    split: str = "train",
) -> bool:
    """Download parquet files and build FTS5 index using a custom text builder."""
    db_path = INDEX_DIR / db_name
    cache_dir = DOWNLOAD_DIR / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and db_path.stat().st_size > 50_000:
        print(f"  [OK] {db_name} already exists ({db_path.stat().st_size/1e6:.1f} MB)")
        return True

    print(f"  Downloading {dataset_id} (config={config}, split={split})...")
    try:
        urls = _get_parquet_urls(dataset_id, config=config, split=split)
    except Exception as exc:
        print(f"  [FAIL] Could not get Parquet URLs: {exc}")
        return False

    if not urls:
        print(f"  [WARN] No Parquet files found for {dataset_id}")
        return False

    print(f"  {len(urls)} Parquet file(s)")

    local_files = []
    for i, url in enumerate(urls):
        local = cache_dir / f"{split}_{i:04d}.parquet"
        print(f"    [{i+1}/{len(urls)}] ", end="", flush=True)
        if local.exists() and local.stat().st_size > 1000:
            print(f"cached ({local.stat().st_size/1e6:.1f} MB)", flush=True)
            local_files.append(local)
        elif _download_parquet(url, local):
            print(f"{local.stat().st_size/1e6:.1f} MB", flush=True)
            local_files.append(local)
        else:
            print("FAILED", flush=True)

    if not local_files:
        print(f"  [FAIL] No files downloaded")
        return False

    print(f"  Building FTS5 index...")
    builder = FTS5Builder(db_path)
    t0 = time.monotonic()

    for f in local_files:
        table = _read_parquet_table(f)
        cols = table.column_names

        for i in range(len(table)):
            row = {c: str(table[c][i]) for c in cols}
            text = text_builder_fn(row, cols)
            if text and len(text) >= 50:
                builder.add(text, f"{prefix}_{builder.total_entries:07d}")

        elapsed = time.monotonic() - t0
        print(f"    {f.name}: {builder.total_entries:,} entries ({elapsed:.0f}s)")

    entries, chunks, size = builder.finish()
    elapsed = time.monotonic() - t0
    print(f"  [OK] {db_name}: {entries:,} entries, {chunks:,} chunks "
          f"({size:.0f} MB, {elapsed:.0f}s)")
    return True


# -----------------------------------------------------------------------
# Dataset processors
# -----------------------------------------------------------------------

def process_cruxeval() -> bool:
    """CRUXEval: 800 Python functions for self-assessment."""
    def build_text(row, cols):
        code = row.get("code", "")
        inp = row.get("input", "")
        output = row.get("output", "")
        if code == "None" or not code:
            return ""
        parts = [f"Code:\n{code}"]
        if inp and inp != "None":
            parts.append(f"Input: {inp}")
        if output and output != "None":
            parts.append(f"Output: {output}")
        return "\n\n".join(parts)

    return _generic_download_and_index(
        dataset_id="cruxeval-org/cruxeval",
        db_name="cruxeval.fts5.db",
        cache_subdir="cruxeval",
        prefix="crux",
        text_builder_fn=build_text,
        split="test",
    )


def process_bigcodebench() -> bool:
    """BigCodeBench: 1,140 tasks with API usage patterns."""
    def build_text(row, cols):
        prompt = row.get("complete_prompt", "") or row.get("instruct_prompt", "")
        solution = row.get("canonical_solution", "")
        libs = row.get("libs", "")
        if prompt == "None":
            prompt = ""
        if solution == "None":
            solution = ""
        parts = []
        if prompt:
            parts.append(f"Task:\n{prompt}")
        if solution:
            parts.append(f"Solution:\n{solution}")
        if libs and libs != "None":
            parts.append(f"Libraries: {libs}")
        return "\n\n".join(parts)

    return _generic_download_and_index(
        dataset_id="bigcode/bigcodebench",
        db_name="bigcodebench.fts5.db",
        cache_subdir="bigcodebench",
        prefix="bcb",
        text_builder_fn=build_text,
        split="v0.1.4",
    )


def process_leetcode() -> bool:
    """LeetCodeDataset: 2,870 problems with solutions and tests."""
    def build_text(row, cols):
        desc = row.get("problem_description", "") or row.get("content", "")
        solution = row.get("completion", "") or row.get("python", "")
        difficulty = row.get("difficulty", "")
        tags = row.get("tags", "")
        if desc == "None":
            desc = ""
        if solution == "None":
            solution = ""
        parts = []
        if difficulty and difficulty != "None":
            parts.append(f"Difficulty: {difficulty}")
        if tags and tags != "None":
            parts.append(f"Tags: {tags}")
        if desc:
            parts.append(f"Problem:\n{desc}")
        if solution:
            parts.append(f"Solution:\n{solution}")
        return "\n\n".join(parts)

    return _generic_download_and_index(
        dataset_id="newfacade/LeetCodeDataset",
        db_name="leetcode.fts5.db",
        cache_subdir="leetcode",
        prefix="lc",
        text_builder_fn=build_text,
    )


def process_debugbench() -> bool:
    """DebugBench: 4,253 bug instances with explanations."""
    def build_text(row, cols):
        question = row.get("question", "")
        buggy = row.get("buggy_code", "")
        solution = row.get("solution", "")
        explanation = row.get("bug_explanation", "")
        category = row.get("category", "")
        subtype = row.get("subtype", "")
        lang = row.get("language", "")
        level = row.get("level", "")
        parts = []
        if lang and lang != "None":
            parts.append(f"Language: {lang}")
        if level and level != "None":
            parts.append(f"Difficulty: {level}")
        if category and category != "None":
            parts.append(f"Bug Category: {category}")
        if subtype and subtype != "None":
            parts.append(f"Bug Type: {subtype}")
        if question and question != "None":
            parts.append(f"Problem:\n{question}")
        if buggy and buggy != "None":
            parts.append(f"Buggy Code:\n{buggy}")
        if explanation and explanation != "None":
            parts.append(f"Bug Explanation:\n{explanation}")
        if solution and solution != "None":
            parts.append(f"Fixed Code:\n{solution}")
        return "\n\n".join(parts)

    return _generic_download_and_index(
        dataset_id="Rtian/DebugBench",
        db_name="debugbench.fts5.db",
        cache_subdir="debugbench",
        prefix="dbg",
        text_builder_fn=build_text,
        split="test",
    )


def process_codeforces() -> bool:
    """Codeforces: 10K problems with editorials and solutions."""
    def build_text(row, cols):
        title = row.get("title", "")
        desc = row.get("description", "")
        editorial = row.get("editorial", "")
        rating = row.get("rating", "")
        tags = row.get("tags", "")
        parts = []
        if title and title != "None":
            parts.append(f"Problem: {title}")
        if rating and rating != "None":
            parts.append(f"Rating: {rating}")
        if tags and tags != "None":
            parts.append(f"Tags: {tags}")
        if desc and desc != "None":
            parts.append(f"Description:\n{desc}")
        if editorial and editorial != "None":
            parts.append(f"Editorial:\n{editorial}")
        return "\n\n".join(parts)

    return _generic_download_and_index(
        dataset_id="open-r1/codeforces",
        db_name="codeforces.fts5.db",
        cache_subdir="codeforces",
        prefix="cf",
        text_builder_fn=build_text,
    )


def process_software_architecture() -> bool:
    """Software Architecture: 450K entries covering design patterns."""
    def build_text(row, cols):
        instruction = row.get("instruction", "")
        inp = row.get("input", "")
        output = row.get("output", "")
        parts = []
        if instruction and instruction != "None":
            parts.append(f"Q: {instruction}")
        if inp and inp != "None" and inp.strip():
            parts.append(f"Context: {inp}")
        if output and output != "None":
            parts.append(f"A: {output}")
        return "\n\n".join(parts)

    return _generic_download_and_index(
        dataset_id="ajibawa-2023/Software-Architecture",
        db_name="software_architecture.fts5.db",
        cache_subdir="software_architecture",
        prefix="arch",
        text_builder_fn=build_text,
    )


def process_opencodereasoning2() -> bool:
    """OpenCodeReasoning-2: 2.5M reasoning traces with critiques."""
    def build_text(row, cols):
        question = row.get("question", "")
        solution = row.get("solution", "")
        critique = row.get("qwq_critique", "")
        difficulty = row.get("difficulty", "")
        parts = []
        if difficulty and difficulty != "None":
            parts.append(f"Difficulty: {difficulty}")
        if question and question != "None":
            parts.append(f"Problem:\n{question[:2000]}")
        if critique and critique != "None":
            parts.append(f"Critique:\n{critique[:1500]}")
        if solution and solution != "None":
            parts.append(f"Solution:\n{solution[:2000]}")
        return "\n\n".join(parts)

    # Config="train", split="python" (HuggingFace parquet structure)
    return _generic_download_and_index(
        dataset_id="nvidia/OpenCodeReasoning-2",
        db_name="opencodereasoning2.fts5.db",
        cache_subdir="opencodereasoning2",
        prefix="ocr2",
        text_builder_fn=build_text,
        config="train",
        split="python",
    )


def process_swesmith() -> bool:
    """SWE-smith: 59K real-world bug fix instances."""
    def build_text(row, cols):
        parts = []
        repo = row.get("repo", "")
        problem = row.get("problem_statement", "")
        patch = row.get("patch", "")
        hints = row.get("hints_text", "")
        if repo:
            parts.append(f"Repo: {repo}")
        if problem:
            parts.append(f"Problem:\n{problem[:2000]}")
        if hints:
            parts.append(f"Hints:\n{hints[:1000]}")
        if patch:
            parts.append(f"Patch:\n{patch[:2000]}")
        return "\n\n".join(parts)

    return _generic_download_and_index(
        dataset_id="SWE-bench/SWE-smith",
        db_name="swesmith.fts5.db",
        cache_subdir="swesmith",
        prefix="swe",
        text_builder_fn=build_text,
    )


def process_xlcost() -> bool:
    """XLCoST: cross-language code pairs (Python program-level)."""
    def build_text(row, cols):
        parts = []
        text = row.get("text", "")
        code = row.get("code", "")
        lang = row.get("language", "")
        if text:
            parts.append(f"Description: {text}")
        if lang:
            parts.append(f"Language: {lang}")
        if code:
            parts.append(f"Code:\n{code[:3000]}")
        return "\n\n".join(parts)

    return _generic_download_and_index(
        dataset_id="codeparrot/xlcost-text-to-code",
        db_name="xlcost.fts5.db",
        cache_subdir="xlcost",
        prefix="xlc",
        text_builder_fn=build_text,
        config="Python-program-level",
        split="train",
    )


def process_opencodeinstruct() -> bool:
    """OpenCodeInstruct: 5M code instruction pairs."""
    def build_text(row, cols):
        parts = []
        instruction = row.get("instruction", "") or row.get("prompt", "")
        response = row.get("response", "") or row.get("output", "")
        lang = row.get("language", "") or row.get("lang", "")
        if lang:
            parts.append(f"Language: {lang}")
        if instruction:
            parts.append(f"Instruction: {instruction[:1500]}")
        if response:
            parts.append(f"Response:\n{response[:2500]}")
        return "\n\n".join(parts)

    return _generic_download_and_index(
        dataset_id="nvidia/OpenCodeInstruct",
        db_name="opencodeinstruct.fts5.db",
        cache_subdir="opencodeinstruct",
        prefix="oci",
        text_builder_fn=build_text,
        config="train",
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

ALL_PROCESSORS = {
    "cruxeval": ("CRUXEval (800 functions)", process_cruxeval),
    "bigcodebench": ("BigCodeBench (1,140 tasks)", process_bigcodebench),
    "leetcode": ("LeetCodeDataset (2,870 problems)", process_leetcode),
    "debugbench": ("DebugBench (4,253 bugs)", process_debugbench),
    "codeforces": ("Codeforces (10K problems)", process_codeforces),
    "software_architecture": ("Software Architecture (450K)", process_software_architecture),
    "opencodereasoning2": ("OpenCodeReasoning-2 (2.5M)", process_opencodereasoning2),
    "swesmith": ("SWE-smith (59K bug fixes)", process_swesmith),
    "xlcost": ("XLCoST (Python program-level)", process_xlcost),
    "opencodeinstruct": ("OpenCodeInstruct (5M instructions)", process_opencodeinstruct),
}


def main():
    parser = argparse.ArgumentParser(
        description="Download and index new coding datasets (2026-03-10)"
    )
    parser.add_argument(
        "--only", default="",
        help="Comma-separated list of dataset keys to process"
    )
    parser.add_argument(
        "--skip-large", action="store_true",
        help="Skip datasets > 100K entries (software_architecture, opencodereasoning2)"
    )
    args = parser.parse_args()

    selected = (
        [s.strip() for s in args.only.split(",") if s.strip()]
        if args.only else list(ALL_PROCESSORS.keys())
    )

    large = {"software_architecture", "opencodereasoning2", "swesmith", "opencodeinstruct"}
    if args.skip_large:
        selected = [s for s in selected if s not in large]

    print("=" * 60)
    print("JCoder New Dataset Download & Ingest")
    print(f"Datasets: {', '.join(selected)}")
    print("=" * 60)

    results = {}
    t0 = time.monotonic()

    for key in selected:
        if key not in ALL_PROCESSORS:
            print(f"\n[WARN] Unknown dataset: {key}")
            continue
        label, processor = ALL_PROCESSORS[key]
        print(f"\n--- {label} ---")
        try:
            ok = processor()
            results[key] = "OK" if ok else "FAIL"
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            results[key] = f"FAIL: {exc}"

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    for key, status in results.items():
        print(f"  {key:30s}: {status}")
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    _close_downloader()


if __name__ == "__main__":
    main()
