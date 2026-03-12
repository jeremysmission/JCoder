#!/usr/bin/env python3
"""
JCoder Phase 6: World-Class Agent Knowledge Downloads
=====================================================
Downloads and indexes datasets to make JCoder the best coding agent on Earth.

Categories (in priority order):
  A. Code Reasoning & Agent Intelligence
  B. API & Library Mastery
  C. Security & Reliability
  D. Coding Instruction (large scale)
  E. Project Structure & DevOps

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\download_phase6_datasets.py
    .venv\\Scripts\\python scripts\\download_phase6_datasets.py --category A
    .venv\\Scripts\\python scripts\\download_phase6_datasets.py --only swesmith_traj,openhands_traj
"""

import argparse
import hashlib
import io
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INDEX_DIR = Path(os.environ.get("JCODER_INDEX_DIR", PROJECT_ROOT / "data" / "indexes"))
DOWNLOAD_DIR = Path(os.environ.get(
    "JCODER_DOWNLOAD_DIR",
    Path("D:/JCoder_Data/downloads") if Path("D:/JCoder_Data").exists()
    else PROJECT_ROOT / "data" / "downloads",
))

INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 5000
MAX_CHARS = 4000

_NORMALIZE_RE = re.compile(r"[_\-./\\:]")
_CAMEL_RE1 = re.compile(r"([a-z])([A-Z])")
_CAMEL_RE2 = re.compile(r"([A-Z]+)([A-Z][a-z])")

_DOWNLOADER = None


def _get_downloader():
    global _DOWNLOADER
    if _DOWNLOADER is None:
        from core.download_manager import DownloadManager
        _DOWNLOADER = DownloadManager(cache_root=str(DOWNLOAD_DIR))
    return _DOWNLOADER


def _close_downloader():
    global _DOWNLOADER
    if _DOWNLOADER is not None:
        _DOWNLOADER.close()
        _DOWNLOADER = None


def _normalize(text):
    out = _NORMALIZE_RE.sub(" ", text)
    out = _CAMEL_RE1.sub(r"\1 \2", out)
    out = _CAMEL_RE2.sub(r"\1 \2", out)
    return out.lower()


def _download_parquet(url, local_path):
    relative_path = local_path.relative_to(DOWNLOAD_DIR)
    result = _get_downloader().download_file(
        url, relative_path, min_existing_bytes=1000, chunk_size=256 * 1024,
    )
    return result.ok


def _get_parquet_urls(dataset_id, config="default", split="train"):
    from core.download_manager import fetch_huggingface_parquet_urls
    return fetch_huggingface_parquet_urls(
        _get_downloader(), dataset_id, config=config, split=split,
    )


def _read_parquet_table(path):
    import pyarrow.parquet as pq
    return pq.read_table(str(path))


def _iter_parquet_batches(path, batch_size=5000):
    """Memory-safe parquet reader for large files (>200 MB)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_size):
        cols = batch.schema.names
        for i in range(batch.num_rows):
            row = {c: str(batch.column(c)[i].as_py()) for c in cols}
            yield row, cols


class FTS5Builder:
    def __init__(self, db_path):
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

    def add(self, text, source_id):
        if not text or not text.strip():
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
    dataset_id, db_name, cache_subdir, prefix, text_builder_fn,
    config="default", split="train",
):
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

    parquet_files = []
    for i, url in enumerate(urls, 1):
        fname = url.split("/")[-1].split("?")[0]
        if not fname.endswith(".parquet"):
            fname = f"{prefix}_{i:04d}.parquet"
        local = cache_dir / fname
        if not _download_parquet(url, local):
            print(f"    [{i}/{len(urls)}] FAILED")
            continue
        size_mb = local.stat().st_size / 1e6
        print(f"    [{i}/{len(urls)}] {size_mb:.1f} MB")
        parquet_files.append(local)

    if not parquet_files:
        print(f"  [FAIL] No files downloaded for {dataset_id}")
        return False

    print("  Building FTS5 index...")
    builder = FTS5Builder(db_path)
    t0 = time.monotonic()

    for pf in sorted(parquet_files):
        try:
            table = _read_parquet_table(pf)
        except Exception as exc:
            print(f"    [WARN] Could not read {pf.name}: {exc}")
            continue
        cols = table.column_names
        for row_idx in range(table.num_rows):
            row = {c: str(table.column(c)[row_idx].as_py()) for c in cols}
            text = text_builder_fn(row, cols)
            if text and len(text) > 50:
                source_id = f"{prefix}_{pf.stem}_{row_idx}"
                builder.add(text, source_id)

        elapsed = time.monotonic() - t0
        print(f"    {pf.name}: {builder.total_entries:,} entries ({elapsed:.0f}s)")

    entries, chunks, size_mb = builder.finish()
    elapsed = time.monotonic() - t0
    print(f"  [OK] {db_name}: {entries:,} entries, {chunks:,} chunks "
          f"({size_mb:.0f} MB, {elapsed:.0f}s)")
    return True


def _generic_download_and_index_batched(
    dataset_id, db_name, cache_subdir, prefix, text_builder_fn,
    config="default", split="train",
):
    """Like _generic_download_and_index but reads parquet in small batches
    to avoid OOM on large files (>200 MB)."""
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

    parquet_files = []
    for i, url in enumerate(urls, 1):
        fname = url.split("/")[-1].split("?")[0]
        if not fname.endswith(".parquet"):
            fname = f"{prefix}_{i:04d}.parquet"
        local = cache_dir / fname
        if not _download_parquet(url, local):
            print(f"    [{i}/{len(urls)}] FAILED")
            continue
        size_mb = local.stat().st_size / 1e6
        print(f"    [{i}/{len(urls)}] {size_mb:.1f} MB")
        parquet_files.append(local)

    if not parquet_files:
        print(f"  [FAIL] No files downloaded for {dataset_id}")
        return False

    print("  Building FTS5 index (batched reader)...")
    builder = FTS5Builder(db_path)
    t0 = time.monotonic()

    for pf in sorted(parquet_files):
        try:
            for row, cols in _iter_parquet_batches(pf, batch_size=2000):
                text = text_builder_fn(row, cols)
                if text and len(text) > 50:
                    source_id = f"{prefix}_{pf.stem}_{builder.total_entries}"
                    builder.add(text, source_id)
        except Exception as exc:
            print(f"    [WARN] Error in {pf.name}: {exc}")
            continue

        elapsed = time.monotonic() - t0
        print(f"    {pf.name}: {builder.total_entries:,} entries ({elapsed:.0f}s)")

    entries, chunks, size_mb = builder.finish()
    elapsed = time.monotonic() - t0
    print(f"  [OK] {db_name}: {entries:,} entries, {chunks:,} chunks "
          f"({size_mb:.0f} MB, {elapsed:.0f}s)")
    return True


# ======================================================================
# CATEGORY A: Code Reasoning & Agent Intelligence
# ======================================================================

def process_humanevalplus():
    """HumanEval+: 164 coding problems with enhanced tests."""
    def build_text(row, cols):
        parts = []
        prompt = row.get("prompt", "")
        canonical = row.get("canonical_solution", "")
        entry_point = row.get("entry_point", "")
        if entry_point and entry_point != "None":
            parts.append(f"Function: {entry_point}")
        if prompt:
            parts.append(f"Problem:\n{prompt}")
        if canonical and canonical != "None":
            parts.append(f"Solution:\n{canonical}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "evalplus/humanevalplus", "humanevalplus.fts5.db",
        "humanevalplus", "hep", build_text, split="test")

def process_mbppplus():
    """MBPP+: 378 basic programming problems with tests."""
    def build_text(row, cols):
        parts = []
        prompt = row.get("prompt", "")
        code = row.get("code", "")
        if prompt:
            parts.append(f"Problem: {prompt}")
        if code and code != "None":
            parts.append(f"Solution:\n{code}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "evalplus/mbppplus", "mbppplus.fts5.db",
        "mbppplus", "mbpp", build_text, split="test")

def process_swesmith_traj():
    """SWE-smith trajectories: 5K agent solving traces."""
    def build_text(row, cols):
        parts = []
        for key in ["instance_id", "repo", "patch", "trajectory", "model_name_or_path"]:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 5:
                parts.append(f"{key}: {val[:3000]}")
        return "\n\n".join(parts)
    # SWE-smith-trajectories uses split names: ticks, tool, xml
    return _generic_download_and_index(
        "SWE-bench/SWE-smith-trajectories", "swesmith_traj.fts5.db",
        "swesmith_traj", "swt", build_text, split="tool")

def process_openhands_traj():
    """OpenHands trajectories: 67K multi-turn agent solving traces."""
    def build_text(row, cols):
        parts = []
        for key in cols:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 10:
                parts.append(f"{key}: {val[:3000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index_batched(
        "nebius/SWE-rebench-openhands-trajectories", "openhands_traj.fts5.db",
        "openhands_traj", "oht", build_text)

def process_sweagent_traj():
    """SWE-agent trajectories: 80K agent solving traces."""
    def build_text(row, cols):
        parts = []
        for key in cols:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 10:
                parts.append(f"{key}: {val[:3000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index_batched(
        "nebius/SWE-agent-trajectories", "sweagent_traj.fts5.db",
        "sweagent_traj", "sat", build_text)

def process_menvdata_traj():
    """MEnvData trajectories: 3.8K multi-language agent traces."""
    def build_text(row, cols):
        parts = []
        for key in cols:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 10:
                parts.append(f"{key}: {val[:3000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "ernie-research/MEnvData-SWE-Trajectory", "menvdata_traj.fts5.db",
        "menvdata_traj", "mev", build_text)

def process_coderforge():
    """CoderForge: 51K agentic coding trajectories."""
    def build_text(row, cols):
        parts = []
        for key in cols:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 10:
                parts.append(f"{key}: {val[:3000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "togethercomputer/CoderForge-Preview", "coderforge.fts5.db",
        "coderforge", "cfp", build_text,
        config="trajectories", split="SWE_Smith")


# ======================================================================
# CATEGORY A2: Code Review Intelligence
# ======================================================================

def process_codereview_python():
    """Code review critique and revision pairs (Python, 9.4K entries)."""
    def build_text(row, cols):
        parts = []
        # Actual cols: prompt, response, body, question_id
        # response format: "ORIGINAL: [code] CRITIQUE: [analysis] REVISED: [improved]"
        prompt = row.get("prompt", "")
        response = row.get("response", "")
        body = row.get("body", "")
        if body and body != "None" and len(body) > 10:
            parts.append(f"Question:\n{body[:1500]}")
        elif prompt and prompt != "None" and len(prompt) > 10:
            parts.append(f"Prompt:\n{prompt[:1500]}")
        if response and response != "None" and len(response) > 10:
            parts.append(f"Review:\n{response[:2500]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "Dahoas/code-review-instruct-critique-revision-python",
        "codereview_python.fts5.db", "codereview_python", "crp", build_text)

def process_codereview_general():
    """General code review dataset."""
    def build_text(row, cols):
        parts = []
        for key in cols:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 10:
                parts.append(f"{key}: {val[:2000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "VatsaDev/code-review", "codereview_general.fts5.db",
        "codereview_general", "crg", build_text)


# ======================================================================
# CATEGORY A3: Agent Trajectories (Extended)
# ======================================================================

def process_code_act():
    """CodeAct: 7K code agent trajectories + 71K general conversations."""
    def build_text(row, cols):
        parts = []
        conv = row.get("conversations", "")
        cid = row.get("id", "")
        if cid and cid != "None":
            parts.append(f"ID: {cid}")
        if conv and conv != "None" and len(conv) > 10:
            parts.append(f"Conversations: {conv[:3000]}")
        return "\n\n".join(parts)
    # codeact split has the agent trajectories (7.1K rows)
    return _generic_download_and_index(
        "xingyaoww/code-act", "code_act.fts5.db",
        "code_act", "cact", build_text, split="codeact")

def process_swe_gym_traj():
    """SWE-Gym sampled trajectories: 6K curated solving traces."""
    def build_text(row, cols):
        parts = []
        for key in cols:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 10:
                parts.append(f"{key}: {val[:3000]}")
        return "\n\n".join(parts)
    # Split is literally "train.raw" (not "train")
    return _generic_download_and_index(
        "SWE-Gym/OpenHands-Sampled-Trajectories", "swe_gym_traj.fts5.db",
        "swe_gym_traj", "sgt", build_text, split="train.raw")

def process_code_refine():
    """Code refinement pairs: 123K before/after code transformations."""
    def build_text(row, cols):
        parts = []
        old_code = row.get("old_code", "") or row.get("buggy", "") or row.get("code_before", "")
        new_code = row.get("new_code", "") or row.get("fixed", "") or row.get("code_after", "")
        comment = row.get("comment", "") or row.get("nl", "")
        if comment and comment != "None":
            parts.append(f"Change description: {comment[:1000]}")
        if old_code and old_code != "None":
            parts.append(f"Before:\n{old_code[:2000]}")
        if new_code and new_code != "None":
            parts.append(f"After:\n{new_code[:2000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "google/code_x_glue_cc_code_refinement", "code_refine.fts5.db",
        "code_refine", "crf", build_text, config="medium")

def process_github_codereview():
    """GitHub code review: 2.58 GB of real code review conversations."""
    def build_text(row, cols):
        parts = []
        for key in ["code", "review", "comment", "diff", "patch", "message", "body"]:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 10:
                parts.append(f"{key}: {val[:2500]}")
        return "\n\n".join(parts)
    return _generic_download_and_index_batched(
        "ronantakizawa/github-codereview", "github_codereview.fts5.db",
        "github_codereview", "gcr", build_text)


# ======================================================================
# CATEGORY B: Coding Instruction (Large Scale)
# ======================================================================

def process_magicoder():
    """Magicoder Evol-Instruct: 110K coding instructions."""
    def build_text(row, cols):
        parts = []
        instruction = row.get("instruction", "")
        response = row.get("response", "")
        if instruction:
            parts.append(f"Instruction: {instruction[:1500]}")
        if response:
            parts.append(f"Response:\n{response[:2500]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "ise-uiuc/Magicoder-Evol-Instruct-110K", "magicoder_110k.fts5.db",
        "magicoder", "mag", build_text)

def process_evol_codealpaca():
    """Evol-CodeAlpaca: evolved code instruction pairs."""
    def build_text(row, cols):
        parts = []
        instruction = row.get("instruction", "")
        output = row.get("output", "")
        if instruction:
            parts.append(f"Instruction: {instruction[:1500]}")
        if output:
            parts.append(f"Response:\n{output[:2500]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "theblackcat102/evol-codealpaca-v1", "evol_codealpaca.fts5.db",
        "evol_codealpaca", "eca", build_text)

def process_code_search_net():
    """CodeSearchNet Python: functions paired with docstrings for search."""
    def build_text(row, cols):
        parts = []
        docstring = row.get("func_documentation_string", "") or row.get("docstring", "")
        code = row.get("whole_func_string", "") or row.get("code", "")
        name = row.get("func_name", "")
        if name and name != "None":
            parts.append(f"Function: {name}")
        if docstring and docstring != "None":
            parts.append(f"Documentation: {docstring[:1500]}")
        if code and code != "None":
            parts.append(f"Code:\n{code[:2500]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "code-search-net/code_search_net", "code_search_net.fts5.db",
        "code_search_net", "csn", build_text, config="python")

def process_opencodeinstruct():
    """OpenCodeInstruct: massive 5M coding instruction set (BEAST only)."""
    def build_text(row, cols):
        parts = []
        instruction = row.get("instruction", "") or row.get("question", "")
        response = row.get("response", "") or row.get("answer", "")
        if instruction:
            parts.append(f"Instruction: {instruction[:1500]}")
        if response:
            parts.append(f"Response:\n{response[:2500]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "nvidia/OpenCodeInstruct", "opencodeinstruct.fts5.db",
        "opencodeinstruct", "oci", build_text)


# ======================================================================
# CATEGORY C: Security & Reliability
# ======================================================================

def process_vuln_cwe_patch():
    """CIRCL: 39K vulnerabilities with CWE and patches."""
    def build_text(row, cols):
        parts = []
        cve = row.get("cve_id", "") or row.get("id", "")
        desc = row.get("description", "")
        cwe = row.get("cwe_id", "") or row.get("cwe", "")
        patch = row.get("patch_url", "") or row.get("patch", "")
        if cve and cve != "None":
            parts.append(f"CVE: {cve}")
        if cwe and cwe != "None":
            parts.append(f"CWE: {cwe}")
        if desc:
            parts.append(f"Description: {desc[:2000]}")
        if patch and patch != "None":
            parts.append(f"Patch: {patch[:2000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index_batched(
        "CIRCL/vulnerability-cwe-patch", "vuln_cwe_patch.fts5.db",
        "vuln_cwe_patch", "vcp", build_text)

def process_cve_cwe_dataset():
    """All CVE+CWE records 1999-2025."""
    def build_text(row, cols):
        parts = []
        for key in cols:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 5:
                parts.append(f"{key}: {val[:2000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "stasvinokur/cve-and-cwe-dataset-1999-2025", "cve_cwe_all.fts5.db",
        "cve_cwe_all", "cca", build_text)

def process_vuln_security_dpo():
    """Code vulnerability security DPO training data."""
    def build_text(row, cols):
        parts = []
        for key in ["prompt", "chosen", "rejected", "question", "answer"]:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 10:
                parts.append(f"{key}: {val[:2000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "CyberNative/Code_Vulnerability_Security_DPO", "vuln_security_dpo.fts5.db",
        "vuln_security_dpo", "vsd", build_text)

def process_securecode_web():
    """SecureCode Web: 1.3K web security code examples."""
    def build_text(row, cols):
        parts = []
        for key in cols:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 10:
                parts.append(f"{key}: {val[:2000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "scthornton/securecode-web", "securecode_web.fts5.db",
        "securecode_web", "scw", build_text)

def process_cve_training():
    """All CVE records as training dataset (300K records)."""
    def build_text(row, cols):
        parts = []
        for key in cols:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 10:
                parts.append(f"{key}: {val[:2000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index_batched(
        "AlicanKiraz0/All-CVE-Records-Training-Dataset", "cve_training.fts5.db",
        "cve_training", "cvt", build_text)


# ======================================================================
# CATEGORY D: Large-Scale Code
# ======================================================================

def process_github_code_2025():
    """GitHub Code 2025: 1.5M+ curated repositories."""
    def build_text(row, cols):
        parts = []
        for key in ["code", "content", "text", "source"]:
            val = row.get(key, "")
            if val and val != "None" and len(val) > 20:
                parts.append(val[:4000])
                break
        lang = row.get("language", "") or row.get("lang", "")
        if lang and lang != "None":
            parts.insert(0, f"Language: {lang}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "nick007x/github-code-2025", "github_code_2025.fts5.db",
        "github_code_2025", "gc25", build_text)


# ======================================================================
# CATEGORY E: DevOps, Schemas & Tool Use
# ======================================================================

def process_dockerfiles():
    """Linted Dockerfiles: 195K real-world container configs."""
    def build_text(row, cols):
        parts = []
        content = row.get("content", "") or row.get("text", "") or row.get("dockerfile", "")
        repo = row.get("repo_name", "") or row.get("repo", "")
        if repo and repo != "None":
            parts.append(f"Repository: {repo}")
        if content and content != "None":
            parts.append(f"Dockerfile:\n{content[:3500]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "LeeSek/dockerfiles-linted", "dockerfiles.fts5.db",
        "dockerfiles", "dkf", build_text)

def process_sql_schemas():
    """SQL create context: 78K table schemas with natural language queries."""
    def build_text(row, cols):
        parts = []
        context = row.get("context", "") or row.get("create_table", "")
        question = row.get("question", "")
        answer = row.get("answer", "") or row.get("query", "")
        if context and context != "None":
            parts.append(f"Schema:\n{context[:2000]}")
        if question and question != "None":
            parts.append(f"Question: {question}")
        if answer and answer != "None":
            parts.append(f"SQL:\n{answer[:1500]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "b-mc2/sql-create-context", "sql_schemas.fts5.db",
        "sql_schemas", "sql", build_text)

def process_function_calling():
    """xLAM function calling: 60K function-calling specifications."""
    def build_text(row, cols):
        parts = []
        query = row.get("query", "") or row.get("instruction", "")
        tools = row.get("tools", "") or row.get("functions", "")
        answers = row.get("answers", "") or row.get("output", "")
        if query and query != "None":
            parts.append(f"Query: {query[:1500]}")
        if tools and tools != "None":
            parts.append(f"Tools:\n{tools[:2000]}")
        if answers and answers != "None":
            parts.append(f"Answer:\n{answers[:1500]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "Salesforce/xlam-function-calling-60k", "function_calling.fts5.db",
        "function_calling", "xlam", build_text)

# ======================================================================
# Category F: Advanced Reasoning & Benchmarks (NEW 2026-03-11)
# ======================================================================

def process_mixture_of_thoughts():
    """Mixture-of-Thoughts: 350K reasoning traces from open-r1."""
    def build_text(row, cols):
        parts = []
        q = row.get("problem", "") or row.get("question", "") or row.get("prompt", "")
        thought = row.get("thought", "") or row.get("reasoning", "") or row.get("chain_of_thought", "")
        answer = row.get("answer", "") or row.get("solution", "") or row.get("response", "")
        if q and q != "None":
            parts.append(f"Problem: {q[:2000]}")
        if thought and thought != "None":
            parts.append(f"Reasoning trace:\n{thought[:3000]}")
        if answer and answer != "None":
            parts.append(f"Answer: {answer[:2000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index_batched(
        "open-r1/Mixture-of-Thoughts", "mixture_of_thoughts.fts5.db",
        "mixture_of_thoughts", "mot", build_text)

def process_swe_rebench_v2():
    """SWE-rebench-V2: 32K multilingual agent trajectories."""
    def build_text(row, cols):
        parts = []
        repo = row.get("repo", "") or row.get("instance_id", "")
        problem = row.get("problem_statement", "") or row.get("description", "")
        patch = row.get("patch", "") or row.get("model_patch", "") or row.get("gold_patch", "")
        lang = row.get("language", "")
        if repo and repo != "None":
            parts.append(f"Repo: {repo}")
        if lang and lang != "None":
            parts.append(f"Language: {lang}")
        if problem and problem != "None":
            parts.append(f"Problem:\n{problem[:2000]}")
        if patch and patch != "None":
            parts.append(f"Patch:\n{patch[:3000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "nebius/SWE-rebench-V2", "swe_rebench_v2.fts5.db",
        "swe_rebench_v2", "swrb", build_text)

def process_jupyter_agent():
    """jupyter-agent: 51K notebook agent interactions.
    Dataset has splits: non_thinking, thinking (not train)."""
    def build_text(row, cols):
        parts = []
        instruction = row.get("instruction", "") or row.get("query", "") or row.get("prompt", "")
        code = row.get("code", "") or row.get("response", "") or row.get("output", "")
        context = row.get("context", "") or row.get("notebook", "")
        conv = row.get("conversations", "") or row.get("messages", "")
        if instruction and instruction != "None":
            parts.append(f"Instruction: {instruction[:1500]}")
        if context and context != "None":
            parts.append(f"Context:\n{context[:2000]}")
        if code and code != "None":
            parts.append(f"Code:\n{code[:3000]}")
        if conv and conv != "None" and not parts:
            parts.append(f"Conversation:\n{str(conv)[:5000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "jupyter-agent/jupyter-agent-dataset", "jupyter_agent.fts5.db",
        "jupyter_agent", "jagt", build_text,
        split="non_thinking")

def process_nemotron_agentic():
    """Nemotron-Agentic-v1: NVIDIA agentic training data."""
    def build_text(row, cols):
        parts = []
        for key in ["instruction", "prompt", "input", "query", "system"]:
            val = row.get(key, "")
            if val and val != "None":
                parts.append(f"{key.title()}: {val[:2000]}")
                break
        for key in ["response", "output", "answer", "completion"]:
            val = row.get(key, "")
            if val and val != "None":
                parts.append(f"Response:\n{val[:3000]}")
                break
        return "\n\n".join(parts)
    return _generic_download_and_index_batched(
        "nvidia/Nemotron-Agentic-v1", "nemotron_agentic.fts5.db",
        "nemotron_agentic", "nemo", build_text)

def process_hermes_function_calling():
    """Hermes function calling: NousResearch function calling dataset."""
    def build_text(row, cols):
        parts = []
        conv = row.get("conversations", "") or row.get("messages", "")
        tools = row.get("tools", "") or row.get("functions", "")
        if tools and tools != "None":
            parts.append(f"Tools: {tools[:2000]}")
        if conv and conv != "None":
            parts.append(f"Conversation:\n{conv[:4000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "NousResearch/hermes-function-calling-v1",
        "hermes_function_calling.fts5.db",
        "hermes_func_call", "hfc", build_text)

def process_glaive_function_calling_v2():
    """Glaive function calling v2: structured function calling data."""
    def build_text(row, cols):
        parts = []
        system = row.get("system", "") or row.get("system_prompt", "")
        chat = row.get("chat", "") or row.get("conversations", "") or row.get("messages", "")
        if system and system != "None":
            parts.append(f"System: {system[:2000]}")
        if chat and chat != "None":
            parts.append(f"Chat:\n{chat[:4000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "glaiveai/glaive-function-calling-v2",
        "glaive_function_calling_v2.fts5.db",
        "glaive_fc_v2", "gfcv", build_text)

def process_openthoughts_agent():
    """OpenThoughts-Agent-v1: agent reasoning with chain-of-thought.
    Actual columns: conversations, agent, model, task, episode, run_id, trial_name."""
    def build_text(row, cols):
        parts = []
        task = row.get("task", "")
        agent = row.get("agent", "")
        model = row.get("model", "")
        conv = row.get("conversations", "")
        if task and task != "None":
            parts.append(f"Task: {task[:500]}")
        if agent and agent != "None":
            parts.append(f"Agent: {agent[:200]}")
        if model and model != "None":
            parts.append(f"Model: {model[:200]}")
        if conv and conv != "None":
            parts.append(f"Conversation:\n{str(conv)[:5000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index_batched(
        "open-thoughts/OpenThoughts-Agent-v1-SFT",
        "openthoughts_agent.fts5.db",
        "openthoughts_agent", "otag", build_text)

def process_trail_agent():
    """TRAIL: PatronusAI agent trajectory evaluation dataset."""
    def build_text(row, cols):
        parts = []
        task = row.get("task", "") or row.get("instruction", "") or row.get("query", "")
        trajectory = row.get("trajectory", "") or row.get("trace", "") or row.get("steps", "")
        result = row.get("result", "") or row.get("outcome", "") or row.get("label", "")
        if task and task != "None":
            parts.append(f"Task: {task[:2000]}")
        if trajectory and trajectory != "None":
            parts.append(f"Trajectory:\n{trajectory[:4000]}")
        if result and result != "None":
            parts.append(f"Result: {result[:500]}")
        return "\n\n".join(parts)
    return _generic_download_and_index(
        "PatronusAI/TRAIL", "trail_agent.fts5.db",
        "trail_agent", "trai", build_text)

def process_verifiable_coding():
    """Verifiable coding problems with test suites from open-r1."""
    def build_text(row, cols):
        parts = []
        pid = row.get("problem_id", "")
        source = row.get("source", "")
        problem = row.get("problem_statement", "")
        solution = row.get("gold_standard_solution", "")
        metadata = row.get("metadata", "")
        verification = row.get("verification_info", "")
        if pid and pid != "None":
            parts.append(f"Problem ID: {pid}")
        if source and source != "None":
            parts.append(f"Source: {source}")
        if metadata and metadata != "None":
            parts.append(f"Metadata: {str(metadata)[:500]}")
        if problem and problem != "None":
            parts.append(f"Problem:\n{problem[:2000]}")
        if solution and solution != "None":
            parts.append(f"Solution:\n{solution[:3000]}")
        if verification and verification != "None":
            parts.append(f"Tests:\n{str(verification)[:2000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index_batched(
        "open-r1/verifiable-coding-problems-python",
        "verifiable_coding.fts5.db",
        "verifiable_coding", "vcod", build_text)

def process_livecodebench():
    """LiveCodeBench: contamination-free recent coding problems.
    Uses datasets library in streaming mode to avoid Windows WinError 32
    file-lock bug in the Arrow cache finalization step."""
    db_name = "livecodebench.fts5.db"
    db_path = INDEX_DIR / db_name
    if db_path.exists() and db_path.stat().st_size > 50_000:
        print(f"  [OK] {db_name} already exists ({db_path.stat().st_size/1e6:.1f} MB)")
        return True
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [FAIL] datasets library not installed (pip install datasets)")
        return False
    print("  Downloading via datasets library (streaming mode)...")
    try:
        ds = load_dataset("livecodebench/code_generation_lite",
                          version_tag="release_v5", split="test",
                          trust_remote_code=True, streaming=True)
    except Exception as exc:
        print(f"  [FAIL] Could not load dataset: {exc}")
        return False
    print("  Streaming problems into FTS5 index...")
    builder = FTS5Builder(db_path)
    t0 = time.monotonic()
    for i, row in enumerate(ds):
        parts = []
        title = str(row.get("question_title", "") or "")
        difficulty = str(row.get("difficulty", "") or "")
        content = str(row.get("question_content", "") or "")
        if title and title != "None":
            parts.append(f"Title: {title}")
        if difficulty and difficulty != "None":
            parts.append(f"Difficulty: {difficulty}")
        if content and content != "None":
            parts.append(f"Problem:\n{content[:3000]}")
        text = "\n\n".join(parts)
        if text and len(text) > 50:
            builder.add(text, f"lcb_{i:05d}")
    entries, chunks, size_mb = builder.finish()
    elapsed = time.monotonic() - t0
    print(f"  [OK] {db_name}: {entries:,} entries, {chunks:,} chunks "
          f"({size_mb:.0f} MB, {elapsed:.0f}s)")
    return True

def process_multi_swe_bench():
    """Multi-SWE-bench: ByteDance multilingual SWE benchmark.
    Uses datasets library in streaming mode (heterogeneous struct schemas
    across JSONL files prevent normal load_dataset from casting).
    Actual columns: org, repo, number, state, title, body, base,
    resolved_issues, fix_patch, test_patch, fixed_tests, instance_id, hints."""
    db_name = "multi_swe_bench.fts5.db"
    db_path = INDEX_DIR / db_name
    if db_path.exists() and db_path.stat().st_size > 50_000:
        print(f"  [OK] {db_name} already exists ({db_path.stat().st_size/1e6:.1f} MB)")
        return True
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [FAIL] datasets library not installed (pip install datasets)")
        return False
    print("  Downloading via datasets library (streaming, JSONL format)...")
    try:
        ds = load_dataset("ByteDance-Seed/Multi-SWE-bench", split="train",
                          trust_remote_code=True, streaming=True)
    except Exception as exc:
        print(f"  [FAIL] Could not load dataset: {exc}")
        return False
    builder = FTS5Builder(db_path)
    t0 = time.monotonic()
    for i, row in enumerate(ds):
        parts = []
        instance = str(row.get("instance_id", "") or "")
        org = str(row.get("org", "") or "")
        repo = str(row.get("repo", "") or "")
        title = str(row.get("title", "") or "")
        body = str(row.get("body", "") or "")
        fix_patch = str(row.get("fix_patch", "") or "")
        hints = str(row.get("hints", "") or "")
        if instance and instance != "None":
            parts.append(f"Instance: {instance}")
        if org and org != "None" and repo and repo != "None":
            parts.append(f"Repo: {org}/{repo}")
        elif repo and repo != "None":
            parts.append(f"Repo: {repo}")
        if title and title != "None":
            parts.append(f"Title: {title[:500]}")
        if body and body != "None":
            parts.append(f"Problem:\n{body[:2000]}")
        if hints and hints != "None":
            parts.append(f"Hints: {hints[:500]}")
        if fix_patch and fix_patch != "None":
            parts.append(f"Fix Patch:\n{fix_patch[:3000]}")
        text = "\n\n".join(parts)
        if text and len(text) > 50:
            builder.add(text, f"mswb_{i:05d}")
        if (i + 1) % 500 == 0:
            elapsed = time.monotonic() - t0
            print(f"    {i + 1:,} rows processed ({elapsed:.0f}s)", flush=True)
    entries, chunks, size_mb = builder.finish()
    elapsed = time.monotonic() - t0
    print(f"  [OK] {db_name}: {entries:,} entries, {chunks:,} chunks "
          f"({size_mb:.0f} MB, {elapsed:.0f}s)")
    return True


# ======================================================================
# Category E continued
# ======================================================================

def process_commit_chronicle():
    """Commit chronicle: 10.7M commit-diff pairs (LARGE)."""
    def build_text(row, cols):
        parts = []
        msg = row.get("message", "") or row.get("commit_message", "")
        diff = row.get("diff", "") or row.get("patch", "")
        repo = row.get("repo", "") or row.get("repository", "")
        if repo and repo != "None":
            parts.append(f"Repo: {repo}")
        if msg and msg != "None":
            parts.append(f"Commit message: {msg[:1000]}")
        if diff and diff != "None":
            parts.append(f"Diff:\n{diff[:3000]}")
        return "\n\n".join(parts)
    return _generic_download_and_index_batched(
        "Cyborg-AI/commit-chronicle", "commit_chronicle.fts5.db",
        "commit_chronicle", "cchr", build_text)


# ======================================================================
# Main
# ======================================================================

# Category -> list of (key, description, function)
CATEGORIES = {
    "A": [
        ("humanevalplus", "HumanEval+ (164 problems)", process_humanevalplus),
        ("mbppplus", "MBPP+ (378 problems)", process_mbppplus),
        ("swesmith_traj", "SWE-smith trajectories (5K)", process_swesmith_traj),
        ("openhands_traj", "OpenHands trajectories (67K)", process_openhands_traj),
        ("sweagent_traj", "SWE-agent trajectories (80K)", process_sweagent_traj),
        ("menvdata_traj", "MEnvData trajectories (3.8K)", process_menvdata_traj),
        ("coderforge", "CoderForge (51K agentic)", process_coderforge),
        ("codereview_python", "Code Review Python", process_codereview_python),
        ("codereview_general", "Code Review General", process_codereview_general),
        ("code_act", "CodeAct trajectories (78K)", process_code_act),
        ("swe_gym_traj", "SWE-Gym trajectories (6K)", process_swe_gym_traj),
        ("code_refine", "Code refinement pairs (123K)", process_code_refine),
        ("github_codereview", "GitHub code review (2.58 GB)", process_github_codereview),
    ],
    "B": [
        ("magicoder", "Magicoder 110K", process_magicoder),
        ("evol_codealpaca", "Evol-CodeAlpaca", process_evol_codealpaca),
        ("code_search_net", "CodeSearchNet Python", process_code_search_net),
        ("opencodeinstruct", "OpenCodeInstruct (5M, BEAST)", process_opencodeinstruct),
    ],
    "C": [
        ("vuln_cwe_patch", "CIRCL Vuln+CWE+Patch (39K)", process_vuln_cwe_patch),
        ("cve_cwe_dataset", "CVE+CWE 1999-2025", process_cve_cwe_dataset),
        ("vuln_security_dpo", "Security DPO", process_vuln_security_dpo),
        ("securecode_web", "SecureCode Web (1.3K)", process_securecode_web),
        ("cve_training", "CVE Training (300K)", process_cve_training),
    ],
    "D": [
        ("github_code_2025", "GitHub Code 2025 (1.5M repos)", process_github_code_2025),
    ],
    "E": [
        ("dockerfiles", "Dockerfiles (195K)", process_dockerfiles),
        ("sql_schemas", "SQL schemas (78K)", process_sql_schemas),
        ("function_calling", "Function calling specs (60K)", process_function_calling),
        ("commit_chronicle", "Commit chronicle (10.7M, LARGE)", process_commit_chronicle),
        ("hermes_function_calling", "Hermes function calling", process_hermes_function_calling),
        ("glaive_function_calling_v2", "Glaive function calling v2", process_glaive_function_calling_v2),
        ("jupyter_agent", "Jupyter agent (51K notebooks)", process_jupyter_agent),
    ],
    "F": [
        ("mixture_of_thoughts", "Mixture-of-Thoughts (350K)", process_mixture_of_thoughts),
        ("verifiable_coding", "Verifiable coding problems", process_verifiable_coding),
        ("livecodebench", "LiveCodeBench (recent)", process_livecodebench),
        ("swe_rebench_v2", "SWE-rebench-V2 (32K multilingual)", process_swe_rebench_v2),
        ("multi_swe_bench", "Multi-SWE-bench (multilingual)", process_multi_swe_bench),
        ("trail_agent", "TRAIL agent eval", process_trail_agent),
        ("openthoughts_agent", "OpenThoughts Agent v1", process_openthoughts_agent),
        ("nemotron_agentic", "Nemotron Agentic v1 (NVIDIA)", process_nemotron_agentic),
    ],
}

ALL_PROCESSORS = {}
for cat_procs in CATEGORIES.values():
    for key, desc, fn in cat_procs:
        ALL_PROCESSORS[key] = (desc, fn)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: World-class agent knowledge downloads"
    )
    parser.add_argument(
        "--only", default="",
        help="Comma-separated list of dataset keys"
    )
    parser.add_argument(
        "--category", default="",
        help="Download entire category (A, B, C, D, or ALL)"
    )
    args = parser.parse_args()

    if args.only:
        selected = [s.strip() for s in args.only.split(",") if s.strip()]
    elif args.category:
        cats = args.category.upper().split(",")
        selected = []
        for cat in cats:
            cat = cat.strip()
            if cat == "ALL":
                selected = list(ALL_PROCESSORS.keys())
                break
            elif cat in CATEGORIES:
                selected.extend(key for key, _, _ in CATEGORIES[cat])
            else:
                print(f"[WARN] Unknown category: {cat}")
        if not selected:
            selected = list(ALL_PROCESSORS.keys())
    else:
        selected = list(ALL_PROCESSORS.keys())

    print("=" * 60)
    print("JCoder Phase 6: World-Class Agent Knowledge")
    print(f"Datasets: {', '.join(selected)}")
    print("=" * 60)

    results = {}
    t0 = time.monotonic()

    for key in selected:
        if key not in ALL_PROCESSORS:
            print(f"\n[WARN] Unknown dataset key: {key}")
            continue
        desc, fn = ALL_PROCESSORS[key]
        print(f"\n--- {desc} ---")
        try:
            ok = fn()
            results[key] = "OK" if ok else "FAIL"
        except Exception as exc:
            results[key] = f"FAIL: {exc}"
            print(f"  [FAIL] {exc}")

    elapsed = time.monotonic() - t0
    _close_downloader()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for key, status in results.items():
        print(f"  {key:30s}: {status}")
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
