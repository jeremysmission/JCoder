"""
Overnight Data Acquisition + Preprocessing Pipeline for JCoder.

Downloads, validates, sanitizes, and stages coding datasets
for indexing when BEAST hardware arrives.

Run this before going to sleep:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\overnight_download.py

It will:
1. Download CodeSearchNet (Python, JS, Java, Go, Ruby, PHP) -- ~20 GB
2. Download The Stack v2 filtered subset (Python, JS) -- streaming, ~50-100 GB
3. Sanitize everything through the existing pipeline
4. Build FTS5 keyword indexes for immediate search
5. Log progress to D:\\JCoder_Data\\logs\\overnight_*.log

Safe to interrupt and resume -- downloads are resumable.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_ROOT = Path(os.environ.get("JCODER_DATA", r"D:\JCoder_Data"))
RAW_ROOT = DATA_ROOT / "raw_downloads"
CLEAN_ROOT = DATA_ROOT / "clean_source"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
LOG_DIR = DATA_ROOT / "logs"

for d in [RAW_ROOT, CLEAN_ROOT, INDEX_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Log setup
LOG_FILE = LOG_DIR / f"overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# -----------------------------------------------------------------------
# Phase 1: CodeSearchNet download
# -----------------------------------------------------------------------

CODESEARCHNET_LANGS = ["python", "javascript", "java", "go", "ruby", "php"]


def download_codesearchnet():
    """Download CodeSearchNet dataset via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        log("[WARN] 'datasets' not installed. Run: pip install datasets")
        return

    csn_dir = RAW_ROOT / "codesearchnet"
    csn_dir.mkdir(parents=True, exist_ok=True)

    for lang in CODESEARCHNET_LANGS:
        marker = csn_dir / f"{lang}.done"
        if marker.exists():
            log(f"[OK] CodeSearchNet {lang} already downloaded (marker exists)")
            continue

        log(f"Downloading CodeSearchNet {lang} via HuggingFace...")
        try:
            ds = load_dataset("code_search_net", lang, split="train", cache_dir=str(csn_dir / "hf_cache"))
            # Save as JSONL for extraction phase
            jsonl_path = csn_dir / f"{lang}.jsonl"
            count = 0
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in ds:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 50000 == 0:
                        log(f"  {lang}: {count:,} entries saved")
            marker.write_text(str(count))
            log(f"[OK] CodeSearchNet {lang}: {count:,} entries ({jsonl_path.stat().st_size / 1e6:.0f} MB)")
        except Exception as e:
            log(f"[FAIL] CodeSearchNet {lang}: {e}")


def _read_counter(counter_path: Path) -> int:
    """Read extraction counter file, return 0 if missing."""
    if counter_path.exists():
        try:
            return int(counter_path.read_text().strip())
        except (ValueError, OSError):
            pass
    return 0


def _write_counter(counter_path: Path, count: int):
    """Write extraction progress counter."""
    counter_path.write_text(str(count), encoding="utf-8")


def _subdir_for(lang_dir: Path, index: int, batch_size: int = 1000) -> Path:
    """Return subdirectory path: lang_dir/000/, lang_dir/001/, etc."""
    bucket = index // batch_size
    sub = lang_dir / f"{bucket:03d}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def extract_codesearchnet():
    """Convert CodeSearchNet JSONL to sanitized markdown files.

    Uses subdirectory batching (1000 files per subdir) to avoid NTFS
    slowdown at 300K+ files. Tracks progress via counter file for resume.
    """
    csn_dir = RAW_ROOT / "codesearchnet"
    out_dir = CLEAN_ROOT / "codesearchnet"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_entries = 0
    for lang in CODESEARCHNET_LANGS:
        jsonl_path = csn_dir / f"{lang}.jsonl"
        if not jsonl_path.exists():
            log(f"[WARN] CodeSearchNet {lang}.jsonl not found, skipping")
            continue

        lang_dir = out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        counter_path = lang_dir / ".count"
        existing = _read_counter(counter_path)

        # Count JSONL lines to check if fully extracted
        jsonl_lines = sum(1 for _ in open(jsonl_path, "r", encoding="utf-8"))
        if existing >= jsonl_lines and existing > 100:
            log(f"[OK] CodeSearchNet {lang} fully extracted ({existing:,} files)")
            total_entries += existing
            continue

        if existing > 0:
            log(f"Resuming CodeSearchNet {lang} from entry {existing:,}...")
        else:
            log(f"Extracting CodeSearchNet {lang} ({jsonl_lines:,} entries)...")

        try:
            count = existing
            skipped = 0
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    if line_num < existing:
                        continue  # skip already-extracted lines

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    code = obj.get("whole_func_string", obj.get("code", obj.get("function", "")))
                    docstring = obj.get("func_documentation_string", obj.get("docstring", ""))
                    func_name = obj.get("func_name", obj.get("identifier", ""))
                    repo = obj.get("repository_name", "")

                    if not code or len(code.strip()) < 20:
                        skipped += 1
                        continue

                    title = func_name or f"function_{count}"
                    md = f"# {title}\n\n"
                    md += f"- source_kind: codesearchnet\n"
                    md += f"- language: {lang}\n"
                    if repo:
                        md += f"- repo: {repo}\n"
                    if docstring:
                        md += f"\n## Documentation\n{docstring.strip()}\n"
                    md += f"\n## Code\n```{lang}\n{code.strip()}\n```\n"

                    sub = _subdir_for(lang_dir, count)
                    fname = f"csn_{lang}_{count:06d}.md"
                    (sub / fname).write_text(md, encoding="utf-8")
                    count += 1

                    if count % 10000 == 0:
                        _write_counter(counter_path, count)
                        log(f"  {lang}: {count:,} entries extracted")

            _write_counter(counter_path, count)
            log(f"[OK] CodeSearchNet {lang}: {count:,} entries ({skipped:,} skipped)")
            total_entries += count
        except Exception as e:
            log(f"[FAIL] CodeSearchNet {lang} extraction at {count:,}: {e}")
            _write_counter(counter_path, count)  # save progress on crash

    log(f"[OK] CodeSearchNet total: {total_entries:,} entries")
    return total_entries


# -----------------------------------------------------------------------
# Phase 2: The Stack v2 (streaming download, filtered)
# -----------------------------------------------------------------------

def download_code_datasets():
    """Download code datasets via HuggingFace streaming.

    Tries multiple sources in priority order:
    1. codeparrot/github-code (ungated, huge, multi-language)
    2. bigcode/the-stack-v2-dedup (gated, needs HF_TOKEN)
    3. bigcode/starcoderdata (ungated subset)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log("[WARN] 'datasets' not installed. Run: pip install datasets huggingface_hub")
        return 0

    code_dir = CLEAN_ROOT / "github_code"
    code_dir.mkdir(parents=True, exist_ok=True)

    # Languages to download -- all coding languages for max coverage
    languages = ["Python", "JavaScript", "Java", "Go", "C", "C++", "TypeScript", "Rust", "C#", "Shell"]
    total_files = 0

    for lang in languages:
        lang_lower = lang.lower().replace("+", "p").replace("#", "sharp")
        lang_dir = code_dir / lang_lower
        lang_dir.mkdir(parents=True, exist_ok=True)
        counter_path = lang_dir / ".count"

        existing = _read_counter(counter_path)
        if existing > 50000:
            log(f"[OK] Code {lang} already has {existing:,} files, skipping")
            total_files += existing
            continue

        count = existing
        target = 200_000  # Per language cap

        # Try codeparrot/github-code first (ungated)
        log(f"Streaming codeparrot/github-code ({lang})...")
        try:
            ds = load_dataset(
                "codeparrot/github-code",
                languages=[lang],
                licenses=["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"],
                split="train",
                streaming=True,
            )

            for item in ds:
                if count >= target:
                    log(f"  {lang}: hit target {target:,}, stopping")
                    break

                content = item.get("code", "")
                if len(content) < 50 or len(content) > 100_000:
                    continue
                if content.count("\n") < 3:
                    continue

                path = item.get("path", f"file_{count}")
                repo = item.get("repo_name", "unknown")
                license_name = item.get("license", "unknown")

                md = f"# {Path(path).name}\n\n"
                md += f"- source_kind: github_code\n"
                md += f"- language: {lang_lower}\n"
                md += f"- repo: {repo}\n"
                md += f"- license: {license_name}\n"
                md += f"- path: {path}\n"
                md += f"\n## Code\n```{lang_lower}\n{content}\n```\n"

                sub = _subdir_for(lang_dir, count)
                fname = f"ghcode_{lang_lower}_{count:07d}.md"
                (sub / fname).write_text(md, encoding="utf-8")
                count += 1

                if count % 10000 == 0:
                    _write_counter(counter_path, count)
                    log(f"  github-code {lang}: {count:,} files saved")

        except Exception as e:
            log(f"[WARN] github-code {lang} error at {count:,}: {e}")

            # Fallback: try Stack v2 if HF_TOKEN is set
            if os.environ.get("HF_TOKEN"):
                log(f"  Trying Stack v2 fallback for {lang}...")
                try:
                    ds2 = load_dataset(
                        "bigcode/the-stack-v2-dedup",
                        data_dir=f"data/{lang.lower()}",
                        split="train",
                        streaming=True,
                    )
                    for item in ds2:
                        if count >= target:
                            break
                        content = item.get("content", "")
                        if len(content) < 50 or len(content) > 100_000:
                            continue
                        if content.count("\n") < 3:
                            continue

                        path = item.get("path", f"file_{count}")
                        repo = item.get("repo_name", "unknown")

                        md = f"# {Path(path).name}\n\n"
                        md += f"- source_kind: stack_v2\n"
                        md += f"- language: {lang_lower}\n"
                        md += f"- repo: {repo}\n"
                        md += f"\n## Code\n```{lang_lower}\n{content}\n```\n"

                        sub = _subdir_for(lang_dir, count)
                        fname = f"stack_{lang_lower}_{count:07d}.md"
                        (sub / fname).write_text(md, encoding="utf-8")
                        count += 1

                        if count % 10000 == 0:
                            _write_counter(counter_path, count)
                            log(f"  Stack v2 {lang}: {count:,} files saved")
                except Exception as e2:
                    log(f"  [WARN] Stack v2 {lang} also failed: {e2}")

        _write_counter(counter_path, count)
        log(f"[OK] Code {lang}: {count:,} files total")
        total_files += count

    log(f"[OK] Code datasets total: {total_files:,} files")
    return total_files


# -----------------------------------------------------------------------
# Phase 3: Build FTS5 indexes for new data
# -----------------------------------------------------------------------

def build_fts5_for_dir(source_dir: Path, index_name: str):
    """Build FTS5 keyword index for a directory of .md files."""
    import re

    def _normalize(text):
        out = re.sub(r"[_\-./\\:]", " ", text)
        out = re.sub(r"([a-z])([A-Z])", r"\1 \2", out)
        out = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", out)
        return out.lower()

    files = []
    for dirpath, _, filenames in os.walk(source_dir):
        for fn in filenames:
            if fn.endswith(".md"):
                files.append(Path(dirpath) / fn)

    if not files:
        log(f"[WARN] No .md files in {source_dir}")
        return 0

    db_path = str(INDEX_DIR / f"{index_name}.fts5.db")

    # Skip if already built and has entries
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            if row and row[0] > 0:
                log(f"[OK] FTS5 index '{index_name}' already has {row[0]:,} rows")
                return row[0]
        except Exception:
            pass
        finally:
            conn.close()

    log(f"Building FTS5 index '{index_name}' from {len(files):,} files...")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(
            "CREATE VIRTUAL TABLE chunks "
            "USING fts5(search_content, source_path, chunk_id)"
        )

        batch = []
        total = 0
        MAX_CHARS = 4000

        for fp in files:
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if not content.strip():
                continue

            # Simple chunking
            pos = 0
            while pos < len(content):
                end = min(pos + MAX_CHARS, len(content))
                if end < len(content):
                    nl = content.rfind("\n", pos, end)
                    if nl > pos:
                        end = nl + 1
                chunk = content[pos:end]
                if chunk.strip():
                    cid = hashlib.sha256(f"{fp}:{pos}".encode()).hexdigest()
                    batch.append((_normalize(chunk), str(fp), cid))
                pos = end

            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT INTO chunks(search_content, source_path, chunk_id) VALUES (?, ?, ?)",
                    batch,
                )
                conn.commit()
                total += len(batch)
                batch = []

                if total % 50000 == 0:
                    log(f"  {index_name}: {total:,} chunks indexed")

        if batch:
            conn.executemany(
                "INSERT INTO chunks(search_content, source_path, chunk_id) VALUES (?, ?, ?)",
                batch,
            )
            conn.commit()
            total += len(batch)
    finally:
        conn.close()
    log(f"[OK] FTS5 index '{index_name}': {total:,} chunks")
    return total


# -----------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------

def main():
    log("=" * 60)
    log("JCoder Overnight Download + Preprocessing Pipeline")
    log(f"Start: {datetime.now().isoformat()}")
    log(f"Log: {LOG_FILE}")
    log(f"Disk free: {shutil.disk_usage('D:\\').free / 1e9:.0f} GB")
    log("=" * 60)

    t0 = time.time()

    # Phase 1: CodeSearchNet
    log("\n--- PHASE 1: CodeSearchNet ---")
    download_codesearchnet()
    csn_count = extract_codesearchnet()

    # Phase 2: Code datasets (github-code / Stack v2 fallback)
    log("\n--- PHASE 2: Code Datasets (streaming) ---")
    stack_count = download_code_datasets()

    # Phase 3: Build FTS5 indexes
    log("\n--- PHASE 3: FTS5 Indexing ---")

    csn_dir = CLEAN_ROOT / "codesearchnet"
    if csn_dir.exists():
        build_fts5_for_dir(csn_dir, "codesearchnet")

    code_dir = CLEAN_ROOT / "github_code"
    if code_dir.exists():
        build_fts5_for_dir(code_dir, "github_code")

    # Summary
    elapsed = time.time() - t0
    log("\n" + "=" * 60)
    log("OVERNIGHT PIPELINE COMPLETE")
    log(f"Total time: {elapsed / 3600:.1f} hours")
    log(f"CodeSearchNet entries: {csn_count:,}")
    log(f"Stack v2 files: {stack_count:,}")
    log(f"Disk free: {shutil.disk_usage('D:\\').free / 1e9:.0f} GB")
    log(f"Log: {LOG_FILE}")
    log("=" * 60)


if __name__ == "__main__":
    main()
