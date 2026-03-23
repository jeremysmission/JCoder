# Extension Allowlist Analysis — JCoder Repository

**Date:** 2026-03-22
**Status:** VULNERABLE — Hardcoded extension lists in tools/ (drift risk)
**Severity:** MEDIUM (tools are utilities, not core indexing pipeline)
**Reference:** HybridRAG3_Educational Bug Fix (2026-03-21)

---

## Executive Summary

JCoder's **core ingestion pipeline** (chunker.py → repo_loader.py → corpus_pipeline.py) is **SAFE** — uses a single source of truth `LANGUAGE_MAP`.

However, **optional utility scripts in tools/** contain hardcoded extension lists that:
- Do NOT use the registry
- Are NOT synchronized with core
- Could drift over time (drift-vulnerable pattern from HybridRAG3_Educational)

**Recommendation:** Refactor tools to import from core registry (like HybridRAG3_Educational did).

---

## Current Architecture

### SAFE: Core Pipeline (Single Source of Truth)

```
chunker.py (LANGUAGE_MAP)
    ↓
    ├─→ repo_loader.py (imports LANGUAGE_MAP.keys())
    ├─→ corpus_pipeline.py (imports LANGUAGE_MAP)
    ├─→ ast_fts5_builder.py (imports LANGUAGE_MAP)
    └─→ build_fts5_indexes.py (imports LANGUAGE_MAP)
```

**File:** `C:\Users\jerem\JCoder\ingestion\chunker.py` (lines 26-47)

**LANGUAGE_MAP Registry:**
```python
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    # Config files -- no AST grammar, char-fallback only
    ".yaml": None,
    ".yml": None,
    # Documentation / harvested markdown -- char-fallback only
    ".md": None,
    ".txt": None,
    ".json": None,
}
```

**Consumers:**
- `repo_loader.py:16` — `from .chunker import Chunker, LANGUAGE_MAP`
- `repo_loader.py:92` — `self.supported_extensions = set(LANGUAGE_MAP.keys())`
- `corpus_pipeline.py:19` — `from ingestion.chunker import Chunker, LANGUAGE_MAP`
- `corpus_pipeline.py:45` — `_CODE_EXTENSIONS: Dict[str, str] = {ext: lang for ext, lang in LANGUAGE_MAP.items() if lang is not None}`
- `ast_fts5_builder.py:27` — `from ingestion.chunker import Chunker, LANGUAGE_MAP`
- `ast_fts5_builder.py:44` — Uses LANGUAGE_MAP in `_detect_language()`
- `build_fts5_indexes.py` — `AST_EXTENSIONS = {ext for ext, lang in LANGUAGE_MAP.items() if lang is not None}`
- `cli/doctor.py:16` — `from ingestion.chunker import CHUNKER_VERSION, LANGUAGE_MAP` (prints inventory)
- `tests/test_ast_chunker_integration.py` — Tests LANGUAGE_MAP consistency

✓ **All import from single source** — no copying, no drift

---

### VULNERABLE: Tools with Hardcoded Lists

#### 1. `tools/site_wiki_builder.py` (lines 35-45)

```python
DOC_TYPES = {
    "Drawings": {".dwg", ".dxf", ".dgn", ".vsd", ".vsdx", ".stp", ".step", ".igs", ".iges"},
    "Documents": {".docx", ".doc", ".pdf", ".rtf", ".odt", ".txt", ".md"},  # ← .odt
    "Spreadsheets": {".xlsx", ".xls", ".csv", ".ods"},  # ← .xls, .ods
    "Presentations": {".pptx", ".ppt", ".odp"},  # ← .ppt, .odp
    "Photos": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".heic"},
    "Email": {".msg", ".eml", ".pst"},
    "Archives": {".zip", ".7z", ".rar", ".tar", ".gz"},
    "Video": {".mp4", ".avi", ".mkv", ".mov", ".wmv"},
    "Code": {".py", ".ps1", ".sh", ".bat", ".js", ".html", ".htm", ".xml", ".json", ".yaml", ".yml"},
}
```

**Problem:** Hardcoded `.xls, .ppt, .odt, .ods, .odp` — if LANGUAGE_MAP adds these, DOC_TYPES is stale.

---

#### 2. `tools/wiki_builder.py` (lines 27-51)

```python
TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".js", ".ts", ".html", ".htm", ".css",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".conf",
    ".sh", ".bash", ".ps1", ".bat", ".cmd", ".xml", ".csv",
    ".rst", ".log", ".env", ".gitignore", ".dockerfile",  # ← .rst
}

BINARY_EXTENSIONS = {
    ".docx": "Word Document",
    # ...
    ".svg": "SVG Image",  # ← .svg
    # ...
}
```

**Problem:** Hardcoded `.rst, .svg` — not imported from core.

---

#### 3. `tools/wiki_builder_v2.py` (lines 33-50)

```python
TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".js", ".ts", ".html", ".htm", ".css",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".conf",
    ".sh", ".bash", ".ps1", ".bat", ".cmd", ".xml", ".csv",
    ".rst", ".log", ".dockerfile", ".sql", ".r", ".go", ".rs",  # ← .rst
}

BINARY_LABELS = {
    # ...
    ".svg": "SVG Image", ".bmp": "Image",  # ← .svg
    # ...
}
```

**Problem:** Hardcoded `.rst, .svg` — stale copy.

---

#### 4. `tools/fts5_demo.py` (line 51)

```python
all_ext_list = [
    ".py", ".js", ".ts", ".cfg", ".ini", ".rst", ".xml", ".csv",  # ← .rst
    # ...
]
```

**Problem:** Hardcoded `.rst` — not using LANGUAGE_MAP.

---

## Missing Critical Formats

**Formats recognized in HybridRAG3_Educational but NOT in JCoder's LANGUAGE_MAP:**

| Format | Status | Missing From | Risk |
|--------|--------|--------------|------|
| `.xls` | ✓ (parsers exist in HybridRAG3) | LANGUAGE_MAP | If JCoder adds support, tools won't know |
| `.ppt` | ✓ (parsers exist in HybridRAG3) | LANGUAGE_MAP | If JCoder adds support, tools won't know |
| `.odt` | ✓ (parsers exist in HybridRAG3) | LANGUAGE_MAP | If JCoder adds support, tools won't know |
| `.ods` | ✓ (parsers exist in HybridRAG3) | LANGUAGE_MAP | If JCoder adds support, tools won't know |
| `.odp` | ✓ (parsers exist in HybridRAG3) | LANGUAGE_MAP | If JCoder adds support, tools won't know |
| `.epub` | ✓ (parsers exist in HybridRAG3) | LANGUAGE_MAP | If JCoder adds support, tools won't know |
| `.drawio` | ✓ (plain text XML in HybridRAG3) | LANGUAGE_MAP | If JCoder adds support, tools won't know |
| `.dia` | ✓ (plain text XML in HybridRAG3) | LANGUAGE_MAP | If JCoder adds support, tools won't know |

**Note:** JCoder currently doesn't support these formats in core, but tools have drifted hardcoded lists that WILL cause problems if:
1. Core adds format support in the future
2. Tools are copy-pasted without updating lists

---

## Drift Risk Assessment

**Scenario:** Developer adds `.xls` support to JCoder core:

```python
# In chunker.py
LANGUAGE_MAP = {
    # ...
    ".xls": None,  # ← NEW
}
```

**What happens:**
- ✓ repo_loader.py discovers .xls files (uses LANGUAGE_MAP.keys())
- ✓ corpus_pipeline.py handles .xls files (checks LANGUAGE_MAP)
- ✗ tools/site_wiki_builder.py STILL classifies .xls as "Other" (uses hardcoded DOC_TYPES)
- ✗ tools/wiki_builder.py STILL treats .xls as unknown (uses hardcoded TEXT_EXTENSIONS)

**Result:** Silent inconsistency — core indexes .xls but tools classify differently.

---

## Fix: Sync All Extension Lists to LANGUAGE_MAP

### Pattern from HybridRAG3_Educational

```python
# registry.py (single source of truth)
SUPPORTED_FORMATS = ['.xls', '.ppt', '.odt', ...]

# config.py (imports at runtime, no copying)
from src.parsers.registry import SUPPORTED_FORMATS

# downloader.py (imports at runtime)
from src.parsers.registry import SUPPORTED_FORMATS

# test_sync.py (enforces parity)
def test_config_synced():
    assert set(config.SUPPORTED_FORMATS) == set(registry.SUPPORTED_FORMATS)
```

### JCoder Approach

**Option A: Minimal (Recommended for tools)**
- Keep LANGUAGE_MAP in chunker.py
- Refactor tools to import from chunker.py instead of hardcoding

**Option B: Full Registry (Optional)**
- Create `src/parsers/registry.py` mirroring HybridRAG3_Educational pattern
- Migrate core and tools to use it
- Add sync tests

---

## Recommendation

Implement **Option A (Minimal)** immediately:

1. **tools/site_wiki_builder.py**
   - Import LANGUAGE_MAP
   - Derive DOC_TYPES from it (where applicable)

2. **tools/wiki_builder.py**
   - Import LANGUAGE_MAP
   - Sync TEXT_EXTENSIONS with LANGUAGE_MAP keys

3. **tools/wiki_builder_v2.py**
   - Import LANGUAGE_MAP
   - Sync TEXT_EXTENSIONS with LANGUAGE_MAP keys

4. **tools/fts5_demo.py**
   - Import LANGUAGE_MAP
   - Use it instead of hardcoded list

5. **Add sync test**
   - Verify all tools import from LANGUAGE_MAP
   - Prevent future hardcoded copies

---

## Recovery Plan

If tools are deployed with stale extension lists:

1. **Audit:** Grep tools/ for hardcoded extension lists
2. **Identify:** Which formats are missing from tools
3. **Refactor:** Point tools to LANGUAGE_MAP (central source)
4. **Test:** Run tools with new LANGUAGE_MAP before deploying
5. **Repeat:** Any time LANGUAGE_MAP is updated

---

## Files to Review

- Core (SAFE):
  - `C:\Users\jerem\JCoder\ingestion\chunker.py`
  - `C:\Users\jerem\JCoder\ingestion\repo_loader.py`
  - `C:\Users\jerem\JCoder\ingestion\corpus_pipeline.py`

- Tools (VULNERABLE):
  - `C:\Users\jerem\JCoder\tools\site_wiki_builder.py`
  - `C:\Users\jerem\JCoder\tools\wiki_builder.py`
  - `C:\Users\jerem\JCoder\tools\wiki_builder_v2.py`
  - `C:\Users\jerem\JCoder\tools\fts5_demo.py`

- Tests (ADD):
  - `C:\Users\jerem\JCoder\tests\test_extension_sync.py` (NEW)

---

## Timeline

- **Detection:** 2026-03-22
- **Root Cause:** Hardcoded lists in tools (not importing from core)
- **Fix:** Refactor tools to use LANGUAGE_MAP
- **Testing:** Add sync test to prevent future drift
- **Deployment:** Follow JCoder's standard test cycle

---

**Severity Justification:**

- **MEDIUM** (not CRITICAL) because:
  - Core pipeline is safe (single source of truth)
  - Tools are utilities, not critical indexing path
  - Drift hasn't caused data loss yet (unlike HybridRAG3)
  - Easy to fix (just import, don't copy)

- **Still needs immediate fix** because:
  - HybridRAG3_Educational shows this pattern leads to large-scale data loss
  - Prevention is cheaper than recovery
  - Sync test is trivial to add

---

**Author:** AI agents (Claude Haiku 4.5)
**Reviewed by:** (pending)
