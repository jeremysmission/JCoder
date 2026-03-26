# Sprint: Direct CUDA Embedding (27h → 36min)

**Priority:** P0 — Blocks 700GB indexing on work desktop
**Estimated Impact:** 45x embedding speedup
**Sprint Owner:** TBD (assign team)
**Repo:** HybridRAG3_Educational
**Date:** 2026-03-25

---

## Background

Embedding 700GB (460K files, 11.5M chunks) currently takes ~27 hours due to sequential HTTP round-trips to Ollama. Direct CUDA embedding via sentence-transformers eliminates the HTTP overhead entirely.

All 3 machines have 12GB+ GPU (Beast: dual 3090 48GB, work desktop: Blackwell, work laptop: Blackwell). No CPU fallback needed.

---

## Slice 1: Requirements + Approvals (1 hour)

**Owner:** Documentation lead
**Files:** `requirements_approved.txt`

- [ ] Add `torch==2.6.0` (BSD-3, Meta/USA — note: Meta BANNED for models, torch library itself is fine)
- [ ] Add `sentence-transformers==4.1.0` (Apache 2.0, UKP Lab/Germany)
- [ ] Add transitive deps: `transformers`, `tokenizers`, `safetensors`, `huggingface-hub`
- [ ] Mark all as APPLYING with license + origin
- [ ] Document: "Added for 45x embedding speedup. Ollama retained for LLM inference only."
- [ ] Verify no China-origin packages in the dependency chain

**Acceptance:** `pip install -r requirements_approved.txt` succeeds cleanly

---

## Slice 2: Embedder Rewrite (2-3 hours)

**Owner:** Core coder
**Files:** `src/core/embedder.py`

- [ ] Add `SentenceTransformer` import with try/except fallback
- [ ] New method `_embed_direct_cuda(texts)` using `model.encode(texts, batch_size=256, device='cuda')`
- [ ] Modify `embed_batch()`: try direct CUDA first, fall back to Ollama HTTP if torch unavailable
- [ ] Auto-detect GPU at init: `torch.cuda.is_available()`
- [ ] Log which path is active: "Embedding via direct CUDA (45x)" or "Embedding via Ollama HTTP (fallback)"
- [ ] Set `OLLAMA_NUM_PARALLEL=1` for Ollama path (bug #6262)
- [ ] Respect `HYBRIDRAG_EMBED_BATCH` env var (default 256 for CUDA, 64 for Ollama)
- [ ] Keep model name configurable: `nomic-embed-text` default, `nomic-embed-text-v2-moe` option
- [ ] Handle OOM gracefully: if CUDA OOM, halve batch size and retry

**Acceptance:**
```python
from src.core.embedder import Embedder
e = Embedder()
# Should print "Embedding via direct CUDA"
vecs = e.embed_batch(["test"] * 256)
assert vecs.shape == (256, 768)
```

---

## Slice 3: Config Update (30 min)

**Owner:** Core coder
**Files:** `config/config.yaml`, `src/core/config.py`

- [ ] Add `embedding.device: auto` (auto/cuda/cpu)
- [ ] Add `embedding.batch_size: 256`
- [ ] Add `embedding.model_path: nomic-ai/nomic-embed-text-v1.5` (HuggingFace model ID)
- [ ] Remove `device: str = "cuda"  # Unused` comment from config.py
- [ ] Wire new config fields into Embedder constructor

**Acceptance:** `jcoder doctor check` shows embedding device correctly

---

## Slice 4: Documentation Updates (2 hours)

**Owner:** Documentation lead
**Files:** 6 docs

- [ ] `docs/01_setup/INSTALL_AND_SETUP.md` — Add "GPU Embedding Setup" section
  - torch install command: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
  - sentence-transformers install
  - First-run model download note (nomic-embed-text auto-downloads ~500MB)
  - Verify: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] `docs/01_setup/MANUAL_INSTALL.md` — Same GPU section
- [ ] `docs/02_architecture/TECHNICAL_THEORY_OF_OPERATION_RevC.md`
  - Update embedding pipeline diagram: "Ollama HTTP" → "Direct CUDA (sentence-transformers)"
  - Add note: "Ollama retained for LLM inference. Embeddings use direct GPU for 45x speedup."
- [ ] `docs/05_security/waiver_reference_sheet.md` — Add torch + sentence-transformers entries
- [ ] `docs/03_guides/AUTOTUNE_CHEAT_SHEET.md` — Update if embedding timing is referenced
- [ ] `CLAUDE.md` — Add: "Embedding uses direct CUDA. Ollama is for LLM only."

**Acceptance:** Fresh reader can follow install guide and get CUDA embedding working

---

## Slice 5: Indexer Integration (1 hour)

**Owner:** Core coder
**Files:** `src/core/indexer.py`

- [ ] Add `multiprocessing.Pool(workers=8)` for file parsing (8x speedup on parsing stage)
- [ ] Add batch SQLite commits (1000 chunks per commit instead of 1)
- [ ] Log embedding throughput: "Embedding: X chunks/sec (direct CUDA)" during indexing
- [ ] Add progress bar with ETA based on throughput

**Acceptance:** Index 1000 files and verify:
- Embedding uses CUDA (check log)
- Parsing uses multiple cores (check CPU usage)
- SQLite commits are batched (check I/O pattern)

---

## Slice 6: Testing (1 hour)

**Owner:** QA
**Files:** `tests/`

- [ ] `test_embedder_cuda.py` — Test direct CUDA embedding produces correct dimensions
- [ ] `test_embedder_fallback.py` — Test Ollama fallback when torch unavailable
- [ ] `test_embedder_batch.py` — Test batch sizes 64, 128, 256, 512
- [ ] `test_embedder_oom.py` — Test OOM recovery (halve batch + retry)
- [ ] Run full regression: all 1,296 tests still pass
- [ ] Benchmark: time 10,000 chunk embedding via CUDA vs Ollama HTTP

**Acceptance:** All new + existing tests pass. Benchmark shows >10x speedup.

---

## Slice 7: Env Var Setup Script (30 min)

**Owner:** DevOps
**Files:** `tools/setup_gpu_embedding.bat` (new)

- [ ] Create one-click setup script:
```batch
@echo off
echo Setting up GPU embedding environment...
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KEEP_ALIVE 24h
setx OLLAMA_MAX_LOADED_MODELS 2
setx OLLAMA_NUM_PARALLEL 1
setx HYBRIDRAG_EMBED_BATCH 256
echo Done. Restart Ollama for changes to take effect.
```
- [ ] Add to INSTALL_AND_SETUP.md as post-install step

**Acceptance:** Running script + restarting Ollama = all env vars set correctly

---

## Timeline

| Slice | Est. Hours | Dependency |
|-------|-----------|------------|
| 1. Requirements | 1h | None |
| 2. Embedder rewrite | 2-3h | Slice 1 |
| 3. Config update | 0.5h | Slice 2 |
| 4. Documentation | 2h | Slice 2 |
| 5. Indexer integration | 1h | Slice 2 |
| 6. Testing | 1h | Slices 2-5 |
| 7. Env var script | 0.5h | None |
| **TOTAL** | **8-9 hours** | |

**Can parallelize:** Slices 1+7 (no deps), then 2+3+4 (parallel after 1), then 5+6.
**Critical path:** Slice 1 → Slice 2 → Slice 6 = ~5 hours minimum.

---

## Success Metrics

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| 700GB embed time | 27 hours | <1 hour | 45x speedup |
| Parsing time | 6.4 hours | <1 hour | 8x speedup |
| Total index time | 35 hours | <3 hours | 12x speedup |
| Tests passing | 1,296 | 1,296+ | No regression |

---

Signed: Claude Opus 4.6 | Sprint Plan | 2026-03-25 19:50 MDT
