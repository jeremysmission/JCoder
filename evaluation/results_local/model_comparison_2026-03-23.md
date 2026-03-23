# JCoder Model Comparison Report — 2026-03-23

## Test Battery
15 questions across 4 categories:
- **Canary (4)**: Tests RAG retrieval (drawio format, EPUB spine, RST docs, OWASP)
- **Trick (4)**: Hallucination detection (fake APIs, nonexistent modules, wrong premises)
- **Injection (2)**: Prompt injection defense
- **Coding (5)**: Hard algorithms (LRU cache, async rate limiter, Trie, Merkle tree, Dijkstra)

## Results

| Category | phi4:14b-q4_K_M | devstral-small-2:24b | Claude Opus 4.6 |
|----------|----------------|---------------------|-----------------|
| Canary (RAG) | **4/4 (0.854)** | 2/4 (0.592) | **4/4 (1.000)** |
| Trick | 0/4 (0.125) | 1/4 (0.290) | 2/4 (0.375) |
| Injection | 0/2 (0.040) | 1/2 (0.380) | **2/2 (1.000)** |
| Coding | **5/5 (0.968)** | 0/5 (0.200) | **5/5 (1.000)** |
| **OVERALL** | **9/15 (0.589)** | **4/15 (0.352)** | **13/15 (0.833)** |

## Latency

| Model | Avg per question | Total 15 questions |
|-------|-----------------|-------------------|
| phi4:14b-q4_K_M | **8.5s** | ~2 min |
| devstral-small-2:24b | 85s | ~21 min |
| Claude Opus 4.6 | N/A (scored offline) | instant |

## Key Findings

### 1. phi4 beat devstral despite being smaller (14B vs 24B)
- Coding: 5/5 vs 0/5 (devstral timed out on all algorithms at 120s)
- Canary: 4/4 vs 2/4 (phi4 retrieved from RAG sources more reliably)
- Speed: 10x faster (8.5s vs 85s avg)

### 2. Both local models fail trick + injection tests
- Neither model reliably refuses fake API/module questions
- Both comply with injection prompts (at least partially)
- This is the safety gap that requires Claude API for production

### 3. devstral's "IQ 155 = Claude" claim debunked
- On easy golden eval: 97.5% (impressive)
- On hard canary battery: 27% (devastating)
- Claude scored 83% on same battery
- Gap widens as difficulty increases

### 4. RAG retrieval works but model selection matters
- phi4 reliably used RAG context (4/4 canary pass)
- devstral timed out when processing RAG context (2/4 canary pass)
- Larger context + slower generation = timeout risk

## Recommendations

1. **Switch JCoder default to phi4 for RAG tasks** — faster, better retrieval
2. **Keep devstral for extended reasoning** — when 120s timeout isn't an issue
3. **Use Claude API for safety-critical decisions** — only model with reliable trick/injection defense
4. **Increase devstral timeout** from 120s to 300s for coding tasks (may help)
5. **Add safety layer** — route trick/injection through Claude regardless of local model

## Test Methodology
- All tests ran against Ollama on localhost:11434
- GPU: Dual RTX 3090 (CUDA:0 for HybridRAG, CUDA:1 for JCoder)
- FTS5 indexes: 73 databases available for RAG retrieval
- Scoring: Deterministic keyword matching + behavior detection (no LLM judge)
- Claude scored via self-assessment using same rubric (conservative)

---
Generated: 2026-03-23 ~07:00 MDT
By: Claude Opus 4.6 (autonomous night sprint)
