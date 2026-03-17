# JCoder Full Review + Research Brief -- 2026-03-15
**Sources:** 5 parallel research agents (coding agents, RAG optimization, self-learning, architecture review, docs review)

## TOP 10 ACTIONABLE FINDINGS (Priority Order)

### 1. CRITICAL: ModelCascade + SmartOrchestrator Not Wired (bridge.py)
- Both imported but NEVER INSTANTIATED in `_try_init_pipeline()`
- Config flags `cascade_enabled` and `smart_orchestrator_enabled` are read but unused
- Sprint 10+12 features appear enabled but are completely non-functional
- **Fix:** Wire instantiation in bridge.py (~40 lines total)

### 2. CRITICAL: Tree-Sitter Repo Map (from Aider)
- Aider uses tree-sitter to build a dependency-graph-ranked summary of classes/functions/signatures
- Sends concise structural context instead of raw code chunks
- Drastically reduces token waste and improves retrieval relevance
- **JCoder already has tree-sitter** -- just needs the repo-map layer on top

### 3. HIGH: AST-Based Chunking (cAST pattern)
- Research shows +4.3 Recall@5 on RepoEval, +2.67 Pass@1 on SWE-bench
- Chunk boundaries align with syntactic units (functions, classes)
- JCoder has tree-sitter chunker but may not be using it optimally
- Open-source: `astchunk` (Python), `code-chunk` (tree-sitter)

### 4. HIGH: Reflexion / Verbal Self-Critique (no weight updates needed)
- After each task, generate verbal critique, store in episodic memory
- 91% pass@1 on HumanEval (vs GPT-4's 80%)
- Maps directly to JCoder's Thompson sampling + experience replay
- **Zero weight updates** -- pure prompt-level learning with any local LLM

### 5. HIGH: V-STaR Verifier Training
- Train a verifier on BOTH correct AND incorrect solutions (STaR normally discards failures)
- Verifier enables best-of-N selection at inference for free
- JCoder already has BestOfN -- adding a trained verifier would dramatically improve it

### 6. HIGH: Granite Code 8B/20B (IBM, Apache 2.0, USA)
- Outperforms CodeLlama 16B at half the size
- USA-origin, Apache 2.0 -- clean NDAA/ITAR
- 8B fits workstation (12GB GPU), 20B fits BEAST
- **New approved model candidate**

### 7. MEDIUM: Self-RAG Reflection Loop
- Generate "reflection tokens" that evaluate retrieval quality
- If confidence low, trigger re-retrieval with reformulated query
- Reduces hallucination significantly
- SmartOrchestrator already designed for this -- just needs wiring (#1)

### 8. MEDIUM: GRPO for RL (no value model needed)
- Group Relative Policy Optimization -- generate N solutions, run tests, compute group-relative advantages
- Test suite IS the reward signal -- no API calls, no human feedback
- Ideal for offline self-improvement on BEAST

### 9. MEDIUM: Error Memory / Anti-Pattern Log (from Darwin Godel Machine)
- Record failed approaches to prevent repetition
- Inject negative examples into future prompts
- Low implementation cost, high value
- Maps to experience replay with negative tagging

### 10. MEDIUM: README + Docs Overhaul
- README claims vLLM but actually uses Ollama
- HANDOVER.md says 35 tests (actual: 1730)
- Architecture.md marks components "NOT YET" that exist
- Missing: getting-started guide, CLI reference, module reference

---

## COMPETITIVE GAPS (What Top Agents Have That JCoder Doesn't)

| Feature | Competitors | JCoder Status |
|---------|-------------|---------------|
| Tree-sitter repo map | Aider | Has tree-sitter, needs repo-map layer |
| MCP (Model Context Protocol) | Cline, Roo Code | Not implemented |
| Plan/Act separation | Cline | Not formalized |
| Auto-commit per change | Aider | Not implemented |
| Git-first workflow (/undo, /diff) | Aider | Partial (git tools exist) |
| IDE integration | Continue, Roo Code | Desktop GUI only |
| Self-evolving scaffold | Live-SWE-agent | Evolution runner exists but offline |

## NEW MODEL CANDIDATES (NDAA/ITAR Safe)

| Model | Origin | License | Size | Why |
|-------|--------|---------|------|-----|
| Granite Code 8B | IBM (USA) | Apache 2.0 | 8B | Workstation alt, beats CodeLlama 16B |
| Granite Code 20B | IBM (USA) | Apache 2.0 | 20B | BEAST tier, evaluate vs phi4:14b |
| Codestral 22B | Mistral (France) | Permissive | 22B | Dedicated code model |
| StarCoder2-3B | BigCode (ServiceNow) | OpenRAIL-M | 3B | Toaster fallback |
| Codestral Embed | Mistral (France) | Permissive | -- | SOTA code embeddings |

## RAG OPTIMIZATION TECHNIQUES

| Technique | Impact | Effort |
|-----------|--------|--------|
| RRF fusion (k=60) over BM25+dense | 15-30% recall | Already have FTS5+embeddings |
| Two-stage reranking (retrieve 20, rerank to 3) | Major precision gain | ollama_reranker exists |
| Context compression (remove boilerplate) | 68% size reduction | Medium |
| Self-RAG reflection loop | Reduces hallucination | SmartOrchestrator needs wiring |
| PKG (Programming Knowledge Graph) | 20-34% accuracy gains | High effort |
| Multi-query expansion | Better recall | Low effort |

## SELF-LEARNING RESEARCH

| Technique | Paper | Applicability |
|-----------|-------|---------------|
| STaR + V-STaR | arXiv 2203.14465, 2402.06457 | Core self-training loop |
| GRPO | DeepSeek 2025 | RL without value model |
| P3ER experience replay | arXiv 2410.12236 | Already implemented (P2Value) |
| WebRL self-evolving curriculum | arXiv 2411.02337 | Task manufacturing from failures |
| Reflexion | arXiv 2303.11366 | Verbal self-critique, no weight updates |
| Live-SWE-agent | arXiv 2511.13646 | Self-modifying agent scaffold |
| Darwin Godel Machine | arXiv 2505.22954 | Evolutionary self-improvement |
| MiniPLM distillation | NeurIPS 2024 | Offline teacher->student |

## ARCHITECTURE REVIEW FINDINGS

### Already Fixed This Session
- [x] SQL injection in weekly_knowledge_update.py
- [x] bridge.py:647 silent error (added DEBUG logging)
- [x] ToolRegistry class size (657->428L)
- [x] SanitizationPipeline class size (620->535L)
- [x] Test coverage for orchestrator.py (13 tests) and runtime.py (18 tests)

### Still Open
- [ ] ModelCascade not wired in bridge.py
- [ ] SmartOrchestrator not wired in bridge.py
- [ ] N+1 Q-value updates in bridge.py:176-178
- [ ] Dead code in active_learner.py (_shannon_entropy, _normalize_answer)
- [ ] 74+ silent exception blocks across codebase
- [ ] 26 core modules without test coverage
- [ ] Config settings with no implementation (devils_advocate, claim_verification)
