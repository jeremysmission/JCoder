# Research-Driven Sprint Plan (2026-03-14)

## Priority Rationale

Based on 40+ paper analysis and codebase gap audit:
1. JCoder Sprints 18-20 fill the UNDEFINED gap with highest-ROI research upgrades
2. HybridRAG3 Sprints 15-16 address top retrieval quality gaps
3. All items are code-only (no BEAST dependency), testable on toaster
4. BEAST deployment (2026-03-15/17) validates everything at scale

---

## JCoder Sprint 18: Autoresearch PromptEvolver

**Module:** core/prompt_evolver.py
**Based on:** Karpathy Autoresearch (MIT, March 2026), DSPy MIPROv2

**Changes:**
- Replace hard tournament (top 50% survive) with keep/discard loop
- Add OperatorArm with Thompson sampling per mutation operator
- Track which operators (rephrase/extend/compress/crossover) produce winners
- Auto-deprioritize underperforming operators via Beta distribution
- SQLite persistence for operator effectiveness stats
- Monotonic champion improvement (champion only changes on genuine gain)

**Gate test:** Operator that consistently produces winning mutations gets selected more

---

## JCoder Sprint 19: P2Value Experience Replay

**Module:** core/experience_replay.py
**Based on:** P2Value (arxiv 2510.07429), RLEP (arxiv 2601.01931)

**Changes:**
- P2Value = alpha * confidence + (1-alpha) * pass_rate
- Near-miss detection: experiences that failed 1 test get priority boost
- Test results tracking (pass_count, fail_count per experience)
- Replay blending: mix new rollouts with replayed successes

**Gate test:** Near-miss experience retrieved before random success

---

## JCoder Sprint 20: Cost-Aware Routing

**Module:** core/meta_cognitive.py
**Based on:** BARP bandit-feedback routing (arxiv 2410.17952)

**Changes:**
- Add cost_per_call tracking per strategy
- cost_weight parameter (0.0-1.0) for quality-cost tradeoff
- Multi-objective Thompson sampling: sample - cost_weight * normalized_cost
- Model cost mapping (phi4-mini cheap, devstral expensive)

**Gate test:** Cheap strategy preferred for easy queries when cost_weight > 0

---

## HybridRAG3 Sprint 15: FTS5 Source Path + Corrective Retrieval

**Modules:** src/core/vector_store.py, src/core/query_engine.py
**Based on:** CRAG paper, structural metadata research

**Changes:**
- FTS5 source_path as second indexed column (requires re-index)
- Corrective retrieval loop: retrieve -> evaluate confidence -> reformulate -> retry
- Max 2 retrieval rounds (latency budget)

---

## HybridRAG3 Sprint 16: Ollama Reranker Revival

**Module:** src/core/retriever.py
**Based on:** Cross-encoder reranking research (+8-48% NDCG)

**Changes:**
- Replace dead sentence-transformers backend with Ollama LLM scoring
- Chunk relevance scoring on 0-1 scale
- Stays opt-in (reranker_enabled=false, NEVER enable for multi-type eval)

---

## Implementation Order

| # | Sprint | Project | Est. Tests |
|---|--------|---------|-----------|
| 1 | S18 | JCoder | ~33 |
| 2 | S19 | JCoder | ~25 |
| 3 | S20 | JCoder | ~25 |
| 4 | S15 | HybridRAG3 | TBD |
| 5 | S16 | HybridRAG3 | TBD |
