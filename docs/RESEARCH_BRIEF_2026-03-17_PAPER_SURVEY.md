# Academic Paper Survey — 2025-2026

## Implemented This Session

| Paper | Technique | JCoder Module | Status |
|-------|-----------|---------------|--------|
| **HybridServe** (arXiv:2505.12566) | Confidence-based skip connections in cascade | `core/cascade.py` | DONE — skip_threshold param, 3 tests |

## High-Priority Next Implementations

| Paper | Technique | JCoder Module | Difficulty |
|-------|-----------|---------------|------------|
| RAP-RAG | Adaptive retrieval planning by query complexity | `layered_triage.py` | Easy |
| Self-Improving at Test-Time (arXiv:2510.07841) | Reflection loop at inference | `star_reasoner.py` | Easy |
| MOPrompt (arXiv:2508.01541) | Multi-objective Pareto front | `prompt_evolver.py` | Easy |
| GEPA (ICLR 2026 Oral, arXiv:2507.19457) | Reflective prompt evolution | `prompt_evolver.py` | Medium |
| Speculative RAG (arXiv:2407.08223) | Parallel draft generation | `speculative_coder.py` | Medium |
| HippoRAG2 | Dual-granularity KG nodes + PPR | `knowledge_graph.py` | Medium |
| PathRAG (arXiv:2502.14902) | Flow-based KG path pruning | `knowledge_graph.py` | Medium |
| RLSR (arXiv:2505.08827) | Self-generated practice problems | `experience_replay.py` | Medium |
| Cascade Routing Unified (ICLR 2025) | Joint routing + cascading policy | `cascade.py` | Medium |
| FOREVER (arXiv:2601.03938) | Forgetting curve memory replay | `experience_replay.py` | Medium |

## Validation Papers (Confirm Architecture)

| Paper | Finding |
|-------|---------|
| RACG Survey (arXiv:2510.04905) | Hybrid retrieval (BM25 + embeddings + structural) outperforms single-mode — validates JCoder's FAISS+FTS5+KG approach |
| CodeCoR (arXiv:2501.07811) | 4-agent reflection pattern for code gen — validates JCoder's multi_agent.py design |
| A-RAG (arXiv:2602.03442) | Agentic retrieval interfaces — validates exposing retrieval as agent tools |

## Additional Findings (Second Research Pass)

| Paper | Venue | Key Technique | JCoder Module | Difficulty |
|-------|-------|---------------|---------------|------------|
| GEPA (arXiv:2507.19457) | ICLR 2026 Oral | Pareto-frontier prompt evolution + reflection | `prompt_evolver.py` | Easy — [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa) |
| Prompt Duel Optimizer (arXiv:2510.13907) | arXiv | Double Thompson Sampling, label-free | `prompt_evolver.py` | Easy |
| A-RAG (arXiv:2602.03442) | arXiv | Agentic retrieval — FTS5/FAISS as agent tools | `retrieval_engine.py` | Medium — [github.com/Ayanami0730/arag](https://github.com/Ayanami0730/arag) |
| MemRL (arXiv:2601.03192) | arXiv | Q-value episodic memory, utility-based retrieval | `experience_replay.py` | Medium |
| Practical GraphRAG (arXiv:2507.03226) | arXiv | RRF + graph traversal as 3rd signal | `knowledge_graph.py` | Medium |
| GFM-RAG (arXiv:2502.01113) | NeurIPS 2025 | 8M-param graph foundation model for KG retrieval | `knowledge_graph.py` | Hard — [github.com/RManLuo/gfm-rag](https://github.com/RManLuo/gfm-rag) |
| LightRAG (arXiv:2410.05779) | EMNLP 2025 | Dual-level entity+topic retrieval, <100 tokens | `retrieval_engine.py` | Easy |
| CodeTree (NAACL 2025) | NAACL | Thinker/Solver/Debugger/Critic tree search | `code_tree.py` | Medium — [github.com/SalesforceAIResearch/CodeTree](https://github.com/SalesforceAIResearch/CodeTree) |
| DAPO (arXiv:2503.14476) | arXiv | Dynamic Sampling + Clip-Higher for RL | `active_learner.py` | Hard |
| SPIN (arXiv:2401.01335) | ICML 2024 | Self-play fine-tuning via self-generated data | `adversarial_self_play.py` | Medium — [github.com/uclaml/SPIN](https://github.com/uclaml/SPIN) |
| UniRoute (arXiv:2502.08773) | arXiv | Learned LLM capability vectors for routing | `cascade.py` | Medium |
| Promptomatix (arXiv:2507.14241) | arXiv | Cost-aware prompt optimization | `prompt_evolver.py` | Easy — [github.com/SalesforceAIResearch/promptomatix](https://github.com/SalesforceAIResearch/promptomatix) |
| CoCA (arXiv:2603.05881) | arXiv | Confidence-before-answering via GRPO | `meta_cognitive.py` | Hard |
| Quiet-STaR (arXiv:2403.09629) | arXiv | Token-level internal reasoning | `star_reasoner.py` | Hard |
| ADAS (arXiv:2408.08435) | ICLR 2025 | Meta-agent evolves agent architectures | `multi_agent.py` | Hard — [github.com/ShengranHu/ADAS](https://github.com/ShengranHu/ADAS) |

## Full Paper List (20 papers)

1. A-RAG (arXiv:2602.03442) — Agentic RAG via hierarchical retrieval interfaces
2. RAGLens (arXiv:2512.08892) — Faithfulness detection via sparse autoencoders
3. Speculative RAG (arXiv:2407.08223) — Parallel draft + verify pattern
4. RAP-RAG — Adaptive retrieval task planning
5. HippoRAG2 — Dual-granularity KG with Personalized PageRank
6. PathRAG (arXiv:2502.14902) — Flow-based KG path pruning
7. HopRAG (arXiv:2502.12442) — Multi-hop passage graph retrieval
8. Darwin Godel Machine (arXiv:2505.22954) — Self-modifying agent
9. Self-Improving Coding Agent (arXiv:2504.15228) — Agent edits itself
10. RLSR (arXiv:2505.08827) — RL from self-reward
11. Self-Improving at Test-Time (arXiv:2510.07841) — Test-time reflection
12. GEPA (arXiv:2507.19457) — Reflective prompt evolution (ICLR 2026 Oral)
13. SCOPE (arXiv:2512.15374) — Dual-stream context optimization
14. MOPrompt (arXiv:2508.01541) — Multi-objective prompt optimization
15. Cascade Routing Unified (ICLR 2025) — Joint routing + cascading
16. HybridServe (arXiv:2505.12566) — Confidence skip connections
17. FOREVER (arXiv:2601.03938) — Forgetting curve memory replay
18. MSSR (arXiv:2603.09892) — Memory-aware adaptive replay
19. RACG Survey (arXiv:2510.04905) — Retrieval-augmented code generation
20. CodeCoR (arXiv:2501.07811) — Self-reflective multi-agent code gen
