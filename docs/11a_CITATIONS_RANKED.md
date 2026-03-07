# Agentic Self-Learning Research: Ranked Citations & Links

Date: 2026-03-07
Companion to: 11_RESEARCH_AGENTIC_SELF_LEARNING.md

---

## Ranking Criteria

Papers ranked by: (1) Novelty of technique, (2) Implementability without fine-tuning,
(3) Relevance to coding agents, (4) Recency, (5) Venue prestige / community impact.

Scale: S-tier (field-defining), A-tier (high impact), B-tier (solid contribution),
C-tier (useful reference).

---

## S-Tier: Field-Defining (Must-Read)

### 1. GVU Framework: Self-Improving AI Agents through Self-Play
- **arXiv**: https://arxiv.org/abs/2512.02731
- **HTML**: https://arxiv.org/html/2512.02731v1
- **Date**: December 2025
- **Why S-tier**: Provides the first rigorous mathematical foundation for ALL self-
  improvement. Proves that verification quality matters more than generation quality.
  Unifies AlphaZero, RLHF, Constitutional AI, and code agents under one operator.
  The Variance Inequality is the E=mc^2 of self-improvement theory.

### 2. GEPA: Reflective Prompt Evolution (ICLR 2026 Oral)
- **arXiv**: https://arxiv.org/abs/2507.19457
- **PDF**: https://arxiv.org/pdf/2507.19457
- **Date**: July 2025 (accepted ICLR 2026)
- **Why S-tier**: ICLR oral = top ~1% of all submissions. Proves automated prompt
  evolution outperforms human experts. The reflective mutation mechanism is directly
  implementable with any LLM API.

### 3. Comprehensive Survey of Self-Evolving AI Agents (MASE + Three Laws)
- **arXiv**: https://arxiv.org/abs/2508.07407
- **GitHub**: https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents
- **Date**: August 2025 (updated through 2026)
- **Why S-tier**: The definitive taxonomy. Three Laws (Endure/Excel/Evolve) are the
  operating principles for any self-evolving system. GitHub repo is the single best
  curated resource in the field.

---

## A-Tier: High Impact (Strongly Recommended)

### 4. PRAXIS: Procedural Recall for Agents with Experiences Indexed by State
- **arXiv**: https://arxiv.org/abs/2511.22074
- **HTML**: https://arxiv.org/html/2511.22074
- **Date**: November 2025
- **Why A-tier**: First practical procedural memory system for LLM agents. No fine-tuning.
  Works with frozen models. Directly implementable. Reduces task completion steps by 20%.

### 5. RISE: Recursive Introspection for Self-Improvement
- **arXiv**: https://arxiv.org/abs/2407.18219
- **HTML**: https://arxiv.org/html/2407.18219v1
- **Project**: https://cohenqu.github.io/rise.github.io/
- **Date**: July 2024 (NeurIPS 2024, widely deployed 2025-2026)
- **Why A-tier**: Multi-turn self-correction with proven gains. Biggest improvements on
  weakest models (+17.7% on LLaMA2-7B). Trivially implementable with any chat API.

### 6. Intrinsic Metacognitive Learning (Position Paper)
- **OpenReview**: https://openreview.net/forum?id=4KhDd0Ozqe
- **PDF**: https://arxiv.org/pdf/2506.05109
- **Date**: June 2025 (ICLR 2026)
- **Why A-tier**: Identifies the fundamental gap in current self-improving agents.
  Three-component metacognition framework (Knowledge/Planning/Evaluation) is the
  blueprint for next-gen agent architectures.

### 7. DEEVO: Tournament of Prompts via Debate-Driven Evolution
- **arXiv**: https://arxiv.org/abs/2506.00178
- **HTML**: https://arxiv.org/html/2506.00178v2
- **Amazon Science**: https://www.amazon.science/publications/tournament-of-prompts-evolving-llm-instructions-through-structured-debates-and-elo-ratings
- **Date**: June 2025
- **Why A-tier**: No ground truth needed. Elo-based fitness tracking is elegant and proven.
  Outperforms manual prompt engineering on both open-ended and close-ended tasks.

### 8. SeRL: Self-Play Reinforcement Learning with Limited Data
- **arXiv**: https://arxiv.org/abs/2505.20347
- **Date**: May 2025
- **Why A-tier**: Solves the data bottleneck. Self-instruction + majority-vote self-reward
  enables bootstrapping from minimal seed data. Critical for domain-specific agents.

### 9. Remember Me, Refine Me: Dynamic Procedural Memory
- **arXiv HTML**: https://arxiv.org/html/2512.10696v1
- **Date**: December 2025
- **Why A-tier**: Complements PRAXIS with the "solidification" step -- turning frequent
  successes into reusable templates. Two-paradigm design (strategic guidance + procedural
  solidification) is the right architecture.

### 10. OpenAI Self-Evolving Agents Cookbook
- **Cookbook**: https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining
- **GitHub**: https://github.com/openai/openai-cookbook/blob/main/examples/partners/self_evolving_agents/autonomous_agent_retraining.ipynb
- **Date**: November 2025
- **Why A-tier**: Production-ready code. The VersionedPrompt + multi-grader + metaprompt
  pattern is immediately adaptable to any provider. Best "how to actually build this" resource.

---

## B-Tier: Solid Contributions (Worth Reading)

### 11. Self-Evolving Curriculum (SEC) for LLM Reasoning
- **arXiv**: https://arxiv.org/abs/2505.14970
- **Date**: May 2025
- **Why B-tier**: Elegant solution to curriculum design. Co-training the curriculum policy
  alongside the main model is novel. Requires RL infrastructure for full implementation.

### 12. RWML: Reinforcement World Model Learning
- **HuggingFace**: https://huggingface.co/papers/2602.05842
- **Date**: February 2026
- **Why B-tier**: World models for text-state agents is frontier research. Sim-to-real
  gap reward is clever. Early-stage but high potential.

### 13. Language Self-Play for Data-Free Training
- **arXiv PDF**: https://arxiv.org/pdf/2509.07414
- **Date**: September 2025
- **Why B-tier**: Challenger/Solver dual-mode self-play is an elegant formulation.
  Data-free training from self-play alone is a strong result.

### 14. A Survey of Self-Evolving Agents (Second Survey)
- **arXiv**: https://arxiv.org/abs/2507.21046
- **HTML**: https://arxiv.org/html/2507.21046v4
- **Date**: July 2025
- **Why B-tier**: Complementary to the MASE survey with different categorization.
  "What, When, How, Where to Evolve" framework is useful for planning.

### 15. Mem^p: Exploring Agent Procedural Memory
- **arXiv HTML**: https://arxiv.org/html/2508.06433v1
- **Date**: August 2025
- **Why B-tier**: Theoretical analysis of procedural memory design space.
  Good companion to PRAXIS for understanding the landscape.

---

## C-Tier: Useful References

### 16. ICLR 2026 Workshop: AI with Recursive Self-Improvement
- **Website**: https://recursive-workshop.github.io/
- **OpenReview**: https://openreview.net/forum?id=OsPQ6zTQXV
- **Why**: Workshop proceedings with accepted papers covering the full RSI landscape.

### 17. ICLR 2026 Workshop: Lifelong Agents
- **Website**: https://lifelongagent.github.io/
- **Why**: Complementary workshop on continuous learning and agent evolution.

### 18. ICLR 2026 Workshop: MemAgents
- **OpenReview**: https://openreview.net/pdf?id=U51WxL382H
- **Why**: Memory-focused workshop for LLM-based agentic systems.

### 19. Memory in the Age of AI Agents (Survey)
- **HuggingFace**: https://huggingface.co/papers/2512.13564
- **GitHub**: https://github.com/Shichun-Liu/Agent-Memory-Paper-List
- **Why**: Comprehensive memory survey covering all agent memory architectures.

### 20. RL for Large Reasoning Models (Survey)
- **GitHub**: https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs
- **Why**: Covers RL techniques (GRPO, DPO, ORPO) applied to reasoning models.

---

## Top 10 Monitoring Sources (Ranked by Signal Quality)

| Rank | Source | Link | Update Freq | Signal |
|------|--------|------|-------------|--------|
| 1 | arXiv cs.AI Recent | https://arxiv.org/list/cs.AI/recent | Daily | Raw papers, highest volume |
| 2 | Awesome Self-Evolving Agents | https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents | Weekly | Best curated taxonomy |
| 3 | HuggingFace Daily Papers | https://huggingface.co/papers | Daily | Community-filtered signal |
| 4 | OpenReview (ICLR/NeurIPS) | https://openreview.net | Per conference | Peer-reviewed, highest quality |
| 5 | Emergent Mind | https://www.emergentmind.com | Daily | Topic clustering, trend detection |
| 6 | Sebastian Raschka Newsletter | https://magazine.sebastianraschka.com | Weekly | Deep technical analysis |
| 7 | VoltAgent AI Agent Papers | https://github.com/VoltAgent/awesome-ai-agent-papers | Weekly | 2026-focused curation |
| 8 | Autonomous Agents Papers | https://github.com/tmgthb/Autonomous-Agents | Daily | Comprehensive daily updates |
| 9 | OpenAI Cookbook | https://developers.openai.com/cookbook/ | Monthly | Production-ready patterns |
| 10 | LessWrong AI | https://www.lesswrong.com | Daily | RSI safety discussion, novel ideas |

---

## Quick-Reference: What to Read Based on Your Goal

| If you want to... | Read these (in order) |
|--------------------|-----------------------|
| Understand the theory | #1 (GVU), #6 (Metacognition), #3 (MASE Survey) |
| Build procedural memory | #4 (PRAXIS), #9 (Remember Me), #15 (Mem^p) |
| Optimize prompts automatically | #2 (GEPA), #7 (DEEVO), #10 (OpenAI Cookbook) |
| Build self-play for code | #8 (SeRL), #13 (Language Self-Play), #1 (GVU) |
| Design a complete self-evolving agent | #3 (Survey), #10 (Cookbook), #5 (RISE), #4 (PRAXIS) |
| Stay current on the field | Monitor sources #1-#5 weekly |

---

## Additional Sources (from Subagent Deep Dives, 2026-03-07)

### Self-Play for Code
- SSR: Self-Play SWE-RL -- https://arxiv.org/abs/2512.18552 (Meta FAIR, Dec 2025)
- PSV: Propose, Solve, Verify -- https://arxiv.org/abs/2512.18160 (Dec 2025)
- PSV GitHub: https://github.com/abwilf/psv
- AlphaVerus baseline: https://arxiv.org/abs/2412.06176

### Prompt Evolution
- GEPA GitHub: https://github.com/gepa-ai/gepa
- GEPA DSPy Tutorial: https://dspy.ai/tutorials/gepa_ai_program/
- GEPA Official Site: https://gepa-ai.github.io/gepa/
- GEPA ICLR Oral: https://iclr.cc/virtual/2026/oral/10009494

### Curriculum Learning
- Actor-Curator: https://arxiv.org/abs/2602.20532 (Feb 2026)
- VCRL: https://arxiv.org/abs/2509.19803
- AdaRFT: https://arxiv.org/abs/2504.05520 (Apr 2025)
- TACLer: https://arxiv.org/abs/2601.21711 (Jan 2026)

### Memory Systems
- ReasoningBank: https://arxiv.org/abs/2509.25140 (Google, Sep 2025, ICLR 2026 poster)
- ReMe GitHub: https://github.com/agentscope-ai/ReMe
- A-MEM (Zettelkasten): https://arxiv.org/abs/2502.12110
- EvolveR: https://arxiv.org/abs/2510.16079

### World Models
- RWML: https://arxiv.org/abs/2602.05842 (Microsoft, Feb 2026)
- DynaWeb: https://arxiv.org/abs/2601.22149 (Jan 2026)
- AWM (Snowflake): https://arxiv.org/abs/2602.10090 | https://github.com/Snowflake-Labs/agent-world-model

### Lifelong Learning
- ELL/StuLife: https://arxiv.org/abs/2508.19005
- Lifelong LLM Roadmap: https://arxiv.org/abs/2501.07278
- Lifelong Agents Workshop: https://lifelongagent.github.io/

### Claude CLI Ecosystem
- memsearch: https://milvus.io/blog/adding-persistent-memory-to-claude-code-with-the-lightweight-memsearch-plugin.md
- metaswarm: https://github.com/dsifry/metaswarm
- Ruflo v3: https://github.com/ruvnet/ruflo
- recursive-decomposition-skill: https://github.com/massimodeluisa/recursive-decomposition-skill
- awesome-claude-code: https://github.com/hesreallyhim/awesome-claude-code
