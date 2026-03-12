"""
GPT-5.4 Knowledge Distillation Tool
------------------------------------
Sends research documents directly to GPT-5.4 via OpenAI API (cloud GPU)
and saves the synthesized knowledge to agent_knowledge.

No local GPU, no FAISS, no Ollama -- pure cloud inference.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

JCODER_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = JCODER_ROOT / "data" / "agent_knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


def call_gpt5(system_prompt: str, user_prompt: str, model: str = "gpt-5.4") -> dict:
    """Call GPT-5.4 via OpenAI API. Returns {content, input_tokens, output_tokens}."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY not set")
        sys.exit(1)

    # Newer models require max_completion_tokens
    token_param = "max_completion_tokens" if any(
        model.startswith(p) for p in ("gpt-5", "gpt-4.1", "o1", "o3", "o4")
    ) else "max_tokens"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        token_param: 8192,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-type": "application/json",
    }

    print(f"[OK] Sending to {model} (cloud GPU)...")
    t0 = time.monotonic()

    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    elapsed = time.monotonic() - t0
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})

    print(f"[OK] Response in {elapsed:.1f}s "
          f"({usage.get('prompt_tokens', 0)} in, "
          f"{usage.get('completion_tokens', 0)} out)")

    return {
        "content": msg.get("content", ""),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "model": data.get("model", model),
        "elapsed_s": elapsed,
    }


def save_knowledge(title: str, content: str, metadata: dict) -> Path:
    """Save synthesized knowledge to agent_knowledge directory."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    slug = title.lower().replace(" ", "_")[:50]
    filename = f"{ts}_{slug}.md"
    path = KNOWLEDGE_DIR / filename

    header = f"""# {title}

Generated: {datetime.now(timezone.utc).isoformat()}
Model: {metadata.get('model', 'unknown')}
Input tokens: {metadata.get('input_tokens', 0)}
Output tokens: {metadata.get('output_tokens', 0)}

---

"""
    path.write_text(header + content, encoding="utf-8")
    print(f"[OK] Saved: {path}")
    return path


def load_docs() -> dict:
    """Load all self-learning research docs."""
    docs_dir = JCODER_ROOT / "docs"
    doc_files = {
        "survey": docs_dir / "11_RESEARCH_AGENTIC_SELF_LEARNING.md",
        "citations": docs_dir / "11a_CITATIONS_RANKED.md",
        "deep_dives": docs_dir / "11b_DEEP_DIVE_SUBAGENT_FINDINGS.md",
        "self_evolution": docs_dir / "09_RESEARCH_SELF_EVOLUTION.md",
    }
    result = {}
    for key, path in doc_files.items():
        if path.exists():
            result[key] = path.read_text(encoding="utf-8")
            print(f"[OK] Loaded {key}: {len(result[key]):,} chars")
        else:
            print(f"[WARN] Missing: {path}")
    return result


def main():
    print("=" * 60)
    print("GPT-5.4 Self-Learning Knowledge Distillation")
    print("=" * 60)

    docs = load_docs()
    if not docs:
        print("[FAIL] No documents found")
        sys.exit(1)

    # Combine all docs into one big context
    combined = "\n\n---\n\n".join(
        f"## Document: {key}\n\n{text}" for key, text in docs.items()
    )
    print(f"\n[OK] Total context: {len(combined):,} chars")

    system_prompt = """You are a world-class AI research scientist specializing in self-learning systems,
meta-cognition, and autonomous agent improvement. You have deep expertise in:
- Reinforcement learning from self-play
- Procedural memory and experience replay
- Evolutionary prompt optimization
- Metacognitive strategy selection
- Catastrophic forgetting prevention

Your task is to synthesize research documents into actionable knowledge. Be specific,
concrete, and practical. Include implementation details, not just high-level concepts.
Prioritize techniques that work WITHOUT fine-tuning (inference-time only) since the
target system uses frozen API models."""

    user_prompt = f"""I'm building JCoder, an offline coding assistant with a self-learning pipeline.
The system has:
- FTS5 keyword search + FAISS vector search with RRF fusion
- Experience replay (SQLite-backed)
- Procedural memory (PRAXIS-style state-action-result triples)
- Meta-cognitive controller (Thompson sampling for strategy selection)
- Prompt evolver (genetic algorithms)
- Adversarial self-play (Sol-Ver pattern)
- Quality-diversity archive (MAP-Elites)
- Continual learner (anti-forgetting baselines)
- Active learner (uncertainty sampling + committee disagreement)

Here are ALL the self-learning research papers and findings I've collected:

{combined}

---

Based on ALL of this research, please provide:

1. **RANKED LIST**: What are the TOP 10 most effective self-learning techniques for a coding agent,
   ranked by expected impact? For each, explain WHY it ranks where it does and give a concrete
   implementation sketch.

2. **CRITICAL PATH**: What is the single most important learning loop to get right FIRST?
   Describe the exact data flow, step by step.

3. **SYNERGIES**: Which techniques amplify each other when combined? What are the best 2-3
   technique combinations and why?

4. **ANTI-PATTERNS**: What should we NEVER do? What are the most common mistakes in
   self-learning systems and how to avoid them?

5. **VERIFICATION IS KING**: The GVU theorem says verification quality matters more than
   generation quality. What is the optimal verification strategy for a coding agent?

6. **MEMORY ARCHITECTURE**: What is the ideal memory system? How should procedural memory,
   episodic memory, and semantic memory interact?

7. **THE 3-ITERATION CLIFF**: Research shows diminishing returns after iteration 3.
   How do we beat this ceiling? What strategies extend the improvement curve?

8. **PRACTICAL SCHEDULE**: Given limited compute (16GB RAM, CPU-only for local, plus
   cloud API access to GPT-5.4), what is the optimal daily/weekly learning schedule?

Be brutally honest about what works and what doesn't. No hype, no hand-waving."""

    result = call_gpt5(system_prompt, user_prompt)

    # Save the synthesis
    save_knowledge(
        "GPT-5.4 Self-Learning Strategy Synthesis",
        result["content"],
        result,
    )

    # Print the full response
    print("\n" + "=" * 60)
    print("GPT-5.4 SYNTHESIS")
    print("=" * 60)
    print(result["content"])

    return result


if __name__ == "__main__":
    main()
