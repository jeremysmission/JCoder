"""
JCoder Systematic Knowledge Distillation
------------------------------------------
Processes ALL indexed data in a logical curriculum order.
Each phase builds on the previous one. GPT-5.4 acts as teacher,
distilling raw material into structured knowledge notes.

Curriculum Order (dependencies flow downward):
  Phase 1: Language Foundations (Python docs, stdlib)
  Phase 2: Code Patterns & Best Practices (magicoder, code_feedback, best_practices)
  Phase 3: Algorithms & Problem Solving (code_exercises, code_contests, math_instruct)
  Phase 4: Real-World Code (codeparrot, commitpack, codesearchnet)
  Phase 5: Community Knowledge (Stack Overflow, Stack Exchange, code_review)
  Phase 6: Instruction Tuning (code_290k, glaive, capybara, openhermes)
  Phase 7: Systems & Infrastructure (RFC docs, security, networking)
  Phase 8: Secondary Languages (Rust, Java, Go, JS, PHP, Ruby)
  Phase 9: Research & Theory (ML papers, arxiv, self-learning -- already done)
  Phase 10: Synthesis (cross-domain connections, master patterns)
"""

import json
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

JCODER_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = JCODER_ROOT / "data" / "agent_knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

# Track what we've already learned
LEARNED_CONTEXT_FILE = KNOWLEDGE_DIR / "_curriculum_progress.json"

random.seed(42)

# Skip databases larger than this on memory-constrained machines (toaster: 16 GB RAM).
# 6.5 GB FTS5 databases cause OFFSET scans that thrash disk for hours.
# Set to 0 to disable the guard (e.g. on BEAST with 128 GB RAM).
MAX_DB_SIZE_MB = int(os.environ.get("JCODER_MAX_DB_MB", "500"))


def call_gpt5(system_p: str, user_p: str, max_tok: int = 4096) -> dict:
    """Call GPT-5.4 via OpenAI cloud API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY not set")
        sys.exit(1)

    payload = {
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": system_p},
            {"role": "user", "content": user_p},
        ],
        "temperature": 0.2,
        "max_completion_tokens": max_tok,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-type": "application/json",
    }
    t0 = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    elapsed = time.monotonic() - t0
    usage = data.get("usage", {})
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    # GPT-5+ can return content=null on max_completion_tokens truncation
    # while still reporting tokens in usage. Fall back to refusal field.
    if not content and msg.get("refusal"):
        content = f"[REFUSAL] {msg['refusal']}"
    if not content:
        finish = data["choices"][0].get("finish_reason", "unknown")
        print(f"  [WARN] Empty content from API (finish_reason={finish}, "
              f"tokens={usage.get('completion_tokens', 0)})")
    return {
        "content": content,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "elapsed_s": elapsed,
    }


def pull_samples(db_path: str, n: int = 20, min_len: int = 150, max_len: int = 3000) -> list:
    """Pull diverse quality samples from an FTS5 index."""
    try:
        p = Path(db_path)
        if not p.exists():
            return []
        size_mb = p.stat().st_size / (1024 * 1024)
        if MAX_DB_SIZE_MB and size_mb > MAX_DB_SIZE_MB:
            print(f"  [SKIP] {p.name}: {size_mb:.0f} MB exceeds {MAX_DB_SIZE_MB} MB limit")
            return []
        conn = sqlite3.connect(db_path)
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total == 0:
            conn.close()
            return []

        # Sample from evenly spaced positions for diversity
        step = max(1, total // (n * 3))
        offsets = list(range(0, total, step))
        random.shuffle(offsets)

        samples = []
        for off in offsets:
            if len(samples) >= n:
                break
            try:
                row = conn.execute(
                    "SELECT search_content FROM chunks LIMIT 1 OFFSET ?", (off,)
                ).fetchone()
                if row and min_len < len(row[0]) < max_len:
                    samples.append(row[0])
            except Exception as exc:
                print(f"  [WARN] Sample query failed: {exc}"); continue

        conn.close()
        return samples
    except Exception as e:
        print(f"  [WARN] get_random_samples failed: {e}"); return []


def get_index_count(db_path: str) -> int:
    """Get chunk count for an index."""
    try:
        p = Path(db_path)
        if not p.exists():
            return 0
        size_mb = p.stat().st_size / (1024 * 1024)
        if MAX_DB_SIZE_MB and size_mb > MAX_DB_SIZE_MB:
            return 0
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        return count
    except Exception as exc:
        print(f"  [WARN] get_index_count failed: {exc}"); return 0


def save_knowledge(phase: str, topic: str, content: str, metadata: dict) -> Path:
    """Save distilled knowledge to agent_knowledge."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    slug = topic.lower().replace(" ", "_").replace("/", "_")[:40]
    filename = f"{ts}_phase{phase}_{slug}.md"
    path = KNOWLEDGE_DIR / filename

    header = (
        f"# Phase {phase}: {topic}\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
        f"Model: gpt-5.4\n"
        f"Input tokens: {metadata.get('input_tokens', 0)}\n"
        f"Output tokens: {metadata.get('output_tokens', 0)}\n"
        f"Elapsed: {metadata.get('elapsed_s', 0):.1f}s\n\n---\n\n"
    )
    path.write_text(header + content, encoding="utf-8")
    return path


def load_progress() -> dict:
    """Load curriculum progress."""
    if LEARNED_CONTEXT_FILE.exists():
        return json.loads(LEARNED_CONTEXT_FILE.read_text(encoding="utf-8"))
    return {"completed": [], "summaries": {}}


def save_progress(progress: dict):
    """Save curriculum progress."""
    LEARNED_CONTEXT_FILE.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build_prior_context(progress: dict, max_chars: int = 3000) -> str:
    """Build a compact summary of what's been learned so far."""
    if not progress["summaries"]:
        return ""
    parts = []
    total = 0
    for phase_topic, summary in progress["summaries"].items():
        chunk = f"[{phase_topic}]: {summary[:300]}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


# =====================================================================
# CURRICULUM DEFINITION
# =====================================================================

# Each entry: (phase, topic, indexes_to_query, samples_per_index, digest_prompt_focus)
# Indexes are tried in order; first found is used. Supports both data locations.

D1 = str(JCODER_ROOT / "data" / "indexes")
D2 = "D:/JCoder_Data/indexes"

CURRICULUM = [
    # Phase 1: Language Foundations
    ("1", "Python Language & Stdlib", [
        (f"{D1}/python_docs.fts5.db", 25),
        (f"{D2}/python_docs.fts5.db", 25),
    ], "Extract Python language patterns, stdlib best practices, and common idioms. "
       "Focus on: data structures, error handling, file I/O, string processing, "
       "itertools/functools patterns, context managers, decorators, generators."),

    # Phase 2: Code Patterns
    ("2a", "Code Best Practices & Patterns", [
        (f"{D1}/magicoder_oss_instruct.fts5.db", 20),
        (f"{D1}/code_feedback.fts5.db", 15),
        (f"{D2}/best_practices.fts5.db", 15),
    ], "Extract coding patterns, design principles, and best practices. "
       "Focus on: clean code principles, SOLID patterns, error handling conventions, "
       "function design, class design, testing patterns."),

    ("2b", "Code Review & Quality", [
        (f"{D2}/code_feedback.fts5.db", 20),
        (f"{D2}/se_codereview.fts5.db", 15),
    ], "Extract code review insights: common mistakes, improvement suggestions, "
       "quality patterns. What makes code good vs bad? What do reviewers catch?"),

    # Phase 3: Algorithms
    ("3a", "Algorithms & Data Structures", [
        (f"{D1}/code_exercises.fts5.db", 15),
        (f"{D1}/code_contests.fts5.db", 15),
        (f"{D2}/code_exercises.fts5.db", 10),
    ], "Extract algorithm patterns: sorting, searching, graph traversal, dynamic "
       "programming, greedy algorithms, divide and conquer. Focus on problem-solving "
       "strategies, time/space complexity patterns, and common algorithm templates."),

    ("3b", "Mathematical Programming", [
        (f"{D1}/math_instruct.fts5.db", 15),
        (f"{D2}/math_instruct.fts5.db", 15),
    ], "Extract mathematical programming patterns: numerical methods, combinatorics, "
       "probability, linear algebra operations, optimization. How to translate math "
       "problems into code correctly."),

    # Phase 4: Real-World Code
    ("4a", "Production Python Code", [
        (f"{D1}/codeparrot_clean.fts5.db", 20),
        (f"{D2}/codeparrot_python.fts5.db", 15),
        (f"{D2}/csn_python.fts5.db", 10),
    ], "Extract real-world Python patterns from production codebases: project structure, "
       "module organization, API design, configuration handling, logging, CLI patterns, "
       "database interaction patterns, HTTP client patterns."),

    ("4b", "Git & Commit Patterns", [
        (f"{D2}/commitpack.fts5.db", 20),
    ], "Extract software evolution patterns from commit data: what kinds of changes "
       "are most common? How do bugfixes differ from features? What patterns appear in "
       "commit messages and diffs? What are common refactoring patterns?"),

    # Phase 5: Community Knowledge
    ("5a", "Stack Overflow Python", [
        (f"{D1}/stackoverflow.fts5.db", 25),
        (f"{D2}/se_cs.fts5.db", 10),
        (f"{D2}/se_datascience.fts5.db", 10),
        (f"{D2}/se_ai.fts5.db", 10),
    ], "Extract the most valuable Stack Overflow patterns for Python development: "
       "common pitfalls, debugging strategies, library usage patterns, performance tips, "
       "concurrency patterns, packaging/deployment. Focus on answers that solved real problems."),

    ("5b", "Systems & DevOps Knowledge", [
        (f"{D2}/se_unix.fts5.db", 10),
        (f"{D2}/se_serverfault.fts5.db", 10),
        (f"{D2}/se_superuser.fts5.db", 10),
    ], "Extract systems administration and DevOps patterns: shell scripting, process "
       "management, networking, file systems, permissions, deployment, monitoring, "
       "troubleshooting. Focus on patterns useful for a coding assistant."),

    ("5c", "Security Patterns", [
        (f"{D2}/se_security.fts5.db", 15),
        (f"{D2}/se_crypto.fts5.db", 10),
    ], "Extract security best practices for software development: input validation, "
       "authentication patterns, encryption usage, common vulnerabilities and how to "
       "prevent them, secure coding patterns."),

    # Phase 6: Instruction Tuning Knowledge
    ("6a", "Code Instruction Patterns", [
        (f"{D2}/code_290k_sharegpt.fts5.db", 15),
        (f"{D2}/glaive_code_v3.fts5.db", 15),
        (f"{D2}/evol_codealpaca.fts5.db", 10),
    ], "Extract instruction-following patterns for code generation: how to interpret "
       "user requests, how to structure solutions, common request patterns, how to "
       "handle ambiguous requirements, how to provide working examples."),

    ("6b", "Chain-of-Thought Coding", [
        (f"{D1}/cot_code_instruct.fts5.db", 15),
        (f"{D2}/evol_instruct_code.fts5.db", 15),
    ], "Extract reasoning patterns for code: how to think through problems step by step, "
       "how to decompose complex tasks, how to verify solutions, how to explain code. "
       "Focus on the reasoning process, not just the final answer."),

    ("6c", "General Instruction Following", [
        (f"{D2}/capybara.fts5.db", 15),
        (f"{D1}/openhermes_2_5.fts5.db", 15),
    ], "Extract general instruction-following patterns: how to handle multi-step requests, "
       "how to clarify ambiguity, how to structure long responses, how to balance "
       "thoroughness with conciseness."),

    # Phase 7: Systems & Infrastructure
    ("7", "RFC & Protocol Standards", [
        (f"{D1}/rfc_docs.fts5.db", 15),
        (f"{D2}/rfc.fts5.db", 15),
    ], "Extract networking and protocol patterns: HTTP, TCP/IP, DNS, TLS, WebSocket, "
       "REST API conventions, status codes, header patterns. Focus on patterns a "
       "coding assistant needs when implementing network code."),

    # Phase 8: Secondary Languages
    ("8a", "Rust Patterns", [
        (f"{D2}/strandset_rust.fts5.db", 20),
    ], "Extract Rust-specific patterns: ownership, borrowing, lifetimes, error handling "
       "with Result/Option, trait patterns, async Rust, common crate patterns. "
       "Focus on what a polyglot developer needs to write correct Rust."),

    ("8b", "Java & Go Patterns", [
        (f"{D2}/csn_java.fts5.db", 12),
        (f"{D2}/stack_java.fts5.db", 12),
        (f"{D2}/csn_go.fts5.db", 12),
        (f"{D2}/stack_go.fts5.db", 10),
    ], "Extract Java and Go patterns: Java class hierarchies, Spring patterns, "
       "Go error handling, goroutines, channels, interface patterns. Focus on "
       "idiomatic usage in each language."),

    ("8c", "JavaScript & TypeScript Patterns", [
        (f"{D2}/csn_javascript.fts5.db", 15),
        (f"{D2}/stack_javascript.fts5.db", 15),
    ], "Extract JavaScript/TypeScript patterns: async/await, promises, event handling, "
       "module patterns, type annotations, React patterns if present, Node.js patterns. "
       "Focus on modern JS idioms."),

    # Phase 9: AI/ML Research (supplement what we already have)
    ("9", "ML & AI Research Patterns", [
        (f"{D2}/ml_arxiv_papers.fts5.db", 15),
        (f"{D2}/arxiv_agentic_ai.fts5.db", 10),
    ], "Extract machine learning implementation patterns: training loops, data loading, "
       "model architecture patterns, evaluation methodology, experiment tracking, "
       "common ML pitfalls. Supplement the self-learning research already digested."),

    # Phase 10: Cross-Domain Synthesis
    ("10", "Master Synthesis", [
        # No new data -- synthesize from all prior phases
    ], "SYNTHESIS PHASE -- no new data to ingest. Instead, review everything learned "
       "in phases 1-9 and produce a master document: the top 25 cross-cutting patterns "
       "that apply across all domains. What are the universal principles of good code? "
       "What patterns transfer between Python, Rust, Java, and JS? What makes an "
       "expert developer different from a junior?"),
]


def process_phase(phase: str, topic: str, indexes: list, focus: str,
                  progress: dict) -> bool:
    """Process one curriculum phase."""
    phase_key = f"Phase {phase}: {topic}"

    # Skip if already completed
    if phase_key in progress["completed"]:
        print(f"  [SKIP] Already completed")
        return True

    # Gather samples from all specified indexes
    all_samples = []
    for db_path, n_samples in indexes:
        samples = pull_samples(db_path, n=n_samples)
        if samples:
            count = get_index_count(db_path)
            db_name = Path(db_path).name
            print(f"  [OK] {db_name}: {len(samples)} samples (from {count:,} chunks)")
            all_samples.extend(samples)
        # Don't warn on missing -- we list multiple fallbacks

    if not all_samples and phase != "10":
        print(f"  [WARN] No samples found for any index -- skipping")
        return False

    # Build context from prior learning
    prior = build_prior_context(progress)
    prior_block = ""
    if prior:
        prior_block = (
            f"\n\nYou have already learned the following from prior phases:\n"
            f"{prior}\n\n"
            f"Build on this knowledge. Do not repeat what you already know. "
            f"Focus on NEW patterns and insights from the current material.\n"
        )

    # Build the digest prompt
    if all_samples:
        code_context = "\n\n---NEXT SAMPLE---\n\n".join(all_samples)
        # Cap at 22K chars to leave room for system prompt
        code_context = code_context[:22000]
        user_prompt = (
            f"Phase {phase}: {topic}\n\n"
            f"Study these {len(all_samples)} samples from the indexed archives:\n\n"
            f"{code_context}\n\n"
            f"---\n\n"
            f"FOCUS: {focus}\n\n"
            f"Produce a structured knowledge document with:\n"
            f"1. Top 10-15 specific patterns (with concrete examples from the samples)\n"
            f"2. Common mistakes/anti-patterns you observed\n"
            f"3. Key principles that transfer to other contexts\n"
            f"4. Anything surprising or non-obvious in the data\n\n"
            f"Be concrete and specific. Cite actual patterns from the samples."
        )
    else:
        # Phase 10: synthesis from prior context only
        user_prompt = (
            f"Phase {phase}: MASTER SYNTHESIS\n\n"
            f"You have now studied material across all domains:\n"
            f"{prior}\n\n"
            f"FOCUS: {focus}\n\n"
            f"Produce the definitive master document."
        )

    system_prompt = (
        "You are JCoder's knowledge distillation engine. You are building a permanent "
        "knowledge base that will be retrieved via FTS5 keyword search during future "
        "coding tasks. Every pattern you extract must be:\n"
        "- Specific enough to match on relevant search queries\n"
        "- Actionable (a developer can use it immediately)\n"
        "- Correct (no hallucinated patterns)\n"
        "- Concise (FTS5 retrieval works best with focused chunks)\n\n"
        "Structure your output with clear headings and bullet points. "
        "Use code examples where they add clarity."
        f"{prior_block}"
    )

    print(f"  Sending to GPT-5.4 ({len(all_samples)} samples, ~{sum(len(s) for s in all_samples):,} chars)...")
    result = call_gpt5(system_prompt, user_prompt, 8192)

    print(
        f"  [{result['elapsed_s']:.1f}s, {result['input_tokens']} in, "
        f"{result['output_tokens']} out]"
    )

    # Guard: do not save or mark complete if content is empty
    content = result["content"].strip()
    if not content:
        print(f"  [WARN] GPT-5.4 returned empty content -- NOT marking phase complete")
        return False

    # Save the knowledge file
    path = save_knowledge(phase, topic, content, result)
    print(f"  [OK] Saved: {path.name}")

    # Update progress
    # Store a 1-line summary for building context in later phases
    summary = content[:400].replace("\n", " ").strip()
    progress["completed"].append(phase_key)
    progress["summaries"][phase_key] = summary
    save_progress(progress)

    return True


def run_quiz(checkpoint_name: str, topics_covered: list, progress: dict) -> dict:
    """
    Run a verification quiz after a group of phases.
    Tests whether JCoder actually retained and can apply what it learned.
    Returns {score, total, passed, details}.
    """
    quiz_key = f"quiz_{checkpoint_name}"
    if quiz_key in progress.get("quizzes", {}):
        prior = progress["quizzes"][quiz_key]
        print(f"  [SKIP] Quiz already passed ({prior['score']}/{prior['total']})")
        return prior

    prior_context = build_prior_context(progress, max_chars=4000)

    system_prompt = (
        "You are a strict programming instructor creating a quiz to verify "
        "that a student has actually internalized specific technical knowledge. "
        "Create questions that require APPLYING knowledge, not just recalling facts. "
        "Each question should have one clear correct answer."
    )

    quiz_gen_prompt = (
        f"Create a 5-question quiz for checkpoint: {checkpoint_name}\n\n"
        f"Topics covered:\n"
        + "\n".join(f"- {t}" for t in topics_covered)
        + f"\n\nKnowledge summary:\n{prior_context[:3000]}\n\n"
        "For each question, provide:\n"
        "1. The question (specific, requires applied knowledge)\n"
        "2. The correct answer (concise, definitive)\n"
        "3. A common wrong answer that someone who didn't learn would give\n\n"
        "Format as JSON array:\n"
        '[{"q": "...", "correct": "...", "wrong": "..."}]\n\n'
        "Mix question types: code output prediction, best practice selection, "
        "bug identification, pattern recognition, edge case analysis.\n"
        "Return ONLY the JSON array, no other text."
    )

    print(f"\n  {'~'*50}")
    print(f"  QUIZ CHECKPOINT: {checkpoint_name}")
    print(f"  {'~'*50}")

    # Generate quiz questions
    print(f"  Generating quiz questions...")
    qresult = call_gpt5(system_prompt, quiz_gen_prompt, 2048)
    try:
        # Strip markdown fences if present
        raw = qresult["content"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        questions = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  [WARN] Failed to parse quiz questions: {e}")
        print(f"  Raw: {qresult['content'][:500]}")
        return {"score": 0, "total": 0, "passed": False, "details": "parse error"}

    # Now have JCoder answer the quiz
    answer_system = (
        "You are JCoder. Answer each question using ONLY what you have learned "
        "from the material you studied. Be specific and concise. "
        "If you are not sure, say so -- do not guess.\n\n"
        f"Your accumulated knowledge:\n{prior_context[:4000]}"
    )

    quiz_text = "\n\n".join(
        f"Q{i+1}: {q['q']}" for i, q in enumerate(questions)
    )

    answer_prompt = (
        f"Answer these {len(questions)} questions. For each, give a short, "
        f"definitive answer.\n\n{quiz_text}\n\n"
        "Format: one answer per line, starting with Q1:, Q2:, etc."
    )

    print(f"  JCoder answering {len(questions)} questions...")
    aresult = call_gpt5(answer_system, answer_prompt, 1024)

    # Now grade the answers
    grade_system = (
        "You are a strict grader. For each question-answer pair, determine if "
        "the student's answer is CORRECT or INCORRECT. An answer is correct if "
        "it captures the essential correct concept, even if worded differently. "
        "Be fair but firm."
    )

    grade_prompt = (
        f"Grade these answers:\n\n"
    )
    for i, q in enumerate(questions):
        grade_prompt += (
            f"Q{i+1}: {q['q']}\n"
            f"Correct answer: {q['correct']}\n"
            f"Student answer: (see below)\n\n"
        )
    grade_prompt += (
        f"Student's answers:\n{aresult['content']}\n\n"
        "For each question, respond with ONLY:\n"
        "Q1: CORRECT or INCORRECT (brief reason)\n"
        "Q2: CORRECT or INCORRECT (brief reason)\n"
        "etc.\n"
        "Then on the last line: SCORE: X/Y"
    )

    print(f"  Grading...")
    gresult = call_gpt5(grade_system, grade_prompt, 1024)

    # Parse score
    grade_text = gresult["content"]
    print(f"\n  QUIZ RESULTS:")
    for line in grade_text.strip().split("\n"):
        line = line.strip()
        if line:
            print(f"    {line}")

    # Extract score
    import re
    score_match = re.search(r"SCORE:\s*(\d+)\s*/\s*(\d+)", grade_text)
    if score_match:
        score = int(score_match.group(1))
        total = int(score_match.group(2))
    else:
        # Count CORRECT occurrences
        score = grade_text.upper().count("CORRECT") - grade_text.upper().count("INCORRECT")
        total = len(questions)
        score = max(0, score)

    passed = score >= (total * 0.6)  # 60% pass threshold
    result = {
        "score": score,
        "total": total,
        "passed": passed,
        "answers": aresult["content"][:500],
        "grades": grade_text[:500],
    }

    print(f"\n  SCORE: {score}/{total} {'PASSED' if passed else 'FAILED'}")
    if not passed:
        print(f"  [WARN] Below 60% threshold -- knowledge may need reinforcement")

    # Save quiz result
    if "quizzes" not in progress:
        progress["quizzes"] = {}
    progress["quizzes"][quiz_key] = result
    save_progress(progress)

    return result


# Quiz checkpoints: (after_phase_index, checkpoint_name, topics_list)
QUIZ_CHECKPOINTS = {
    3: ("Foundations & Patterns",
        ["Python stdlib", "Code patterns", "Code review", "Best practices"]),
    5: ("Algorithms & Real-World Code",
        ["Algorithms", "Data structures", "Math programming",
         "Production Python", "Git patterns"]),
    9: ("Community & Instruction Knowledge",
        ["Stack Overflow", "Systems/DevOps", "Security",
         "Code instructions", "Chain-of-thought"]),
    13: ("Systems & Languages",
         ["RFC/Protocols", "Rust", "Java", "Go", "JavaScript"]),
    17: ("Full Curriculum",
         ["All phases 1-9", "Cross-domain patterns", "Master synthesis"]),
}


def main():
    print("=" * 70)
    print("  JCODER SYSTEMATIC KNOWLEDGE DISTILLATION")
    print("  Curriculum: 10 phases, ~20 topics, quiz checkpoints")
    print("=" * 70)

    progress = load_progress()
    completed_count = len(progress["completed"])
    if completed_count > 0:
        print(f"\n[OK] Resuming -- {completed_count} phases already completed")

    total_phases = len(CURRICULUM)
    t_start = time.monotonic()

    for i, (phase, topic, indexes, focus) in enumerate(CURRICULUM):
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{total_phases}] Phase {phase}: {topic}")
        print(f"{'='*70}")

        success = process_phase(phase, topic, indexes, focus, progress)

        if not success:
            print(f"  [WARN] Phase {phase} had issues -- continuing")

        # Check if we hit a quiz checkpoint
        if (i + 1) in QUIZ_CHECKPOINTS:
            checkpoint_name, topics = QUIZ_CHECKPOINTS[i + 1]
            quiz_result = run_quiz(checkpoint_name, topics, progress)

    elapsed_total = time.monotonic() - t_start

    print(f"\n{'='*70}")
    print(f"  CURRICULUM COMPLETE")
    print(f"  Total phases: {len(progress['completed'])}/{total_phases}")
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"{'='*70}")

    # Print quiz summary
    if progress.get("quizzes"):
        print(f"\n  QUIZ SUMMARY:")
        total_score = 0
        total_qs = 0
        for name, result in progress["quizzes"].items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"    [{status}] {name}: {result['score']}/{result['total']}")
            total_score += result["score"]
            total_qs += result["total"]
        if total_qs > 0:
            pct = total_score / total_qs * 100
            print(f"\n  Overall quiz score: {total_score}/{total_qs} ({pct:.0f}%)")

    # List all generated knowledge files
    knowledge_files = sorted(KNOWLEDGE_DIR.glob("*_phase*"))
    if knowledge_files:
        print(f"\n  Knowledge files generated:")
        for f in knowledge_files:
            size = f.stat().st_size
            print(f"    {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
