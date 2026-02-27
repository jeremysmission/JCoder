# Multi-Model Knowledge Distillation Research

Date: 2026-02-26
Status: Research Summary -- Comprehensive Findings

---

## 1. Executive Summary

Knowledge distillation from frontier LLMs into a smaller, locally-runnable code model
is the core training strategy for JCoder. This document consolidates research on
multi-teacher distillation, budget math, data generation techniques, quality filtering,
terms-of-service constraints, and alignment methods (DPO, GRPO). The target student
model is Phi-4 14B (MIT license, NDAA-compliant), fine-tuned on dual RTX 3090 GPUs
(48 GB total VRAM) using QLoRA via Axolotl.

---

## 2. Multi-Teacher Distillation

### 2.1 Optimal Teacher Count

Recent work (arXiv:2602.01064, February 2026) establishes that **2-3 teachers is
optimal** for multi-teacher distillation. Beyond three teachers, knowledge conflicts
between disagreeing teachers actively degrade student quality rather than improving it.
The degradation is not gradual -- it is a sharp cliff after the third teacher, because
the student model lacks the capacity to reconcile fundamentally different reasoning
strategies.

### 2.2 Knowledge Purification

The **Knowledge Purification** technique addresses teacher disagreements head-on.
Rather than naively averaging teacher logits or randomly sampling from teachers, the
method works as follows:

1. Collect rationales from all teachers for the same prompt.
2. Identify agreement zones (where teachers converge) and conflict zones.
3. Condense the rationales into a single purified rationale that preserves the
   consensus reasoning and discards contradictory artifacts.
4. Distill from the purified rationale, not from raw teacher outputs.

This approach resolves the N>3 degradation problem partially, but the diminishing
returns still make 2-3 teachers the practical sweet spot.

### 2.3 Assigned Teacher Roles

Each teacher is selected for a distinct capability where it demonstrably outperforms
the others:

| Teacher             | Role                        | Strength                                  |
|---------------------|-----------------------------|-------------------------------------------|
| Claude Opus         | Reasoning, debugging        | Best chain-of-thought, catches subtle bugs |
| GPT-5.2-Codex       | Implementation, SWE-bench   | Strongest on real-world repo tasks         |
| Gemini 3.1 Pro      | Long-context, architecture  | 2M token context, system design            |
| Grok Code Fast 1    | Bulk generation             | 10x cheaper per token, high throughput     |

**Grok Code Fast 1** is the workhorse for volume. Its output quality is lower than the
other three, but at 10x the cost efficiency it is used for generating the bulk of
candidate pairs that are then filtered and scored by the higher-quality teachers.

---

## 3. Budget Math

### 3.1 Claude Max -- Critical Clarification

**Claude Max at $200/month is NOT API credits.** It provides interactive chat/console
use only. There is no programmatic API access included. Distillation requires a
**separate API budget** through Anthropic's API billing, which is usage-based and
billed per token. Do not plan around Claude Max as a distillation budget line item.

### 3.2 Batch Pricing Estimates

All three major providers offer batch/async pricing at roughly 50% of real-time rates.
The following estimates use batch pricing where available:

**$300/month budget:**

| Provider          | Allocation | Raw Pairs Generated | After Filtering (~50%) |
|-------------------|------------|--------------------:|----------------------:|
| Claude Opus       | $90        | ~9,000              | ~4,500                |
| GPT-5.2-Codex     | $90        | ~18,000             | ~9,000                |
| Gemini 3.1 Pro    | $60        | ~12,000             | ~6,000                |
| Grok Code Fast 1  | $60        | ~24,000             | ~12,000               |
| **Total**         | **$300**   | **~63,000**         | **~31,500**           |

**$500/month budget:**

| Provider          | Allocation | Raw Pairs Generated | After Filtering (~50%) |
|-------------------|------------|--------------------:|----------------------:|
| Claude Opus       | $150       | ~15,000             | ~7,500                |
| GPT-5.2-Codex     | $150       | ~30,000             | ~15,000               |
| Gemini 3.1 Pro    | $100       | ~20,000             | ~10,000               |
| Grok Code Fast 1  | $100       | ~27,000             | ~13,500               |
| **Total**         | **$500**   | **~92,000**         | **~46,000**           |

The ~50% yield after filtering is consistent with published numbers from WizardCoder
and Magicoder pipelines. Budget should be treated as a monthly burn rate, with data
accumulating across months toward the fine-tuning threshold.

---

## 4. Training Data Generation Techniques

### 4.1 Evol-Instruct (WizardCoder, ICLR 2024)

Evol-Instruct takes a seed instruction and evolves it through successive complexity
increases: adding constraints, requiring edge case handling, combining sub-problems,
and demanding optimization. Each evolution step produces a harder variant of the
original prompt. The key insight is that frontier models can reliably increase problem
difficulty even when they cannot always solve the harder problem -- the evolution and
solution steps are decoupled.

### 4.2 OSS-Instruct (Magicoder, ICML 2024)

OSS-Instruct generates training data by having a teacher model read real open-source
code and then create instruction-response pairs inspired by that code. This grounds the
generated data in realistic programming patterns rather than synthetic toy problems. The
resulting data has higher diversity and better coverage of real-world coding idioms than
purely synthetic generation.

### 4.3 UnitCoder (EMNLP 2025)

UnitCoder adds an execution verification step: each generated code sample is paired
with unit tests, and only samples that pass their tests are retained. This is
particularly valuable because it provides a ground-truth correctness signal that does
not depend on another model's judgment. The execution-verified subset has significantly
higher quality than the unverified superset.

### 4.4 Multi-Turn Debugging

Generate a first-pass solution with intentional or natural errors, then capture the
debugging conversation where the model identifies and fixes the error. Multi-turn
debugging data teaches the student model to recognize mistakes and self-correct, which
is a distinct and valuable skill from one-shot code generation.

### 4.5 Error Injection

Deliberately introduce common bug patterns (off-by-one, null reference, type
confusion, race conditions) into correct code, then train the model to identify and fix
them. This is the inverse of code generation -- it teaches code repair, which is often
more practically useful than writing code from scratch.

---

## 5. Quality Filtering Pipeline

A 5-stage filtering pipeline is applied to all generated data. Each stage is
independent and can be run in parallel where applicable. The expected cumulative yield
is approximately 50% of raw generated pairs.

### Stage 1: Syntax Validation
Parse the generated code with the language's AST parser. Reject anything that does not
parse. This catches truncated outputs, malformed syntax, and hallucinated language
constructs. Cost: near zero (CPU only). Expected pass rate: ~85%.

### Stage 2: Static Analysis
Run linters (pylint, ruff for Python; ESLint for JS/TS) and type checkers (mypy,
pyright). Reject code with critical errors. Warnings are allowed but logged for later
review. Expected pass rate: ~80% of Stage 1 survivors.

### Stage 3: Execution Verification
Run the code in a sandboxed environment. For function-level code, execute with provided
test cases. For script-level code, verify it runs without errors. This is the most
expensive stage but provides the strongest signal. Expected pass rate: ~75% of Stage 2
survivors.

### Stage 4: Consistency Check
For multi-teacher data, check that the instruction-response pair is internally
consistent: the response actually addresses the instruction, variable names match,
imports are present, etc. This catches cases where a teacher generates plausible but
wrong code. Expected pass rate: ~90% of Stage 3 survivors.

### Stage 5: Deduplication
Hash-based and semantic deduplication. Exact duplicates are trivial to catch. Near-
duplicates (same logic, different variable names) are caught with MinHash or embedding
similarity. This stage does not reject bad code but reduces redundancy to improve
training efficiency. Expected removal: ~10-15% of Stage 4 survivors.

**Cumulative yield:** 0.85 * 0.80 * 0.75 * 0.90 * 0.87 = ~40-50%.

---

## 6. Terms of Service Analysis

### 6.1 Current Restrictions

**All three major providers explicitly restrict distillation in their terms of service:**

- **Anthropic (Claude):** Usage Policy prohibits using outputs to train competing
  models. The Acceptable Use Policy and API Terms of Service both include language
  restricting use of outputs to develop models that compete with Anthropic's services.

- **OpenAI (GPT-5.2-Codex):** Terms of Service Section 2(c) restricts using outputs
  to develop models that compete with OpenAI. The Business Terms add additional
  restrictions for enterprise customers.

- **Google (Gemini):** API Terms restrict using outputs to train or improve other AI
  models, with specific callouts for distillation and synthetic data generation.

### 6.2 Gray Areas

Personal and educational use occupies a legally ambiguous zone. None of the providers
have publicly enforced ToS against individual researchers doing small-scale
distillation for personal projects. However, the risk is non-zero, and the terms are
written broadly enough to cover this use case if a provider chose to enforce.

### 6.3 Recommended Blend Strategy

To minimize legal exposure while still benefiting from frontier model quality:

| Data Source                | Proportion | Purpose                                |
|----------------------------|------------|----------------------------------------|
| Open-source datasets       | 60-70%     | Bulk training data, no ToS risk        |
| Hand-crafted examples      | 20-30%     | Domain-specific, high-quality seed data |
| Frontier model outputs     | Judging/scoring only | Quality filtering, preference pairs |

Under this strategy, frontier models are used primarily as **judges and scorers**
rather than as direct data generators. They evaluate and rank open-source or
hand-crafted data rather than generating the training data itself. This substantially
reduces ToS exposure because the frontier outputs (scores, rankings) are not themselves
used as training targets.

---

## 7. Fine-Tuning Target

### 7.1 Student Model: Phi-4 14B

- **License:** MIT (fully permissive, no usage restrictions)
- **NDAA Compliance:** Microsoft/USA origin, no restricted-country components
- **Size:** 14B parameters, ~9.1 GB in Q4_K_M quantization
- **Context:** 16K tokens (sufficient for function-level code generation)
- **Base performance:** Strong on code benchmarks for its size class

### 7.2 Hardware: Dual RTX 3090

- **Total VRAM:** 48 GB (2x 24 GB)
- **QLoRA memory footprint:** ~18 GB for Phi-4 14B (leaves headroom for batch size)
- **Framework:** Axolotl (supports FSDP + QLoRA for multi-GPU training)
- **Expected throughput:** ~500 samples/hour for 14B with QLoRA

---

## 8. Alignment Methods

### 8.1 DPO (Direct Preference Optimization)

Standard DPO requires paired preference data: a chosen (good) response and a rejected
(bad) response for the same prompt. For code, this is generated by producing multiple
solutions and using test results to determine which is better.

**Minimum dataset size:** 10,000 preference pairs for meaningful alignment.

**Focused-DPO** is a variant that concentrates preference pairs on areas where the
model makes the most errors, rather than uniformly sampling. This is more data-
efficient because it targets the model's actual weaknesses. For a code model, this
means over-representing error-prone areas like edge case handling, concurrency, and
type-system interactions.

### 8.2 GRPO (Group Relative Policy Optimization)

GRPO is the best alignment method for code because it uses **verifiable test-result
rewards** and requires **no critic model.** The process:

1. For each prompt, generate N candidate solutions (typically N=4-8).
2. Run all solutions against test cases to get binary pass/fail scores.
3. Compute the group average score.
4. Reinforce solutions scoring above the group average.
5. Penalize solutions scoring below the group average.

The key advantage is that code correctness is objectively verifiable through test
execution, which eliminates the need for a learned reward model or critic. This makes
GRPO both simpler and more reliable than RLHF for code tasks.

### 8.3 Recommended Sequence

1. **SFT** (supervised fine-tuning) on filtered instruction data -- builds base capability.
2. **DPO** (Focused-DPO) on 10K+ preference pairs -- aligns output style and fixes common errors.
3. **GRPO** on execution-verified problems -- maximizes pass@1 through test-based reinforcement.

---

## 9. Sources

- arXiv:2602.01064 -- Multi-teacher distillation scaling analysis (February 2026)
- Anthropic Terms of Service and Acceptable Use Policy
- OpenAI Terms of Service (Section 2c, API Terms)
- Google Gemini API Terms of Service
- WizardCoder: Empowering Code Large Language Models with Evol-Instruct (ICLR 2024)
- Magicoder: Empowering Code Generation with OSS-Instruct (ICML 2024)
- UnitCoder: Unit Test-Driven Code Generation with Execution Feedback (EMNLP 2025)
- Focused-DPO: Targeted Preference Optimization for Error-Prone Regions (ACL 2025)
- GRPO: Group Relative Policy Optimization (DeepSeek-R1 Technical Report)
