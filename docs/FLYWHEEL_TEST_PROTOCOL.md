# Flywheel Self-Improvement Test Protocol

Based on Absolute Zero methodology + practitioner pitfalls research.

## The Trifecta (what makes improvement REAL)
1. **Procedural generation** — generate test problems algorithmically so they can't be memorized
2. **External verification** — Python execution validates correctness, never self-judgment
3. **Cross-domain transfer** — if coding self-study improves math reasoning, improvement is real

## 5-Step Protocol

### Step 1: Clean Baseline
Evaluate on 3 categories: (a) in-domain (coding), (b) near-transfer (math), (c) far-transfer (logic).
Record pass@1 at temperature=0.0. This is the immutable reference.

### Step 2: Externally Verified Training Tasks
Self-improvement loop must use code executor as verifier.
Difficulty frontier: problems solved 30-50% of the time (maximum learnability).
Never use self-judgment for verification.

### Step 3: Contamination Firewall
Three separated eval sets:
- **Canary set** (50 problems model HAS seen) — if these rise disproportionately = memorization
- **Held-out set** (created before training, never touched) — the true test
- **Live-generated set** (created AFTER training with new random seeds) — confirms transfer

### Step 4: Cross-Domain Transfer
Strongest signal: self-study on coding → improvement on math reasoning = genuine growth.
If only in-domain improves → may be template memorization.

### Step 5: Ablations
Compare against: (a) random data (not curriculum), (b) solutions-only (no self-play),
(c) shuffled difficulty. Must beat all three to prove the mechanism works.

## Key Pitfalls
- Small models plateau fast (watch for flatline after initial gains)
- Elo inflation in self-play (benchmark against fixed external standard)
- Style overfitting (learning format, not reasoning)
- Pass@1 vs Pass@k distortion (report both)

## Our First Result
- Bank A: 40% (2/5) — baseline
- Bank B: 80% (4/5) — post-feedback-routing
- Delta: +40% (promising but needs more problems + contamination controls)
- Consistent weakness: cache implementations (0/2 across both banks)

Signed: Claude Opus 4.6 | 2026-03-25
