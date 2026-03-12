# Eval A/B Comparison Report

Baseline: `results_local` (200 questions)
Candidate: `results_api_gpt-4_1-mini` (200 questions)

## Overall

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| Questions | 200 | 200 | -- |
| Pass rate | 195/200 (97.5%) | 200/200 (100.0%) | +5 |
| Avg score | 0.8684 | 0.8982 | +0.0298 |
| Avg latency | 38.7s | 8.4s | -30.3s |

## Per Category

| Category | Baseline Avg | Candidate Avg | Delta | Baseline Pass | Candidate Pass |
|----------|-------------|---------------|-------|--------------|----------------|
| shell | 0.737 | 0.804 | +0.067 | 83% | 100% |
| algorithms | 0.887 | 0.944 | +0.057 | 97% | 100% |
| debugging | 0.842 | 0.899 | +0.057 | 93% | 100% |
| python | 0.938 | 0.968 | +0.030 | 100% | 100% |
| security | 0.891 | 0.921 | +0.030 | 100% | 100% |
| systems | 0.804 | 0.830 | +0.026 | 100% | 100% |
| rust | 0.825 | 0.850 | +0.025 | 100% | 100% |
| javascript | 0.906 | 0.901 | -0.005 | 98% | 100% |
| go | 0.914 | 0.907 | -0.007 | 100% | 100% |

## Distillation Targets (Weakest in Baseline)

1. **shell** -- baseline avg=0.737, candidate avg=0.804, gap=+0.067
2. **systems** -- baseline avg=0.804, candidate avg=0.830, gap=+0.026
3. **rust** -- baseline avg=0.825, candidate avg=0.850, gap=+0.025
4. **debugging** -- baseline avg=0.842, candidate avg=0.899, gap=+0.057
5. **algorithms** -- baseline avg=0.887, candidate avg=0.944, gap=+0.057

## Improved Questions (66)

- **alg_150** (algorithms): 0.10 -> 0.93 (+0.83)
- **dbg_171** (debugging): 0.00 -> 0.80 (+0.80)
- **js_071** (javascript): 0.10 -> 0.62 (+0.53)
- **dbg_158** (debugging): 0.38 -> 0.80 (+0.43)
- **sec_118** (security): 0.65 -> 1.00 (+0.35)
- **multi_197** (shell): 0.40 -> 0.72 (+0.32)
- **js_009** (javascript): 0.60 -> 0.92 (+0.32)
- **dbg_002** (debugging): 0.70 -> 1.00 (+0.30)
- **multi_188** (rust): 0.70 -> 1.00 (+0.30)
- **sys_003** (systems): 0.50 -> 0.80 (+0.30)
- **algo_003** (algorithms): 0.72 -> 1.00 (+0.28)
- **alg_154** (algorithms): 0.75 -> 1.00 (+0.25)
- **sys_095** (systems): 0.55 -> 0.80 (+0.25)
- **sys_009** (systems): 0.68 -> 0.90 (+0.22)
- **dbg_004** (debugging): 0.70 -> 0.90 (+0.20)
- ... and 51 more

## Regressed Questions (39)

- **js_069** (javascript): 1.00 -> 0.70 (-0.30)
- **alg_145** (algorithms): 0.90 -> 0.70 (-0.20)
- **dbg_157** (debugging): 0.93 -> 0.72 (-0.20)
- **js_062** (javascript): 0.93 -> 0.72 (-0.20)
- **js_066** (javascript): 0.93 -> 0.72 (-0.20)
- **multi_189** (rust): 0.93 -> 0.72 (-0.20)
- **multi_193** (rust): 0.93 -> 0.72 (-0.20)
- **sys_096** (systems): 0.90 -> 0.70 (-0.20)
- **dbg_169** (debugging): 1.00 -> 0.80 (-0.20)
- **js_001** (javascript): 1.00 -> 0.80 (-0.20)

## Summary

- **66** questions improved
- **39** questions regressed
- **95** questions unchanged
- Net score delta: +0.0298

