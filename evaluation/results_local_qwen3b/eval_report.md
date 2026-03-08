# Agent Evaluation Report

## Overall

| Metric | Value |
|--------|-------|
| Total questions | 200 |
| Passed | 198 |
| Pass rate | 99.0% |
| Avg score | 0.8504 |
| Min score | 0.1000 |
| Max score | 1.0000 |
| Avg latency | 43.406s |
| Avg tokens | 615 |

## Per Category

| Category | Count | Avg Score | Pass Rate |
|----------|-------|-----------|-----------|
| algorithms | 30 | 0.9058 | 100.0% |
| debugging | 30 | 0.8433 | 100.0% |
| go | 7 | 0.8643 | 100.0% |
| javascript | 40 | 0.8912 | 100.0% |
| python | 15 | 0.8394 | 93.3% |
| rust | 7 | 0.8500 | 100.0% |
| security | 30 | 0.8700 | 100.0% |
| shell | 6 | 0.7250 | 83.3% |
| systems | 35 | 0.7689 | 100.0% |

## Worst 5 Questions

- **py_005**: 0.1000
- **multi_197**: 0.4000
- **sys_081**: 0.5500
- **sec_124**: 0.5500
- **js_007**: 0.5750

## Failed Question Details

### py_005 (score: 0.1000)

**Question:** How do I read and write JSON files in Python using the json module?

**Subscores:** {'has_code': 0.0, 'has_correct_api': 0.0, 'has_imports': 1.0, 'is_runnable': 0.0, 'cites_source': 0.0, 'weighted_total': 0.1}

**Answer preview:** How do I read and write JSON files in Python using the json module?

### multi_197 (score: 0.4000)

**Question:** Explain bash parameter expansion including default values, substring extraction, pattern replacement, and variable indirection.

**Subscores:** {'has_code': 1.0, 'has_correct_api': 0.0, 'has_imports': 1.0, 'is_runnable': 0.0, 'cites_source': 0.0, 'weighted_total': 0.4}

**Answer preview:** Bash parameter expansion is a powerful feature in the Bash shell that allows you to manipulate strings and variables in various ways. It provides a concise syntax for extracting substrings, replacing patterns, and performing other string manipulations. Here's a detailed explanation of the main compo...

