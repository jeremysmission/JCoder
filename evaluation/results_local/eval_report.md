# Agent Evaluation Report

## Overall

| Metric | Value |
|--------|-------|
| Total questions | 200 |
| Passed | 195 |
| Pass rate | 97.5% |
| Avg score | 0.8684 |
| Min score | 0.0000 |
| Max score | 1.0000 |
| Avg latency | 38.713s |
| Avg tokens | 400 |

## Per Category

| Category | Count | Avg Score | Pass Rate |
|----------|-------|-----------|-----------|
| algorithms | 30 | 0.8875 | 96.7% |
| debugging | 30 | 0.8421 | 93.3% |
| go | 7 | 0.9143 | 100.0% |
| javascript | 40 | 0.9064 | 97.5% |
| python | 15 | 0.9383 | 100.0% |
| rust | 7 | 0.8250 | 100.0% |
| security | 30 | 0.8908 | 100.0% |
| shell | 6 | 0.7375 | 83.3% |
| systems | 35 | 0.8039 | 100.0% |

## Worst 5 Questions

- **dbg_171**: 0.0000
- **js_071**: 0.1000
- **alg_150**: 0.1000
- **dbg_158**: 0.3750
- **multi_197**: 0.4000

## Failed Question Details

### dbg_171 (score: 0.0000)

**Question:** How do I use cProfile to find performance bottlenecks in a Python script and interpret the output?

**Subscores:** {'has_code': 0.0, 'has_correct_api': 0.0, 'has_imports': 0.0, 'is_runnable': 0.0, 'cites_source': 0.0, 'weighted_total': 0.0}

**Answer preview:** [ERROR: timed out]

### js_071 (score: 0.1000)

**Question:** How do I use TypeScript mapped types to create a readonly version of an interface where all methods return Promises?

**Subscores:** {'has_code': 0.0, 'has_correct_api': 0.0, 'has_imports': 1.0, 'is_runnable': 0.0, 'cites_source': 0.0, 'weighted_total': 0.1}

**Answer preview:** [ERROR: timed out]

### alg_150 (score: 0.1000)

**Question:** What is amortized analysis? Explain it using the example of dynamic array resizing and its amortized O(1) append.

**Subscores:** {'has_code': 0.0, 'has_correct_api': 0.0, 'has_imports': 1.0, 'is_runnable': 0.0, 'cites_source': 0.0, 'weighted_total': 0.1}

**Answer preview:** [ERROR: timed out]

### dbg_158 (score: 0.3750)

**Question:** What does this error mean and how do I fix it?

ImportError: No module named 'my_package'

But the file exists in my project directory.

**Subscores:** {'has_code': 1.0, 'has_correct_api': 0.25, 'has_imports': 0.0, 'is_runnable': 0.0, 'cites_source': 0.0, 'weighted_total': 0.375}

**Answer preview:** The error `ImportError: No module named 'my_package'` indicates that Python cannot find the module named 'my_package' in your project directory. This could be due to several reasons:

1. The module is not installed in your Python environment.
2. The module is not installed in the site-packages direc...

### multi_197 (score: 0.4000)

**Question:** Explain bash parameter expansion including default values, substring extraction, pattern replacement, and variable indirection.

**Subscores:** {'has_code': 1.0, 'has_correct_api': 0.0, 'has_imports': 1.0, 'is_runnable': 0.0, 'cites_source': 0.0, 'weighted_total': 0.4}

**Answer preview:** Bash parameter expansion is a powerful feature that allows you to manipulate strings and variables in various ways. Here are some common uses:

1. **Default Values**: You can provide a default value for a variable if it is unset or null.

```bash
name=${name:-"Default Name"}
echo $name  # Outputs "D...

