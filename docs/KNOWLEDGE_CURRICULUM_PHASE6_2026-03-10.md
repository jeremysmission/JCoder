# JCoder Phase 6: World-Class Coding Agent Curriculum

Generated: 2026-03-10
Purpose: Transform JCoder from "knows a lot of code" to "thinks like a senior engineer."
Strategy: Group by category, basics first within each, highest-impact categories first.

---

## Curriculum Design Principles

1. **Each category starts with fundamentals, works up to advanced**
2. **Priority order: reasoning > patterns > reference > raw data**
3. **Small high-value datasets before large noisy ones**
4. **Agent trajectory data is the #1 priority** (teaches HOW to solve, not just WHAT)

---

## Category A: Code Reasoning & Agent Intelligence (HIGHEST PRIORITY)

Why first: This is the single biggest gap. JCoder has 11M chunks of code but almost
no data on HOW to reason about code problems step by step.

### A1. Agent Trajectories (How agents solve real bugs)
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | HumanEval+ solutions | Simple function-level problem solving | 1 |
| Basic | MBPP+ solutions | Basic programming puzzles with tests | 2 |
| Medium | SWE-bench Verified trajectories | Real bug diagnosis and fixing | 3 |
| Medium | OpenHands/CodeAct trajectories | Multi-step agent reasoning | 4 |
| Advanced | SWE-agent trajectories | Tool-augmented code editing | 5 |

### A2. Code Review Intelligence (What experts catch)
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | CodeReviewer (Microsoft) | What reviewers comment on | 6 |
| Medium | Code review discussions | How review conversations evolve | 7 |
| Advanced | Refactoring pairs | Before/after code transformations | 8 |

### A3. Commit Reasoning (Why code changed)
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | CommitPackFT (already have) | Commit messages + diffs | DONE |
| Medium | High-quality commit-diff pairs | Change reasoning at scale | 9 |

---

## Category B: API & Library Mastery (HIGH PRIORITY)

Why second: A coding assistant that doesn't know the APIs it's working with
produces wrong code. This is high-value, small-size data.

### B1. Python Standard Library
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | Python stdlib docs | Every function, every module | 10 |
| Medium | Python stdlib examples | Real usage patterns | 11 |

### B2. Popular Library Documentation
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | FastAPI docs | Modern Python web framework | 12 |
| Basic | Pydantic docs | Data validation patterns | 13 |
| Basic | pytest docs | Testing framework mastery | 14 |
| Basic | httpx docs | HTTP client patterns | 15 |
| Basic | click/typer docs | CLI framework patterns | 16 |
| Medium | SQLAlchemy docs | ORM and database patterns | 17 |
| Medium | asyncio docs | Async programming reference | 18 |
| Advanced | Django/Flask docs | Full web framework patterns | 19 |

---

## Category C: Security & Reliability (MEDIUM-HIGH PRIORITY)

Why third: Security awareness separates professional code from amateur code.
Production reliability separates demos from real systems.

### C1. Security Vulnerability Patterns
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | CWE Top 25 examples | Most common vulnerability patterns | 20 |
| Medium | OWASP code examples | Web security anti-patterns | 21 |
| Advanced | CVE fix patterns | How real vulnerabilities get patched | 22 |

### C2. Production Reliability
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | Error message -> fix mappings | Troubleshooting patterns | 23 |
| Medium | Postmortem/incident reports | How production systems fail | 24 |
| Advanced | SRE patterns | Google-style reliability engineering | 25 |

---

## Category D: Project Structure & DevOps (MEDIUM PRIORITY)

Why fourth: An agent that can't scaffold a proper project or set up CI/CD
can't work autonomously on real projects.

### D1. Project Scaffolding
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | GitHub project structures | How top repos are organized | 26 |
| Medium | cookiecutter templates | Project template patterns | 27 |

### D2. CI/CD & Build Systems
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | GitHub Actions workflows | CI/CD pipeline patterns | 28 |
| Medium | Dockerfile best practices | Container patterns | 29 |
| Advanced | Multi-stage build configs | Production deployment | 30 |

### D3. Database & Schema Design
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | Common schema patterns | Standard table designs | 31 |
| Medium | Migration sequences | Schema evolution over time | 32 |

---

## Category E: Architecture & Design (LOWER PRIORITY -- already partially covered)

### E1. Architecture Patterns
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | ADRs from open-source projects | Decision reasoning | 33 |
| Medium | API specs (OpenAPI/Swagger) | API design by example | 34 |
| Advanced | System design examples | Large-scale architecture | 35 |

### E2. Performance Engineering
| Level | Dataset | What It Teaches | Priority |
|-------|---------|-----------------|----------|
| Basic | Python performance anti-patterns | What NOT to do | 36 |
| Medium | Benchmark comparisons | Which approach is faster | 37 |

---

## Download Queue (VERIFIED -- all HuggingFace IDs confirmed)

### Category A: Code Reasoning & Agent Intelligence -- DOWNLOADING

| # | Dataset | HuggingFace ID | Entries | Status |
|---|---------|---------------|---------|--------|
| 1 | HumanEval+ | evalplus/humanevalplus | 164 | DOWNLOADING |
| 2 | MBPP+ | evalplus/mbppplus | 378 | DOWNLOADING |
| 3 | SWE-smith Trajectories | SWE-bench/SWE-smith-trajectories | 5,017 | DOWNLOADING |
| 4 | OpenHands Trajectories | nebius/SWE-rebench-openhands-trajectories | 67,074 | DOWNLOADING |
| 5 | SWE-agent Trajectories | nebius/SWE-agent-trajectories | 80,036 | DOWNLOADING |
| 6 | MEnvData Trajectories | ernie-research/MEnvData-SWE-Trajectory | 3,872 | DOWNLOADING |
| 7 | CoderForge Preview | togethercomputer/CoderForge-Preview | 51,000 | DOWNLOADING |
| 8 | Code Review Python | Dahoas/code-review-instruct-critique-revision-python | varies | DOWNLOADING |
| 9 | Code Review General | VatsaDev/code-review | varies | DOWNLOADING |

### Category B: Coding Instruction -- QUEUED

| # | Dataset | HuggingFace ID | Entries | Status |
|---|---------|---------------|---------|--------|
| 10 | Magicoder Evol-Instruct | ise-uiuc/Magicoder-Evol-Instruct-110K | 110,000 | QUEUED |
| 11 | Evol-CodeAlpaca | theblackcat102/evol-codealpaca-v1 | varies | QUEUED |

### Category C: Security & Reliability -- QUEUED

| # | Dataset | HuggingFace ID | Entries | Status |
|---|---------|---------------|---------|--------|
| 12 | CIRCL Vuln+CWE+Patch | CIRCL/vulnerability-cwe-patch | 39,260 | QUEUED |
| 13 | CVE+CWE 1999-2025 | stasvinokur/cve-and-cwe-dataset-1999-2025 | all CVEs | QUEUED |
| 14 | Security DPO | CyberNative/Code_Vulnerability_Security_DPO | varies | QUEUED |
| 15 | SecureCode Web | scthornton/securecode-web | 1,378 | QUEUED |
| 16 | CVE Training | AlicanKiraz0/All-CVE-Records-Training-Dataset | 300,000 | QUEUED |

### Category D: Large-Scale Code -- QUEUED

| # | Dataset | HuggingFace ID | Entries | Status |
|---|---------|---------------|---------|--------|
| 17 | GitHub Code 2025 | nick007x/github-code-2025 | 1.5M+ | QUEUED |

### Category E: Docs & Reference -- PLANNED (manual scrape)

| # | Dataset | Source | Status |
|---|---------|--------|--------|
| 18 | Python stdlib docs | docs.python.org | PLANNED |
| 19 | FastAPI docs | fastapi.tiangolo.com | PLANNED |
| 20 | Pydantic docs | docs.pydantic.dev | PLANNED |
| 21 | pytest docs | docs.pytest.org | PLANNED |
| 22 | httpx docs | www.python-httpx.org | PLANNED |

---

## Disk Budget

- **3 TB available** -- no constraints, max out all downloads
- Current usage: ~41 GB
- Strategy: Download everything. Clean download cache after indexing only if needed.

---

## Success Metrics

JCoder will be "best on earth" when it can:
1. Diagnose a bug from error output and fix it autonomously (agent trajectories)
2. Review code and catch security, performance, and style issues (code review data)
3. Know every Python stdlib function and popular library API (docs ingestion)
4. Scaffold a complete project with proper structure, tests, CI/CD (project templates)
5. Reason about architectural tradeoffs with evidence (ADRs, design patterns)
6. Write secure code by default (CWE/OWASP awareness)
7. Coordinate subagents effectively for multi-file projects (agent coordination data)
