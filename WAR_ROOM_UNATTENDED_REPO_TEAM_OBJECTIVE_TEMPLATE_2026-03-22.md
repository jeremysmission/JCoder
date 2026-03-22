# Unattended Repo Team Objective Template - 2026-03-22

Use this template when assigning repo work while away from the machine.

## Core rule

The team may analyze, plan, validate, document, and prepare bounded patches or PR-ready outputs.

The team may not:
- create scheduled tasks
- edit shell profiles
- edit registry startup entries
- open Windows Terminal tabs or shell windows automatically
- create background persistence
- spawn additional teams without approval
- perform cross-repo actions without approval
- merge directly to protected branches

## Standard objective template

```text
Repo: <repo name>
Mode: unattended, constrained
Primary objective: <single objective>

Allowed work:
- analyze current state
- inspect code and docs
- identify defects and root causes
- draft bounded code changes inside approved paths
- prepare PR-ready patches
- write reports and rollback notes

Forbidden work:
- no scheduler installs
- no shell/profile/startup edits
- no terminal/window spawning
- no registry edits
- no cross-repo writes
- no autonomous team generation
- no direct merges

Deliverables required:
- summary of findings
- exact files changed or proposed
- test/validation results
- rollback instructions
- open questions/blockers

Exit condition:
- stop after producing report, patch, or PR-ready output
- do not persist in background
- do not relaunch yourself
```

## Recommended repo objectives

### HybridRAG

```text
Repo: HybridRAG3_Educational
Mode: unattended, constrained
Primary objective: audit source corpus integrity and prepare a repair and reindex plan before any retrieval or query retuning.

Specific goals:
- identify suspect, duplicate, stale, or malformed source inputs
- map trusted source replacements where available
- use Hustle coordinator findings where relevant to identify correct redownload sources
- document exact repair order: source repair, canonical rebuild, reindex, retrieval validation, then retuning
- prepare bounded scripts or docs for review, but do not run destructive bulk rewrites without approval
```

### JCoder

```text
Repo: JCoder
Mode: unattended, constrained
Primary objective: audit automation and coordinator logic for unsafe persistence, hidden fanout, and weak rollback paths.

Specific goals:
- identify any host-level automation risks
- propose manifest-driven constraints
- prepare PR-ready guardrail changes
- document approval and rollback requirements
```

### Side Hustle RAG Business

```text
Repo: Side Hustle RAG Business
Mode: unattended, constrained
Primary objective: consolidate coordinator findings about trusted source acquisition and business-relevant research inputs.

Specific goals:
- identify strongest trusted source channels
- flag weak or questionable corpus inputs
- produce a prioritized reacquisition plan
- write only reports or PR-ready docs
```

### LimitlessApp_V2

```text
Repo: LimitlessApp_V2
Mode: unattended, constrained
Primary objective: identify bounded reliability fixes and prepare patch-ready outputs without any host-level automation.
```

### Career Moves

```text
Repo: Career Moves
Mode: unattended, constrained
Primary objective: audit current docs, data flow, and automation assumptions; produce a safe prioritized queue of fixes.
```

## Best practice

When away from the machine:
- assign one clear objective per repo
- prefer read/analyze/report mode
- allow bounded patch preparation only where rollback is clear
- require PR-ready outputs instead of autonomous completion
- review everything before merging

## Final rule

Unattended work should create clarity and prepared changes, not hidden background behavior.
