# Repo Handoff And Recovery Priorities - 2026-03-22

## First read

The recent workstation incident proved that automation and RAG integrity must both be handled as substrate problems first.

Two hard rules now apply:
- do not optimize on top of an untrusted substrate
- do not automate on top of an untrusted control plane

## Automation rule

Before any multi-agent or coordinator design:
1. search for known failure modes
2. search for the standard safe architecture
3. only then propose a design
4. if the work touches startup, scheduling, shells, terminals, or registry, require a verification pass against prior art before implementation

## RAG recovery rule

Before demo or retuning:
1. repair source downloads from trusted sources
2. rebuild canonical corpus
3. reindex
4. validate queries and retrieval on the rebuilt index
5. run button-smash and regression tests
6. only then retune retrieval, ranking, and query logic

## Visible minimum required to continue

Every repo participating in recovery should surface:
- trusted source status
- corpus integrity status
- reindex status
- query validation status
- regression status
- approval status for retuning

## What not to do

- no `schtasks` launching `pwsh`
- no profile edits for orchestration
- no recurring `wt.exe` automation
- no autonomous team regeneration without prompt
- no cross-repo fanout without approval
- no retuning on top of suspect source data
