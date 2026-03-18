# JCoder -- Agent Standing Orders

## YOUR ROLE
You are a development and testing agent for JCoder, a local-first AI coding
assistant with an autonomous agent, self-learning pipeline, and model cascade.

## PROJECT STRUCTURE
- `main.py` -- CLI entry point (Click)
- `core/` -- RAG pipeline, reasoning modules, research, self-learning (60+ modules)
- `agent/` -- Autonomous agent framework, tools, goals, sessions
- `ingestion/` -- Document parsing, chunking, PII scanning, dedup
- `evaluation/` -- 200-question golden eval set and scoring
- `scripts/` -- Learning cycle, weekly scraper, downloads, data prep
- `gui/` -- Tkinter command center (HybridRAG3 dark palette)
- `config/` -- YAML configs (models, ports, policies, agent, memory)
- `tests/` -- pytest suite
- `.venv/` -- virtual environment

## BEFORE DOING ANYTHING
1. Read `config/agent.yaml` for current module enable/disable state.
2. Read the target module(s) you will modify.
3. Check `tests/` for existing coverage before adding code.
4. Run `.venv\Scripts\python.exe -m pytest tests/ -q` to verify baseline.

## CODE RULES
- Max 500 lines of code per module. Split into focused files.
- No hardcoded drive paths. Use `JCODER_DATA` env var or config-driven paths.
- All self-learning modules must be optional (graceful degradation via try/import).
- Prefer atomic writes for JSON/state files.
- ASCII-only source files.

## TESTING RULES
- Full regression (`pytest tests/ -q`) must pass before any commit.
- Write new tests in `tests/test_*.py` for any new functionality.
- Eval smoke tests (`tests/test_eval_smoke.py`) must pass without Ollama.

## GIT RULES
- Pre-push hook blocks AI co-author trailers. Do not use Co-Authored-By lines.
- Commit messages: imperative mood, concise subject, body for context.

## ARCHITECTURE NOTES
- Agent bridge (`agent/bridge.py`) wires optional modules via try/import.
- Model cascade routes queries by complexity (simple -> local, complex -> cloud).
- Self-learning pipeline: 6-phase cycle (baseline -> study -> distill -> re-eval).
- All network access gated by `core/network_gate.py`.

*Documentation review: 2026-03-18 -- Documentation Engineer*
