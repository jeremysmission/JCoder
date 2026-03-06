"""
JCoder Agent Stack Smoke Test
------------------------------
Quick health check: verifies all agent modules import, wire together,
and pass basic functional tests using mocks (no live model needed).

Usage:  python scripts/smoke_test.py
Exit 0 = all critical checks pass, 1 = any failure.
"""
from __future__ import annotations
import os, sys, tempfile, shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_pass = _fail = _warn = _total = 0

def _ok(msg):
    global _pass, _total; _pass += 1; _total += 1; print(f"[OK]   {msg}")

def _fail_msg(msg, detail=""):
    global _fail, _total; _fail += 1; _total += 1
    print(f"[FAIL] {msg}" + (f" -- {detail}" if detail else ""))

def _warn_msg(msg, detail=""):
    global _warn, _total; _warn += 1; _total += 1
    print(f"[WARN] {msg}" + (f" -- {detail}" if detail else ""))

# -- 1. Import checks (critical=True means FAIL, False means WARN) --------

_IMPORTS = [
    ("agent.llm_backend",
     "create_backend, OpenAIBackend, AnthropicBackend, ChatResponse, ToolCall",
     True),
    ("agent.tools",        "ToolRegistry, ToolResult, TOOL_SCHEMAS", True),
    ("agent.core",         "Agent, AgentResult, AgentStep",          True),
    ("agent.goals",        "GoalQueue, StudyEngine, Goal",           True),
    ("agent.bridge",       "AgentBridge",                            True),
    ("agent.memory",       "AgentMemory, MemoryEntry",              True),
    ("agent.prompts",
     "PromptBuilder, FIMFormatter, AGENT_SYSTEM_PROMPT, CODE_QA_PROMPT, "
     "CODE_REVIEW_PROMPT, CODE_EXPLAIN_PROMPT, DEBUG_PROMPT, REFACTOR_PROMPT",
     True),
    ("core.federated_search",    "FederatedSearch, SearchResult",    True),
    ("ingestion.corpus_pipeline","CorpusPipeline, IngestStats",      True),
    ("ingestion.pii_scanner",    "PIIScanner, sanitize_for_ingest",  True),
    ("ingestion.dedup",          "MinHashDedup, DedupStats",         False),
    ("agent.config_loader",
     "AgentConfig, load_agent_config, build_agent_from_config", True),
    ("agent.session",      "SessionStore, SessionInfo",                  True),
    ("agent.logger",       "AgentLogger, AgentLogEntry",                 True),
    ("agent.web_tools",    "WebSearcher",                                True),
]

def run_import_checks():
    for mod, names, critical in _IMPORTS:
        try:
            m = __import__(mod, fromlist=names.split(", "))
            for n in names.split(", "):
                getattr(m, n)
            extra = ""
            if mod == "agent.tools":
                extra = f" ({len(m.TOOL_SCHEMAS)} tools)"
            elif mod == "agent.prompts":
                extra = f" ({len(m.PromptBuilder.available_modes())} modes)"
            _ok(f"{mod} imports{extra}")
        except Exception as exc:
            if critical:
                _fail_msg(f"{mod} imports", str(exc))
            else:
                _warn_msg(f"{mod} not available yet", str(exc))

# -- 2. Tool registry count -----------------------------------------------

def check_tool_registry():
    try:
        from agent.tools import ToolRegistry, TOOL_SCHEMAS
        ToolRegistry(working_dir=tempfile.gettempdir())
        n = len({t["function"]["name"] for t in TOOL_SCHEMAS})
        _ok(f"Tool registry: {n} tools registered") if n >= 15 else \
            _fail_msg(f"Tool registry: expected >=15, got {n}")
    except Exception as exc:
        _fail_msg("Tool registry creation", str(exc))

# -- 3. Mock agent run ----------------------------------------------------

def check_mock_agent_run():
    try:
        from agent.llm_backend import LLMBackend, ChatResponse
        from agent.tools import ToolRegistry
        from agent.core import Agent

        class _Mock(LLMBackend):
            def chat(self, messages, tools=None, temperature=0.1,
                     max_tokens=4096):
                return ChatResponse(content="Hello! Task complete.",
                                    tool_calls=[], model="mock",
                                    input_tokens=10, output_tokens=5)
            def close(self): pass

        with tempfile.TemporaryDirectory() as td:
            result = Agent(_Mock(), ToolRegistry(working_dir=td),
                           max_iterations=3).run("say hello")
            if result.success:
                _ok("Mock agent run completed")
            else:
                _fail_msg("Mock agent run", f"success=False: {result.summary}")
    except Exception as exc:
        _fail_msg("Mock agent run", str(exc))

# -- 4. Goal queue CRUD ---------------------------------------------------

def check_goal_queue():
    try:
        from agent.goals import GoalQueue, Goal
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "goals.json")
            gq = GoalQueue(persist_path=path)
            g = gq.add("Learn pytest fixtures",
                        description="Understand conftest and autouse",
                        priority=3)
            assert isinstance(g, Goal) and g.title == "Learn pytest fixtures"
            assert len(gq.list()) >= 1
            gq.complete(g.id, "Learned about fixtures and conftest.py")
            assert gq.get(g.id).status == "completed"
            assert GoalQueue(persist_path=path).get(g.id).status == "completed"
            _ok("Goal queue CRUD works")
    except Exception as exc:
        _fail_msg("Goal queue CRUD", str(exc))

# -- 5. Memory ingest + search (FTS5 only) --------------------------------

def check_memory_fts5():
    td = tempfile.mkdtemp()
    try:
        from agent.memory import AgentMemory, MemoryEntry
        mem = AgentMemory(embedding_engine=None,
                          index_dir=os.path.join(td, "indexes"),
                          knowledge_dir=os.path.join(td, "knowledge"))
        entry = mem.ingest(content="Use pytest.fixture with autouse=True for setup",
                           source_task="learn-pytest", tags=["pytest", "testing"])
        assert isinstance(entry, MemoryEntry)
        results = mem.search("pytest fixture autouse", top_k=3)
        assert len(results) >= 1 and "autouse" in results[0].get("content", "")
        _ok("Memory ingest + search works")
        mem.close()
    except Exception as exc:
        _fail_msg("Memory ingest + search", str(exc))
    finally:
        shutil.rmtree(td, ignore_errors=True)

# -- 6. PII scanner -------------------------------------------------------

def check_pii_detection():
    try:
        from ingestion.pii_scanner import PIIScanner
        result = PIIScanner(redact=True).scan(
            'api_key = "sk-proj-abc123XYZdef456GHIjklmnopqrs"')
        if not result.is_clean:
            _ok("PII scanner detects secrets")
        else:
            _fail_msg("PII scanner", "failed to detect api_key in assignment")
    except Exception as exc:
        _fail_msg("PII scanner", str(exc))

# -- 7. Prompt modes ------------------------------------------------------

def check_prompt_modes():
    try:
        from agent.prompts import PromptBuilder, FIMFormatter
        modes = PromptBuilder.available_modes()
        errors = []
        for mode in modes:
            pb = PromptBuilder(mode=mode)
            msgs = (pb.build_messages("", fim_prefix="def foo():",
                                      fim_suffix="    pass")
                    if mode == "fim"
                    else pb.build_messages("How do I sort a list?"))
            if not msgs or not isinstance(msgs, list):
                errors.append(f"{mode}: empty")
        if not FIMFormatter("devstral").format_completion("def f():", "\n    pass"):
            errors.append("FIMFormatter empty")
        if errors:
            _fail_msg("Prompt modes", "; ".join(errors))
        else:
            _ok(f"All prompt modes valid ({len(modes)} modes)")
    except Exception as exc:
        _fail_msg("Prompt modes", str(exc))

# -- 8. Session store round-trip -------------------------------------------

def check_session_store():
    td = tempfile.mkdtemp()
    try:
        from agent.session import SessionStore
        store = SessionStore(store_dir=os.path.join(td, "sessions"))
        store.save("smoke_test_session", "Smoke test task",
                   [{"role": "user", "content": "hello"}],
                   status="active", iterations=1, tokens=42)
        data = store.load("smoke_test_session")
        assert data["task"] == "Smoke test task"
        assert data["total_tokens"] == 42
        infos = store.list_sessions()
        assert len(infos) == 1
        store.delete("smoke_test_session")
        _ok("Session store CRUD works")
    except Exception as exc:
        _fail_msg("Session store", str(exc))
    finally:
        shutil.rmtree(td, ignore_errors=True)

# -- 9. Agent logger -------------------------------------------------------

def check_agent_logger():
    td = tempfile.mkdtemp()
    try:
        from agent.logger import AgentLogger
        lg = AgentLogger(log_dir=os.path.join(td, "logs"))
        lg.log_task_start("smoke_sess", "Smoke test")
        lg.log_tool_call("smoke_sess", "read_file", {"path": "/test.py"}, iteration=1)
        lg.log_task_complete("smoke_sess", True, "Done", total_tokens=100, iterations=1)
        entries = lg.query(session_id="smoke_sess")
        assert len(entries) == 3
        summary = lg.session_summary("smoke_sess")
        assert summary["success"] is True
        lg.close()
        _ok("Agent logger works")
    except Exception as exc:
        _fail_msg("Agent logger", str(exc))
    finally:
        shutil.rmtree(td, ignore_errors=True)

# -- 10. Config loader defaults --------------------------------------------

def check_config_loader():
    try:
        from agent.config_loader import AgentConfig, load_agent_config
        cfg = AgentConfig()
        assert cfg.max_iterations == 50
        assert cfg.memory_enabled is True
        assert cfg.backend == "openai"
        _ok("Config loader defaults valid")
    except Exception as exc:
        _fail_msg("Config loader", str(exc))

# -- 11. CLI agent commands ------------------------------------------------

def check_cli_commands():
    try:
        from cli.agent_cmd import agent_cmd
        assert agent_cmd is not None and agent_cmd.name == "agent"
        _ok("CLI agent commands registered")
    except Exception as exc:
        _fail_msg("CLI agent commands", str(exc))

# -- main ------------------------------------------------------------------

def main() -> int:
    print("\nJCoder Agent Smoke Test")
    print("=" * 40)
    run_import_checks()
    check_tool_registry()
    check_mock_agent_run()
    check_goal_queue()
    check_memory_fts5()
    check_pii_detection()
    check_prompt_modes()
    check_session_store()
    check_agent_logger()
    check_config_loader()
    check_cli_commands()
    print("=" * 40)
    parts = [f"{_pass}/{_total} passed"]
    if _warn:
        parts.append(f"{_warn} warning{'s' if _warn > 1 else ''}")
    if _fail:
        parts.append(f"{_fail} FAILED")
    print(", ".join(parts) + "\n")
    return 1 if _fail else 0

if __name__ == "__main__":
    sys.exit(main())
