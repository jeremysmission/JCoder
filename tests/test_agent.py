"""
Tests for the JCoder autonomous agent framework.
Covers: tool safety, tool registry, LLM backends, agent core loop, goal queue.
"""

import json
import os
import subprocess

import httpx
import pytest

from agent.llm_backend import (
    AnthropicBackend,
    ChatResponse,
    LLMBackend,
    OpenAIBackend,
    ToolCall,
    create_backend,
)
from agent.session import SessionStore
from agent.tools import ToolRegistry, ToolResult, TOOL_SCHEMAS, _is_command_safe
from agent.core import Agent, AgentResult, AgentStep
from agent.goals import Goal, GoalQueue, PENDING, COMPLETED, FAILED


# ---------------------------------------------------------------------------
# Mock backend for agent tests
# ---------------------------------------------------------------------------

class MockBackend(LLMBackend):
    """Returns pre-configured responses in sequence."""

    def __init__(self, responses: list):
        self._responses = responses
        self._call_count = 0

    def chat(self, messages, tools=None, temperature=0.1, max_tokens=4096):
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    def close(self):
        pass


# ===========================================================================
# TestToolSafety
# ===========================================================================

class TestToolSafety:
    """Tests for _is_command_safe() guard rails."""

    def test_safe_commands(self):
        for cmd in ("python --version", "git status", "pip list"):
            safe, reason = _is_command_safe(cmd)
            assert safe, f"{cmd!r} should be safe, got: {reason}"

    def test_blocked_commands(self):
        for cmd in ("format C:", "shutdown /s", "diskpart"):
            safe, reason = _is_command_safe(cmd)
            assert not safe, f"{cmd!r} should be blocked"
            assert "Blocked command" in reason

    def test_blocked_patterns(self):
        # rm -rf / is caught by the _BLOCKED_PATTERNS regex
        safe, reason = _is_command_safe("rm -rf /")
        assert not safe, "'rm -rf /' should be blocked"
        assert "Blocked pattern" in reason

    def test_empty_command(self):
        safe, reason = _is_command_safe("")
        assert not safe
        assert "Empty" in reason

    def test_blocked_shell_operators(self):
        safe, reason = _is_command_safe("git status && whoami")
        assert not safe
        assert "Shell operator" in reason


# ===========================================================================
# TestToolRegistry
# ===========================================================================

class TestToolRegistry:
    """Tests for ToolRegistry file/search/rag operations."""

    def test_read_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        reg = ToolRegistry(working_dir=str(tmp_path))
        result = reg.execute("read_file", {"path": str(f)})
        assert result.success
        assert "hello world" in result.output

    def test_write_file(self, tmp_path):
        reg = ToolRegistry(working_dir=str(tmp_path))
        target = str(tmp_path / "out.txt")
        result = reg.execute("write_file", {"path": target, "content": "abc123"})
        assert result.success
        assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "abc123"

    def test_edit_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("x = 1\ny = 2\n", encoding="utf-8")
        reg = ToolRegistry(working_dir=str(tmp_path))
        result = reg.execute("edit_file", {
            "path": str(f), "old_text": "x = 1", "new_text": "x = 99",
        })
        assert result.success
        assert "x = 99" in f.read_text(encoding="utf-8")

    def test_edit_file_not_found(self, tmp_path):
        reg = ToolRegistry(working_dir=str(tmp_path))
        result = reg.execute("edit_file", {
            "path": str(tmp_path / "nope.py"),
            "old_text": "a", "new_text": "b",
        })
        assert not result.success
        assert "not found" in result.error.lower()

    def test_edit_file_ambiguous(self, tmp_path):
        f = tmp_path / "dup.py"
        f.write_text("val = 1\nval = 1\n", encoding="utf-8")
        reg = ToolRegistry(working_dir=str(tmp_path))
        result = reg.execute("edit_file", {
            "path": str(f), "old_text": "val = 1", "new_text": "val = 2",
        })
        assert not result.success
        assert "2 times" in result.error

    def test_search_files(self, tmp_path):
        (tmp_path / "a.py").write_text("pass", encoding="utf-8")
        (tmp_path / "b.py").write_text("pass", encoding="utf-8")
        (tmp_path / "c.txt").write_text("pass", encoding="utf-8")
        reg = ToolRegistry(working_dir=str(tmp_path))
        result = reg.execute("search_files", {"pattern": "*.py"})
        assert result.success
        assert "a.py" in result.output
        assert "b.py" in result.output

    def test_search_content(self, tmp_path):
        (tmp_path / "one.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (tmp_path / "two.py").write_text("class Bar:\n    pass\n", encoding="utf-8")
        reg = ToolRegistry(working_dir=str(tmp_path))
        result = reg.execute("search_content", {"pattern": "def foo"})
        assert result.success
        assert "one.py" in result.output
        assert "two.py" not in result.output

    def test_unknown_tool(self, tmp_path):
        reg = ToolRegistry(working_dir=str(tmp_path))
        result = reg.execute("nonexistent_tool", {})
        assert not result.success
        assert "Unknown tool" in result.error

    def test_path_restriction(self, tmp_path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        reg = ToolRegistry(
            working_dir=str(allowed),
            allowed_dirs=[str(allowed)],
        )
        outside = str(tmp_path / "secret.txt")
        result = reg.execute("read_file", {"path": outside})
        assert not result.success
        assert "outside" in result.error.lower() or "PermissionError" in result.error

    def test_prefix_match_escape_is_blocked(self, tmp_path):
        allowed = tmp_path / "safe"
        sibling = tmp_path / "safe_evil"
        allowed.mkdir()
        sibling.mkdir()
        target = sibling / "secret.txt"
        target.write_text("nope", encoding="utf-8")

        reg = ToolRegistry(working_dir=str(allowed), allowed_dirs=[str(allowed)])
        result = reg.execute("read_file", {"path": str(target)})

        assert not result.success
        assert "outside allowed" in result.error.lower()

    def test_symlink_escape_is_blocked(self, tmp_path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        link = allowed / "link.txt"

        try:
            os.symlink(outside, link)
        except (AttributeError, NotImplementedError, OSError):
            pytest.skip("symlinks unavailable in this environment")

        reg = ToolRegistry(working_dir=str(allowed), allowed_dirs=[str(allowed)])
        result = reg.execute("read_file", {"path": str(link)})

        assert not result.success
        assert "outside allowed" in result.error.lower()

    def test_run_command_uses_shell_false(self, tmp_path, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["shell"] = kwargs.get("shell")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr("agent.tools.subprocess.run", fake_run)
        reg = ToolRegistry(working_dir=str(tmp_path))
        result = reg.execute("run_command", {"command": 'python -c "print(1)"'})

        assert result.success
        assert captured["shell"] is False
        assert captured["cmd"] == ["python", "-c", "print(1)"]

    def test_rag_query_no_callback(self, tmp_path):
        reg = ToolRegistry(working_dir=str(tmp_path), rag_callback=None)
        result = reg.execute("rag_query", {"query": "how to sort a list"})
        assert not result.success
        assert "not configured" in result.error.lower()

    def test_rag_query_with_callback(self, tmp_path):
        def fake_rag(query: str) -> str:
            return f"Answer for: {query}"

        reg = ToolRegistry(working_dir=str(tmp_path), rag_callback=fake_rag)
        result = reg.execute("rag_query", {"query": "how to sort"})
        assert result.success
        assert "Answer for: how to sort" in result.output

    def test_schemas_valid(self):
        assert len(TOOL_SCHEMAS) > 0
        for schema in TOOL_SCHEMAS:
            fn = schema.get("function", schema)
            assert "name" in fn, f"Schema missing 'name': {schema}"
            assert "description" in fn, f"Schema missing 'description' for {fn['name']}"
            assert "parameters" in fn, f"Schema missing 'parameters' for {fn['name']}"


# ===========================================================================
# TestLLMBackend
# ===========================================================================

class TestLLMBackend:
    """Tests for backend factory and ChatResponse properties."""

    def test_create_openai_backend(self):
        backend = create_backend("openai", endpoint="http://localhost:11434/v1",
                                 model="test-model")
        assert isinstance(backend, OpenAIBackend)
        backend.close()

    def test_create_ollama_backend(self):
        backend = create_backend("ollama", model="test-model")
        assert isinstance(backend, OpenAIBackend)
        backend.close()

    def test_create_anthropic_backend(self):
        backend = create_backend("anthropic", model="claude-test",
                                 api_key="sk-fake")
        assert isinstance(backend, AnthropicBackend)
        backend.close()

    def test_create_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("foo")

    def test_chat_response_has_tool_calls(self):
        resp = ChatResponse(
            content="",
            tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "x"})],
        )
        assert resp.has_tool_calls is True

    def test_chat_response_no_tool_calls(self):
        resp = ChatResponse(content="done")
        assert resp.has_tool_calls is False

    def test_openai_backend_rejects_malformed_tool_json(self):
        payload = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "tc1",
                        "function": {"name": "read_file", "arguments": "{not json"},
                    }],
                }
            }],
            "usage": {},
        }
        backend = OpenAIBackend(endpoint="http://example.test/v1", model="demo")
        backend._client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload, request=request)
            )
        )
        with pytest.raises(ValueError, match="Malformed tool arguments"):
            backend.chat([{"role": "user", "content": "hi"}])
        backend.close()

    def test_anthropic_backend_rejects_malformed_tool_json(self):
        backend = AnthropicBackend(model="claude-test", api_key="sk-test")
        with pytest.raises(ValueError, match="Malformed tool arguments"):
            backend._convert_messages([{
                "role": "assistant",
                "tool_calls": [{
                    "id": "tc1",
                    "function": {"name": "read_file", "arguments": "{not json"},
                }],
            }])
        backend.close()


# ===========================================================================
# TestAgent
# ===========================================================================

class TestAgent:
    """Tests for the Agent core loop using MockBackend."""

    def test_agent_simple_response(self, tmp_path):
        """Backend returns text with no tool calls -- agent returns immediately."""
        backend = MockBackend([
            ChatResponse(content="The answer is 42.", input_tokens=10, output_tokens=5),
        ])
        tools = ToolRegistry(working_dir=str(tmp_path))
        agent = Agent(backend=backend, tools=tools, max_iterations=5)

        result = agent.run("What is the answer?")
        assert result.success
        assert "42" in result.summary
        assert result.iterations == 1
        assert len(result.steps) == 0

    def test_agent_tool_call(self, tmp_path):
        """Backend requests a tool call, then returns text. Verify tool executed."""
        target = tmp_path / "data.txt"
        target.write_text("secret content", encoding="utf-8")

        # Iteration 1: LLM asks to read a file
        # Iteration 2: LLM returns final text
        backend = MockBackend([
            ChatResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="read_file",
                                     arguments={"path": str(target)})],
                input_tokens=20, output_tokens=10,
            ),
            ChatResponse(
                content="The file contains secret content.",
                input_tokens=30, output_tokens=15,
            ),
        ])
        tools = ToolRegistry(working_dir=str(tmp_path))
        agent = Agent(backend=backend, tools=tools, max_iterations=10)

        result = agent.run("Read data.txt")
        assert result.success
        assert len(result.steps) == 1
        assert result.steps[0].tool_name == "read_file"
        assert result.steps[0].tool_success
        assert result.iterations == 2

    def test_agent_max_iterations(self, tmp_path):
        """Backend always returns tool calls -- agent stops at iteration limit."""
        backend = MockBackend([
            ChatResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="search_files",
                                     arguments={"pattern": "*.py"})],
                input_tokens=10, output_tokens=5,
            ),
        ])
        tools = ToolRegistry(working_dir=str(tmp_path))
        agent = Agent(backend=backend, tools=tools, max_iterations=3)

        result = agent.run("Find all Python files forever")
        assert not result.success
        assert result.iterations == 3
        assert result.timed_out

    def test_agent_task_complete(self, tmp_path):
        """Backend calls task_complete tool -- agent returns with success."""
        backend = MockBackend([
            ChatResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="task_complete",
                                     arguments={"summary": "All done."})],
                input_tokens=10, output_tokens=5,
            ),
        ])
        tools = ToolRegistry(working_dir=str(tmp_path))
        agent = Agent(backend=backend, tools=tools, max_iterations=10)

        result = agent.run("Do something and signal done")
        assert result.success
        assert "All done" in result.summary
        assert result.iterations == 1
        assert len(result.steps) == 1
        assert result.steps[0].tool_name == "task_complete"

    def test_agent_resume_restores_prior_token_totals(self, tmp_path):
        store = SessionStore(store_dir=str(tmp_path / "sessions"))
        store.save(
            session_id="resume_tokens",
            task="Continue prior work",
            history=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "continue"},
            ],
            status="active",
            iterations=1,
            tokens=168,
            input_tokens=123,
            output_tokens=45,
        )
        backend = MockBackend([
            ChatResponse(content="done", input_tokens=11, output_tokens=7),
        ])
        tools = ToolRegistry(working_dir=str(tmp_path))
        agent = Agent(
            backend=backend,
            tools=tools,
            max_iterations=5,
            session_store=store,
        )

        result = agent.resume("resume_tokens")

        assert result.success
        assert result.total_input_tokens == 134
        assert result.total_output_tokens == 52
        assert result.tokens == 186

    def test_agent_resume_missing_session_releases_running_flag(self, tmp_path):
        store = SessionStore(store_dir=str(tmp_path / "sessions"))
        backend = MockBackend([
            ChatResponse(content="fresh run", input_tokens=3, output_tokens=2),
        ])
        tools = ToolRegistry(working_dir=str(tmp_path))
        agent = Agent(
            backend=backend,
            tools=tools,
            max_iterations=5,
            session_store=store,
        )

        with pytest.raises(FileNotFoundError):
            agent.resume("missing-session")

        result = agent.run("start over")

        assert result.success
        assert result.summary == "fresh run"


# ===========================================================================
# TestGoalQueue
# ===========================================================================

class TestGoalQueue:
    """Tests for GoalQueue persistence and priority ordering."""

    def test_add_and_list(self, tmp_path):
        path = str(tmp_path / "goals.json")
        q = GoalQueue(persist_path=path)
        q.add("Goal A", "Desc A")
        q.add("Goal B", "Desc B")
        assert len(q.list()) == 2
        assert q.list()[0].title == "Goal A"

    def test_next_by_priority(self, tmp_path):
        path = str(tmp_path / "goals.json")
        q = GoalQueue(persist_path=path)
        q.add("Low priority", "desc", priority=10)
        q.add("High priority", "desc", priority=1)
        q.add("Medium priority", "desc", priority=5)
        nxt = q.next()
        assert nxt is not None
        assert nxt.title == "High priority"
        assert nxt.priority == 1

    def test_complete_goal(self, tmp_path):
        path = str(tmp_path / "goals.json")
        q = GoalQueue(persist_path=path)
        g = q.add("Task", "Do the thing")
        assert g.status == PENDING
        q.complete(g.id, "Done successfully", tokens_used=500)
        updated = q.get(g.id)
        assert updated.status == COMPLETED
        assert updated.result_summary == "Done successfully"
        assert updated.tokens_used == 500
        assert updated.completed_at is not None

    def test_fail_goal(self, tmp_path):
        path = str(tmp_path / "goals.json")
        q = GoalQueue(persist_path=path)
        g = q.add("Risky task", "Might fail")
        q.fail(g.id, "Out of memory")
        updated = q.get(g.id)
        assert updated.status == FAILED
        assert "Out of memory" in updated.result_summary

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "goals.json")
        q1 = GoalQueue(persist_path=path)
        q1.add("Alpha", "first", priority=2)
        q1.add("Beta", "second", priority=1)
        q1.complete(q1.list()[0].id, "Alpha done")

        # Load fresh instance from same file
        q2 = GoalQueue(persist_path=path)
        assert len(q2.list()) == 2
        assert q2.list()[0].title == "Alpha"
        assert q2.list()[0].status == COMPLETED
        nxt = q2.next()
        assert nxt is not None
        assert nxt.title == "Beta"
