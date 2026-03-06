"""Tests for agent.prompts -- system prompts, FIM formatting, PromptBuilder."""

import pytest

from agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    CODE_EXPLAIN_PROMPT,
    CODE_QA_PROMPT,
    CODE_REVIEW_PROMPT,
    DEBUG_PROMPT,
    REFACTOR_PROMPT,
    FIMFormatter,
    PromptBuilder,
)

ALL_PROMPTS = [
    AGENT_SYSTEM_PROMPT,
    CODE_QA_PROMPT,
    CODE_REVIEW_PROMPT,
    CODE_EXPLAIN_PROMPT,
    DEBUG_PROMPT,
    REFACTOR_PROMPT,
]


# -------------------------------------------------------------------
# System prompt constants
# -------------------------------------------------------------------

class TestSystemPrompts:
    def test_all_prompts_are_strings(self):
        for p in ALL_PROMPTS:
            assert isinstance(p, str)
            assert len(p) > 0

    def test_agent_prompt_has_tool_instructions(self):
        assert "memory_search" in AGENT_SYSTEM_PROMPT
        assert "task_complete" in AGENT_SYSTEM_PROMPT

    def test_qa_prompt_mentions_context(self):
        assert "context" in CODE_QA_PROMPT.lower()
        assert "I don't" in CODE_QA_PROMPT

    def test_review_prompt_mentions_security(self):
        text = CODE_REVIEW_PROMPT.lower()
        assert "owasp" in text or "security" in text

    def test_debug_prompt_mentions_root_cause(self):
        text = DEBUG_PROMPT.lower()
        assert "root cause" in text or "error" in text


# -------------------------------------------------------------------
# FIM formatting
# -------------------------------------------------------------------

class TestFIMFormatter:
    def test_devstral_format(self):
        fmt = FIMFormatter("devstral")
        tokens = FIMFormatter.FORMATS["devstral"]
        assert tokens["prefix"] == "[PREFIX]"
        assert tokens["suffix"] == "[SUFFIX]"
        assert tokens["middle"] == "[MIDDLE]"

    def test_codellama_format(self):
        fmt = FIMFormatter("codellama")
        tokens = FIMFormatter.FORMATS["codellama"]
        assert tokens["prefix"] == "<PRE>"
        assert tokens["suffix"] == " <SUF>"
        assert tokens["middle"] == " <MID>"

    def test_starcoder_format(self):
        fmt = FIMFormatter("starcoder")
        tokens = FIMFormatter.FORMATS["starcoder"]
        assert tokens["prefix"] == "<fim_prefix>"
        assert tokens["suffix"] == "<fim_suffix>"
        assert tokens["middle"] == "<fim_middle>"

    def test_deepseek_format(self):
        tokens = FIMFormatter.FORMATS["deepseek"]
        assert tokens["prefix"] == "<|fim_begin|>"
        assert tokens["suffix"] == "<|fim_hole|>"
        assert tokens["middle"] == "<|fim_end|>"

    def test_generic_fallback(self):
        fmt = FIMFormatter("unknown_model_xyz")
        # Should silently fall back to generic
        result = fmt.format_completion("pre", "suf")
        assert "### PREFIX" in result
        assert "### SUFFIX" in result
        assert "### MIDDLE" in result

    def test_format_completion(self):
        # Devstral puts suffix before prefix
        fmt = FIMFormatter("devstral")
        result = fmt.format_completion("def hello():", "    return 42")
        assert result == "[SUFFIX]    return 42[PREFIX]def hello():[MIDDLE]"

        # Non-devstral: prefix, suffix, middle order
        fmt2 = FIMFormatter("starcoder")
        result2 = fmt2.format_completion("def hello():", "    return 42")
        assert result2 == "<fim_prefix>def hello():<fim_suffix>    return 42<fim_middle>"

    def test_extract_completion(self):
        fmt = FIMFormatter("devstral")
        raw = "[PREFIX]some junk[MIDDLE]\n    x = 1\n"
        cleaned = fmt.extract_completion(raw)
        assert "[PREFIX]" not in cleaned
        assert "[MIDDLE]" not in cleaned
        assert "x = 1" in cleaned

    def test_supported_formats(self):
        formats = FIMFormatter.supported_formats()
        assert isinstance(formats, list)
        assert len(formats) >= 5
        for name in ("devstral", "codellama", "starcoder", "deepseek", "generic"):
            assert name in formats


# -------------------------------------------------------------------
# PromptBuilder
# -------------------------------------------------------------------

class TestPromptBuilder:
    def test_agent_mode(self):
        pb = PromptBuilder("agent")
        msgs = pb.build_messages("Fix the bug")
        assert msgs[0]["role"] == "system"
        assert "JCoder" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert "Fix the bug" in msgs[1]["content"]

    def test_qa_mode_with_context(self):
        pb = PromptBuilder("qa")
        msgs = pb.build_messages("What does foo do?", context="def foo(): pass")
        user_text = msgs[1]["content"]
        assert "Context:" in user_text
        assert "def foo(): pass" in user_text

    def test_review_mode_with_code(self):
        pb = PromptBuilder("review")
        msgs = pb.build_messages("Review this", code="x = eval(input())")
        user_text = msgs[1]["content"]
        assert "```" in user_text
        assert "eval(input())" in user_text

    def test_debug_mode_with_error(self):
        pb = PromptBuilder("debug")
        msgs = pb.build_messages("Help", error="ZeroDivisionError: division by zero")
        user_text = msgs[1]["content"]
        assert "Error:" in user_text
        assert "ZeroDivisionError" in user_text

    def test_fim_mode(self):
        pb = PromptBuilder("fim")
        msgs = pb.build_messages("", fim_prefix="def add(a, b):", fim_suffix="    return result")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "[PREFIX]" in msgs[0]["content"]
        assert "[SUFFIX]" in msgs[0]["content"]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            PromptBuilder("nonexistent_mode")

    def test_available_modes(self):
        modes = PromptBuilder.available_modes()
        assert len(modes) == 7
        for m in ("agent", "qa", "review", "explain", "debug", "refactor", "fim"):
            assert m in modes

    def test_messages_format(self):
        for mode in ("agent", "qa", "review", "explain", "debug", "refactor"):
            pb = PromptBuilder(mode)
            msgs = pb.build_messages("test query")
            for msg in msgs:
                assert "role" in msg
                assert "content" in msg
                assert msg["role"] in ("system", "user")
                assert isinstance(msg["content"], str)
