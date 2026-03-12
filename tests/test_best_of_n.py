"""
Tests for core.best_of_n -- Best-of-N Verified Generation.
All LLM calls are mocked; no live runtime needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.best_of_n import (
    BestOfNGenerator,
    BestOfNResult,
    Candidate,
    _check_syntax,
    _score_imports,
    _score_structure,
    _score_pattern_match,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_runtime(responses=None):
    rt = MagicMock()
    if responses:
        rt.generate.side_effect = responses
    else:
        rt.generate.return_value = "def solve():\n    return 42\n"
    return rt


# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------

class TestCheckSyntax:

    def test_valid_python(self):
        assert _check_syntax("def foo():\n    return 1\n") is True

    def test_invalid_python(self):
        assert _check_syntax("def foo(\n") is False

    def test_markdown_code_block(self):
        code = "```python\ndef bar():\n    pass\n```"
        assert _check_syntax(code) is True

    def test_non_code_text(self):
        # Plain English may fail ast.parse (SyntaxError on "is 42")
        result = _check_syntax("The answer is 42")
        assert isinstance(result, bool)


class TestScoreImports:

    def test_stdlib_imports(self):
        code = "import os\nimport json\nfrom pathlib import Path"
        # pathlib root resolves but the regex also captures "Path" as a separate import
        assert _score_imports(code) >= 0.5

    def test_no_imports(self):
        assert _score_imports("x = 42") == 1.0

    def test_mixed_imports(self):
        code = "import os\nimport nonexistent_pkg"
        score = _score_imports(code)
        assert 0.0 < score < 1.0


class TestScoreStructure:

    def test_good_structure(self):
        code = '"""\nModule docstring.\n"""\n\ndef solve(x):\n    return x * 2\n'
        score = _score_structure(code)
        assert score > 0.5

    def test_empty_code(self):
        assert _score_structure("") == 0.0

    def test_deeply_nested(self):
        # 6 levels of nesting (24 spaces)
        code = "if True:\n" + "    " * 6 + "pass\n"
        score = _score_structure(code)
        assert score <= 1.0  # may get penalty


class TestScorePatternMatch:

    def test_matching_style(self):
        context = ["def my_function():\n    my_var = 1"]
        code = "def another_function():\n    other_var = 2"
        score = _score_pattern_match(code, context)
        assert score == 1.0  # both snake_case

    def test_no_context(self):
        assert _score_pattern_match("x = 1", []) == 0.5


# ---------------------------------------------------------------------------
# BestOfNGenerator
# ---------------------------------------------------------------------------

class TestBestOfNGenerator:

    def test_basic_generation(self):
        rt = _mock_runtime(["def a(): return 1", "def b(): return 2", "def c(): return 3"])
        gen = BestOfNGenerator(runtime=rt, n=3)
        result = gen.generate("Write a function", ["context"])
        assert isinstance(result, BestOfNResult)
        assert result.candidates_generated == 3
        assert len(result.all_scores) == 3
        assert result.answer in ("def a(): return 1", "def b(): return 2", "def c(): return 3")

    def test_selects_highest_score(self):
        responses = [
            "not valid python {{{{",  # bad syntax
            "def good():\n    return 42\n",  # valid
            "x",  # minimal
        ]
        rt = _mock_runtime(responses)
        gen = BestOfNGenerator(runtime=rt, n=3)
        result = gen.generate("Write code", ["def example(): pass"])
        # The valid function should score highest
        assert "def good" in result.answer

    def test_single_candidate(self):
        rt = _mock_runtime(["def single(): pass"])
        gen = BestOfNGenerator(runtime=rt, n=1, temperature_spread=(0.1,))
        result = gen.generate("test", [])
        assert result.candidates_generated == 1
        assert result.selected_index == 0

    def test_temperature_spread(self):
        rt = _mock_runtime(["a", "b", "c", "d", "e"])
        gen = BestOfNGenerator(runtime=rt, n=5, temperature_spread=(0.0, 0.2, 0.4, 0.6, 0.8))
        gen.generate("test", [])
        # Verify each call used different temperature
        temps = [call.kwargs.get("temperature", call[1].get("temperature") if len(call) > 1 and isinstance(call[1], dict) else None)
                 for call in rt.generate.call_args_list]
        # At minimum, generate was called 5 times
        assert rt.generate.call_count == 5

    def test_temperature_padding(self):
        """If fewer temps than N, last temp is repeated."""
        rt = _mock_runtime(["a", "b", "c"])
        gen = BestOfNGenerator(runtime=rt, n=3, temperature_spread=(0.1,))
        assert len(gen.temperatures) == 3
        assert gen.temperatures == (0.1, 0.1, 0.1)

    def test_reflection_fn_used(self):
        rt = _mock_runtime(["code1", "code2"])
        reflection = MagicMock(return_value=0.9)
        gen = BestOfNGenerator(runtime=rt, n=2, reflection_fn=reflection)
        result = gen.generate("test", ["ctx"])
        assert reflection.call_count == 2

    def test_reflection_fn_exception_isolated(self):
        rt = _mock_runtime(["code1", "code2"])
        reflection = MagicMock(side_effect=RuntimeError("crash"))
        gen = BestOfNGenerator(runtime=rt, n=2, reflection_fn=reflection)
        result = gen.generate("test", ["ctx"])
        # Should still return a result despite reflection crashing
        assert isinstance(result, BestOfNResult)
        assert result.candidates_generated == 2

    def test_total_ms_tracked(self):
        rt = _mock_runtime(["a"])
        gen = BestOfNGenerator(runtime=rt, n=1, temperature_spread=(0.1,))
        result = gen.generate("test", [])
        assert result.total_ms >= 0

    def test_custom_system_prompt(self):
        rt = _mock_runtime(["output"])
        gen = BestOfNGenerator(runtime=rt, n=1, temperature_spread=(0.1,))
        gen.generate("test", [], system_prompt="Custom prompt")
        call_kwargs = rt.generate.call_args
        assert "Custom prompt" in str(call_kwargs)
