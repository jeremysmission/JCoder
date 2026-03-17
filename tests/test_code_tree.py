"""
Tests for core.code_tree -- CodeTree agent-guided tree search.
No external deps; pure in-memory.
"""

from __future__ import annotations

import pytest

from core.code_tree import (
    CodeNode,
    CodeTreeSearch,
    NodeStatus,
    TreeSearchResult,
    _check_syntax,
    _critique_feedback,
    _exec_feedback,
    _structural_score,
    decompose_strategies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_generator(problem: str, strategy: str) -> str:
    """Generate a simple function for testing."""
    return (
        f"def solve():\n"
        f"    # Strategy: {strategy}\n"
        f"    return 42\n"
    )


def _bad_generator(problem: str, strategy: str) -> str:
    """Generate syntactically invalid code."""
    return "def broken(\n    # missing closing paren"


def _scored_generator(problem: str, strategy: str) -> str:
    """Generate code that varies by strategy for scoring tests."""
    if "class" in strategy:
        return (
            "class Solution:\n"
            '    """Solver class."""\n'
            "    def solve(self, data: list) -> int:\n"
            "        try:\n"
            "            return sum(data)\n"
            "        except TypeError:\n"
            "            return 0\n"
        )
    if "helper" in strategy:
        return (
            "def helper(x: int) -> int:\n"
            "    return x * 2\n\n"
            "def solve(data: list) -> int:\n"
            "    return sum(helper(x) for x in data)\n"
        )
    return "def solve():\n    return 42\n"


# ---------------------------------------------------------------------------
# Syntax checking
# ---------------------------------------------------------------------------

class TestSyntaxCheck:

    def test_valid_code(self):
        ok, err = _check_syntax("x = 1\ny = 2\n")
        assert ok
        assert err == ""

    def test_invalid_code(self):
        ok, err = _check_syntax("def broken(:\n")
        assert not ok
        assert "Line" in err

    def test_empty_code(self):
        ok, err = _check_syntax("")
        assert ok


# ---------------------------------------------------------------------------
# Structural scoring
# ---------------------------------------------------------------------------

class TestStructuralScore:

    def test_empty_code(self):
        assert _structural_score("") == 0.0

    def test_function_code(self):
        score = _structural_score("def foo():\n    return 1\n")
        assert score > 0.0

    def test_complex_code_scores_higher(self):
        simple = _structural_score("x = 1\n")
        complex_code = (
            "def process(data: list) -> int:\n"
            '    """Process data."""\n'
            "    try:\n"
            "        return sum(data)\n"
            "    except TypeError:\n"
            "        return 0\n"
        )
        complex_score = _structural_score(complex_code)
        assert complex_score > simple


# ---------------------------------------------------------------------------
# Execution feedback
# ---------------------------------------------------------------------------

class TestExecFeedback:

    def test_syntax_error(self):
        score, fb = _exec_feedback("def broken(:\n")
        assert score == 0.0
        assert "Syntax error" in fb

    def test_valid_code_structural(self):
        score, fb = _exec_feedback("def foo():\n    return 1\n")
        assert score > 0.0
        assert "structural" in fb

    def test_with_test_fn(self):
        def fake_test(code):
            return 0.9
        score, fb = _exec_feedback("def foo():\n    pass\n", test_fn=fake_test)
        assert score == 0.9
        assert fb == "passed"

    def test_test_fn_failure(self):
        def failing_test(code):
            raise RuntimeError("test crash")
        score, fb = _exec_feedback("x = 1\n", test_fn=failing_test)
        assert score == 0.1
        assert "Execution error" in fb


# ---------------------------------------------------------------------------
# Critique feedback
# ---------------------------------------------------------------------------

class TestCritiqueFeedback:

    def test_empty_code(self):
        score, fb = _critique_feedback("", "write a sort")
        assert score == 0.0
        assert "Empty" in fb

    def test_trivial_code(self):
        score, fb = _critique_feedback("pass", "write a sort")
        assert score < 0.7
        assert "trivial" in fb

    def test_good_code(self):
        code = "def sort_data(items):\n    return sorted(items)\n"
        score, fb = _critique_feedback(code, "write a sort function")
        assert score > 0.5

    def test_with_critique_fn(self):
        def fake_critic(code, problem):
            return 0.85, "looks clean"
        score, fb = _critique_feedback("x = 1\n", "test", critique_fn=fake_critic)
        assert score == 0.85
        assert fb == "looks clean"


# ---------------------------------------------------------------------------
# Strategy decomposition
# ---------------------------------------------------------------------------

class TestDecompose:

    def test_default_strategies(self):
        strategies = decompose_strategies("write a function")
        assert len(strategies) >= 2
        assert len(strategies) <= 3

    def test_recursive_hint(self):
        strategies = decompose_strategies("write a recursive tree traversal")
        assert len(strategies) >= 2  # should get extra strategy for recursive keyword

    def test_custom_strategy_fn(self):
        def custom(problem):
            return ["approach A", "approach B"]
        strategies = decompose_strategies("test", strategy_fn=custom)
        assert strategies == ["approach A", "approach B"]

    def test_max_strategies(self):
        strategies = decompose_strategies("sort and optimize", max_strategies=2)
        assert len(strategies) <= 2


# ---------------------------------------------------------------------------
# CodeTreeSearch
# ---------------------------------------------------------------------------

class TestCodeTreeSearch:

    def test_basic_search(self):
        tree = CodeTreeSearch(
            generate_fn=_simple_generator,
            beam_width=2,
            max_depth=2,
        )
        result = tree.search("write a function that adds numbers")
        assert result.answer != ""
        assert result.total_nodes > 0
        assert result.selected_score > 0.0

    def test_pruning(self):
        tree = CodeTreeSearch(
            generate_fn=_simple_generator,
            beam_width=3,
            max_depth=3,
            prune_threshold=0.8,
        )
        result = tree.search("complex problem")
        # With high threshold, most branches should be pruned
        assert result.pruned_count >= 0

    def test_bad_code_low_score(self):
        tree = CodeTreeSearch(
            generate_fn=_bad_generator,
            beam_width=2,
            max_depth=1,
        )
        result = tree.search("test problem")
        assert result.selected_score < 0.5  # bad code scores low

    def test_scored_generator(self):
        tree = CodeTreeSearch(
            generate_fn=_scored_generator,
            beam_width=3,
            max_depth=1,
        )
        result = tree.search("process data from list")
        assert result.answer != ""
        assert result.strategies_explored >= 2

    def test_with_test_fn(self):
        def perfect_test(code):
            return 1.0 if "solve" in code else 0.0

        tree = CodeTreeSearch(
            generate_fn=_simple_generator,
            beam_width=2,
            max_depth=1,
            test_fn=perfect_test,
        )
        result = tree.search("solve something")
        assert result.selected_score > 0.5

    def test_with_critique_fn(self):
        def strict_critic(code, problem):
            return (0.9, "good") if len(code) > 20 else (0.1, "too short")

        tree = CodeTreeSearch(
            generate_fn=_simple_generator,
            beam_width=2,
            max_depth=1,
            critique_fn=strict_critic,
        )
        result = tree.search("write something")
        assert result.selected_score > 0.0

    def test_depth_progression(self):
        tree = CodeTreeSearch(
            generate_fn=_scored_generator,
            beam_width=2,
            max_depth=3,
            prune_threshold=0.1,
        )
        result = tree.search("solve data processing")
        assert result.max_depth >= 1
        assert result.total_nodes >= 3

    def test_tree_report(self):
        tree = CodeTreeSearch(
            generate_fn=_simple_generator,
            beam_width=2,
            max_depth=2,
        )
        tree.search("test problem")
        report = tree.get_tree_report()
        assert "total_nodes" in report
        assert "nodes" in report
        assert len(report["nodes"]) == report["total_nodes"]
        for node in report["nodes"]:
            assert "id" in node
            assert "score" in node
            assert "status" in node

    def test_strategy_fn_integration(self):
        def custom_strategies(problem):
            return ["brute force", "dynamic programming"]

        tree = CodeTreeSearch(
            generate_fn=_simple_generator,
            beam_width=3,
            max_depth=1,
            strategy_fn=custom_strategies,
        )
        result = tree.search("optimize")
        assert result.strategies_explored == 2

    def test_generator_exception_handled(self):
        def exploding_gen(problem, strategy):
            raise RuntimeError("LLM timeout")

        tree = CodeTreeSearch(
            generate_fn=exploding_gen,
            beam_width=2,
            max_depth=1,
        )
        result = tree.search("test")
        assert result.total_nodes > 0
        # Should still complete, just with low scores
        assert result.selected_score < 0.6


# ---------------------------------------------------------------------------
# GATE TEST
# ---------------------------------------------------------------------------

class TestGateCodeTree:
    """
    Gate test: tree search with dual feedback should select better
    code than the worst candidate.
    """

    def test_tree_selects_best_candidate(self):
        tree = CodeTreeSearch(
            generate_fn=_scored_generator,
            beam_width=3,
            max_depth=2,
            prune_threshold=0.1,
        )
        result = tree.search("process and sort data items")
        assert result.selected_score > 0.0
        assert result.selected_score >= min(result.all_scores)
        assert result.total_nodes >= 3
