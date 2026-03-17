"""Tests for cascade answer confidence (merged into core.cascade)."""

from __future__ import annotations

import pytest

from core.cascade import estimate_answer_confidence


# ---------------------------------------------------------------------------
# Answer confidence estimation
# ---------------------------------------------------------------------------

class TestEstimateAnswerConfidence:

    def test_empty_answer(self):
        assert estimate_answer_confidence("") == 0.0

    def test_short_answer(self):
        c = estimate_answer_confidence("I don't know.")
        assert c < 0.3

    def test_code_answer(self):
        answer = (
            "Here's how to sort a list:\n"
            "```python\ndef sort_list(items):\n    return sorted(items)\n```\n"
            "This uses Python's built-in `sorted()` function."
        )
        c = estimate_answer_confidence(answer)
        assert c > 0.5

    def test_hedging_penalized(self):
        confident = "Use `sorted()` to sort a list. It returns a new list."
        hedging = "Maybe you could perhaps try sorted(), but I'm not sure."
        assert estimate_answer_confidence(confident) > estimate_answer_confidence(hedging)

    def test_specific_references_rewarded(self):
        generic = "You should use the library function for this."
        specific = "Call `os.path.join()` and `pathlib.Path.resolve()` to handle paths."
        assert estimate_answer_confidence(specific) >= estimate_answer_confidence(generic)

    def test_bounded_zero_one(self):
        assert 0.0 <= estimate_answer_confidence("x") <= 1.0
        long_hedge = "maybe " * 100 + "I don't know " * 50
        assert estimate_answer_confidence(long_hedge) >= 0.0
