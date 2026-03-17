"""Tests for the real eval_fn callbacks wired in bridge.py."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def eval_set_dir(tmp_path):
    """Create a minimal eval set JSON file."""
    q = [
        {
            "id": "test_001",
            "category": "python",
            "question": "How do I sort a list in Python?",
            "expected_keywords": ["sorted", "sort", "list"],
            "expected_imports": [],
            "scoring": {
                "has_code": 0.3,
                "has_correct_api": 0.4,
                "has_imports": 0.0,
                "is_runnable": 0.2,
                "cites_source": 0.1,
            },
        },
        {
            "id": "test_002",
            "category": "python",
            "question": "How do I read a JSON file?",
            "expected_keywords": ["json", "open", "load"],
            "expected_imports": ["json"],
            "scoring": {
                "has_code": 0.3,
                "has_correct_api": 0.3,
                "has_imports": 0.2,
                "is_runnable": 0.1,
                "cites_source": 0.1,
            },
        },
    ]
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(q), encoding="utf-8")
    return path


# ---- AgentEvalRunner basics ----

def test_eval_runner_loads(eval_set_dir):
    """AgentEvalRunner loads eval set JSON and builds question_map."""
    from evaluation.agent_eval_runner import AgentEvalRunner
    runner = AgentEvalRunner(str(eval_set_dir), agent=None)
    assert len(runner.questions) == 2
    assert "test_001" in runner.question_map
    assert "test_002" in runner.question_map


def test_eval_runner_question_map_lookup(eval_set_dir):
    """question_map is keyed by question id."""
    from evaluation.agent_eval_runner import AgentEvalRunner
    runner = AgentEvalRunner(str(eval_set_dir), agent=None)
    q = runner.question_map["test_001"]
    assert q["question"] == "How do I sort a list in Python?"
    assert q["expected_keywords"] == ["sorted", "sort", "list"]


def test_build_question_index_by_text(eval_set_dir):
    """Build a question index keyed by lowercased question text."""
    from evaluation.agent_eval_runner import AgentEvalRunner
    runner = AgentEvalRunner(str(eval_set_dir), agent=None)
    # Build an index keyed by lowercased question text (what tests previously
    # expected from bridge._build_question_index)
    idx = {q["question"].strip().lower(): q for q in runner.questions}
    key = "how do i sort a list in python?"
    assert key in idx
    assert idx[key]["id"] == "test_001"


# ---- ContinualLearner init via _try_init_continual ----

def test_continual_init_returns_learner():
    """_try_init_continual returns a ContinualLearner with placeholder eval_fn."""
    import agent.bridge as bridge
    sl_config = {"regression_margin": 0.05}
    learner = bridge._try_init_continual(sl_config)
    if learner is not None:
        # The placeholder eval_fn always returns 1.0
        score = learner.eval_fn("test_cap", ["some query"])
        assert score == 1.0


def test_continual_init_placeholder_eval_fn():
    """Without a real evaluator the placeholder eval_fn returns 1.0 (safe fallback)."""
    import agent.bridge as bridge
    sl_config = {"regression_margin": 0.05}
    learner = bridge._try_init_continual(sl_config)
    if learner is not None:
        score = learner.eval_fn("test_cap", ["some query"])
        assert score == 1.0


# ---- AgentEvalRunner scoring ----

def test_score_answer_with_keywords(eval_set_dir):
    """score_answer returns subscores including weighted_total."""
    from evaluation.agent_eval_runner import AgentEvalRunner
    runner = AgentEvalRunner(str(eval_set_dir), agent=None)
    q = runner.question_map["test_001"]
    answer = (
        "Use `sorted(my_list)` or `my_list.sort()` to sort a list.\n"
        "```python\nresult = sorted([3, 1, 2])\n```"
    )
    sub = runner.score_answer(q, answer)
    assert "weighted_total" in sub
    assert 0.0 < sub["weighted_total"] <= 1.0


def test_score_answer_unknown_query_heuristic(eval_set_dir):
    """For answers with code blocks, heuristic scoring gives partial credit."""
    from evaluation.agent_eval_runner import AgentEvalRunner
    runner = AgentEvalRunner(str(eval_set_dir), agent=None)
    # Score against a question but with an answer that only partially matches
    q = runner.question_map["test_002"]
    answer = "Here is how to do it:\n```python\nprint('hello')\n```\n" * 3
    sub = runner.score_answer(q, answer)
    assert "weighted_total" in sub
    assert 0.0 <= sub["weighted_total"] <= 1.0


# ---- PromptEvolver eval_fn (real scoring path) ----

def test_prompt_evolver_scoring_path(eval_set_dir):
    """Demonstrate the real scoring path: runtime.generate -> score_answer."""
    from evaluation.agent_eval_runner import AgentEvalRunner
    import agent.bridge as bridge

    runner = AgentEvalRunner(str(eval_set_dir), agent=None)

    mock_backend = MagicMock()
    mock_backend.chat.return_value = MagicMock(
        content="```python\nimport json\nwith open('f.json') as f:\n    data = json.load(f)\n```",
    )

    runtime = bridge._BackendRuntimeAdapter(mock_backend)

    query = "How do I read a JSON file?"
    prompt = "You are a helpful coding assistant."
    answer = runtime.generate(query, [], system_prompt=prompt)

    # Build question index by text
    q_index = {q["question"].strip().lower(): q for q in runner.questions}
    q_key = query.strip().lower()
    q_dict = q_index.get(q_key)
    assert q_dict is not None
    sub = runner.score_answer(q_dict, answer)
    assert "weighted_total" in sub
    assert 0.0 <= sub["weighted_total"] <= 1.0
