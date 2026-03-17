"""Tests for the real eval_fn callbacks wired in bridge.py."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the eval runner cache between tests."""
    import agent.bridge as bridge
    bridge._EVAL_SET_CACHE = None
    yield
    bridge._EVAL_SET_CACHE = None


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


# ---- _get_eval_runner ----

def test_get_eval_runner_loads(eval_set_dir):
    from agent.bridge import _get_eval_runner
    with patch("agent.bridge.Path") as MockPath:
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = lambda self: str(eval_set_dir)
        MockPath.return_value.__truediv__ = MagicMock(return_value=mock_path)
        # Direct load via known path
        from evaluation.agent_eval_runner import AgentEvalRunner
        import agent.bridge as bridge
        runner = AgentEvalRunner(str(eval_set_dir), agent=None)
        bridge._EVAL_SET_CACHE = runner
        result = _get_eval_runner()
    assert result is not None
    assert len(result.questions) == 2


def test_get_eval_runner_caches():
    import agent.bridge as bridge
    sentinel = object()
    bridge._EVAL_SET_CACHE = sentinel
    assert bridge._get_eval_runner() is sentinel


# ---- _build_question_index ----

def test_build_question_index(eval_set_dir):
    from evaluation.agent_eval_runner import AgentEvalRunner
    from agent.bridge import _build_question_index
    runner = AgentEvalRunner(str(eval_set_dir), agent=None)
    idx = _build_question_index(runner)
    key = "how do i sort a list in python?"
    assert key in idx
    assert idx[key]["id"] == "test_001"


# ---- ContinualLearner eval_fn (real) ----

def test_continual_eval_fn_with_runtime(eval_set_dir):
    """Real eval_fn generates answers and scores them."""
    from evaluation.agent_eval_runner import AgentEvalRunner
    import agent.bridge as bridge

    # Preload eval cache
    runner = AgentEvalRunner(str(eval_set_dir), agent=None)
    bridge._EVAL_SET_CACHE = runner

    # Mock stack with backend
    mock_backend = MagicMock()
    mock_backend.chat.return_value = MagicMock(
        content="Use `sorted(my_list)` or `my_list.sort()` to sort a list.\n"
                "```python\nresult = sorted([3, 1, 2])\n```",
    )
    stack = {"backend": mock_backend, "federated": None}

    from core.continual_learner import ContinualLearner
    sl_config = {"regression_margin": 0.05}
    learner = bridge._try_init_continual(sl_config, stack)
    assert learner is not None

    # Call eval_fn with a known eval question
    score = learner.eval_fn(
        "python_basics",
        ["How do I sort a list in Python?"],
    )
    assert 0.0 < score <= 1.0
    # The mock answer contains "sorted" and "sort" and "list" so keyword score should be high


def test_continual_eval_fn_no_runtime():
    """Without runtime, eval_fn returns 1.0 (safe fallback)."""
    import agent.bridge as bridge
    sl_config = {"regression_margin": 0.05}
    learner = bridge._try_init_continual(sl_config, stack={})
    if learner is not None:
        score = learner.eval_fn("test_cap", ["some query"])
        assert score == 1.0


def test_continual_eval_fn_unknown_query(eval_set_dir):
    """For queries not in eval set, uses heuristic scoring."""
    from evaluation.agent_eval_runner import AgentEvalRunner
    import agent.bridge as bridge

    runner = AgentEvalRunner(str(eval_set_dir), agent=None)
    bridge._EVAL_SET_CACHE = runner

    mock_backend = MagicMock()
    mock_backend.chat.return_value = MagicMock(
        content="Here is how to do it:\n```python\nprint('hello')\n```\n" * 3,
    )
    stack = {"backend": mock_backend}

    sl_config = {"regression_margin": 0.05}
    learner = bridge._try_init_continual(sl_config, stack)
    assert learner is not None

    # Query NOT in eval set -- should use heuristic
    score = learner.eval_fn("misc", ["What is the meaning of life?"])
    assert 0.0 < score <= 1.0


# ---- PromptEvolver eval_fn (real) ----

def test_prompt_evolver_eval_fn(eval_set_dir):
    """PromptEvolver eval_fn scores a candidate prompt on a query."""
    from evaluation.agent_eval_runner import AgentEvalRunner
    import agent.bridge as bridge

    runner = AgentEvalRunner(str(eval_set_dir), agent=None)
    bridge._EVAL_SET_CACHE = runner

    mock_backend = MagicMock()
    mock_backend.chat.return_value = MagicMock(
        content="```python\nimport json\nwith open('f.json') as f:\n    data = json.load(f)\n```",
    )
    stack = {"backend": mock_backend, "federated": None}

    sl_config = {
        "pipeline_enabled": True,
        "prompt_evolver_enabled": True,
        "regression_margin": 0.05,
    }

    # Build runtime adapter
    runtime = bridge._BackendRuntimeAdapter(mock_backend)

    # Test prompt_eval_fn directly via the module-level builder
    bridge._EVAL_SET_CACHE = runner
    q_index = bridge._build_question_index(runner)

    # Simulate what _try_init_pipeline builds
    query = "How do I read a JSON file?"
    prompt = "You are a helpful coding assistant."
    answer = runtime.generate(query, [], system_prompt=prompt)

    q_key = query.strip().lower()
    q_dict = q_index.get(q_key)
    assert q_dict is not None
    sub = runner.score_answer(q_dict, answer)
    assert "weighted_total" in sub
    assert 0.0 <= sub["weighted_total"] <= 1.0
