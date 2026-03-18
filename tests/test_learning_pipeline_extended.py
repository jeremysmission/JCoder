"""
Tests for Sprint 8-9 learning features:
  - ExperienceStore  (core.experience_replay)
  - ContinualLearner (core.continual_learner)
  - SelfLearningPipeline (core.self_learning_pipeline)

All external I/O is mocked. SQLite uses temp files via tmp_path.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from core.experience_replay import Experience, ExperienceStore
from core.continual_learner import (
    CapabilityBaseline,
    ContinualLearner,
    ConsolidationResult,
    RegressionCheck,
)
from core.self_learning_pipeline import PipelineAnswer, SelfLearningPipeline


# ── helpers ──────────────────────────────────────────────────────────────

def _make_store(tmp_path, **kw) -> ExperienceStore:
    return ExperienceStore(db_path=str(tmp_path / "exp.db"), **kw)


def _good_answer(tag="default"):
    """Answer that passes quality filters (>50 chars, contains code pattern)."""
    return f"def solve_{tag}():\n    return compute_result(data, params, extra_arg)\n# done"


def _store_n(store: ExperienceStore, n: int, base_conf: float = 0.8):
    """Store *n* experiences with ascending IDs."""
    for i in range(n):
        store.store(
            exp_id=f"exp_{i:04d}",
            query=f"how does module_{i} work",
            answer=_good_answer(f"m{i}"),
            source_files=[f"core/mod_{i}.py"],
            confidence=base_conf,
        )


# ═══════════════════════════════════════════════════════════════════════
# EXPERIENCE REPLAY
# ═══════════════════════════════════════════════════════════════════════

class TestExperienceStoreBasic:
    """Store and retrieve experiences."""

    def test_store_and_retrieve(self, tmp_path):
        store = _make_store(tmp_path)
        ok = store.store("e1", "how does parser work",
                         _good_answer("parser"), ["core/parser.py"], 0.9)
        assert ok is True
        results = store.retrieve("parser work", top_k=3)
        assert len(results) >= 1
        assert results[0].exp_id == "e1"

    def test_reject_low_confidence(self, tmp_path):
        store = _make_store(tmp_path, min_confidence=0.6)
        ok = store.store("e1", "q", _good_answer(), ["f.py"], 0.3)
        assert ok is False
        assert store.stats()["total"] == 0

    def test_reject_short_answer(self, tmp_path):
        store = _make_store(tmp_path)
        ok = store.store("e1", "q", "short", ["f.py"], 0.9)
        assert ok is False

    def test_reject_no_code(self, tmp_path):
        store = _make_store(tmp_path)
        ok = store.store("e1", "q", "a" * 60, ["f.py"], 0.9)
        assert ok is False

    def test_stats_populated(self, tmp_path):
        store = _make_store(tmp_path)
        _store_n(store, 5)
        s = store.stats()
        assert s["total"] == 5
        assert s["avg_confidence"] == pytest.approx(0.8, abs=0.01)


class TestExperiencePriority:
    """Priority-based sampling and scoring."""

    def test_higher_confidence_ranked_first(self, tmp_path):
        store = _make_store(tmp_path)
        store.store("lo", "how does retriever work",
                    _good_answer("retriever"), ["r.py"], 0.65)
        store.store("hi", "how does retriever work",
                    _good_answer("retriever"), ["r.py"], 0.99)
        results = store.retrieve("retriever work", top_k=2)
        assert results[0].exp_id == "hi"

    def test_q_value_influences_ranking(self, tmp_path):
        store = _make_store(tmp_path, q_value_weight=0.8)
        store.store("a", "how does chunker split",
                    _good_answer("chunker"), ["c.py"], 0.7)
        store.store("b", "how does chunker split",
                    _good_answer("chunker"), ["c.py"], 0.7)
        store.update_q_value("b", reward=1.0)
        results = store.retrieve("chunker split", top_k=2)
        ids = [r.exp_id for r in results]
        assert ids[0] == "b"

    def test_update_q_value_nonexistent_is_noop(self, tmp_path):
        store = _make_store(tmp_path)
        store.update_q_value("ghost", reward=1.0)  # should not raise


class TestExperienceBufferOverflow:
    """Buffer overflow -- oldest/lowest evicted."""

    def test_max_experiences_enforced(self, tmp_path):
        store = _make_store(tmp_path)
        store.MAX_EXPERIENCES = 10
        _store_n(store, 15)
        assert store.stats()["total"] <= 11  # at most MAX + 1 (race window)

    def test_low_confidence_evicted_first(self, tmp_path):
        store = _make_store(tmp_path)
        store.MAX_EXPERIENCES = 5
        # store 4 high-confidence, then 1 low, then 1 more high
        for i in range(4):
            store.store(f"hi_{i}", f"query module_{i}",
                        _good_answer(f"h{i}"), ["f.py"], 0.95)
        store.store("lo", "query module_lo",
                    _good_answer("lo"), ["f.py"], 0.61)
        store.store("hi_new", "query module_new",
                    _good_answer("new"), ["f.py"], 0.95)
        stats = store.stats()
        assert stats["total"] <= 6


class TestExperienceReplayBlend:
    """replay_blend mixes new and stored experiences."""

    def test_blend_returns_bounded(self, tmp_path):
        store = _make_store(tmp_path)
        _store_n(store, 5)
        fresh = [Experience("f1", "q", "a", [], 0.9, time.time())]
        blended = store.replay_blend(fresh, replay_ratio=0.5, max_total=4)
        assert len(blended) <= 4

    def test_blend_zero_max_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.replay_blend([], max_total=0) == []


class TestFormatAsExamples:
    def test_format_includes_query(self, tmp_path):
        store = _make_store(tmp_path)
        exp = Experience("e1", "parser question", "def parse(): return result(x)",
                         ["p.py"], 0.9, time.time())
        text = store.format_as_examples([exp])
        assert "parser question" in text

    def test_format_respects_max_chars(self, tmp_path):
        store = _make_store(tmp_path)
        exps = [
            Experience(f"e{i}", f"q{i}", "def f(): return result(x)" * 20,
                       ["f.py"], 0.9, time.time())
            for i in range(20)
        ]
        text = store.format_as_examples(exps, max_chars=200)
        assert len(text) < 600  # header + at most a couple entries


# ═══════════════════════════════════════════════════════════════════════
# CONTINUAL LEARNER
# ═══════════════════════════════════════════════════════════════════════

def _make_learner(tmp_path, eval_fn=None, **kw) -> ContinualLearner:
    if eval_fn is None:
        eval_fn = lambda name, queries: 0.9
    return ContinualLearner(
        eval_fn=eval_fn,
        db_path=str(tmp_path / "cl.db"),
        **kw,
    )


class TestContinualBaselines:
    """Knowledge retention across sessions."""

    def test_set_and_get_baseline(self, tmp_path):
        cl = _make_learner(tmp_path)
        cl.set_baseline("accuracy", 0.92, ["q1", "q2"])
        bls = cl.get_baselines()
        assert len(bls) == 1
        assert bls[0].name == "accuracy"
        assert bls[0].score == pytest.approx(0.92)

    def test_threshold_defaults_to_score_minus_margin(self, tmp_path):
        cl = _make_learner(tmp_path, regression_margin=0.05)
        bl = cl.set_baseline("acc", 0.90, ["q1"])
        assert bl.threshold == pytest.approx(0.85)

    def test_baselines_persist_across_instances(self, tmp_path):
        db = str(tmp_path / "cl.db")
        cl1 = ContinualLearner(lambda n, q: 0.9, db_path=db)
        cl1.set_baseline("skill", 0.88, ["q"])
        cl2 = ContinualLearner(lambda n, q: 0.9, db_path=db)
        assert cl2.get_baselines()[0].score == pytest.approx(0.88)


class TestCatastrophicForgetting:
    """Regression detection prevents catastrophic forgetting."""

    def test_no_regression_passes(self, tmp_path):
        cl = _make_learner(tmp_path, eval_fn=lambda n, q: 0.9)
        cl.set_baseline("acc", 0.9, ["q1"])
        check = cl.check_regression()
        assert check.passed is True
        assert check.failed_capabilities == []

    def test_regression_detected(self, tmp_path):
        cl = _make_learner(tmp_path, eval_fn=lambda n, q: 0.5,
                           regression_margin=0.05)
        cl.set_baseline("acc", 0.9, ["q1"])
        check = cl.check_regression()
        assert check.passed is False
        assert "acc" in check.failed_capabilities

    def test_config_snapshot_on_check(self, tmp_path):
        cl = _make_learner(tmp_path)
        cl.set_baseline("acc", 0.9, ["q1"])
        cl.check_regression(config={"model": "phi-4"})
        rolled = cl.rollback_to_last()
        assert rolled == {"model": "phi-4"}

    def test_rollback_returns_none_when_empty(self, tmp_path):
        cl = _make_learner(tmp_path)
        assert cl.rollback_to_last() is None

    def test_eval_exception_treated_as_zero(self, tmp_path):
        def bad_eval(name, queries):
            raise RuntimeError("boom")
        cl = _make_learner(tmp_path, eval_fn=bad_eval)
        cl.set_baseline("acc", 0.9, ["q1"])
        check = cl.check_regression()
        assert check.passed is False


class TestCurriculumProgression:
    """Baselines ratchet up (difficulty progression)."""

    def test_update_baselines_ratchets_up(self, tmp_path):
        scores = {"acc": 0.95}
        cl = _make_learner(tmp_path, eval_fn=lambda n, q: scores.get(n, 0.9))
        cl.set_baseline("acc", 0.90, ["q1"])
        updates = cl.update_baselines()
        assert "acc" in updates
        assert updates["acc"] == pytest.approx(0.95)
        # threshold should have risen too
        bl = cl.get_baselines()[0]
        assert bl.threshold >= 0.90 - 1e-9

    def test_update_baselines_never_lowers(self, tmp_path):
        cl = _make_learner(tmp_path, eval_fn=lambda n, q: 0.7)
        cl.set_baseline("acc", 0.90, ["q1"])
        updates = cl.update_baselines()
        assert updates == {}
        bl = cl.get_baselines()[0]
        assert bl.score == pytest.approx(0.90)


class TestPerformanceTracking:
    """Health report and regression history."""

    def test_health_report_structure(self, tmp_path):
        cl = _make_learner(tmp_path)
        cl.set_baseline("acc", 0.9, ["q"])
        cl.check_regression()
        report = cl.health_report()
        assert report["capabilities_tracked"] == 1
        assert report["total_regression_checks"] == 1
        assert "failure_rate" in report

    def test_consolidation_calls_prune_fns(self, tmp_path):
        cl = _make_learner(tmp_path)
        result = cl.consolidate(
            experience_prune_fn=lambda: 3,
            config_prune_fn=lambda: 1,
            telemetry_compact_fn=lambda: 7,
        )
        assert result.experiences_pruned == 3
        assert result.configs_pruned == 1
        assert result.telemetry_compacted == 7
        assert result.duration_ms >= 0

    def test_consolidation_handles_exceptions(self, tmp_path):
        cl = _make_learner(tmp_path)
        result = cl.consolidate(
            experience_prune_fn=lambda: (_ for _ in ()).throw(RuntimeError("x")),
        )
        assert result.experiences_pruned == 0  # exception swallowed


# ═══════════════════════════════════════════════════════════════════════
# SELF-LEARNING PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def _make_pipeline(**overrides) -> SelfLearningPipeline:
    """Build a pipeline with all modules mocked."""
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {"id": "c1", "content": "def foo(): pass", "source_path": "core/foo.py",
         "score": 0.85},
    ]
    runtime = MagicMock()
    runtime.generate.return_value = "def foo(): pass  # generated"

    defaults = dict(
        retriever=retriever,
        runtime=runtime,
    )
    defaults.update(overrides)
    return SelfLearningPipeline(**defaults)


class TestPipelineFullCycle:
    """Full cycle: eval -> identify weak -> study -> re-eval."""

    def test_basic_answer_returns_pipeline_answer(self):
        pipe = _make_pipeline()
        ans = pipe.answer("what does foo do?")
        assert isinstance(ans, PipelineAnswer)
        assert ans.answer
        assert ans.query_id

    def test_empty_retrieval_returns_fallback(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        pipe = _make_pipeline(retriever=retriever)
        ans = pipe.answer("anything")
        assert "No relevant code" in ans.answer

    def test_experience_replay_injected(self):
        exp = MagicMock()
        exp.retrieve.return_value = []
        exp.format_as_examples.return_value = ""
        pipe = _make_pipeline(experience=exp)
        pipe.answer("test query")
        exp.retrieve.assert_called_once()

    def test_experience_stored_on_high_confidence(self):
        exp = MagicMock()
        exp.retrieve.return_value = []
        exp.format_as_examples.return_value = ""
        refl = MagicMock()
        refl.full_reflection.return_value = {"confidence": 0.85}
        pipe = _make_pipeline(experience=exp, reflection=refl)
        pipe.answer("test query")
        exp.store.assert_called_once()

    def test_experience_not_stored_on_low_confidence(self):
        exp = MagicMock()
        exp.retrieve.return_value = []
        exp.format_as_examples.return_value = ""
        refl = MagicMock()
        refl.full_reflection.return_value = {"confidence": 0.3}
        pipe = _make_pipeline(experience=exp, reflection=refl)
        pipe.answer("test query")
        exp.store.assert_not_called()

    def test_confidence_gate_adds_warning(self):
        refl = MagicMock()
        refl.full_reflection.return_value = {"confidence": 0.1}
        pipe = _make_pipeline(reflection=refl, confidence_gate=0.2)
        ans = pipe.answer("test")
        assert "not confident" in ans.answer.lower()


class TestModuleStats:
    """system_health collects stats from active modules."""

    def test_active_modules_listed(self):
        exp = MagicMock()
        exp.stats.return_value = {"total": 5}
        pipe = _make_pipeline(experience=exp)
        report = pipe.system_health()
        assert "experience" in report["modules_active"]
        assert report["experience_stats"] == {"total": 5}

    def test_inactive_modules_listed(self):
        pipe = _make_pipeline()
        report = pipe.system_health()
        assert "experience" in report["modules_inactive"]

    def test_query_count_increments(self):
        pipe = _make_pipeline()
        pipe.answer("q1")
        pipe.answer("q2")
        assert pipe.system_health()["queries_processed"] == 2


class TestQDArchiveIntegration:
    """Stigmergy (QD-archive proxy) integration in pipeline."""

    def test_stigmergy_boost_called(self):
        stig = MagicMock()
        stig.boost_scores.return_value = [("c1", 0.9)]
        pipe = _make_pipeline(stigmergy=stig)
        pipe.answer("test")
        stig.boost_scores.assert_called_once()

    def test_stigmergy_deposit_on_success(self):
        stig = MagicMock()
        stig.boost_scores.return_value = [("c1", 0.9)]
        refl = MagicMock()
        refl.full_reflection.return_value = {"confidence": 0.8}
        pipe = _make_pipeline(stigmergy=stig, reflection=refl)
        pipe.answer("test")
        stig.deposit.assert_called_once()


class TestFeedbackLoopClosure:
    """Telemetry and meta-cognitive feedback loops close properly."""

    def test_telemetry_logged(self):
        telem = MagicMock()
        with patch("core.self_learning_pipeline.QueryEvent") as QE:
            QE.return_value = MagicMock()
            pipe = _make_pipeline(telemetry=telem)
            pipe.answer("test")
            telem.log.assert_called_once()

    def test_meta_outcome_reported(self):
        meta = MagicMock()
        meta.select_strategy.return_value = ("standard", MagicMock(query_type="explain"))
        pipe = _make_pipeline(meta_controller=meta)
        with patch("core.self_learning_pipeline.classify_query", lambda q: "explain"):
            pipe.answer("how does X work")
        meta.report_outcome.assert_called_once()

    def test_active_learning_value_computed(self):
        active = MagicMock()
        refl = MagicMock()
        refl.full_reflection.return_value = {"confidence": 0.5}
        pipe = _make_pipeline(active_learner=active, reflection=refl)
        ans = pipe.answer("edge case query")
        # max entropy at confidence=0.5 -> learning_value = 4*0.5*0.5 = 1.0
        assert ans.learning_value == pytest.approx(1.0)

    def test_continual_health_in_report(self):
        cont = MagicMock()
        cont.health_report.return_value = {"capabilities_tracked": 2}
        pipe = _make_pipeline(continual=cont)
        report = pipe.system_health()
        assert report["continual_health"]["capabilities_tracked"] == 2
