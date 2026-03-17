"""
Tests for core.surrogate_scorer -- Surrogate-Assisted Evaluation (Sprint 30).
Verifies feature extraction, linear surrogate model, persistent store,
and pre-filtering logic.
"""

from __future__ import annotations

import math
import pytest

from core.surrogate_scorer import (
    SurrogateEvalStore,
    SurrogateModel,
    SurrogateWeights,
    _solve_linear,
    extract_features,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:

    def test_numeric_values(self):
        features = extract_features({"x": 1, "y": 2.5})
        assert features["x"] == 1.0
        assert features["y"] == 2.5

    def test_boolean_values(self):
        features = extract_features({"enabled": True, "debug": False})
        assert features["enabled"] == 1.0
        assert features["debug"] == 0.0

    def test_string_becomes_length(self):
        features = extract_features({"name": "hello"})
        assert features["name_len"] == 5.0

    def test_list_becomes_length(self):
        features = extract_features({"items": [1, 2, 3]})
        assert features["items_len"] == 3.0

    def test_nested_dict(self):
        features = extract_features({"model": {"temp": 0.7, "top_k": 5}})
        assert features["model.temp"] == 0.7
        assert features["model.top_k"] == 5.0

    def test_empty_dict(self):
        features = extract_features({})
        assert features == {}

    def test_mixed_types(self):
        features = extract_features({
            "val": 42,
            "flag": True,
            "name": "test",
            "tags": [1, 2],
            "nested": {"a": 1.0},
        })
        assert features["val"] == 42.0
        assert features["flag"] == 1.0
        assert features["name_len"] == 4.0
        assert features["tags_len"] == 2.0
        assert features["nested.a"] == 1.0


# ---------------------------------------------------------------------------
# Linear solver
# ---------------------------------------------------------------------------

class TestLinearSolver:

    def test_simple_2x2(self):
        # 2x + 3y = 8, x + y = 3 => x=1, y=2
        A = [[2, 3], [1, 1]]
        b = [8, 3]
        x = _solve_linear(A, b)
        assert x is not None
        assert abs(x[0] - 1.0) < 1e-6
        assert abs(x[1] - 2.0) < 1e-6

    def test_identity(self):
        A = [[1, 0], [0, 1]]
        b = [5, 7]
        x = _solve_linear(A, b)
        assert x is not None
        assert abs(x[0] - 5.0) < 1e-6
        assert abs(x[1] - 7.0) < 1e-6

    def test_singular_returns_none(self):
        A = [[1, 2], [2, 4]]  # rank 1
        b = [3, 6]
        x = _solve_linear(A, b)
        assert x is None

    def test_3x3(self):
        A = [[1, 1, 1], [0, 2, 5], [2, 5, -1]]
        b = [6, -4, 27]
        x = _solve_linear(A, b)
        assert x is not None
        assert abs(x[0] - 5.0) < 1e-6
        assert abs(x[1] - 3.0) < 1e-6
        assert abs(x[2] - (-2.0)) < 1e-6


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

class TestSurrogateModel:

    def test_unfitted_returns_neutral(self):
        model = SurrogateModel()
        assert not model.is_fitted
        assert model.predict({"x": 1.0}) == 0.5

    def test_fit_simple_linear(self):
        """y = 0.1 * x (roughly) should be learnable."""
        model = SurrogateModel()
        features = [{"x": float(i)} for i in range(20)]
        scores = [0.1 * i for i in range(20)]
        r2 = model.fit(features, scores)
        assert r2 > 0.9  # near-perfect linear fit
        assert model.is_fitted
        assert model.n_samples == 20

    def test_predict_after_fit(self):
        model = SurrogateModel()
        features = [{"x": float(i)} for i in range(10)]
        scores = [0.05 * i for i in range(10)]
        model.fit(features, scores)

        # Predict near middle of training range
        pred = model.predict({"x": 5.0})
        assert 0.1 < pred < 0.5  # should be in reasonable range

    def test_fit_too_few_samples(self):
        model = SurrogateModel()
        r2 = model.fit([{"x": 1.0}], [0.5])
        assert r2 == 0.0
        assert not model.is_fitted

    def test_predict_config(self):
        model = SurrogateModel()
        features = [{"x": float(i)} for i in range(10)]
        scores = [0.05 * i for i in range(10)]
        model.fit(features, scores)

        pred = model.predict_config({"x": 5})
        assert 0.0 <= pred <= 1.0

    def test_to_dict_from_dict_roundtrip(self):
        model = SurrogateModel()
        features = [{"x": float(i), "y": float(i * 2)} for i in range(10)]
        scores = [0.05 * i + 0.01 * i for i in range(10)]
        model.fit(features, scores)

        data = model.to_dict()
        assert "bias" in data
        assert "weights" in data

        model2 = SurrogateModel()
        model2.from_dict(data)
        assert model2.is_fitted
        assert model2.n_samples == 10

        # Predictions should match
        pred1 = model.predict({"x": 3.0, "y": 6.0})
        pred2 = model2.predict({"x": 3.0, "y": 6.0})
        assert abs(pred1 - pred2) < 1e-6

    def test_predictions_clamped(self):
        """Predictions should be clamped to [0, 1]."""
        model = SurrogateModel()
        features = [{"x": float(i)} for i in range(10)]
        scores = [0.1 * i for i in range(10)]
        model.fit(features, scores)

        # Extreme input should still be clamped
        pred = model.predict({"x": 1000.0})
        assert 0.0 <= pred <= 1.0

        pred = model.predict({"x": -1000.0})
        assert 0.0 <= pred <= 1.0

    def test_multi_feature_fit(self):
        """Model should handle multiple features."""
        model = SurrogateModel()
        features = [
            {"x": float(i), "y": float(i * 0.5), "z": float(i % 3)}
            for i in range(20)
        ]
        scores = [0.03 * i + 0.02 * (i * 0.5) for i in range(20)]
        r2 = model.fit(features, scores)
        assert r2 > 0.5
        assert len(model._weights.weights) == 3


# ---------------------------------------------------------------------------
# Evaluation store
# ---------------------------------------------------------------------------

class TestSurrogateEvalStore:

    def test_record_and_stats(self, tmp_path):
        store = SurrogateEvalStore(
            db_path=tmp_path / "test.db",
            refit_interval=100,
            min_samples=3,
        )
        store.record({"val": 10}, 0.8)
        store.record({"val": 20}, 0.9)

        stats = store.stats()
        assert stats["total_evaluations"] == 2
        assert stats["avg_true_score"] > 0.0
        assert not stats["surrogate_fitted"]
        store.close()

    def test_auto_refit(self, tmp_path):
        store = SurrogateEvalStore(
            db_path=tmp_path / "test.db",
            refit_interval=5,
            min_samples=3,
        )
        for i in range(6):
            store.record({"val": float(i * 10)}, 0.1 * i)

        # After 5+ records, surrogate should have been refit
        assert store.model.is_fitted
        store.close()

    def test_predict_before_fit(self, tmp_path):
        store = SurrogateEvalStore(
            db_path=tmp_path / "test.db",
            min_samples=3,
        )
        pred = store.predict({"val": 50})
        assert pred == 0.5  # unfitted default
        store.close()

    def test_should_evaluate_unfitted(self, tmp_path):
        store = SurrogateEvalStore(
            db_path=tmp_path / "test.db",
            min_samples=3,
        )
        # Without a fitted model, should always evaluate
        assert store.should_evaluate({"val": 0}) is True
        store.close()

    def test_should_evaluate_fitted(self, tmp_path):
        store = SurrogateEvalStore(
            db_path=tmp_path / "test.db",
            refit_interval=5,
            min_samples=3,
        )
        # Add enough data to fit
        for i in range(6):
            store.record({"val": float(i * 10)}, 0.1 * i)

        # Config with high predicted score should pass
        assert store.should_evaluate({"val": 50.0}, threshold=0.0) is True
        store.close()

    def test_manual_refit(self, tmp_path):
        store = SurrogateEvalStore(
            db_path=tmp_path / "test.db",
            refit_interval=1000,  # won't auto-refit
            min_samples=3,
        )
        for i in range(10):
            store.record({"val": float(i * 10)}, 0.05 * i)

        assert not store.model.is_fitted
        r2 = store.refit()
        assert r2 > 0.0
        assert store.model.is_fitted
        store.close()

    def test_refit_too_few_samples(self, tmp_path):
        store = SurrogateEvalStore(
            db_path=tmp_path / "test.db",
            min_samples=10,
        )
        store.record({"val": 1}, 0.5)
        r2 = store.refit()
        assert r2 == 0.0
        store.close()


# ---------------------------------------------------------------------------
# GATE TEST
# ---------------------------------------------------------------------------

class TestGateSurrogate:
    """
    Gate test: surrogate should learn a meaningful mapping after
    enough training data, and successfully pre-filter bad candidates.
    """

    def test_surrogate_learns_linear_trend(self, tmp_path):
        """After training on linear data, surrogate predictions should
        correlate with true scores."""
        store = SurrogateEvalStore(
            db_path=tmp_path / "gate.db",
            refit_interval=100,
            min_samples=3,
        )

        # Generate training data: score = 0.01 * val (roughly)
        for i in range(30):
            store.record({"val": float(i * 3)}, min(1.0, 0.01 * i * 3))

        store.refit()
        assert store.model.is_fitted

        # Low val should predict lower than high val
        low_pred = store.predict({"val": 10.0})
        high_pred = store.predict({"val": 80.0})
        assert high_pred > low_pred

        store.close()
