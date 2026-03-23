"""
Surrogate-Assisted Evaluation (Sprint 30)
-------------------------------------------
Lightweight scoring model that estimates eval scores without running the
full eval pipeline. Used to pre-filter candidates before expensive
LLM-based evaluation.

Based on:
- Surrogate-assisted evolutionary optimization (SEC 2019)
- AlphaEvolve (DeepMind 2025): cheap surrogate filters before expensive eval
- GigaEvo (2025): surrogate scoring for async DAG evaluation

Key insight: Full evaluation (LLM inference + retrieval) costs 5-30 seconds
per candidate. A surrogate model using historical eval data can estimate
scores in microseconds, filtering out clearly bad candidates before they
reach the expensive eval pipeline.

Architecture:
  1. COLLECT  -> store (features, true_score) pairs from real evaluations
  2. FIT      -> train a simple linear model on accumulated data
  3. PREDICT  -> estimate score for new candidates from features alone
  4. FILTER   -> only pass candidates above a surrogate threshold to real eval
  5. UPDATE   -> after real eval, add new data point and periodically refit
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.sqlite_owner import SQLiteConnectionOwner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(config: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric features from a config dict for surrogate scoring.

    Converts a config dict into a flat dict of float features. Handles:
    - Numeric values directly
    - Boolean as 0.0/1.0
    - String length as feature
    - List length as feature
    - Nested dicts are flattened with dot notation
    """
    features: Dict[str, float] = {}
    _flatten(config, "", features)
    return features


def _flatten(obj: Any, prefix: str, out: Dict[str, float]) -> None:
    """Recursively flatten a nested dict to float features."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            _flatten(v, key, out)
    elif isinstance(obj, bool):
        out[prefix] = 1.0 if obj else 0.0
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    elif isinstance(obj, str):
        out[prefix + "_len"] = float(len(obj))
    elif isinstance(obj, (list, tuple)):
        out[prefix + "_len"] = float(len(obj))


# ---------------------------------------------------------------------------
# Linear surrogate model
# ---------------------------------------------------------------------------

@dataclass
class SurrogateWeights:
    """Learned weights for the linear surrogate."""
    bias: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    r_squared: float = 0.0
    fitted_at: float = 0.0


class SurrogateModel:
    """Simple linear regression surrogate for fast candidate scoring.

    Learns a mapping from config features to eval scores using ordinary
    least squares. Fast to train (pure Python, no numpy needed), fast
    to predict (dot product).
    """

    def __init__(self):
        self._weights = SurrogateWeights()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_samples(self) -> int:
        return self._weights.n_samples

    @property
    def r_squared(self) -> float:
        return self._weights.r_squared

    def fit(
        self,
        feature_matrix: List[Dict[str, float]],
        scores: List[float],
    ) -> float:
        """Fit the surrogate on historical (features, score) pairs.

        Uses regularized least squares (ridge regression) with feature
        standardization. Returns R-squared on training data.

        Args:
            feature_matrix: List of feature dicts from extract_features()
            scores: Corresponding true eval scores

        Returns:
            R-squared value (0.0 to 1.0)
        """
        if len(feature_matrix) < 3 or len(scores) < 3:
            self._fitted = False
            return 0.0

        n = len(scores)

        # Collect all feature names
        all_features = set()
        for fm in feature_matrix:
            all_features.update(fm.keys())
        feature_names = sorted(all_features)

        if not feature_names:
            self._fitted = False
            return 0.0

        # Compute means and stds for standardization
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for fname in feature_names:
            vals = [fm.get(fname, 0.0) for fm in feature_matrix]
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
            means[fname] = mean
            stds[fname] = math.sqrt(var) if var > 0 else 1.0

        # Build standardized X matrix (list of lists)
        X = []
        for fm in feature_matrix:
            row = []
            for fname in feature_names:
                raw = fm.get(fname, 0.0)
                row.append((raw - means[fname]) / stds[fname])
            X.append(row)

        y_mean = sum(scores) / n
        y = [s - y_mean for s in scores]

        # Ridge regression: w = (X^T X + lambda I)^-1 X^T y
        # For small feature sets, solve normal equations directly
        d = len(feature_names)
        ridge_lambda = 0.01 * n  # regularization strength

        # X^T X + lambda I
        XtX = [[0.0] * d for _ in range(d)]
        for row in X:
            for i in range(d):
                for j in range(d):
                    XtX[i][j] += row[i] * row[j]
        for i in range(d):
            XtX[i][i] += ridge_lambda

        # X^T y
        Xty = [0.0] * d
        for row_idx, row in enumerate(X):
            for i in range(d):
                Xty[i] += row[i] * y[row_idx]

        # Solve via Gaussian elimination
        w = _solve_linear(XtX, Xty)

        if w is None:
            self._fitted = False
            return 0.0

        # Store weights
        self._weights = SurrogateWeights(
            bias=y_mean,
            weights={fname: w[i] for i, fname in enumerate(feature_names)},
            feature_means=means,
            feature_stds=stds,
            n_samples=n,
            fitted_at=time.time(),
        )

        # Compute R-squared
        ss_res = 0.0
        ss_tot = 0.0
        for row_idx, row in enumerate(X):
            pred = sum(row[i] * w[i] for i in range(d))
            ss_res += (y[row_idx] - pred) ** 2
            ss_tot += y[row_idx] ** 2

        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        self._weights.r_squared = max(0.0, min(1.0, r2))
        self._fitted = True

        return self._weights.r_squared

    def predict(self, features: Dict[str, float]) -> float:
        """Predict eval score from features.

        Returns estimated score, or 0.5 (neutral) if not fitted.
        """
        if not self._fitted:
            return 0.5

        w = self._weights
        score = w.bias
        for fname, weight in w.weights.items():
            raw = features.get(fname, 0.0)
            mean = w.feature_means.get(fname, 0.0)
            std = w.feature_stds.get(fname, 1.0)
            score += weight * (raw - mean) / std

        return max(0.0, min(1.0, score))

    def predict_config(self, config: Dict[str, Any]) -> float:
        """Convenience: extract features and predict in one call."""
        return self.predict(extract_features(config))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize weights for storage."""
        w = self._weights
        return {
            "bias": w.bias,
            "weights": w.weights,
            "feature_means": w.feature_means,
            "feature_stds": w.feature_stds,
            "n_samples": w.n_samples,
            "r_squared": w.r_squared,
            "fitted_at": w.fitted_at,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load weights from dict."""
        self._weights = SurrogateWeights(
            bias=data.get("bias", 0.0),
            weights=data.get("weights", {}),
            feature_means=data.get("feature_means", {}),
            feature_stds=data.get("feature_stds", {}),
            n_samples=data.get("n_samples", 0),
            r_squared=data.get("r_squared", 0.0),
            fitted_at=data.get("fitted_at", 0.0),
        )
        self._fitted = bool(self._weights.weights)


# ---------------------------------------------------------------------------
# Linear solver (Gaussian elimination, no numpy)
# ---------------------------------------------------------------------------

def _solve_linear(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Returns None if the system is singular.
    """
    n = len(b)
    # Augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row

        if max_val < 1e-12:
            return None  # Singular

        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        if abs(aug[row][row]) < 1e-12:
            return None
        x[row] = aug[row][n]
        for j in range(row + 1, n):
            x[row] -= aug[row][j] * x[j]
        x[row] /= aug[row][row]

    return x


# ---------------------------------------------------------------------------
# Surrogate-assisted evaluation store
# ---------------------------------------------------------------------------

_STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_history (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_json TEXT NOT NULL,
    features_json TEXT NOT NULL,
    true_score REAL NOT NULL,
    surrogate_score REAL DEFAULT NULL,
    timestamp REAL NOT NULL
);
"""


class SurrogateEvalStore:
    """Persistent store for evaluation history + surrogate model.

    Accumulates (config, features, score) triples from real evaluations.
    Periodically refits the surrogate model to predict scores from features.
    """

    def __init__(
        self,
        db_path: str | Path = "_surrogate/eval_store.db",
        refit_interval: int = 10,
        min_samples: int = 5,
    ):
        """
        Args:
            db_path: SQLite path for persistent history
            refit_interval: Refit surrogate every N new samples
            min_samples: Minimum samples before fitting
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = SQLiteConnectionOwner(self._db_path)
        self._refit_interval = refit_interval
        self._min_samples = max(3, min_samples)
        self._since_last_fit = 0
        self.model = SurrogateModel()

        conn = self._owner.connect()
        conn.executescript(_STORE_SCHEMA)
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        return self._owner.connect()

    def record(self, config: Dict[str, Any], true_score: float) -> float:
        """Record a real evaluation result and return surrogate prediction.

        Also triggers refit if enough new samples have accumulated.
        """
        features = extract_features(config)
        surrogate_score = self.model.predict(features) if self.model.is_fitted else None

        conn = self._conn()
        conn.execute(
            "INSERT INTO eval_history "
            "(config_json, features_json, true_score, surrogate_score, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                json.dumps(config, default=str),
                json.dumps(features, default=str),
                true_score,
                surrogate_score,
                time.time(),
            ),
        )
        conn.commit()

        self._since_last_fit += 1
        if self._since_last_fit >= self._refit_interval:
            self.refit()

        return surrogate_score if surrogate_score is not None else true_score

    def predict(self, config: Dict[str, Any]) -> float:
        """Get surrogate score estimate for a config."""
        return self.model.predict_config(config)

    def should_evaluate(self, config: Dict[str, Any], threshold: float = 0.3) -> bool:
        """Pre-filter: should this config go through expensive real eval?

        Returns True if surrogate is not fitted or predicts above threshold.
        """
        if not self.model.is_fitted:
            return True  # no surrogate yet, evaluate everything
        return self.model.predict_config(config) >= threshold

    def refit(self) -> float:
        """Refit the surrogate model on all accumulated data.

        Returns R-squared of the fit.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT features_json, true_score FROM eval_history "
            "ORDER BY timestamp DESC LIMIT 1000"
        ).fetchall()

        if len(rows) < self._min_samples:
            return 0.0

        feature_matrix = [json.loads(r[0]) for r in rows]
        scores = [r[1] for r in rows]

        r2 = self.model.fit(feature_matrix, scores)
        self._since_last_fit = 0

        log.info(
            "Surrogate refit: R^2=%.3f on %d samples",
            r2, len(rows),
        )
        return r2

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        conn = self._conn()
        total = (conn.execute(
            "SELECT COUNT(*) FROM eval_history"
        ).fetchone() or (0,))[0]
        avg_score = (conn.execute(
            "SELECT AVG(true_score) FROM eval_history"
        ).fetchone() or (0,))[0]

        return {
            "total_evaluations": total,
            "avg_true_score": round(avg_score or 0.0, 3),
            "surrogate_fitted": self.model.is_fitted,
            "surrogate_r_squared": round(self.model.r_squared, 3),
            "surrogate_n_samples": self.model.n_samples,
        }

    def close(self) -> None:
        self._owner.close()
