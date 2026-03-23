"""
Adaptive Research -- Data Models & Support Classes
---------------------------------------------------
Data classes, UCB1 source bandit, and yield tracker used by the
adaptive research engine. Split from adaptive_research.py for the
500 LOC-per-module rule.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    """Result of fetching from a single source."""
    source_name: str
    query: str
    papers_found: int
    novel_papers: int  # papers not seen before
    avg_novelty: float
    best_novelty: float
    fetch_time_ms: float
    timestamp: float = 0.0


@dataclass
class QueryPerformance:
    """Tracks how well a query term performs across sources."""
    query_term: str
    times_used: int = 0
    total_novel_papers: int = 0
    avg_novelty_yield: float = 0.0
    best_paper_title: str = ""
    last_used: float = 0.0


@dataclass
class SnowballResult:
    """Result of citation snowball traversal."""
    seed_paper: str
    papers_found: int
    forward_refs: int  # papers that cite the seed
    backward_refs: int  # papers the seed cites
    novel_found: int
    depth_reached: int


@dataclass
class SynthesisResult:
    """Result of cross-paper synthesis."""
    paper_a: str
    paper_b: str
    novel_combination: str
    implementation_idea: str
    confidence: float
    timestamp: float = 0.0


@dataclass
class AdaptiveStats:
    """Overall adaptive research statistics."""
    total_fetches: int
    total_papers_seen: int
    total_novel_papers: int
    source_rankings: List[Dict[str, Any]]
    best_queries: List[Dict[str, Any]]
    synthesis_count: int
    snowball_depth_avg: float
    yield_trend: str  # "improving", "stable", "declining"


# ---------------------------------------------------------------------------
# UCB1 Source Bandit
# ---------------------------------------------------------------------------

class SourceBandit:
    """
    UCB1 multi-armed bandit for research source selection.

    Each source is an arm. Reward = novelty yield per fetch.
    UCB1 balances exploitation (best sources) with exploration
    (under-sampled sources).

    UCB1 score = avg_reward + C * sqrt(ln(total_pulls) / arm_pulls)
    """

    def __init__(self, exploration_weight: float = 1.41):
        self.c = exploration_weight
        self._arms: Dict[str, Dict[str, float]] = {}
        self._total_pulls = 0

    def register_arm(self, name: str) -> None:
        if name not in self._arms:
            self._arms[name] = {
                "pulls": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
                "max_reward": 0.0,
            }

    def select(self, k: int = 3) -> List[str]:
        """Select top-k sources to fetch from using UCB1."""
        if not self._arms:
            return []

        # Ensure every arm is tried at least once
        untried = [
            name for name, data in self._arms.items()
            if data["pulls"] == 0
        ]
        if untried:
            return untried[:k]

        # UCB1 scoring
        scores = {}
        for name, data in self._arms.items():
            avg = data["avg_reward"]
            exploration = self.c * math.sqrt(
                math.log(max(1, self._total_pulls)) / max(1, data["pulls"])
            )
            scores[name] = avg + exploration

        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:k]

    def update(self, name: str, reward: float) -> None:
        """Update arm with observed reward."""
        if name not in self._arms:
            self.register_arm(name)

        arm = self._arms[name]
        arm["pulls"] += 1
        arm["total_reward"] += reward
        arm["avg_reward"] = arm["total_reward"] / arm["pulls"]
        arm["max_reward"] = max(arm["max_reward"], reward)
        self._total_pulls += 1

    def rankings(self) -> List[Dict[str, Any]]:
        """Return sources ranked by UCB1 score."""
        if not self._arms or self._total_pulls == 0:
            return []

        results = []
        for name, data in self._arms.items():
            if data["pulls"] > 0:
                ucb = data["avg_reward"] + self.c * math.sqrt(
                    math.log(max(1, self._total_pulls)) / max(1, data["pulls"])
                )
            else:
                ucb = float("inf")

            results.append({
                "source": name,
                "avg_reward": round(data["avg_reward"], 4),
                "pulls": data["pulls"],
                "ucb_score": round(ucb, 4),
                "max_reward": round(data["max_reward"], 4),
            })

        results.sort(key=lambda x: x["ucb_score"], reverse=True)
        return results

    def to_dict(self) -> Dict:
        return {
            "arms": self._arms,
            "total_pulls": self._total_pulls,
            "c": self.c,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SourceBandit":
        b = cls(exploration_weight=data.get("c", 1.41))
        b._arms = data.get("arms", {})
        b._total_pulls = data.get("total_pulls", 0)
        return b


# ---------------------------------------------------------------------------
# Yield Tracker
# ---------------------------------------------------------------------------

class YieldTracker:
    """
    Tracks research yield over time to detect improving/declining trends.

    Yield = novel_papers_found / total_papers_fetched (per session)
    Tracks running averages to detect if the research process
    is getting better or worse.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._yields: List[Tuple[float, float]] = []  # (timestamp, yield)

    def record(self, novel_found: int, total_fetched: int) -> None:
        y = novel_found / max(1, total_fetched)
        self._yields.append((time.time(), y))
        # Keep bounded
        if len(self._yields) > self.window_size * 3:
            self._yields = self._yields[-self.window_size * 2:]

    def trend(self) -> str:
        """Detect yield trend: 'improving', 'stable', 'declining'."""
        if len(self._yields) < 4:
            return "insufficient_data"

        recent = [y for _, y in self._yields[-self.window_size:]]
        mid = len(recent) // 2
        first_half_avg = sum(recent[:mid]) / max(1, mid)
        second_half_avg = sum(recent[mid:]) / max(1, len(recent) - mid)

        diff = second_half_avg - first_half_avg
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"

    def current_yield(self) -> float:
        if not self._yields:
            return 0.0
        recent = [y for _, y in self._yields[-5:]]
        return sum(recent) / len(recent)

    def to_dict(self) -> Dict:
        return {"yields": self._yields, "window": self.window_size}

    @classmethod
    def from_dict(cls, data: Dict) -> "YieldTracker":
        t = cls(window_size=data.get("window", 20))
        t._yields = data.get("yields", [])
        return t
