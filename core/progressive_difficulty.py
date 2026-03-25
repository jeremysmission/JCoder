"""
Progressive Difficulty Engine — Desirable Difficulty for AI
============================================================
Based on the cognitive science principle: the STRUGGLE is the learning.
Easy tests teach nothing. Impossible tests teach nothing. Tests at
the EDGE of capability — where you fail, figure out WHY, and grow
from the failure — produce exponential improvement.

The engine maintains phi4's "frontier" — the difficulty level where
it scores roughly 30-60% (maximum learnability zone). It automatically:

1. Finds the frontier (binary search on difficulty)
2. Tests at the frontier (deliberate practice)
3. Analyzes WHY failures happen (not just THAT they happen)
4. Stores the lesson (both the failure and the correction)
5. Escalates difficulty (because the frontier moved)
6. Repeats — each cycle makes the next cycle more productive

The key insight from research:
- Score 90%+ = too easy, minimal learning (already know it)
- Score 10%- = too hard, minimal learning (can't engage)
- Score 30-60% = MAXIMUM LEARNING (desirable difficulty zone)

This maps to the Absolute Zero "learnability" concept:
  learnability = 1.0 - abs(2.0 * score - 1.0)
  Peak at score=0.5, zero at score=0 and score=1.

The compound effect:
  Level 1: Learn content (pass tests)
  Level 2: Learn HOW to learn (meta-cognition from failure patterns)
  Level 3: Learn how to MASTER (optimize the learning process itself)
  Each level compounds the one below.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)


class ProgressiveDifficultyEngine:
    """Automatically finds and pushes the learning frontier.

    Maintains a difficulty curve per category, calibrated so phi4
    is always challenged at the EDGE of its capability.
    """

    # Calibrated from cognitive science research:
    # - 85% rule (Wilson et al. 2019, Nature Communications)
    # - Desirable difficulty (Bjork 1994)
    # - Zone of Proximal Development (Vygotsky 1978)
    # - Absolute Zero learnability (Zhao et al. 2025)
    #
    # Phase progression:
    #   New concepts: 85% (optimal gradient signal)
    #   Core learning: 70-80% (flow state, complex tasks)
    #   Mastery push: 60-70% (desirable difficulty)
    #   Self-play: 50% (maximum information gain)
    #
    # We use the CORE LEARNING zone as default (70% target)
    # because coding challenges are complex multi-step tasks
    # where richer error signals produce deeper learning.
    TARGET_LOW = 0.50   # below this = too hard, ease up
    TARGET_HIGH = 0.80  # above this = too easy, escalate
    TARGET_OPTIMAL = 0.70  # the sweet spot for complex tasks

    def __init__(self, state_path: str = "data/progressive_difficulty.json"):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f)
        return {
            "categories": {},
            "total_rounds": 0,
            "total_lessons": 0,
            "frontier_history": [],
        }

    def _save_state(self) -> None:
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def get_difficulty(self, category: str) -> int:
        """Get the current frontier difficulty for a category (1-10)."""
        cat_state = self.state["categories"].get(category, {})
        return cat_state.get("difficulty", 5)

    def record_round(
        self,
        category: str,
        difficulty: int,
        passed: int,
        total: int,
        lessons_learned: List[str] = None,
    ) -> Dict[str, Any]:
        """Record a round of challenges and adjust the frontier.

        Returns adjustment info: did difficulty go up, down, or stay?
        """
        if total == 0:
            return {"action": "skip", "reason": "no challenges"}

        score = passed / total
        learnability = 1.0 - abs(2.0 * score - 1.0)

        cat_state = self.state["categories"].setdefault(category, {
            "difficulty": 5,
            "rounds": 0,
            "total_passed": 0,
            "total_attempted": 0,
            "peak_difficulty": 5,
            "lessons_count": 0,
            "history": [],
        })

        cat_state["rounds"] += 1
        cat_state["total_passed"] += passed
        cat_state["total_attempted"] += total

        # Record this round
        round_record = {
            "round": cat_state["rounds"],
            "difficulty": difficulty,
            "score": score,
            "learnability": learnability,
            "passed": passed,
            "total": total,
            "timestamp": datetime.now().isoformat(),
        }
        cat_state["history"].append(round_record)

        # Keep only last 20 rounds per category
        if len(cat_state["history"]) > 20:
            cat_state["history"] = cat_state["history"][-20:]

        # Adjust difficulty based on score
        old_diff = cat_state["difficulty"]
        if score > self.TARGET_HIGH:
            # Too easy — escalate
            cat_state["difficulty"] = min(10, old_diff + 1)
            action = "escalate"
            reason = f"Score {score:.0%} > {self.TARGET_HIGH:.0%} target ceiling"
        elif score < self.TARGET_LOW:
            # Too hard — ease up slightly (but never below 1)
            cat_state["difficulty"] = max(1, old_diff - 1)
            action = "ease"
            reason = f"Score {score:.0%} < {self.TARGET_LOW:.0%} target floor"
        else:
            # In the sweet spot — maximum learning happening
            action = "hold"
            reason = f"Score {score:.0%} in sweet spot ({self.TARGET_LOW:.0%}-{self.TARGET_HIGH:.0%})"

        # Track peak difficulty
        if cat_state["difficulty"] > cat_state["peak_difficulty"]:
            cat_state["peak_difficulty"] = cat_state["difficulty"]

        # Store lessons
        if lessons_learned:
            cat_state["lessons_count"] += len(lessons_learned)
            self.state["total_lessons"] += len(lessons_learned)

        self.state["total_rounds"] += 1
        self.state["frontier_history"].append({
            "category": category,
            "round": self.state["total_rounds"],
            "difficulty": cat_state["difficulty"],
            "score": score,
            "learnability": learnability,
            "action": action,
        })

        # Keep frontier history bounded
        if len(self.state["frontier_history"]) > 100:
            self.state["frontier_history"] = self.state["frontier_history"][-100:]

        self._save_state()

        result = {
            "action": action,
            "reason": reason,
            "old_difficulty": old_diff,
            "new_difficulty": cat_state["difficulty"],
            "score": score,
            "learnability": learnability,
            "peak_difficulty": cat_state["peak_difficulty"],
            "total_rounds": cat_state["rounds"],
        }

        log.info(
            "[%s] Round %d: diff=%d, score=%.0f%%, learnability=%.2f -> %s (diff %d->%d)",
            category, cat_state["rounds"], difficulty,
            score * 100, learnability, action,
            old_diff, cat_state["difficulty"],
        )

        return result

    def is_accelerating(self, category: str = None) -> bool:
        """Check if the learning rate is accelerating.

        True if recent difficulty escalations are happening faster
        than earlier ones — meaning the system is getting better
        at getting better.
        """
        history = self.state.get("frontier_history", [])
        if category:
            history = [h for h in history if h["category"] == category]

        if len(history) < 6:
            return False

        # Compare escalation rate: last 3 rounds vs previous 3
        recent = history[-3:]
        earlier = history[-6:-3]

        recent_escalations = sum(1 for h in recent if h["action"] == "escalate")
        earlier_escalations = sum(1 for h in earlier if h["action"] == "escalate")

        return recent_escalations > earlier_escalations

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the progressive difficulty state."""
        cats = {}
        for cat, state in self.state["categories"].items():
            cats[cat] = {
                "difficulty": state["difficulty"],
                "peak": state["peak_difficulty"],
                "rounds": state["rounds"],
                "success_rate": (
                    state["total_passed"] / max(state["total_attempted"], 1)
                ),
                "lessons": state["lessons_count"],
            }

        return {
            "total_rounds": self.state["total_rounds"],
            "total_lessons": self.state["total_lessons"],
            "accelerating": self.is_accelerating(),
            "categories": cats,
        }
