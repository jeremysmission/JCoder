"""
Autonomous Research Lab (Sprint 26)
--------------------------------------
JCoder identifies research gaps autonomously from eval failures,
proposes hypotheses from cross-domain synthesis, and implements
prototypes during evolution cycles.

Architecture:
  1. GapDetector        -- analyzes eval failures to find research gaps
  2. HypothesisEngine   -- proposes hypotheses from cross-domain synthesis
  3. PrototypeRunner     -- implements and tests prototype techniques
  4. ResearchLedger      -- audit trail of autonomous research cycles
  5. AutonomousLab       -- top-level orchestrator

Gate: JCoder independently discovers and implements a novel technique
      that improves benchmark scores.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.sqlite_owner import SQLiteConnectionOwner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ResearchStatus(str, Enum):
    PROPOSED = "proposed"
    INVESTIGATING = "investigating"
    PROTOTYPING = "prototyping"
    EVALUATING = "evaluating"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class ResearchGap:
    """A detected gap in current system capabilities."""
    gap_id: str
    category: str  # "accuracy", "coverage", "robustness", "speed", "novel"
    description: str
    evidence: List[str] = field(default_factory=list)
    severity: float = 0.0  # 0-1, higher = more impactful
    detected_at: float = field(default_factory=time.time)
    source: str = ""  # "eval_failure", "pattern_analysis", "cross_domain"


@dataclass
class Hypothesis:
    """A proposed technique to address a research gap."""
    hypothesis_id: str
    gap_id: str
    title: str
    description: str
    technique_type: str = ""  # "retrieval", "generation", "scoring", "pipeline", "novel"
    expected_improvement: float = 0.0
    confidence: float = 0.0
    inspiration_sources: List[str] = field(default_factory=list)
    proposed_at: float = field(default_factory=time.time)
    status: str = ResearchStatus.PROPOSED


@dataclass
class PrototypeResult:
    """Result from running a prototype implementation."""
    prototype_id: str
    hypothesis_id: str
    baseline_score: float
    prototype_score: float
    improvement: float = 0.0
    implementation_notes: str = ""
    passed_gate: bool = False
    evaluated_at: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchCycle:
    """Record of one autonomous research cycle."""
    cycle_id: str
    started_at: float
    completed_at: float = 0.0
    gaps_detected: int = 0
    hypotheses_proposed: int = 0
    prototypes_run: int = 0
    discoveries: int = 0
    best_improvement: float = 0.0
    status: str = ResearchStatus.PROPOSED


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_RESEARCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_gaps (
    gap_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    evidence_json TEXT DEFAULT '[]',
    severity REAL DEFAULT 0,
    detected_at REAL NOT NULL,
    source TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    gap_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    technique_type TEXT DEFAULT '',
    expected_improvement REAL DEFAULT 0,
    confidence REAL DEFAULT 0,
    inspiration_json TEXT DEFAULT '[]',
    proposed_at REAL NOT NULL,
    status TEXT DEFAULT 'proposed'
);

CREATE TABLE IF NOT EXISTS prototypes (
    prototype_id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    baseline_score REAL DEFAULT 0,
    prototype_score REAL DEFAULT 0,
    improvement REAL DEFAULT 0,
    notes TEXT DEFAULT '',
    passed_gate INTEGER DEFAULT 0,
    evaluated_at REAL NOT NULL,
    metrics_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS research_cycles (
    cycle_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    completed_at REAL DEFAULT 0,
    gaps_detected INTEGER DEFAULT 0,
    hypotheses_proposed INTEGER DEFAULT 0,
    prototypes_run INTEGER DEFAULT 0,
    discoveries INTEGER DEFAULT 0,
    best_improvement REAL DEFAULT 0,
    status TEXT DEFAULT 'proposed'
);
"""


# ---------------------------------------------------------------------------
# Research Ledger
# ---------------------------------------------------------------------------

class ResearchLedger:
    """Audit trail for autonomous research."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = SQLiteConnectionOwner(self._db_path)
        conn = self._owner.connect()
        conn.executescript(_RESEARCH_SCHEMA)
        conn.commit()

    @property
    def _conn(self):
        return self._owner.connect()

    def record_gap(self, gap: ResearchGap) -> None:
        conn = self._conn
        conn.execute(
            "INSERT OR REPLACE INTO research_gaps "
            "(gap_id, category, description, evidence_json, severity, "
            "detected_at, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                gap.gap_id, gap.category, gap.description,
                json.dumps(gap.evidence, default=str),
                gap.severity, gap.detected_at, gap.source,
            ),
        )
        conn.commit()

    def record_hypothesis(self, hyp: Hypothesis) -> None:
        conn = self._conn
        conn.execute(
            "INSERT OR REPLACE INTO hypotheses "
            "(hypothesis_id, gap_id, title, description, technique_type, "
            "expected_improvement, confidence, inspiration_json, "
            "proposed_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                hyp.hypothesis_id, hyp.gap_id, hyp.title, hyp.description,
                hyp.technique_type, hyp.expected_improvement, hyp.confidence,
                json.dumps(hyp.inspiration_sources, default=str),
                hyp.proposed_at, hyp.status,
            ),
        )
        conn.commit()

    def record_prototype(self, proto: PrototypeResult) -> None:
        conn = self._conn
        conn.execute(
            "INSERT OR REPLACE INTO prototypes "
            "(prototype_id, hypothesis_id, baseline_score, prototype_score, "
            "improvement, notes, passed_gate, evaluated_at, metrics_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                proto.prototype_id, proto.hypothesis_id,
                proto.baseline_score, proto.prototype_score,
                proto.improvement, proto.implementation_notes,
                1 if proto.passed_gate else 0,
                proto.evaluated_at,
                json.dumps(proto.metrics, default=str),
            ),
        )
        conn.commit()

    def record_cycle(self, cycle: ResearchCycle) -> None:
        conn = self._conn
        conn.execute(
            "INSERT OR REPLACE INTO research_cycles "
            "(cycle_id, started_at, completed_at, gaps_detected, "
            "hypotheses_proposed, prototypes_run, discoveries, "
            "best_improvement, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cycle.cycle_id, cycle.started_at, cycle.completed_at,
                cycle.gaps_detected, cycle.hypotheses_proposed,
                cycle.prototypes_run, cycle.discoveries,
                cycle.best_improvement, cycle.status,
            ),
        )
        conn.commit()

    def get_gaps(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT gap_id, category, description, severity, source "
            "FROM research_gaps ORDER BY severity DESC LIMIT ?",
            (min(limit, 500),),
        ).fetchall()
        return [
            {"gap_id": r[0], "category": r[1], "description": r[2],
             "severity": r[3], "source": r[4]}
            for r in rows
        ]

    def get_hypotheses(self, status: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT hypothesis_id, gap_id, title, status, confidence "
                "FROM hypotheses WHERE status=? ORDER BY confidence DESC LIMIT ?",
                (status, min(limit, 500)),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT hypothesis_id, gap_id, title, status, confidence "
                "FROM hypotheses ORDER BY confidence DESC LIMIT ?",
                (min(limit, 500),),
            ).fetchall()
        return [
            {"hypothesis_id": r[0], "gap_id": r[1], "title": r[2],
             "status": r[3], "confidence": r[4]}
            for r in rows
        ]

    def get_discoveries(self) -> List[Dict[str, Any]]:
        """Get prototypes that passed the gate (novel discoveries)."""
        rows = self._conn.execute(
            "SELECT p.prototype_id, p.hypothesis_id, h.title, "
            "p.baseline_score, p.prototype_score, p.improvement "
            "FROM prototypes p JOIN hypotheses h "
            "ON p.hypothesis_id = h.hypothesis_id "
            "WHERE p.passed_gate = 1 "
            "ORDER BY p.improvement DESC LIMIT 100",
        ).fetchall()
        return [
            {"prototype_id": r[0], "hypothesis_id": r[1], "title": r[2],
             "baseline": r[3], "prototype": r[4], "improvement": r[5]}
            for r in rows
        ]

    def stats(self) -> Dict[str, Any]:
        conn = self._conn
        gaps = (conn.execute("SELECT COUNT(*) FROM research_gaps").fetchone() or (0,))[0]
        hyps = (conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone() or (0,))[0]
        protos = (conn.execute("SELECT COUNT(*) FROM prototypes").fetchone() or (0,))[0]
        discoveries = (conn.execute(
            "SELECT COUNT(*) FROM prototypes WHERE passed_gate=1"
        ).fetchone() or (0,))[0]
        cycles = (conn.execute("SELECT COUNT(*) FROM research_cycles").fetchone() or (0,))[0]

        return {
            "total_gaps": gaps,
            "total_hypotheses": hyps,
            "total_prototypes": protos,
            "discoveries": discoveries,
            "research_cycles": cycles,
            "discovery_rate": round(discoveries / max(protos, 1), 3),
        }

    def close(self) -> None:
        self._owner.close()


# ---------------------------------------------------------------------------
# Gap Detector
# ---------------------------------------------------------------------------

class GapDetector:
    """Analyzes eval failures to identify research gaps."""

    CATEGORIES = ["accuracy", "coverage", "robustness", "speed", "novel"]

    def __init__(self):
        self._analyzers: List[Callable[[List[Dict]], List[ResearchGap]]] = []

    def add_analyzer(
        self,
        analyzer_fn: Callable[[List[Dict]], List[ResearchGap]],
    ) -> None:
        """Register a gap analysis function."""
        self._analyzers.append(analyzer_fn)

    def detect(self, eval_failures: List[Dict[str, Any]]) -> List[ResearchGap]:
        """Detect research gaps from eval failures."""
        gaps: List[ResearchGap] = []

        # Built-in pattern analysis
        gaps.extend(self._pattern_gaps(eval_failures))

        # Custom analyzers
        for analyzer in self._analyzers:
            try:
                gaps.extend(analyzer(eval_failures))
            except Exception as exc:
                log.debug("Gap analyzer failed: %s", exc)

        return gaps

    def _pattern_gaps(self, failures: List[Dict[str, Any]]) -> List[ResearchGap]:
        """Detect gaps from failure patterns."""
        gaps = []
        if not failures:
            return gaps

        # Group failures by category
        categories: Dict[str, List[Dict]] = {}
        for f in failures:
            cat = f.get("category", "unknown")
            categories.setdefault(cat, []).append(f)

        for cat, cat_failures in categories.items():
            if len(cat_failures) >= 3:
                gaps.append(ResearchGap(
                    gap_id=f"gap_{uuid.uuid4().hex[:8]}",
                    category=cat if cat in self.CATEGORIES else "novel",
                    description=f"Recurring failures in {cat}: {len(cat_failures)} instances",
                    evidence=[f.get("id", "") for f in cat_failures[:10]],
                    severity=min(1.0, len(cat_failures) / 10.0),
                    source="eval_failure",
                ))

        return gaps


# ---------------------------------------------------------------------------
# Hypothesis Engine
# ---------------------------------------------------------------------------

class HypothesisEngine:
    """Proposes hypotheses from cross-domain synthesis."""

    def __init__(
        self,
        propose_fn: Optional[Callable[[ResearchGap], List[Hypothesis]]] = None,
    ):
        self._propose_fn = propose_fn
        self._templates: Dict[str, List[str]] = {
            "accuracy": [
                "Improve retrieval precision with adaptive re-ranking",
                "Add confidence calibration to generation pipeline",
                "Implement ensemble scoring across retrieval methods",
            ],
            "coverage": [
                "Expand indexing to cover additional document types",
                "Add cross-reference linking between knowledge bases",
                "Implement gap-filling via targeted data acquisition",
            ],
            "robustness": [
                "Add adversarial input detection layer",
                "Implement graceful degradation under partial index failure",
                "Add consistency verification across response batches",
            ],
            "speed": [
                "Cache frequently accessed retrieval results",
                "Implement speculative retrieval for predictable queries",
                "Add progressive response generation",
            ],
            "novel": [
                "Apply cross-domain technique transfer",
                "Implement self-supervised quality estimation",
                "Add emergent capability detection",
            ],
        }

    def propose(self, gap: ResearchGap) -> List[Hypothesis]:
        """Propose hypotheses to address a research gap."""
        if self._propose_fn:
            return self._propose_fn(gap)

        # Template-based proposal
        templates = self._templates.get(gap.category, self._templates["novel"])
        hypotheses = []

        for i, template in enumerate(templates):
            hypotheses.append(Hypothesis(
                hypothesis_id=f"hyp_{uuid.uuid4().hex[:8]}",
                gap_id=gap.gap_id,
                title=template,
                description=f"Addressing {gap.category} gap: {gap.description}",
                technique_type=gap.category,
                expected_improvement=gap.severity * (3 - i) * 0.5,
                confidence=0.3 + gap.severity * 0.3,
                inspiration_sources=[gap.source],
            ))

        return hypotheses


# ---------------------------------------------------------------------------
# Prototype Runner
# ---------------------------------------------------------------------------

class PrototypeRunner:
    """Implements and evaluates prototype techniques."""

    def __init__(
        self,
        eval_fn: Optional[Callable[[Hypothesis, Dict], PrototypeResult]] = None,
        min_improvement: float = 0.5,
    ):
        self._eval_fn = eval_fn
        self._min_improvement = min_improvement

    def run(
        self,
        hypothesis: Hypothesis,
        baseline_score: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> PrototypeResult:
        """Run a prototype for a hypothesis."""
        if self._eval_fn:
            result = self._eval_fn(hypothesis, context or {})
            result.baseline_score = baseline_score
            result.improvement = result.prototype_score - baseline_score
            result.passed_gate = result.improvement >= self._min_improvement
            return result

        # Default: simulate prototype evaluation
        return PrototypeResult(
            prototype_id=f"proto_{uuid.uuid4().hex[:8]}",
            hypothesis_id=hypothesis.hypothesis_id,
            baseline_score=baseline_score,
            prototype_score=baseline_score,
            improvement=0.0,
            implementation_notes="Simulated prototype (no eval function provided)",
            passed_gate=False,
        )

    @property
    def min_improvement(self) -> float:
        return self._min_improvement


# ---------------------------------------------------------------------------
# Autonomous Lab
# ---------------------------------------------------------------------------

class AutonomousLab:
    """Top-level autonomous research orchestrator.

    Connects gap detection, hypothesis generation, prototyping,
    and evaluation into a self-directed research loop.
    """

    def __init__(
        self,
        ledger: Optional[ResearchLedger] = None,
        ledger_path: str | Path = "_research/lab.db",
        gap_detector: Optional[GapDetector] = None,
        hypothesis_engine: Optional[HypothesisEngine] = None,
        prototype_runner: Optional[PrototypeRunner] = None,
    ):
        self._ledger = ledger or ResearchLedger(ledger_path)
        self._detector = gap_detector or GapDetector()
        self._engine = hypothesis_engine or HypothesisEngine()
        self._runner = prototype_runner or PrototypeRunner()

    def run_cycle(
        self,
        eval_failures: List[Dict[str, Any]],
        baseline_score: float,
        max_hypotheses: int = 5,
        max_prototypes: int = 3,
    ) -> ResearchCycle:
        """Run one autonomous research cycle.

        1. Detect gaps from eval failures
        2. Propose hypotheses for top gaps
        3. Prototype and evaluate top hypotheses
        4. Record discoveries
        """
        cycle = ResearchCycle(
            cycle_id=f"research_{uuid.uuid4().hex[:8]}",
            started_at=time.time(),
        )

        # Step 1: Detect gaps
        gaps = self._detector.detect(eval_failures)
        cycle.gaps_detected = len(gaps)
        for gap in gaps:
            self._ledger.record_gap(gap)

        if not gaps:
            cycle.status = ResearchStatus.REJECTED
            cycle.completed_at = time.time()
            self._ledger.record_cycle(cycle)
            return cycle

        # Step 2: Propose hypotheses for top gaps (by severity)
        gaps.sort(key=lambda g: g.severity, reverse=True)
        all_hypotheses: List[Hypothesis] = []
        for gap in gaps[:3]:  # Top 3 gaps
            hyps = self._engine.propose(gap)
            all_hypotheses.extend(hyps)

        # Keep top N by expected improvement
        all_hypotheses.sort(
            key=lambda h: h.expected_improvement, reverse=True,
        )
        selected = all_hypotheses[:max_hypotheses]
        cycle.hypotheses_proposed = len(selected)

        for hyp in selected:
            hyp.status = ResearchStatus.INVESTIGATING
            self._ledger.record_hypothesis(hyp)

        # Step 3: Prototype top hypotheses
        prototypes: List[PrototypeResult] = []
        for hyp in selected[:max_prototypes]:
            hyp.status = ResearchStatus.PROTOTYPING
            result = self._runner.run(hyp, baseline_score)
            prototypes.append(result)
            self._ledger.record_prototype(result)

            if result.passed_gate:
                hyp.status = ResearchStatus.ACCEPTED
                cycle.discoveries += 1
                if result.improvement > cycle.best_improvement:
                    cycle.best_improvement = result.improvement
            else:
                hyp.status = ResearchStatus.REJECTED

            self._ledger.record_hypothesis(hyp)

        cycle.prototypes_run = len(prototypes)

        # Final status
        if cycle.discoveries > 0:
            cycle.status = ResearchStatus.ACCEPTED
        else:
            cycle.status = ResearchStatus.REJECTED

        cycle.completed_at = time.time()
        self._ledger.record_cycle(cycle)
        return cycle

    def get_gaps(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._ledger.get_gaps(limit=limit)

    def get_hypotheses(self, status: str = "") -> List[Dict[str, Any]]:
        return self._ledger.get_hypotheses(status=status)

    def get_discoveries(self) -> List[Dict[str, Any]]:
        return self._ledger.get_discoveries()

    def stats(self) -> Dict[str, Any]:
        return self._ledger.stats()

    def close(self) -> None:
        self._ledger.close()
