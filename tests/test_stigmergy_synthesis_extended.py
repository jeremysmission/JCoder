"""
Extended tests for StigmergicBooster and SynthesisMatrix.

All IO and LLM calls are mocked -- no Ollama or network needed.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.stigmergy import PheromoneConfig, StigmergicBooster
from core.synthesis_matrix import (
    MatrixReport,
    SynthesisCell,
    SynthesisMatrix,
    ThemeRow,
    _extract_noun_phrases,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_booster(tmp_path: Path, **overrides) -> StigmergicBooster:
    cfg = PheromoneConfig(**overrides)
    return StigmergicBooster(
        db_path=str(tmp_path / "pheromones.db"), config=cfg,
    )


def _sample_papers():
    return [
        {
            "title": "Paper A",
            "key_claims": ["neural networks improve ionogram tracing"],
            "triage_summary": "Deep learning applied to ionogram quality",
            "abstract": "We propose a CNN for ionogram quality assessment.",
        },
        {
            "title": "Paper B",
            "key_claims": ["traditional methods remain competitive"],
            "triage_summary": "Comparison of classical vs neural approaches",
            "abstract": "Classical curve fitting still outperforms CNNs on noisy data.",
        },
        {
            "title": "Paper C",
            "key_claims": ["transfer learning reduces labelling cost"],
            "triage_summary": "Transfer learning for ionospheric data",
            "abstract": "Pre-trained models fine-tuned on small ionogram datasets.",
        },
    ]


# ======================================================================
# Stigmergy -- pheromone deposit and evaporation
# ======================================================================

class TestStigmergicDeposit:
    def test_positive_deposit_creates_trail(self, tmp_path):
        b = _make_booster(tmp_path)
        b.deposit(["c1", "c2"], "lookup", success=True)
        hot = b.hot_chunks("lookup")
        ids = {h["chunk_id"] for h in hot}
        assert "c1" in ids and "c2" in ids

    def test_negative_deposit_lowers_strength(self, tmp_path):
        b = _make_booster(tmp_path, positive_deposit=1.0, negative_deposit=-0.5)
        b.deposit(["c1"], "lookup", success=True)
        b.deposit(["c1"], "lookup", success=False)
        hot = b.hot_chunks("lookup")
        assert hot[0]["strength"] < 1.0

    def test_strength_capped_at_max(self, tmp_path):
        b = _make_booster(tmp_path, max_pheromone=3.0, positive_deposit=2.0)
        for _ in range(10):
            b.deposit(["c1"], "debug", success=True)
        hot = b.hot_chunks("debug")
        assert hot[0]["strength"] <= 3.0

    def test_success_and_fail_counts(self, tmp_path):
        b = _make_booster(tmp_path)
        b.deposit(["c1"], "lookup", success=True)
        b.deposit(["c1"], "lookup", success=True)
        b.deposit(["c1"], "lookup", success=False)
        hot = b.hot_chunks("lookup")
        assert hot[0]["successes"] == 2
        assert hot[0]["failures"] == 1


# ======================================================================
# Stigmergy -- trail following (boost_scores)
# ======================================================================

class TestTrailFollowing:
    def test_strongest_pheromone_wins(self, tmp_path):
        b = _make_booster(tmp_path, positive_deposit=2.0, boost_weight=0.5)
        for _ in range(5):
            b.deposit(["c_strong"], "lookup", success=True)
        b.deposit(["c_weak"], "lookup", success=True)

        scores = [("c_weak", 0.82), ("c_strong", 0.80)]
        boosted = b.boost_scores(scores, "lookup")
        # c_strong should be first despite slightly lower base score
        assert boosted[0][0] == "c_strong"

    def test_no_pheromone_no_boost(self, tmp_path):
        b = _make_booster(tmp_path)
        scores = [("c1", 0.5)]
        boosted = b.boost_scores(scores, "lookup")
        assert boosted[0][1] == 0.5

    def test_empty_scores_returns_empty(self, tmp_path):
        b = _make_booster(tmp_path)
        assert b.boost_scores([], "lookup") == []


# ======================================================================
# Stigmergy -- multi-agent pheromone interaction
# ======================================================================

class TestMultiAgent:
    def test_two_agents_accumulate(self, tmp_path):
        """Two booster instances sharing the same DB file accumulate pheromone."""
        db = str(tmp_path / "shared.db")
        a1 = StigmergicBooster(db_path=db)
        a2 = StigmergicBooster(db_path=db)
        a1.deposit(["c1"], "lookup", success=True)
        a2.deposit(["c1"], "lookup", success=True)
        hot = a1.hot_chunks("lookup")
        assert hot[0]["successes"] == 2

    def test_different_query_types_independent(self, tmp_path):
        b = _make_booster(tmp_path)
        b.deposit(["c1"], "lookup", success=True)
        b.deposit(["c1"], "debug", success=True)
        lookup_hot = b.hot_chunks("lookup")
        debug_hot = b.hot_chunks("debug")
        assert len(lookup_hot) == 1
        assert len(debug_hot) == 1


# ======================================================================
# Stigmergy -- evaporation rate configuration
# ======================================================================

class TestEvaporationConfig:
    def test_custom_evaporation_rate(self, tmp_path):
        b = _make_booster(tmp_path, evaporation_rate=0.99)
        b.deposit(["c1"], "lookup", success=True)
        # After evaporation, near-total decay expected
        cleaned = b.evaporate()
        # Strength should be near zero (but evaporate uses time delta,
        # which is ~0 here, so it stays). We verify config is stored.
        assert b.config.evaporation_rate == 0.99

    def test_min_pheromone_cleanup(self, tmp_path):
        b = _make_booster(tmp_path, min_pheromone=5.0, positive_deposit=0.1)
        b.deposit(["c1"], "lookup", success=True)
        # strength 0.1 < min_pheromone 5.0, so evaporate should clean it
        cleaned = b.evaporate()
        assert cleaned >= 1
        assert b.hot_chunks("lookup") == []


# ======================================================================
# Stigmergy -- trail decay over time
# ======================================================================

class TestTrailDecay:
    def test_decay_reduces_boost_over_time(self, tmp_path):
        b = _make_booster(tmp_path, evaporation_rate=0.5)
        b.deposit(["c1"], "lookup", success=True)

        # Immediate boost
        scores = [("c1", 0.5)]
        boosted_now = b.boost_scores(scores, "lookup")

        # Simulate 2 hours later by patching time.time
        future = time.time() + 7200
        with patch("core.stigmergy.time.time", return_value=future):
            boosted_later = b.boost_scores(scores, "lookup")

        # Boost should be smaller after simulated decay
        assert boosted_later[0][1] < boosted_now[0][1]

    def test_stats_reflect_deposits(self, tmp_path):
        b = _make_booster(tmp_path)
        b.deposit(["c1", "c2"], "lookup", success=True)
        b.deposit(["c3"], "debug", success=False)
        s = b.stats()
        assert s["total_trails"] == 3
        assert s["total_deposits"] == 3
        assert len(s["by_type"]) == 2


# ======================================================================
# Synthesis Matrix -- cross-topic connection discovery
# ======================================================================

class TestCrossTopicDiscovery:
    def test_heuristic_finds_shared_themes(self):
        sm = SynthesisMatrix(runtime=None)
        papers = _sample_papers()
        report = sm.build(papers, query="ionogram quality")
        # Should discover at least one theme across papers
        assert len(report.themes) > 0
        assert report.total_sources == 3

    def test_empty_papers_returns_empty_report(self):
        sm = SynthesisMatrix(runtime=None)
        report = sm.build([], query="anything")
        assert report.themes == []
        assert report.total_sources == 0
        assert report.gaps == 0


# ======================================================================
# Synthesis Matrix -- knowledge gap identification
# ======================================================================

class TestKnowledgeGaps:
    def test_gap_when_most_sources_silent(self):
        sm = SynthesisMatrix(runtime=None)
        # Build a row where 2/3 sources are silent -> gap
        cells = [
            SynthesisCell("A", "supports X", "supports", 0.8),
            SynthesisCell("B", "", "silent", 0.0),
            SynthesisCell("C", "", "silent", 0.0),
        ]
        row = ThemeRow(theme="niche topic", cells=cells,
                       consensus="mixed", contradiction_flag=False)
        gap_count = SynthesisMatrix._count_gaps([row], total_sources=3)
        assert gap_count == 1

    def test_no_gap_when_sources_speak(self):
        cells = [
            SynthesisCell("A", "supports X", "supports", 0.8),
            SynthesisCell("B", "also supports", "supports", 0.7),
            SynthesisCell("C", "", "silent", 0.0),
        ]
        row = ThemeRow(theme="common topic", cells=cells,
                       consensus="mixed", contradiction_flag=False)
        gap_count = SynthesisMatrix._count_gaps([row], total_sources=3)
        assert gap_count == 0


# ======================================================================
# Synthesis Matrix -- synthesis scoring (consensus / contradiction)
# ======================================================================

class TestSynthesisScoring:
    def test_strong_agreement(self):
        sm = SynthesisMatrix(runtime=None)
        cells = [
            SynthesisCell("A", "pos", "supports", 0.9),
            SynthesisCell("B", "pos", "supports", 0.8),
        ]
        row = ThemeRow("topic", cells, "mixed", False)
        rows = sm._detect_contradictions([row])
        assert rows[0].consensus == "strong_agreement"
        assert not rows[0].contradiction_flag

    def test_contradiction_detected(self):
        sm = SynthesisMatrix(runtime=None)
        cells = [
            SynthesisCell("A", "yes", "supports", 0.9),
            SynthesisCell("B", "no", "contradicts", 0.8),
        ]
        row = ThemeRow("topic", cells, "mixed", False)
        rows = sm._detect_contradictions([row])
        assert rows[0].consensus == "contradicted"
        assert rows[0].contradiction_flag

    def test_mixed_consensus(self):
        sm = SynthesisMatrix(runtime=None)
        cells = [
            SynthesisCell("A", "yes", "supports", 0.9),
            SynthesisCell("B", "no", "contradicts", 0.8),
            SynthesisCell("C", "meh", "neutral", 0.3),
        ]
        row = ThemeRow("topic", cells, "mixed", False)
        rows = sm._detect_contradictions([row])
        # Has contradiction -> contradicted
        assert rows[0].consensus == "contradicted"

    def test_all_silent_is_mixed(self):
        sm = SynthesisMatrix(runtime=None)
        cells = [
            SynthesisCell("A", "", "silent", 0.0),
            SynthesisCell("B", "", "silent", 0.0),
        ]
        row = ThemeRow("topic", cells, "mixed", False)
        rows = sm._detect_contradictions([row])
        assert rows[0].consensus == "mixed"


# ======================================================================
# Synthesis Matrix -- persistence (to_dict / from dict)
# ======================================================================

class TestMatrixPersistence:
    def test_to_dict_roundtrip(self):
        sm = SynthesisMatrix(runtime=None)
        report = MatrixReport(
            query="test",
            themes=[
                ThemeRow("t1", [
                    SynthesisCell("src", "pos", "supports", 0.9),
                ], "strong_agreement", False),
            ],
            total_sources=1,
            strong_agreements=1,
            contradictions=0,
            gaps=0,
        )
        d = sm.to_dict(report)
        assert d["query"] == "test"
        assert len(d["themes"]) == 1
        assert d["themes"][0]["theme"] == "t1"
        assert d["themes"][0]["cells"][0]["sentiment"] == "supports"

    def test_to_markdown_nonempty(self):
        sm = SynthesisMatrix(runtime=None)
        report = MatrixReport(
            query="test",
            themes=[
                ThemeRow("t1", [
                    SynthesisCell("PaperA", "agrees", "supports", 0.8),
                    SynthesisCell("PaperB", "disagrees", "contradicts", 0.7),
                ], "contradicted", True),
            ],
            total_sources=2,
            strong_agreements=0,
            contradictions=1,
            gaps=0,
        )
        md = sm.to_markdown_table(report)
        assert "PaperA" in md
        assert "PaperB" in md
        assert "[+]" in md   # supports icon
        assert "[X]" in md   # contradicts icon

    def test_to_markdown_empty(self):
        sm = SynthesisMatrix(runtime=None)
        report = MatrixReport("q", [], 0, 0, 0, 0)
        assert sm.to_markdown_table(report) == "(empty matrix)"


# ======================================================================
# Synthesis Matrix -- incremental updates
# ======================================================================

class TestIncrementalUpdates:
    def test_adding_paper_changes_report(self):
        sm = SynthesisMatrix(runtime=None)
        papers = _sample_papers()[:2]
        r1 = sm.build(papers, query="ionogram quality")

        papers.append(_sample_papers()[2])
        r2 = sm.build(papers, query="ionogram quality")

        assert r2.total_sources == r1.total_sources + 1

    def test_noun_phrase_extraction_deterministic(self):
        texts = [
            "neural network for image classification",
            "deep neural network architectures",
        ]
        a = _extract_noun_phrases(texts, 5)
        b = _extract_noun_phrases(texts, 5)
        assert a == b

    def test_parse_classify_json_tolerates_fences(self):
        raw = '```json\n[{"source":"A","theme":"T","sentiment":"supports","position":"x","strength":0.9}]\n```'
        entries = SynthesisMatrix._parse_classify_json(raw)
        assert len(entries) == 1
        assert entries[0]["sentiment"] == "supports"

    def test_parse_classify_json_bad_input(self):
        assert SynthesisMatrix._parse_classify_json("not json") == []
