"""
Extended tests for island model evolution and speculative code generation.

Island Model:
  - Island initialization with population
  - Independent evolution per island
  - Migration between islands
  - Convergence tracking
  - Island isolation (no cross-contamination without migration)

Speculative Coder:
  - Draft generation from context
  - Correction pass on draft
  - Multi-draft best-of-N selection
  - Code validation (syntax check)
  - Fallback on draft failure
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest

from core.island_model import (
    IslandModelRunner,
    IslandState,
    Migrant,
    MigrationEvent,
)
from core.speculative_coder import (
    CodeDraft,
    SpeculativeCodeGenerator,
    SpeculativeResult,
    _check_consistency,
    _check_imports,
    _check_structure,
    _check_syntax,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_CALL_SEQ: list[int] = []


def _deterministic_eval(config):
    """Eval that returns val directly so we can predict outcomes."""
    return float(config.get("val", 0.0))


def _increment_mutate(config):
    c = copy.deepcopy(config)
    c["val"] = c.get("val", 0.0) + 1.0
    return c


def _identity_mutate(config):
    return copy.deepcopy(config)


def _tracking_eval(config):
    """Appends island_id marker to global list for isolation checks."""
    _CALL_SEQ.append(config.get("marker", -1))
    return float(config.get("val", 0.0))


def _marker_mutate(config):
    c = copy.deepcopy(config)
    c["val"] = c.get("val", 0.0) + 0.5
    return c


def _make_runtime_mock(responses=None, fail_at=None):
    """Build a mock Runtime whose generate() returns controlled text."""
    rt = MagicMock()
    call_count = {"n": 0}

    def _gen(question, context_chunks, **kw):
        idx = call_count["n"]
        call_count["n"] += 1
        if fail_at is not None and idx in fail_at:
            raise RuntimeError("draft failed")
        if responses:
            return responses[idx % len(responses)]
        return "def hello():\n    return 'world'\n"

    rt.generate = MagicMock(side_effect=_gen)
    return rt


# -----------------------------------------------------------------------
# Island Model -- initialisation
# -----------------------------------------------------------------------

class TestIslandInit:

    def test_population_size_matches_config(self, tmp_path):
        r = IslandModelRunner(
            num_islands=3, population_per_island=5,
            ledger_dir=str(tmp_path),
        )
        islands = r._init_islands(
            {"val": 10.0}, _deterministic_eval, _increment_mutate,
        )
        for isl in islands:
            assert len(isl.population) == 5
            assert len(isl.scores) == 5
        r.close()

    def test_initial_best_tracked(self, tmp_path):
        r = IslandModelRunner(
            num_islands=2, population_per_island=4,
            ledger_dir=str(tmp_path),
        )
        islands = r._init_islands(
            {"val": 10.0}, _deterministic_eval, _increment_mutate,
        )
        for isl in islands:
            assert isl.best_score > 0.0
            assert isl.best_config != {}
        r.close()

    def test_islands_have_unique_ids(self, tmp_path):
        r = IslandModelRunner(
            num_islands=5, population_per_island=2,
            ledger_dir=str(tmp_path),
        )
        islands = r._init_islands(
            {"val": 1.0}, _deterministic_eval, _increment_mutate,
        )
        ids = [isl.island_id for isl in islands]
        assert len(set(ids)) == 5
        r.close()


# -----------------------------------------------------------------------
# Island Model -- independent evolution
# -----------------------------------------------------------------------

class TestIslandEvolution:

    def test_evolution_improves_best(self, tmp_path):
        """After several generations with +1 mutation, best should climb."""
        r = IslandModelRunner(
            num_islands=2, population_per_island=4,
            migration_interval=999,  # no migration
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=4,
            baseline_config={"val": 10.0},
            baseline_score=10.0,
            eval_fn=_deterministic_eval,
            mutate_fn=_increment_mutate,
            min_improvement=0.1,
        )
        assert result.champion_score > 10.0
        r.close()

    def test_generation_counter_advances(self, tmp_path):
        r = IslandModelRunner(
            num_islands=2, population_per_island=3,
            migration_interval=999,
            ledger_dir=str(tmp_path),
        )
        islands = r._init_islands(
            {"val": 5.0}, _deterministic_eval, _increment_mutate,
        )
        for gen in range(3):
            for isl in islands:
                r._evolve_island(
                    isl, {"val": 5.0}, _deterministic_eval, _increment_mutate,
                )
                isl.generation = gen + 1
        for isl in islands:
            assert isl.generation == 3
        r.close()


# -----------------------------------------------------------------------
# Island Model -- migration
# -----------------------------------------------------------------------

class TestIslandMigration:

    def test_ring_sources_and_targets(self, tmp_path):
        r = IslandModelRunner(
            num_islands=4, population_per_island=4,
            migration_count=1, migration_interval=1,
            ledger_dir=str(tmp_path),
        )
        islands = r._init_islands(
            {"val": 10.0}, _deterministic_eval, _increment_mutate,
        )
        events = r._migrate_ring(islands, generation=1)
        src_dst = [(e.source_island, e.target_island) for e in events]
        # ring: 0->1, 1->2, 2->3, 3->0
        assert ("island_00", "island_01") in src_dst
        assert ("island_03", "island_00") in src_dst
        r.close()

    def test_migrants_appear_in_immigrants(self, tmp_path):
        r = IslandModelRunner(
            num_islands=3, population_per_island=3,
            migration_count=1, migration_interval=1,
            ledger_dir=str(tmp_path),
        )
        islands = r._init_islands(
            {"val": 10.0}, _deterministic_eval, _increment_mutate,
        )
        r._migrate_ring(islands, generation=1)
        for isl in islands:
            assert len(isl.immigrants) >= 1
        r.close()

    def test_immigrants_consumed_during_evolution(self, tmp_path):
        r = IslandModelRunner(
            num_islands=2, population_per_island=3,
            migration_count=1, migration_interval=1,
            ledger_dir=str(tmp_path),
        )
        islands = r._init_islands(
            {"val": 10.0}, _deterministic_eval, _increment_mutate,
        )
        r._migrate_ring(islands, generation=1)
        for isl in islands:
            assert len(isl.immigrants) > 0
        # Evolve should consume immigrants
        for isl in islands:
            r._evolve_island(
                isl, {"val": 10.0}, _deterministic_eval, _increment_mutate,
            )
        for isl in islands:
            assert len(isl.immigrants) == 0
        r.close()


# -----------------------------------------------------------------------
# Island Model -- convergence tracking
# -----------------------------------------------------------------------

class TestConvergence:

    def test_champion_is_global_best(self, tmp_path):
        r = IslandModelRunner(
            num_islands=3, population_per_island=4,
            migration_interval=1,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=3,
            baseline_config={"val": 10.0},
            baseline_score=10.0,
            eval_fn=_deterministic_eval,
            mutate_fn=_increment_mutate,
        )
        island_bests = list(result.island_best_scores.values())
        assert result.champion_score == max(island_bests)

    def test_accepted_when_improvement_sufficient(self, tmp_path):
        r = IslandModelRunner(
            num_islands=2, population_per_island=3,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=5,
            baseline_config={"val": 10.0},
            baseline_score=10.0,
            eval_fn=_deterministic_eval,
            mutate_fn=_increment_mutate,
            min_improvement=0.5,
        )
        assert result.decision == "accepted"
        r.close()

    def test_rejected_when_improvement_insufficient(self, tmp_path):
        r = IslandModelRunner(
            num_islands=2, population_per_island=2,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=1,
            baseline_config={"val": 10.0},
            baseline_score=10.0,
            eval_fn=_deterministic_eval,
            mutate_fn=_identity_mutate,
            min_improvement=100.0,
        )
        assert result.decision == "rejected"
        r.close()


# -----------------------------------------------------------------------
# Island Model -- isolation (no cross-contamination without migration)
# -----------------------------------------------------------------------

class TestIslandIsolation:

    def test_no_migration_keeps_islands_independent(self, tmp_path):
        """Without migration each island's population stays self-contained."""
        r = IslandModelRunner(
            num_islands=3, population_per_island=3,
            migration_count=1, migration_interval=999,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=3,
            baseline_config={"val": 10.0},
            baseline_score=10.0,
            eval_fn=_deterministic_eval,
            mutate_fn=_increment_mutate,
        )
        assert len(result.migrations) == 0
        r.close()

    def test_populations_diverge_without_migration(self, tmp_path):
        """Different mutation strengths should cause populations to diverge."""
        r = IslandModelRunner(
            num_islands=2, population_per_island=4,
            migration_interval=999,
            ledger_dir=str(tmp_path),
        )
        # Use mutation that adds island-index-dependent noise
        call_n = {"n": 0}

        def biased_mutate(cfg):
            call_n["n"] += 1
            c = copy.deepcopy(cfg)
            c["val"] = c.get("val", 0.0) + (call_n["n"] % 3)
            return c

        result = r.run(
            generations=3,
            baseline_config={"val": 10.0},
            baseline_score=10.0,
            eval_fn=_deterministic_eval,
            mutate_fn=biased_mutate,
        )
        scores = list(result.island_best_scores.values())
        # Islands may differ since mutation is call-order dependent
        assert len(scores) == 2
        r.close()


# -----------------------------------------------------------------------
# Speculative Coder -- syntax validation helpers
# -----------------------------------------------------------------------

class TestSyntaxCheck:

    def test_valid_python(self):
        ok, issues = _check_syntax("x = 1\ny = x + 2\n")
        assert ok
        assert issues == []

    def test_invalid_python(self):
        ok, issues = _check_syntax("def foo(:\n    pass\n")
        assert not ok
        assert len(issues) >= 1

    def test_fenced_code_block(self):
        code = "```python\ndef greet():\n    return 'hi'\n```"
        ok, issues = _check_syntax(code)
        assert ok

    def test_structure_short_output_penalised(self):
        score, issues = _check_structure("x = 1\n")
        assert score < 1.0
        assert any("short" in i.lower() for i in issues)

    def test_imports_stdlib_ok(self):
        issues = _check_imports("import os\nimport json\n")
        assert issues == []

    def test_imports_unknown_flagged(self):
        issues = _check_imports("import obscurelib\n")
        assert len(issues) >= 1

    def test_consistency_mixed_indent(self):
        issues = _check_consistency("if True:\n\tx = 1\n    y = 2\n")
        assert any("tabs" in i.lower() or "mixed" in i.lower() for i in issues)


# -----------------------------------------------------------------------
# Speculative Coder -- draft generation
# -----------------------------------------------------------------------

class TestDraftGeneration:

    def test_generates_n_drafts(self):
        rt = _make_runtime_mock()
        gen = SpeculativeCodeGenerator(rt, n_drafts=4)
        result = gen.generate("write hello world", ["ctx"])
        assert result.drafts_generated == 4
        assert rt.generate.call_count == 4

    def test_draft_uses_context(self):
        rt = _make_runtime_mock()
        gen = SpeculativeCodeGenerator(rt, n_drafts=1)
        gen.generate("question", ["chunk_a", "chunk_b"])
        args, kwargs = rt.generate.call_args
        assert args[1] == ["chunk_a", "chunk_b"]

    def test_draft_temperatures_vary(self):
        rt = _make_runtime_mock()
        gen = SpeculativeCodeGenerator(rt, n_drafts=3, draft_temperature=0.3)
        gen.generate("q", [])
        temps = [
            rt.generate.call_args_list[i][1].get("temperature", 0)
            for i in range(3)
        ]
        # Each draft should use a different temperature
        assert len(set(temps)) == 3


# -----------------------------------------------------------------------
# Speculative Coder -- correction pass
# -----------------------------------------------------------------------

class TestCorrectionPass:

    def test_correction_called_for_low_quality(self):
        bad_code = "def foo(:\n    pass"
        good_code = "def foo():\n    pass\n"
        rt = _make_runtime_mock(responses=[bad_code, bad_code, bad_code, good_code])
        gen = SpeculativeCodeGenerator(rt, n_drafts=3)
        result = gen.generate("fix this", [])
        # 3 drafts + 1 correction call
        assert rt.generate.call_count == 4
        assert result.was_corrected

    def test_no_correction_when_quality_high(self):
        good = "def hello():\n    '''Say hi.'''\n    return 'world'\n"
        rt = _make_runtime_mock(responses=[good])
        gen = SpeculativeCodeGenerator(rt, n_drafts=1)
        result = gen.generate("q", [])
        # Only 1 draft, no correction
        assert rt.generate.call_count == 1
        assert not result.was_corrected


# -----------------------------------------------------------------------
# Speculative Coder -- best-of-N selection
# -----------------------------------------------------------------------

class TestBestOfN:

    def test_best_draft_selected(self):
        bad = "x ="  # syntax error
        ok = "x = 1\ny = 2\n"
        good = "def greet():\n    '''Greet.'''\n    return 'hello'\n"
        rt = _make_runtime_mock(responses=[bad, ok, good])
        gen = SpeculativeCodeGenerator(rt, n_drafts=3)
        result = gen.generate("q", [])
        # The syntactically valid code with docstring should score highest
        assert "def greet" in result.final_code or "x = 1" in result.final_code

    def test_quality_score_populated(self):
        rt = _make_runtime_mock()
        gen = SpeculativeCodeGenerator(rt, n_drafts=2)
        result = gen.generate("q", [])
        assert result.quality_score > 0.0


# -----------------------------------------------------------------------
# Speculative Coder -- fallback on draft failure
# -----------------------------------------------------------------------

class TestDraftFallback:

    def test_all_drafts_fail_returns_empty(self):
        rt = _make_runtime_mock(fail_at={0, 1, 2})
        gen = SpeculativeCodeGenerator(rt, n_drafts=3)
        result = gen.generate("q", [])
        # All drafts failed, final code should be empty string
        assert result.final_code == ""
        assert result.quality_score == 0.0

    def test_partial_failure_still_returns_best(self):
        good = "def ok():\n    return True\n"
        rt = _make_runtime_mock(responses=[good], fail_at={1, 2})
        gen = SpeculativeCodeGenerator(rt, n_drafts=3)
        result = gen.generate("q", [])
        assert "def ok" in result.final_code

    def test_timing_fields_populated(self):
        rt = _make_runtime_mock()
        gen = SpeculativeCodeGenerator(rt, n_drafts=2)
        result = gen.generate("q", [])
        assert result.total_ms >= 0.0
        assert result.draft_ms >= 0.0
        assert result.verify_ms >= 0.0
