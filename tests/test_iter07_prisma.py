"""Iteration 7: PRISMA audit trail integrity tests."""

import pytest
import time
from core.prisma_tracker import PrismaTracker


# ---------------------------------------------------------------------------
# Audit trail completeness
# ---------------------------------------------------------------------------

def test_every_identify_logged(tmp_path):
    """Every identified paper appears in the log."""
    tracker = PrismaTracker(db_path=str(tmp_path / "audit.db"))
    for i in range(100):
        tracker.identify(f"Paper {i}", f"source_{i % 5}", f"hash{i}")
    counts = tracker.flow_counts()
    assert counts["identified"] == 100
    tracker.close()


def test_screening_preserves_identify_count(tmp_path):
    """Screening doesn't reduce identified count."""
    tracker = PrismaTracker(db_path=str(tmp_path / "screen.db"))
    for i in range(20):
        tracker.identify(f"Paper {i}", "arxiv", f"h{i}")
    for i in range(20):
        tracker.screen(f"h{i}", passed=(i % 2 == 0), reason="relevance")
    counts = tracker.flow_counts()
    assert counts["identified"] == 20
    assert counts["screened"] == 10  # half passed
    assert counts["excluded"] == 10  # half excluded
    tracker.close()


def test_exclusion_reasons_grouped(tmp_path):
    """Exclusion reasons are correctly grouped by stage."""
    tracker = PrismaTracker(db_path=str(tmp_path / "reasons.db"))
    for i in range(15):
        tracker.identify(f"Paper {i}", "src", f"h{i}")
    # Exclude 5 at screening for "low_relevance"
    for i in range(5):
        tracker.screen(f"h{i}", passed=False, reason="low_relevance")
    # Exclude 3 at screening for "duplicate"
    for i in range(5, 8):
        tracker.screen(f"h{i}", passed=False, reason="duplicate")
    # Pass rest through screening
    for i in range(8, 15):
        tracker.screen(f"h{i}", passed=True, reason="relevant")
    # Exclude 2 at eligibility for "no_data"
    for i in range(8, 10):
        tracker.eligible(f"h{i}", passed=False, reason="no_data")

    screen_reasons = tracker.exclusion_reasons("screened")
    eligible_reasons = tracker.exclusion_reasons("eligible")
    # Screen exclusions: 5 low_relevance + 3 duplicate = 8 total
    total_screen = sum(count for _, count in screen_reasons)
    assert total_screen == 8
    # Eligible exclusions: 2
    total_eligible = sum(count for _, count in eligible_reasons)
    assert total_eligible == 2
    tracker.close()


def test_flow_diagram_counts_match(tmp_path):
    """Flow diagram text has consistent counts."""
    tracker = PrismaTracker(db_path=str(tmp_path / "diagram.db"))
    for i in range(10):
        tracker.identify(f"P{i}", "src", f"h{i}")
    for i in range(10):
        tracker.screen(f"h{i}", passed=(i < 7), reason="r")
    for i in range(7):
        tracker.eligible(f"h{i}", passed=(i < 4), reason="r")
    for i in range(4):
        tracker.include(f"h{i}", reason="final")

    diagram = tracker.flow_diagram_text()
    assert "10" in diagram  # identified
    assert "7" in diagram   # screened
    assert "4" in diagram   # eligible or included
    tracker.close()


# ---------------------------------------------------------------------------
# Timeline ordering
# ---------------------------------------------------------------------------

def test_timestamps_monotonic(tmp_path):
    """Timestamps are monotonically increasing."""
    tracker = PrismaTracker(db_path=str(tmp_path / "time.db"))
    tracker.identify("P1", "src", "h1")
    tracker.screen("h1", passed=True, reason="ok")
    tracker.eligible("h1", passed=True, reason="ok")
    tracker.include("h1", reason="final")
    tracker.flow_counts()

    cur = tracker._conn.execute(
        "SELECT timestamp FROM prisma_log WHERE content_hash='h1' ORDER BY rowid"
    )
    timestamps = [row[0] for row in cur.fetchall()]
    assert timestamps == sorted(timestamps), "Timestamps not monotonic"
    assert len(timestamps) == 4
    tracker.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_screen_unknown_hash(tmp_path):
    """Screening an unknown hash doesn't crash (title lookup returns empty)."""
    tracker = PrismaTracker(db_path=str(tmp_path / "unknown.db"))
    # Screen without identifying first
    tracker.screen("nonexistent", passed=True, reason="test")
    counts = tracker.flow_counts()
    assert counts["screened"] == 1
    tracker.close()


def test_include_without_screening(tmp_path):
    """Including directly (skipping screen/eligible) still logs."""
    tracker = PrismaTracker(db_path=str(tmp_path / "skip.db"))
    tracker.identify("Paper", "src", "h1")
    tracker.include("h1", reason="direct")
    counts = tracker.flow_counts()
    assert counts["identified"] == 1
    assert counts["included"] == 1
    tracker.close()


def test_same_paper_multiple_exclusions(tmp_path):
    """A paper can be excluded at multiple stages (logs all)."""
    tracker = PrismaTracker(db_path=str(tmp_path / "multi.db"))
    tracker.identify("Paper", "src", "h1")
    tracker.screen("h1", passed=False, reason="first_fail")
    # Hypothetically re-evaluated and excluded again
    tracker.exclude("h1", "eligible", "second_fail")
    counts = tracker.flow_counts()
    assert counts["excluded"] == 2
    tracker.close()


def test_special_chars_in_reason(tmp_path):
    """Reasons with special characters don't break SQL or counting."""
    tracker = PrismaTracker(db_path=str(tmp_path / "special.db"))
    tracker.identify("Paper", "src", "h1")
    tracker.screen("h1", passed=False, reason="reason with 'quotes' and \"double\"")
    tracker.identify("Paper2", "src", "h2")
    tracker.screen("h2", passed=False, reason="reason: with; semicolons")
    counts = tracker.flow_counts()
    assert counts["excluded"] == 2
    tracker.close()


def test_very_long_title(tmp_path):
    """Very long title doesn't break PRISMA logging."""
    tracker = PrismaTracker(db_path=str(tmp_path / "long.db"))
    long_title = "A" * 50000
    tracker.identify(long_title, "src", "h1")
    counts = tracker.flow_counts()
    assert counts["identified"] == 1
    tracker.close()


def test_concurrent_prisma_reads(tmp_path):
    """Multiple flow_counts calls don't interfere."""
    tracker = PrismaTracker(db_path=str(tmp_path / "concurrent.db"))
    for i in range(50):
        tracker.identify(f"P{i}", "src", f"h{i}")
    # Read counts multiple times
    c1 = tracker.flow_counts()
    c2 = tracker.flow_counts()
    c3 = tracker.flow_counts()
    assert c1 == c2 == c3
    assert c1["identified"] == 50
    tracker.close()


def test_empty_database_flow_counts(tmp_path):
    """Flow counts on empty database returns all zeros."""
    tracker = PrismaTracker(db_path=str(tmp_path / "empty.db"))
    counts = tracker.flow_counts()
    assert all(v == 0 for v in counts.values())
    assert len(counts) == 5  # all 5 stages
    tracker.close()


def test_exclusion_reasons_empty_stage(tmp_path):
    """Exclusion reasons for a stage with no exclusions returns empty list."""
    tracker = PrismaTracker(db_path=str(tmp_path / "noreason.db"))
    tracker.identify("P1", "src", "h1")
    tracker.screen("h1", passed=True, reason="ok")
    reasons = tracker.exclusion_reasons("screened")
    assert reasons == []
    tracker.close()
