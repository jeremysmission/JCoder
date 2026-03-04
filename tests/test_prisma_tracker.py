"""PRISMA tracker must log papers through every pipeline stage."""

from core.prisma_tracker import PrismaTracker


def test_identify_creates_record(tmp_path):
    tracker = PrismaTracker(str(tmp_path / "prisma.db"))
    tracker.identify("Paper A", "arxiv", "hash_a")

    counts = tracker.flow_counts()
    assert counts["identified"] == 1
    assert counts["screened"] == 0
    tracker.close()


def test_screen_pass(tmp_path):
    tracker = PrismaTracker(str(tmp_path / "prisma.db"))
    tracker.identify("Paper B", "arxiv", "hash_b")
    tracker.screen("hash_b", passed=True, reason="relevant title")

    counts = tracker.flow_counts()
    assert counts["identified"] == 1
    assert counts["screened"] == 1
    assert counts["excluded"] == 0
    tracker.close()


def test_screen_fail_excludes(tmp_path):
    tracker = PrismaTracker(str(tmp_path / "prisma.db"))
    tracker.identify("Paper C", "arxiv", "hash_c")
    tracker.screen("hash_c", passed=False, reason="off-topic")

    counts = tracker.flow_counts()
    assert counts["identified"] == 1
    assert counts["screened"] == 0
    assert counts["excluded"] == 1
    tracker.close()


def test_full_pipeline(tmp_path):
    tracker = PrismaTracker(str(tmp_path / "prisma.db"))

    # 5 papers identified
    for i in range(5):
        tracker.identify(f"Paper {i}", "semantic_scholar", f"hash_{i}")

    # 3 pass screening, 2 fail
    for i in range(3):
        tracker.screen(f"hash_{i}", passed=True, reason="relevant")
    tracker.screen("hash_3", passed=False, reason="off-topic")
    tracker.screen("hash_4", passed=False, reason="duplicate")

    # 2 pass eligibility, 1 fails
    tracker.eligible("hash_0", passed=True, reason="full text available")
    tracker.eligible("hash_1", passed=True, reason="full text available")
    tracker.eligible("hash_2", passed=False, reason="retracted")

    # 1 included
    tracker.include("hash_0", reason="meets all criteria")

    counts = tracker.flow_counts()
    assert counts["identified"] == 5
    assert counts["screened"] == 3
    assert counts["eligible"] == 2
    assert counts["included"] == 1
    assert counts["excluded"] == 3  # 2 screen + 1 eligible
    tracker.close()


def test_exclusion_reasons(tmp_path):
    tracker = PrismaTracker(str(tmp_path / "prisma.db"))

    tracker.identify("Paper X", "arxiv", "hash_x")
    tracker.identify("Paper Y", "arxiv", "hash_y")
    tracker.identify("Paper Z", "arxiv", "hash_z")

    tracker.screen("hash_x", passed=False, reason="off-topic")
    tracker.screen("hash_y", passed=False, reason="off-topic")
    tracker.screen("hash_z", passed=False, reason="duplicate")

    reasons = tracker.exclusion_reasons("screened")
    reason_dict = {r[0]: r[1] for r in reasons}

    assert reason_dict["screened: off-topic"] == 2
    assert reason_dict["screened: duplicate"] == 1
    tracker.close()


def test_flow_diagram(tmp_path):
    tracker = PrismaTracker(str(tmp_path / "prisma.db"))

    for i in range(4):
        tracker.identify(f"Paper {i}", "arxiv", f"hash_{i}")

    tracker.screen("hash_0", passed=True, reason="ok")
    tracker.screen("hash_1", passed=True, reason="ok")
    tracker.screen("hash_2", passed=False, reason="off-topic")
    tracker.screen("hash_3", passed=False, reason="off-topic")

    tracker.eligible("hash_0", passed=True, reason="full text")
    tracker.eligible("hash_1", passed=False, reason="retracted")

    tracker.include("hash_0", reason="meets criteria")

    diagram = tracker.flow_diagram_text()

    assert "Identified:  4" in diagram
    assert "Screened:    2" in diagram
    assert "Eligible:    1" in diagram
    assert "Included:    1" in diagram
    assert "PRISMA Flow Diagram" in diagram
    tracker.close()


def test_duplicate_identify(tmp_path):
    tracker = PrismaTracker(str(tmp_path / "prisma.db"))

    tracker.identify("Paper D", "arxiv", "hash_d")
    tracker.identify("Paper D", "arxiv", "hash_d")

    counts = tracker.flow_counts()
    assert counts["identified"] == 2  # both logged
    tracker.close()
