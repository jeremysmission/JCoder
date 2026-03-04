"""Devil's advocate engine must generate counter-evidence and balance reports."""

from unittest.mock import MagicMock

from core.devils_advocate import BalanceReport, DevilsAdvocate


def _make_papers(n, prefix="Paper"):
    """Helper: build a list of n paper dicts."""
    return [{"title": f"{prefix} {i}", "abstract": f"Abstract {i}"} for i in range(n)]


# -----------------------------------------------------------------------
# Counter-query generation
# -----------------------------------------------------------------------

def test_counter_queries_heuristic():
    """No runtime -- heuristic appends ' criticism' / ' limitations' / ' debunked'."""
    da = DevilsAdvocate()
    queries = da._generate_counter_queries("Transformers outperform RNNs")

    assert len(queries) == 3
    assert queries[0] == "Transformers outperform RNNs criticism"
    assert queries[1] == "Transformers outperform RNNs limitations"
    assert queries[2] == "Transformers outperform RNNs debunked"


# -----------------------------------------------------------------------
# Balance ratio
# -----------------------------------------------------------------------

def test_balance_ratio_all_supporting():
    """5 supporting + 0 opposing => ratio 0.0."""
    da = DevilsAdvocate()
    ratio = da._compute_balance(_make_papers(5), [])
    assert ratio == 0.0


def test_balance_ratio_balanced():
    """3 supporting + 3 opposing => ratio 0.5."""
    da = DevilsAdvocate()
    ratio = da._compute_balance(_make_papers(3), _make_papers(3, "Counter"))
    assert ratio == 0.5


def test_balance_ratio_no_sources():
    """Empty lists => ratio 0.0."""
    da = DevilsAdvocate()
    ratio = da._compute_balance([], [])
    assert ratio == 0.0


# -----------------------------------------------------------------------
# Fetch function integration
# -----------------------------------------------------------------------

def test_fetch_opposing_calls_fetch_fn():
    """fetch_fn must be called once per counter query."""
    fetch_fn = MagicMock(return_value=[{"title": "Counter Paper"}])
    da = DevilsAdvocate(fetch_fn=fetch_fn)

    queries = ["q1", "q2", "q3"]
    results = da._fetch_opposing(queries)

    assert fetch_fn.call_count == 3
    fetch_fn.assert_any_call("q1")
    fetch_fn.assert_any_call("q2")
    fetch_fn.assert_any_call("q3")
    # Dedup keeps only 1 since title is identical across calls.
    assert len(results) == 1
    assert results[0]["title"] == "Counter Paper"


# -----------------------------------------------------------------------
# Full challenge flow
# -----------------------------------------------------------------------

def test_challenge_full_flow():
    """Mock runtime + mock fetch_fn, verify all BalanceReport fields."""
    runtime = MagicMock()
    runtime.generate.return_value = (
        "query against claim\n"
        "claim debunked evidence\n"
        "claim failure cases"
    )

    def fake_fetch(query):
        return [{"title": f"Opposing: {query}", "abstract": "opp"}]

    da = DevilsAdvocate(runtime=runtime, fetch_fn=fake_fetch)
    supporting = _make_papers(4)

    report = da.challenge("RAG is always better", supporting, max_counter_queries=3)

    assert isinstance(report, BalanceReport)
    assert report.original_claim == "RAG is always better"
    assert len(report.counter_queries) == 3
    assert report.supporting_count == 4
    assert report.opposing_count == 3
    assert 0.0 <= report.balance_ratio <= 1.0
    assert report.strongest_counter.startswith("Opposing:")


# -----------------------------------------------------------------------
# Strongest counter
# -----------------------------------------------------------------------

def test_strongest_counter():
    """First opposing paper title is captured as strongest_counter."""
    def fake_fetch(query):
        return [
            {"title": "Strong Counter Evidence", "abstract": "first"},
            {"title": "Weak Counter", "abstract": "second"},
        ]

    da = DevilsAdvocate(fetch_fn=fake_fetch)
    supporting = _make_papers(2)

    report = da.challenge("Hypothesis X", supporting, max_counter_queries=1)

    assert report.strongest_counter == "Strong Counter Evidence"


# -----------------------------------------------------------------------
# Bare-minimum instantiation
# -----------------------------------------------------------------------

def test_no_runtime_no_fetch():
    """DevilsAdvocate() with no args still produces a valid report."""
    da = DevilsAdvocate()
    report = da.challenge("Some claim", [])

    assert isinstance(report, BalanceReport)
    assert report.original_claim == "Some claim"
    assert len(report.counter_queries) == 3  # heuristic fallback
    assert report.supporting_count == 0
    assert report.opposing_count == 0
    assert report.balance_ratio == 0.0
    assert report.strongest_counter == ""
    assert report.blind_spots == []
