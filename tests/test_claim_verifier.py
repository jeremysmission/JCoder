"""Claim verifier must cross-reference sources and produce valid results."""

from __future__ import annotations

from typing import Dict, List

import pytest

from core.claim_verifier import ClaimVerifier, VerificationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(title: str, abstract: str) -> Dict:
    """Build a minimal source dict."""
    return {"title": title, "abstract": abstract}


def _fetch_corroborating(query: str) -> List[Dict]:
    """Return sources whose keywords overlap heavily with any query."""
    return [
        _make_source(
            "Distributed consensus in cloud systems",
            "Raft consensus algorithm achieves distributed agreement across nodes",
        ),
        _make_source(
            "Consensus protocols survey",
            "Raft protocol provides leader-based distributed consensus",
        ),
        _make_source(
            "Distributed systems reliability",
            "The Raft algorithm is widely used for distributed consensus",
        ),
    ]


def _fetch_contradicting(query: str) -> List[Dict]:
    """Return sources that share almost no keywords with a Raft claim."""
    return [
        _make_source(
            "Quantum entanglement in photonic lattices",
            "We observe Bell inequality violations in silicon photonic chips",
        ),
        _make_source(
            "Spectral analysis of gamma-ray bursts",
            "Fermi telescope detects high-energy photon emission from magnetars",
        ),
    ]


def _fetch_mixed(query: str) -> List[Dict]:
    """Return one corroborating and two contradicting sources."""
    return [
        _make_source(
            "Raft consensus explained",
            "Raft is a distributed consensus algorithm for replicated logs",
        ),
        _make_source(
            "Quantum entanglement in photonic lattices",
            "We observe Bell inequality violations in silicon photonic chips",
        ),
        _make_source(
            "Spectral analysis of gamma-ray bursts",
            "Fermi telescope detects high-energy photon emission from magnetars",
        ),
    ]


# ---------------------------------------------------------------------------
# test_verify_no_sources
# ---------------------------------------------------------------------------

def test_verify_no_sources():
    """No fetch_fn means nothing can be verified."""
    verifier = ClaimVerifier(runtime=None, fetch_fn=None)
    result = verifier.verify("Raft uses a leader-based protocol")

    assert isinstance(result, VerificationResult)
    assert result.verified is False
    assert result.confidence == 0.0
    assert result.verification_method == "unverifiable"
    assert result.corroborating_sources == []
    assert result.contradicting_sources == []


# ---------------------------------------------------------------------------
# test_verify_corroborated
# ---------------------------------------------------------------------------

def test_verify_corroborated():
    """2+ corroborating sources should mark the claim as verified."""
    verifier = ClaimVerifier(runtime=None, fetch_fn=_fetch_corroborating)
    result = verifier.verify("Raft is a distributed consensus algorithm")

    assert result.verified is True
    assert len(result.corroborating_sources) >= 2
    assert result.confidence > 0.0
    assert result.verification_method == "cross_reference"


# ---------------------------------------------------------------------------
# test_verify_contradicted
# ---------------------------------------------------------------------------

def test_verify_contradicted():
    """When sources do not corroborate, verified should be False."""
    verifier = ClaimVerifier(runtime=None, fetch_fn=_fetch_contradicting)
    result = verifier.verify("Raft is a distributed consensus algorithm")

    assert result.verified is False
    assert result.verification_method == "cross_reference"


# ---------------------------------------------------------------------------
# test_confidence_scaling
# ---------------------------------------------------------------------------

def test_confidence_scaling():
    """Confidence scales with corroborating count: 0=0.0, 1=0.4, 2=0.7, 3+=0.9."""
    verifier = ClaimVerifier(runtime=None)

    conf_0 = verifier._compute_confidence([], [])
    conf_1 = verifier._compute_confidence([{"t": 1}], [])
    conf_2 = verifier._compute_confidence([{"t": 1}, {"t": 2}], [])
    conf_3 = verifier._compute_confidence(
        [{"t": 1}, {"t": 2}, {"t": 3}], []
    )

    assert conf_0 == pytest.approx(0.0)
    assert conf_1 == pytest.approx(0.4)
    assert conf_2 == pytest.approx(0.7)
    assert conf_3 == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# test_confidence_reduction
# ---------------------------------------------------------------------------

def test_confidence_reduction():
    """Contradicting sources reduce confidence by 0.15 each, clamped to 0."""
    verifier = ClaimVerifier(runtime=None)

    # 2 corroborating (base 0.7) minus 1 contradicting (0.15) = 0.55
    conf = verifier._compute_confidence(
        [{"t": 1}, {"t": 2}],
        [{"t": 3}],
    )
    assert conf == pytest.approx(0.55)

    # 1 corroborating (base 0.4) minus 3 contradicting (0.45) = 0.0 (clamped)
    conf_clamped = verifier._compute_confidence(
        [{"t": 1}],
        [{"t": 2}, {"t": 3}, {"t": 4}],
    )
    assert conf_clamped == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# test_verification_queries_heuristic
# ---------------------------------------------------------------------------

def test_verification_queries_heuristic():
    """Without a runtime, key nouns should be extracted into queries."""
    verifier = ClaimVerifier(runtime=None)
    queries = verifier._generate_verification_queries(
        "The Raft algorithm achieves consensus through leader election"
    )

    assert len(queries) >= 1
    # Stop words stripped -- "The" and "through" should not appear alone
    joined = " ".join(queries).lower()
    assert "raft" in joined
    assert "consensus" in joined or "algorithm" in joined


# ---------------------------------------------------------------------------
# test_verify_batch
# ---------------------------------------------------------------------------

def test_verify_batch():
    """Batch verification returns one result per claim."""
    verifier = ClaimVerifier(runtime=None, fetch_fn=_fetch_corroborating)
    claims = [
        "Raft uses leader election",
        "Paxos is a consensus protocol",
        "PBFT tolerates Byzantine faults",
    ]
    results = verifier.verify_batch(claims)

    assert len(results) == 3
    assert all(isinstance(r, VerificationResult) for r in results)


# ---------------------------------------------------------------------------
# test_cross_reference_heuristic
# ---------------------------------------------------------------------------

def test_cross_reference_heuristic():
    """Heuristic classification: high overlap = corroborating, low = contradicting."""
    verifier = ClaimVerifier(runtime=None)

    candidates = [
        _make_source(
            "Raft consensus protocol",
            "Raft achieves distributed consensus via leader election and log replication",
        ),
        _make_source(
            "Quantum tunneling in superconductors",
            "Josephson junctions exhibit macroscopic quantum tunneling effects",
        ),
    ]

    corr, contra = verifier._cross_reference_heuristic(
        "Raft is a distributed consensus algorithm", candidates
    )

    # First source should corroborate (heavy keyword overlap)
    assert len(corr) >= 1
    assert corr[0]["title"] == "Raft consensus protocol"

    # Second source should contradict or be irrelevant (near-zero overlap)
    # Either way it should NOT appear in corroborating
    corr_titles = {s["title"] for s in corr}
    assert "Quantum tunneling in superconductors" not in corr_titles
