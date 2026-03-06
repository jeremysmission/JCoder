"""Tests for MinHash deduplication module."""

import pytest

from ingestion.dedup import DedupStats, MinHashDedup


class TestMinHashDedup:
    def test_exact_duplicate(self):
        d = MinHashDedup()
        assert d.is_duplicate("the quick brown fox jumps over the lazy dog") is False
        assert d.is_duplicate("the quick brown fox jumps over the lazy dog") is True

    def test_unique_texts(self):
        d = MinHashDedup()
        assert d.add("alpha bravo charlie delta echo") is True
        assert d.add("foxtrot golf hotel india juliet") is True
        assert d.add("kilo lima mike november oscar") is True

    def test_near_duplicate(self):
        d = MinHashDedup()
        original = "the quick brown fox jumps over the lazy dog near the river bank on a sunny day"
        tweaked = "the quick brown fox jumps over the lazy dog near the river bank on a rainy day"
        assert d.is_duplicate(original) is False
        assert d.is_duplicate(tweaked) is True

    def test_below_threshold(self):
        d = MinHashDedup(threshold=0.8)
        assert d.add("apples oranges bananas grapes kiwis mangoes") is True
        assert d.add("python java rust golang typescript haskell") is True

    def test_is_duplicate_and_add(self):
        d = MinHashDedup()
        text = "some reusable content that will be added twice to the index"
        assert d.is_duplicate(text) is False  # first time: not a dupe
        assert d.is_duplicate(text) is True   # exact dupe
        assert d.add(text) is False           # add returns False for dupes

    def test_stats(self):
        d = MinHashDedup()
        d.add("unique text number one for stats test")
        d.add("unique text number two for stats test")
        d.add("unique text number one for stats test")  # exact dupe
        s = d.stats()
        assert s.total_seen == 3
        assert s.unique == 2
        assert s.exact_dupes == 1
        assert isinstance(s, DedupStats)

    def test_persistence(self, tmp_path):
        p = str(tmp_path / "dedup_state.json")
        d1 = MinHashDedup(persist_path=p)
        d1.add("persistent content alpha bravo charlie delta")
        d1.save()

        d2 = MinHashDedup(persist_path=p)  # auto-loads in __init__
        assert d2.is_duplicate("persistent content alpha bravo charlie delta") is True
        assert d2.stats().total_seen == 2
        assert d2.stats().exact_dupes == 1

    def test_reset(self):
        d = MinHashDedup()
        text = "text that will be cleared from the dedup index"
        d.add(text)
        assert d.is_duplicate(text) is True
        d.reset()
        assert d.is_duplicate(text) is False
        assert d.stats().unique == 1

    def test_shingle_generation(self):
        shingles = MinHashDedup._shingle("abcdef", k=3)
        assert shingles == {"abc", "bcd", "cde", "def"}

    def test_jaccard_similarity(self):
        import numpy as np

        sig_a = np.array([1, 2, 3, 4], dtype=np.uint32)
        assert MinHashDedup.jaccard_similarity(sig_a, sig_a) == 1.0
        sig_b = np.array([1, 2, 9, 9], dtype=np.uint32)
        assert MinHashDedup.jaccard_similarity(sig_a, sig_b) == 0.5

    def test_deterministic(self):
        d = MinHashDedup()
        text = "determinism check with fixed seed value"
        sh = d._shingle(text)
        sig1 = d._minhash(sh)
        sig2 = d._minhash(sh)
        assert (sig1 == sig2).all()

    def test_empty_text(self):
        d = MinHashDedup()
        assert d.add("") is True  # empty is "unique" (no prior empty)
        assert d.add("") is False  # second empty is exact dupe

    def test_short_text(self):
        d = MinHashDedup()
        assert d.add("hi") is True  # shorter than default shingle k=5
        shingles = MinHashDedup._shingle("hi", k=5)
        assert shingles == {"hi"}

    def test_custom_threshold(self):
        base = "the quick brown fox jumps over the lazy dog by the riverbank"
        small_edit = "the quick brown cat jumps over the lazy dog by the riverbank"

        strict = MinHashDedup(threshold=0.9)
        strict.add(base)
        strict_flags = strict.is_duplicate(small_edit)

        loose = MinHashDedup(threshold=0.5)
        loose.add(base)
        loose_flags = loose.is_duplicate(small_edit)

        # Loose threshold should catch at least as many dupes as strict
        if strict_flags:
            assert loose_flags

    def test_large_batch(self):
        import hashlib as _hl

        d = MinHashDedup()
        for i in range(1000):
            h = _hl.sha256(str(i).encode()).hexdigest()
            text = f"document {h} category {h[:8]} region {h[8:16]} payload {h[16:32]}"
            assert d.add(text) is True, f"false positive at i={i}"
        s = d.stats()
        assert s.unique == 1000
        assert s.exact_dupes == 0
        assert s.near_dupes == 0
