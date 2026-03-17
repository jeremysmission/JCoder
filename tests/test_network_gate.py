"""Tests for core.network_gate -- URL access policy enforcement."""

import pytest
from core.network_gate import NetworkGate


class TestModes:

    def test_valid_modes(self):
        for mode in ("offline", "localhost", "allowlist"):
            gate = NetworkGate(mode=mode)
            assert gate.mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            NetworkGate(mode="permissive")


class TestOffline:

    def test_blocks_everything(self):
        gate = NetworkGate(mode="offline")
        assert gate.allow("http://localhost:8000/v1") is False
        assert gate.allow("https://api.openai.com/v1") is False
        assert gate.allow("http://127.0.0.1:11434") is False

    def test_guard_raises(self):
        gate = NetworkGate(mode="offline")
        with pytest.raises(PermissionError, match="blocked"):
            gate.guard("http://localhost:8000")


class TestLocalhost:

    def test_allows_localhost(self):
        gate = NetworkGate(mode="localhost")
        assert gate.allow("http://localhost:8000/v1/chat/completions") is True

    def test_allows_127(self):
        gate = NetworkGate(mode="localhost")
        assert gate.allow("http://127.0.0.1:8001/v1/embeddings") is True

    def test_allows_ipv6_loopback(self):
        gate = NetworkGate(mode="localhost")
        assert gate.allow("http://[::1]:8000/v1") is True

    def test_blocks_external(self):
        gate = NetworkGate(mode="localhost")
        assert gate.allow("https://api.openai.com/v1") is False
        assert gate.allow("https://huggingface.co") is False

    def test_guard_allows_localhost(self):
        gate = NetworkGate(mode="localhost")
        gate.guard("http://localhost:8000")  # should not raise

    def test_guard_blocks_external(self):
        gate = NetworkGate(mode="localhost")
        with pytest.raises(PermissionError):
            gate.guard("https://api.openai.com/v1")


class TestAllowlist:

    def test_allows_listed_hosts(self):
        gate = NetworkGate(mode="allowlist", allowlist=["api.openai.com", "localhost"])
        assert gate.allow("https://api.openai.com/v1/chat") is True
        assert gate.allow("http://localhost:8000") is True

    def test_blocks_unlisted_hosts(self):
        gate = NetworkGate(mode="allowlist", allowlist=["api.openai.com"])
        assert gate.allow("https://evil.com/steal") is False
        assert gate.allow("http://localhost:8000") is False

    def test_empty_allowlist_blocks_all(self):
        gate = NetworkGate(mode="allowlist", allowlist=[])
        assert gate.allow("http://localhost:8000") is False

    def test_guard_raises_for_unlisted(self):
        gate = NetworkGate(mode="allowlist", allowlist=["safe.com"])
        with pytest.raises(PermissionError):
            gate.guard("https://evil.com/payload")


class TestEdgeCases:

    def test_malformed_url(self):
        gate = NetworkGate(mode="localhost")
        # No hostname parses to empty string
        assert gate.allow("not-a-url") is False

    def test_none_allowlist(self):
        gate = NetworkGate(mode="allowlist", allowlist=None)
        assert gate.allow("http://anything.com") is False
