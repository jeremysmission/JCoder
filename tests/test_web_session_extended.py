"""
Extended tests for agent.web_tools and agent.session
-----------------------------------------------------
Covers edge cases and scenarios not in the base test suites:

Web Tools:
  - URL validation via NetworkGate allowlist
  - Content extraction: nested/malformed HTML, encoding
  - Search result parsing: missing fields, duplicates
  - Timeout handling on fetch
  - Blocked-domain filtering via gate modes

Session:
  - Full create/save/load cycle
  - Resume from disk
  - Corrupted session file handling
  - Session metadata (timestamps, query count)
  - Multiple sessions isolation
  - Session cleanup/expiry
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ------------------------------------------------------------------
# Shim for jcoder.core.* imports (same as test_web_tools.py)
# ------------------------------------------------------------------
import core.http_factory as _real_http_factory
import core.network_gate as _real_network_gate

_jcoder_pkg = types.ModuleType("jcoder")
_jcoder_pkg.__path__ = []
_jcoder_core_pkg = types.ModuleType("jcoder.core")
_jcoder_core_pkg.__path__ = []

sys.modules.setdefault("jcoder", _jcoder_pkg)
sys.modules.setdefault("jcoder.core", _jcoder_core_pkg)
sys.modules.setdefault("jcoder.core.http_factory", _real_http_factory)
sys.modules.setdefault("jcoder.core.network_gate", _real_network_gate)

from agent.web_tools import WebSearcher, _extract_text, _strip_html  # noqa: E402
from agent.session import SessionStore, SessionInfo  # noqa: E402
from core.network_gate import NetworkGate  # noqa: E402


# ===================================================================
# Helpers
# ===================================================================

def _mock_response(status_code=200, text="", raise_on_status=False):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    if raise_on_status:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.fixture()
def searcher():
    with patch("agent.web_tools.make_client") as mf:
        mock_client = MagicMock(spec=httpx.Client)
        mf.return_value = mock_client
        ws = WebSearcher(network_gate=None, timeout_s=10, max_results=5)
    ws._mock = mock_client
    return ws


@pytest.fixture()
def session_dir(tmp_path):
    return str(tmp_path / "sessions")


@pytest.fixture()
def store(session_dir):
    return SessionStore(store_dir=session_dir)


# ===================================================================
# WEB TOOLS -- URL validation / sanitization via NetworkGate
# ===================================================================

class TestURLValidation:
    """Gate-based URL filtering catches blocked domains."""

    def test_allowlist_rejects_subdomain_mismatch(self):
        gate = NetworkGate(mode="allowlist", allowlist=["api.example.com"])
        with patch("agent.web_tools.make_client") as mf:
            mf.return_value = MagicMock(spec=httpx.Client)
            ws = WebSearcher(network_gate=gate)
        with pytest.raises(PermissionError):
            ws.fetch_page("https://evil.example.com/path")

    def test_allowlist_accepts_exact_host(self):
        gate = NetworkGate(mode="allowlist", allowlist=["docs.python.org"])
        with patch("agent.web_tools.make_client") as mf:
            mc = MagicMock(spec=httpx.Client)
            mf.return_value = mc
            ws = WebSearcher(network_gate=gate)
        mc.get.return_value = _mock_response(text="<p>docs</p>")
        text = ws.fetch_page("https://docs.python.org/3/library/")
        assert "docs" in text

    def test_offline_gate_blocks_all_urls(self):
        gate = NetworkGate(mode="offline")
        with patch("agent.web_tools.make_client") as mf:
            mf.return_value = MagicMock(spec=httpx.Client)
            ws = WebSearcher(network_gate=gate)
        for url in ["https://google.com", "http://localhost", "ftp://x"]:
            with pytest.raises(PermissionError):
                ws.fetch_page(url)

    def test_localhost_allows_127(self):
        gate = NetworkGate(mode="localhost")
        with patch("agent.web_tools.make_client") as mf:
            mc = MagicMock(spec=httpx.Client)
            mf.return_value = mc
            ws = WebSearcher(network_gate=gate)
        mc.get.return_value = _mock_response(text="<p>ok</p>")
        text = ws.fetch_page("http://127.0.0.1:11434/api")
        assert "ok" in text


# ===================================================================
# WEB TOOLS -- Content extraction edge cases
# ===================================================================

class TestContentExtraction:
    """HTML-to-text extraction for tricky markup."""

    def test_deeply_nested_tags(self):
        html = "<div><div><div><span><b><i>deep</i></b></span></div></div></div>"
        assert "deep" in _extract_text(html)

    def test_malformed_unclosed_tags(self):
        html = "<p>hello <b>world</p>"
        text = _extract_text(html)
        assert "hello" in text
        assert "world" in text

    def test_multiple_script_blocks_removed(self):
        html = (
            "<script>a()</script><p>keep</p>"
            "<script type='module'>b()</script><p>also</p>"
        )
        text = _extract_text(html)
        assert "keep" in text
        assert "also" in text
        assert "a()" not in text
        assert "b()" not in text

    def test_html_entities_decoded(self):
        html = "<p>5 &gt; 3 &amp; 2 &lt; 4</p>"
        text = _extract_text(html)
        assert "5 > 3 & 2 < 4" in text

    def test_style_with_media_queries_removed(self):
        html = "<style>@media(max-width:600px){body{font-size:12px}}</style><p>vis</p>"
        text = _extract_text(html)
        assert "vis" in text
        assert "font-size" not in text

    def test_empty_page_returns_empty(self):
        assert _extract_text("") == ""
        assert _extract_text("   ") == ""


# ===================================================================
# WEB TOOLS -- Search result parsing edge cases
# ===================================================================

class TestSearchParsingEdge:
    """Edge cases in DuckDuckGo result parsing."""

    def test_html_with_no_results_block(self, searcher):
        searcher._mock.post.return_value = _mock_response(
            text="<html><body>No results</body></html>"
        )
        results = searcher.search_duckduckgo("gibberish")
        assert results == []

    def test_result_with_special_chars_in_url(self, searcher):
        html = (
            '<a class="result__a" href="https://example.com/page?q=a%20b&x=1">'
            "Special Page</a>"
            '<td class="result__snippet">snippet</td>'
        )
        searcher._mock.post.return_value = _mock_response(text=html)
        results = searcher.search_duckduckgo("special")
        assert len(results) == 1
        assert "q=a%20b" in results[0]["url"]

    def test_more_links_than_snippets(self, searcher):
        html = (
            '<a class="result__a" href="https://a.com">A</a>'
            '<a class="result__a" href="https://b.com">B</a>'
            '<td class="result__snippet">only one snippet</td>'
        )
        searcher._mock.post.return_value = _mock_response(text=html)
        results = searcher.search_duckduckgo("test")
        assert len(results) == 2
        assert results[0]["snippet"] != ""
        assert results[1]["snippet"] == ""


# ===================================================================
# WEB TOOLS -- Timeout handling
# ===================================================================

class TestTimeoutHandling:
    """Timeout errors propagate correctly."""

    def test_search_read_timeout(self, searcher):
        searcher._mock.post.side_effect = httpx.ReadTimeout("timed out")
        with pytest.raises(httpx.ReadTimeout):
            searcher.search_duckduckgo("slow query")

    def test_fetch_connect_timeout(self, searcher):
        searcher._mock.get.side_effect = httpx.ConnectTimeout("connect timeout")
        with pytest.raises(httpx.ConnectTimeout):
            searcher.fetch_page("https://slow.example.com")

    def test_search_and_summarize_handles_timeout_gracefully(self, searcher):
        ddg_html = (
            '<a class="result__a" href="https://example.com">Title</a>'
            '<td class="result__snippet">snip</td>'
        )
        searcher._mock.post.return_value = _mock_response(text=ddg_html)
        searcher._mock.get.side_effect = httpx.ReadTimeout("timed out")
        text = searcher.search_and_summarize("timeout test")
        assert "Failed to fetch" in text


# ===================================================================
# WEB TOOLS -- Blocked domain filtering
# ===================================================================

class TestBlockedDomainFiltering:
    """Allowlist gate acts as a domain blocklist (inverse)."""

    def test_gate_blocks_search_to_non_ddg(self):
        gate = NetworkGate(mode="allowlist", allowlist=["safe.example.com"])
        with patch("agent.web_tools.make_client") as mf:
            mf.return_value = MagicMock(spec=httpx.Client)
            ws = WebSearcher(network_gate=gate)
        # DDG host is html.duckduckgo.com -- not in allowlist
        with pytest.raises(PermissionError):
            ws.search_duckduckgo("test")

    def test_multiple_allowed_domains(self):
        gate = NetworkGate(
            mode="allowlist",
            allowlist=["a.com", "b.com", "c.com"],
        )
        with patch("agent.web_tools.make_client") as mf:
            mc = MagicMock(spec=httpx.Client)
            mf.return_value = mc
            ws = WebSearcher(network_gate=gate)
        mc.get.return_value = _mock_response(text="<p>ok</p>")
        # All three should succeed
        for host in ["a.com", "b.com", "c.com"]:
            ws.fetch_page(f"https://{host}/page")
        assert mc.get.call_count == 3


# ===================================================================
# SESSION -- Create / save / load cycle
# ===================================================================

class TestSessionCycle:
    """Basic create, save, load round-trip."""

    def test_save_and_load(self, store):
        history = [{"role": "user", "content": "hello"}]
        store.save("s1", task="test task", history=history, status="active")
        data = store.load("s1")
        assert data["session_id"] == "s1"
        assert data["task"] == "test task"
        assert data["history"] == history
        assert data["status"] == "active"
        assert data["message_count"] == 1

    def test_save_overwrites_existing(self, store):
        store.save("s1", task="v1", history=[])
        store.save("s1", task="v2", history=[{"role": "user", "content": "x"}])
        data = store.load("s1")
        assert data["task"] == "v2"
        assert data["message_count"] == 1

    def test_load_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_save_creates_directory(self, tmp_path):
        deep = str(tmp_path / "a" / "b" / "c")
        s = SessionStore(store_dir=deep)
        s.save("x", task="t", history=[])
        assert s.load("x")["task"] == "t"


# ===================================================================
# SESSION -- Resume from disk
# ===================================================================

class TestSessionResume:
    """Resume history from a previously saved session."""

    def test_resume_history_returns_messages(self, store):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        store.save("r1", task="resume test", history=msgs)
        resumed = store.resume_history("r1")
        assert resumed == msgs

    def test_resume_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.resume_history("ghost")

    def test_resume_preserves_order(self, store):
        msgs = [{"role": "user", "content": str(i)} for i in range(20)]
        store.save("r2", task="order", history=msgs)
        assert store.resume_history("r2") == msgs


# ===================================================================
# SESSION -- Corrupted session file handling
# ===================================================================

class TestCorruptedSession:
    """Graceful handling of corrupted JSON on disk."""

    def test_load_corrupted_json_raises(self, store, session_dir):
        path = Path(session_dir) / "bad.json"
        path.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            store.load("bad")

    def test_list_skips_corrupted_files(self, store, session_dir):
        store.save("good", task="ok", history=[])
        bad_path = Path(session_dir) / "bad.json"
        bad_path.write_text("NOT JSON", encoding="utf-8")
        sessions = store.list_sessions()
        ids = [s.session_id for s in sessions]
        assert "good" in ids
        assert "bad" not in ids

    def test_save_over_corrupted_succeeds(self, store, session_dir):
        bad_path = Path(session_dir) / "fix.json"
        bad_path.write_text("{broken", encoding="utf-8")
        store.save("fix", task="repaired", history=[])
        data = store.load("fix")
        assert data["task"] == "repaired"

    def test_cleanup_skips_corrupted(self, store, session_dir):
        bad_path = Path(session_dir) / "corrupt.json"
        bad_path.write_text("nope", encoding="utf-8")
        deleted = store.cleanup(max_age_days=0)
        # Corrupted file has no parseable date, so cleanup skips it
        assert bad_path.exists()


# ===================================================================
# SESSION -- Metadata (timestamps, query count)
# ===================================================================

class TestSessionMetadata:
    """Verify timestamps and counters are stored correctly."""

    def test_created_at_preserved_on_update(self, store):
        store.save("m1", task="t", history=[])
        first = store.load("m1")["created_at"]
        store.save("m1", task="t", history=[{"role": "user", "content": "x"}])
        second = store.load("m1")
        assert second["created_at"] == first
        assert second["updated_at"] >= first

    def test_iterations_and_tokens_stored(self, store):
        store.save("m2", task="t", history=[], iterations=5, tokens=1234)
        data = store.load("m2")
        assert data["iterations"] == 5
        assert data["total_tokens"] == 1234

    def test_message_count_matches_history(self, store):
        msgs = [{"role": "user", "content": str(i)} for i in range(7)]
        store.save("m3", task="t", history=msgs)
        data = store.load("m3")
        assert data["message_count"] == 7

    def test_status_values(self, store):
        for status in ("active", "completed", "failed"):
            sid = f"status_{status}"
            store.save(sid, task="t", history=[], status=status)
            assert store.load(sid)["status"] == status

    def test_optional_token_breakdown(self, store):
        store.save("m4", task="t", history=[], input_tokens=100, output_tokens=200)
        data = store.load("m4")
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 200


# ===================================================================
# SESSION -- Multiple sessions isolation
# ===================================================================

class TestSessionIsolation:
    """Multiple sessions do not interfere with each other."""

    def test_distinct_sessions_have_distinct_data(self, store):
        store.save("iso1", task="task A", history=[{"role": "user", "content": "A"}])
        store.save("iso2", task="task B", history=[{"role": "user", "content": "B"}])
        a = store.load("iso1")
        b = store.load("iso2")
        assert a["task"] == "task A"
        assert b["task"] == "task B"
        assert a["history"] != b["history"]

    def test_delete_one_preserves_others(self, store):
        store.save("keep", task="keep", history=[])
        store.save("gone", task="gone", history=[])
        store.delete("gone")
        assert store.load("keep")["task"] == "keep"
        with pytest.raises(FileNotFoundError):
            store.load("gone")

    def test_list_sessions_returns_all(self, store):
        for i in range(5):
            store.save(f"ls{i}", task=f"task {i}", history=[])
        listed = store.list_sessions()
        ids = {s.session_id for s in listed}
        for i in range(5):
            assert f"ls{i}" in ids

    def test_list_sessions_filter_by_status(self, store):
        store.save("a", task="t", history=[], status="active")
        store.save("c", task="t", history=[], status="completed")
        active = store.list_sessions(status="active")
        assert all(s.status == "active" for s in active)
        assert any(s.session_id == "a" for s in active)

    def test_search_finds_matching_task(self, store):
        store.save("s1", task="fix login bug", history=[])
        store.save("s2", task="add feature X", history=[])
        results = store.search("login")
        assert len(results) == 1
        assert results[0].session_id == "s1"


# ===================================================================
# SESSION -- Cleanup / expiry
# ===================================================================

class TestSessionCleanup:
    """Session cleanup removes old sessions."""

    def test_cleanup_removes_old_sessions(self, store, session_dir):
        # Write a session with an old updated_at
        old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        payload = {
            "session_id": "old1",
            "task": "ancient",
            "created_at": old_time,
            "updated_at": old_time,
            "status": "completed",
            "iterations": 0,
            "total_tokens": 0,
            "message_count": 0,
            "history": [],
        }
        path = Path(session_dir) / "old1.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        # Save a recent session normally
        store.save("new1", task="recent", history=[])
        deleted = store.cleanup(max_age_days=30)
        assert deleted == 1
        with pytest.raises(FileNotFoundError):
            store.load("old1")
        assert store.load("new1")["task"] == "recent"

    def test_cleanup_zero_days_keeps_fresh_sessions(self, store):
        # max_age_days=0 means cutoff is "now", so sessions just created
        # are not older than now and should survive.
        store.save("a", task="t", history=[])
        store.save("b", task="t", history=[])
        deleted = store.cleanup(max_age_days=0)
        assert deleted == 0

    def test_delete_returns_true_if_existed(self, store):
        store.save("d1", task="t", history=[])
        assert store.delete("d1") is True

    def test_delete_returns_false_if_missing(self, store):
        assert store.delete("nope") is False

    def test_list_sessions_respects_limit(self, store):
        for i in range(10):
            store.save(f"lim{i}", task="t", history=[])
        listed = store.list_sessions(limit=3)
        assert len(listed) == 3
