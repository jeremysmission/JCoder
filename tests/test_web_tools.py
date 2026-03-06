"""
Tests for agent.web_tools
-------------------------
Covers: WebSearcher initialisation, HTML stripping, DuckDuckGo search
parsing, page fetching, truncation, NetworkGate enforcement, and
error handling.  All tests are self-contained with no real network calls.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ------------------------------------------------------------------
# Shim: web_tools.py imports from jcoder.core.{http_factory,network_gate}
# but the real modules live at core.{http_factory,network_gate}.
# Patch sys.modules so the jcoder.core.* imports resolve correctly.
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
from core.network_gate import NetworkGate  # noqa: E402


# ===================================================================
# Helpers -- fake HTTP responses
# ===================================================================


def _make_response(
    status_code: int = 200,
    text: str = "",
    raise_on_status: bool = False,
) -> MagicMock:
    """Return a mock that quacks like httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    if raise_on_status:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# Minimal DuckDuckGo-style HTML fragment with two results.
_DDG_HTML_TWO_RESULTS = """
<div class="result">
  <a class="result__a" href="https://example.com/page1">
    <b>First</b> Result Title
  </a>
  <td class="result__snippet">Snippet for the <b>first</b> result.</td>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/page2">Second Result</a>
  <td class="result__snippet">Snippet for the second result.</td>
</div>
"""

# DuckDuckGo result with a redirect-wrapped URL (uddg= parameter).
_DDG_HTML_REDIRECT = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.example.com%2Fpage&amp;rut=abc">
    Redirect Result
  </a>
  <td class="result__snippet">Snippet with redirect.</td>
</div>
"""


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def searcher():
    """WebSearcher with no gate and a mocked HTTP client."""
    with patch("agent.web_tools.make_client") as mock_factory:
        mock_client = MagicMock(spec=httpx.Client)
        mock_factory.return_value = mock_client
        ws = WebSearcher(network_gate=None, timeout_s=10, max_results=5)
    # Expose mock for per-test patching
    ws._test_client_mock = mock_client
    return ws


@pytest.fixture()
def offline_gate():
    """A NetworkGate in offline mode (blocks everything)."""
    return NetworkGate(mode="offline")


@pytest.fixture()
def localhost_gate():
    """A NetworkGate in localhost mode."""
    return NetworkGate(mode="localhost")


# ===================================================================
# 1. Initialisation
# ===================================================================


class TestInit:
    """WebSearcher construction."""

    def test_default_gate_is_none(self):
        with patch("agent.web_tools.make_client"):
            ws = WebSearcher()
        assert ws._gate is None

    def test_stores_max_results(self):
        with patch("agent.web_tools.make_client"):
            ws = WebSearcher(max_results=10)
        assert ws._max_results == 10

    def test_default_max_results(self):
        with patch("agent.web_tools.make_client"):
            ws = WebSearcher()
        assert ws._max_results == 5

    def test_accepts_gate(self, offline_gate):
        with patch("agent.web_tools.make_client"):
            ws = WebSearcher(network_gate=offline_gate)
        assert ws._gate is offline_gate


# ===================================================================
# 2. _strip_html
# ===================================================================


class TestStripHtml:
    """Module-level _strip_html function."""

    def test_removes_tags(self):
        assert _strip_html("<b>bold</b>") == "bold"

    def test_removes_nested_tags(self):
        assert _strip_html("<div><p>hello <b>world</b></p></div>") == "hello world"

    def test_unescapes_entities(self):
        assert _strip_html("&amp; &lt; &gt;") == "& < >"

    def test_collapses_whitespace(self):
        result = _strip_html("too    many     spaces")
        assert "  " not in result
        assert "too many spaces" == result

    def test_collapses_blank_lines(self):
        result = _strip_html("a\n\n\n\n\nb")
        assert result.count("\n") <= 2

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_plain_text_passthrough(self):
        assert _strip_html("no tags here") == "no tags here"

    def test_self_closing_tags(self):
        result = _strip_html("line<br/>break")
        assert "line" in result
        assert "break" in result
        assert "<br/>" not in result


# ===================================================================
# 3. _extract_text
# ===================================================================


class TestExtractText:
    """Module-level _extract_text function."""

    def test_removes_script_blocks(self):
        html = "<html><script>var x=1;</script><p>Keep this</p></html>"
        assert "var x" not in _extract_text(html)
        assert "Keep this" in _extract_text(html)

    def test_removes_style_blocks(self):
        html = "<style>body{color:red}</style><p>visible</p>"
        assert "color:red" not in _extract_text(html)
        assert "visible" in _extract_text(html)

    def test_removes_noscript_blocks(self):
        html = "<noscript>Enable JS</noscript><p>content</p>"
        assert "Enable JS" not in _extract_text(html)
        assert "content" in _extract_text(html)

    def test_strips_blank_lines(self):
        html = "<p>a</p><p></p><p></p><p>b</p>"
        text = _extract_text(html)
        assert "\n\n\n" not in text


# ===================================================================
# 4. search_duckduckgo -- result parsing
# ===================================================================


class TestSearchResults:
    """Verify search_duckduckgo parses DDG HTML correctly."""

    def test_returns_list_of_dicts(self, searcher):
        searcher._test_client_mock.post.return_value = _make_response(
            text=_DDG_HTML_TWO_RESULTS,
        )
        results = searcher.search_duckduckgo("test query")
        assert isinstance(results, list)
        for r in results:
            assert "title" in r
            assert "url" in r
            assert "snippet" in r

    def test_parses_two_results(self, searcher):
        searcher._test_client_mock.post.return_value = _make_response(
            text=_DDG_HTML_TWO_RESULTS,
        )
        results = searcher.search_duckduckgo("test query")
        assert len(results) == 2
        assert "First" in results[0]["title"]
        assert results[0]["url"] == "https://example.com/page1"
        assert "first" in results[0]["snippet"].lower()

    def test_unwraps_uddg_redirect(self, searcher):
        searcher._test_client_mock.post.return_value = _make_response(
            text=_DDG_HTML_REDIRECT,
        )
        results = searcher.search_duckduckgo("redirect test")
        assert len(results) == 1
        assert results[0]["url"] == "https://real.example.com/page"

    def test_respects_max_results_param(self, searcher):
        searcher._test_client_mock.post.return_value = _make_response(
            text=_DDG_HTML_TWO_RESULTS,
        )
        results = searcher.search_duckduckgo("test", max_results=1)
        assert len(results) == 1

    def test_respects_instance_max_results(self):
        with patch("agent.web_tools.make_client") as mock_factory:
            mock_client = MagicMock(spec=httpx.Client)
            mock_factory.return_value = mock_client
            ws = WebSearcher(max_results=1)
        mock_client.post.return_value = _make_response(text=_DDG_HTML_TWO_RESULTS)
        results = ws.search_duckduckgo("test")
        assert len(results) == 1

    def test_empty_body_returns_empty_list(self, searcher):
        searcher._test_client_mock.post.return_value = _make_response(text="")
        results = searcher.search_duckduckgo("nothing")
        assert results == []

    def test_no_snippets_still_works(self, searcher):
        html_no_snippet = """
        <a class="result__a" href="https://example.com/x">Title Only</a>
        """
        searcher._test_client_mock.post.return_value = _make_response(
            text=html_no_snippet,
        )
        results = searcher.search_duckduckgo("test")
        assert len(results) == 1
        assert results[0]["snippet"] == ""


# ===================================================================
# 5. search_duckduckgo -- error handling
# ===================================================================


class TestSearchErrors:
    """Network and HTTP errors from search_duckduckgo."""

    def test_http_error_propagates(self, searcher):
        searcher._test_client_mock.post.return_value = _make_response(
            status_code=500,
            raise_on_status=True,
        )
        with pytest.raises(httpx.HTTPStatusError):
            searcher.search_duckduckgo("boom")

    def test_connect_error_propagates(self, searcher):
        searcher._test_client_mock.post.side_effect = httpx.ConnectError(
            "Connection refused"
        )
        with pytest.raises(httpx.ConnectError):
            searcher.search_duckduckgo("offline")

    def test_timeout_error_propagates(self, searcher):
        searcher._test_client_mock.post.side_effect = httpx.ReadTimeout(
            "Read timed out"
        )
        with pytest.raises(httpx.ReadTimeout):
            searcher.search_duckduckgo("slow")


# ===================================================================
# 6. fetch_page
# ===================================================================


class TestFetchPage:
    """WebSearcher.fetch_page behaviour."""

    def test_returns_stripped_text(self, searcher):
        html = "<html><body><p>Hello <b>World</b></p></body></html>"
        searcher._test_client_mock.get.return_value = _make_response(text=html)
        text = searcher.fetch_page("https://example.com")
        assert "Hello" in text
        assert "World" in text
        assert "<p>" not in text

    def test_removes_script_tags(self, searcher):
        html = "<script>alert(1)</script><p>safe</p>"
        searcher._test_client_mock.get.return_value = _make_response(text=html)
        text = searcher.fetch_page("https://example.com")
        assert "alert" not in text
        assert "safe" in text

    def test_truncates_to_max_chars(self, searcher):
        long_body = "<p>" + "A" * 500 + "</p>"
        searcher._test_client_mock.get.return_value = _make_response(text=long_body)
        text = searcher.fetch_page("https://example.com", max_chars=100)
        assert len(text) <= 100 + len("\n... [truncated]")
        assert text.endswith("... [truncated]")

    def test_no_truncation_when_under_limit(self, searcher):
        html = "<p>short</p>"
        searcher._test_client_mock.get.return_value = _make_response(text=html)
        text = searcher.fetch_page("https://example.com", max_chars=50_000)
        assert "[truncated]" not in text

    def test_default_max_chars_is_50000(self, searcher):
        long_body = "<p>" + "B" * 60_000 + "</p>"
        searcher._test_client_mock.get.return_value = _make_response(text=long_body)
        text = searcher.fetch_page("https://example.com")
        assert text.endswith("... [truncated]")

    def test_http_error_propagates(self, searcher):
        searcher._test_client_mock.get.return_value = _make_response(
            status_code=404,
            raise_on_status=True,
        )
        with pytest.raises(httpx.HTTPStatusError):
            searcher.fetch_page("https://example.com/missing")

    def test_connect_error_propagates(self, searcher):
        searcher._test_client_mock.get.side_effect = httpx.ConnectError(
            "Connection refused"
        )
        with pytest.raises(httpx.ConnectError):
            searcher.fetch_page("https://example.com")


# ===================================================================
# 7. search_and_summarize
# ===================================================================


class TestSearchAndSummarize:
    """WebSearcher.search_and_summarize combines search + fetch."""

    def test_returns_combined_text(self, searcher):
        searcher._test_client_mock.post.return_value = _make_response(
            text=_DDG_HTML_TWO_RESULTS,
        )
        searcher._test_client_mock.get.return_value = _make_response(
            text="<p>Page body</p>",
        )
        text = searcher.search_and_summarize("query")
        assert "Result 1" in text
        assert "Page body" in text

    def test_no_results_message(self, searcher):
        searcher._test_client_mock.post.return_value = _make_response(text="")
        text = searcher.search_and_summarize("nothing")
        assert text == "No search results found."

    def test_fetch_failure_captured_inline(self, searcher):
        searcher._test_client_mock.post.return_value = _make_response(
            text=_DDG_HTML_TWO_RESULTS,
        )
        searcher._test_client_mock.get.side_effect = httpx.ConnectError(
            "Connection refused"
        )
        text = searcher.search_and_summarize("query")
        assert "Failed to fetch" in text

    def test_includes_url_in_output(self, searcher):
        """Each result section should contain the source URL."""
        searcher._test_client_mock.post.return_value = _make_response(
            text=_DDG_HTML_TWO_RESULTS,
        )
        searcher._test_client_mock.get.return_value = _make_response(
            text="<p>body</p>",
        )
        text = searcher.search_and_summarize("test")
        assert "https://example.com/page1" in text


# ===================================================================
# 8. NetworkGate enforcement
# ===================================================================


class TestNetworkGateBlocking:
    """When a NetworkGate is attached, blocked URLs raise PermissionError."""

    def test_offline_gate_blocks_search(self, offline_gate):
        with patch("agent.web_tools.make_client") as mf:
            mf.return_value = MagicMock(spec=httpx.Client)
            ws = WebSearcher(network_gate=offline_gate)
        with pytest.raises(PermissionError, match="NetworkGate blocked"):
            ws.search_duckduckgo("anything")

    def test_offline_gate_blocks_fetch(self, offline_gate):
        with patch("agent.web_tools.make_client") as mf:
            mf.return_value = MagicMock(spec=httpx.Client)
            ws = WebSearcher(network_gate=offline_gate)
        with pytest.raises(PermissionError, match="NetworkGate blocked"):
            ws.fetch_page("https://example.com")

    def test_localhost_gate_blocks_external(self, localhost_gate):
        with patch("agent.web_tools.make_client") as mf:
            mf.return_value = MagicMock(spec=httpx.Client)
            ws = WebSearcher(network_gate=localhost_gate)
        with pytest.raises(PermissionError):
            ws.fetch_page("https://example.com")

    def test_localhost_gate_allows_localhost(self, localhost_gate):
        with patch("agent.web_tools.make_client") as mf:
            mock_client = MagicMock(spec=httpx.Client)
            mf.return_value = mock_client
            ws = WebSearcher(network_gate=localhost_gate)
        mock_client.get.return_value = _make_response(text="<p>local</p>")
        text = ws.fetch_page("http://localhost:8080/api")
        assert "local" in text

    def test_allowlist_gate_permits_listed_host(self):
        gate = NetworkGate(mode="allowlist", allowlist=["example.com"])
        with patch("agent.web_tools.make_client") as mf:
            mock_client = MagicMock(spec=httpx.Client)
            mf.return_value = mock_client
            ws = WebSearcher(network_gate=gate)
        mock_client.get.return_value = _make_response(text="<p>allowed</p>")
        text = ws.fetch_page("https://example.com/page")
        assert "allowed" in text

    def test_allowlist_gate_blocks_unlisted_host(self):
        gate = NetworkGate(mode="allowlist", allowlist=["safe.com"])
        with patch("agent.web_tools.make_client") as mf:
            mf.return_value = MagicMock(spec=httpx.Client)
            ws = WebSearcher(network_gate=gate)
        with pytest.raises(PermissionError):
            ws.fetch_page("https://evil.com/steal")

    def test_no_gate_means_no_blocking(self, searcher):
        """With gate=None the _guard method is a no-op."""
        searcher._test_client_mock.get.return_value = _make_response(
            text="<p>open</p>",
        )
        text = searcher.fetch_page("https://anywhere.example.com")
        assert "open" in text


# ===================================================================
# 9. close()
# ===================================================================


class TestClose:
    """WebSearcher.close tears down the HTTP client."""

    def test_close_delegates_to_client(self, searcher):
        searcher.close()
        searcher._test_client_mock.close.assert_called_once()


# ===================================================================
# 10. Edge cases
# ===================================================================


class TestEdgeCases:
    """Miscellaneous edge-case coverage."""

    def test_strip_html_with_only_tags(self):
        assert _strip_html("<br><hr><img src='x'>") == ""

    def test_strip_html_preserves_inner_text(self):
        assert _strip_html("<a href='#'>click</a>") == "click"

    def test_extract_text_empty_page(self):
        assert _extract_text("") == ""

    def test_fetch_page_empty_body(self, searcher):
        searcher._test_client_mock.get.return_value = _make_response(text="")
        text = searcher.fetch_page("https://example.com/empty")
        assert text == ""

    def test_search_html_entities_in_title(self, searcher):
        html = """
        <a class="result__a" href="https://example.com">Tom &amp; Jerry</a>
        <td class="result__snippet">A &lt;classic&gt; show.</td>
        """
        searcher._test_client_mock.post.return_value = _make_response(text=html)
        results = searcher.search_duckduckgo("cartoon")
        assert results[0]["title"] == "Tom & Jerry"
        # &lt; / &gt; are unescaped to literal < > by html.unescape
        assert "<classic>" in results[0]["snippet"]
        assert "classic" in results[0]["snippet"]

    def test_search_preserves_result_order(self, searcher):
        """Results should come back in the same order as the HTML."""
        searcher._test_client_mock.post.return_value = _make_response(
            text=_DDG_HTML_TWO_RESULTS,
        )
        results = searcher.search_duckduckgo("order test")
        assert "First" in results[0]["title"]
        assert "Second" in results[1]["title"]

    def test_max_results_zero_uses_instance_default(self, searcher):
        """Passing max_results=0 should fall back to self._max_results."""
        searcher._test_client_mock.post.return_value = _make_response(
            text=_DDG_HTML_TWO_RESULTS,
        )
        results = searcher.search_duckduckgo("test", max_results=0)
        # Instance default is 5, HTML has 2 results, so we get 2
        assert len(results) == 2
