"""
Web Tools
---------
Lightweight web search and page fetcher for the JCoder agent.
Uses DuckDuckGo HTML search (no API key needed) and regex-based
HTML stripping (no BeautifulSoup dependency).

Requires: httpx (already a project dependency).
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urljoin

import httpx

from core.http_factory import make_client
from core.network_gate import NetworkGate

log = logging.getLogger(__name__)

_USER_AGENT = "JCoder/1.0"
_DDG_URL = "https://html.duckduckgo.com/html/"

# Regex patterns for HTML processing
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")

# DuckDuckGo result parsing
_RESULT_BLOCK_RE = re.compile(
    r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div|span)>',
    re.DOTALL,
)


def _strip_html(raw: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


def _extract_text(raw_html: str) -> str:
    """Extract readable text from a full HTML page."""
    # Remove script and style blocks first
    cleaned = re.sub(
        r"<(script|style|noscript)[^>]*>.*?</\1>",
        "",
        raw_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = _strip_html(cleaned)
    # Collapse runs of blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


class WebSearcher:
    """Lightweight web search and page fetcher for the agent."""

    def __init__(
        self,
        network_gate: Optional[NetworkGate] = None,
        timeout_s: int = 30,
        max_results: int = 5,
    ):
        self._gate = network_gate
        self._max_results = max_results
        self._client: httpx.Client = make_client(
            timeout_s=timeout_s,
            headers={"User-Agent": _USER_AGENT},
        )

    # -- public API --------------------------------------------------------

    def search_duckduckgo(
        self, query: str, max_results: int = 0
    ) -> List[Dict[str, str]]:
        """Search DuckDuckGo via HTML (no API key).

        Returns list of dicts with keys: title, url, snippet.
        """
        cap = max_results or self._max_results
        self._guard(_DDG_URL)

        resp = self._client.post(
            _DDG_URL,
            data={"q": query, "b": ""},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        body = resp.text

        results: List[Dict[str, str]] = []
        links = _RESULT_BLOCK_RE.findall(body)
        snippets = _SNIPPET_RE.findall(body)

        for i, (url, raw_title) in enumerate(links):
            if i >= cap:
                break
            title = _strip_html(raw_title)
            snippet = _strip_html(snippets[i]) if i < len(snippets) else ""
            # DuckDuckGo wraps URLs in a redirect; extract the real URL
            if "uddg=" in url:
                from urllib.parse import parse_qs, urlparse

                qs = parse_qs(urlparse(url).query)
                url = qs.get("uddg", [url])[0]
            results.append({"title": title, "url": url, "snippet": snippet})

        return results

    def fetch_page(self, url: str, max_chars: int = 50_000) -> str:
        """Fetch a web page and return extracted text, truncated to max_chars."""
        self._guard(url)

        resp = self._client.get(url, follow_redirects=True)
        resp.raise_for_status()
        text = _extract_text(resp.text)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        return text

    def search_and_summarize(self, query: str) -> str:
        """Search DuckDuckGo, fetch top 2-3 results, return combined text."""
        hits = self.search_duckduckgo(query, max_results=3)
        if not hits:
            return "No search results found."

        sections: List[str] = []
        for i, hit in enumerate(hits, 1):
            header = f"## Result {i}: {hit['title']}\nURL: {hit['url']}"
            try:
                content = self.fetch_page(hit["url"], max_chars=15_000)
            except Exception as exc:
                content = f"[Failed to fetch: {exc}]"
            sections.append(f"{header}\n{content}")

        return "\n\n".join(sections)

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()

    # -- internals ---------------------------------------------------------

    def _guard(self, url: str):
        """Enforce network policy before any outbound request."""
        if self._gate is not None:
            self._gate.guard(url)
