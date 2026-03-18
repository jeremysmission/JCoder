"""Extended tests for scripts/weekly_scraper.py -- offline, fully mocked."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RSS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Python Blog</title>
    <item>
      <title>Python 3.14 Released</title>
      <description>&lt;p&gt;Big release with pattern matching v2.&lt;/p&gt;</description>
      <link>https://blog.python.org/2026/03/python-3-14.html</link>
    </item>
    <item>
      <title>Security Patch 3.13.4</title>
      <description>&lt;b&gt;Critical&lt;/b&gt; CVE fix</description>
      <link>https://blog.python.org/2026/02/security.html</link>
    </item>
  </channel>
</rss>
"""

SAMPLE_ATOM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Real Python</title>
  <entry>
    <title>Async Best Practices</title>
    <content type="html">&lt;p&gt;Guide to async Python.&lt;/p&gt;</content>
    <link href="https://realpython.com/async/" />
  </entry>
</feed>
"""

SAMPLE_PYPI_JSON = {
    "info": {
        "version": "2.32.0",
        "summary": "HTTP for Humans",
        "requires_python": ">=3.8",
    },
    "releases": {
        "2.30.0": [],
        "2.31.0": [],
        "2.32.0": [],
    },
}

SAMPLE_SE_JSON = {
    "items": [
        {
            "body": "<p>" + ("High quality answer. " * 20) + "</p>",
            "score": 42,
            "question_id": 99999,
        },
        {
            "body": "<p>Short</p>",  # < 100 chars, should be skipped
            "score": 15,
            "question_id": 11111,
        },
    ],
}


def _mock_httpx_response(status_code: int = 200, text: str = "",
                         json_data: dict | None = None):
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _make_mock_client(response=None, side_effect=None):
    """Create a mock httpx.Client whose .get() returns response."""
    client = MagicMock()
    if side_effect is not None:
        client.get.side_effect = side_effect
    elif response is not None:
        client.get.return_value = response
    return client


# ---------------------------------------------------------------------------
# Config / constants validation
# ---------------------------------------------------------------------------

class TestScraperConfig:

    def test_rss_feeds_list_not_empty(self):
        from scripts.weekly_scraper import RSS_FEEDS
        assert len(RSS_FEEDS) >= 1

    def test_rss_feeds_are_tuples_with_urls(self):
        from scripts.weekly_scraper import RSS_FEEDS
        for name, url in RSS_FEEDS:
            assert isinstance(name, str) and name
            assert url.startswith("https://")

    def test_pypi_packages_not_empty(self):
        from scripts.weekly_scraper import PYPI_PACKAGES
        assert len(PYPI_PACKAGES) >= 10

    def test_se_api_base_is_https(self):
        from scripts.weekly_scraper import SE_API_BASE
        assert SE_API_BASE.startswith("https://")


# ---------------------------------------------------------------------------
# Source URL validation
# ---------------------------------------------------------------------------

class TestSourceURLValidation:

    def test_all_rss_urls_parseable(self):
        from urllib.parse import urlparse
        from scripts.weekly_scraper import RSS_FEEDS
        for _, url in RSS_FEEDS:
            parsed = urlparse(url)
            assert parsed.scheme in ("http", "https")
            assert "." in parsed.netloc

    def test_pypi_url_pattern(self):
        from scripts.weekly_scraper import PYPI_PACKAGES
        for pkg in PYPI_PACKAGES:
            expected = f"https://pypi.org/pypi/{pkg}/json"
            assert pkg.isidentifier() or "-" in pkg or "_" in pkg
            assert "pypi.org" in expected


# ---------------------------------------------------------------------------
# RSS scraping with mocked HTTP
# ---------------------------------------------------------------------------

class TestRssScraping:

    @patch("httpx.Client")
    def test_parse_rss_feed(self, MockClient):
        from scripts.weekly_scraper import scrape_rss_feeds
        MockClient.return_value = _make_mock_client(
            _mock_httpx_response(200, SAMPLE_RSS_XML))

        chunks = scrape_rss_feeds(max_per_feed=5)
        assert len(chunks) >= 2
        assert any("Python 3.14" in c["title"] for c in chunks)
        assert all("content" in c for c in chunks)

    @patch("httpx.Client")
    def test_rss_http_error_skips_feed(self, MockClient):
        from scripts.weekly_scraper import scrape_rss_feeds
        MockClient.return_value = _make_mock_client(
            _mock_httpx_response(503, ""))

        chunks = scrape_rss_feeds()
        assert chunks == []

    @patch("httpx.Client")
    def test_rss_parse_atom(self, MockClient):
        from scripts.weekly_scraper import scrape_rss_feeds
        MockClient.return_value = _make_mock_client(
            _mock_httpx_response(200, SAMPLE_ATOM_XML))

        chunks = scrape_rss_feeds(max_per_feed=5)
        assert any("Async" in c.get("title", "") for c in chunks)


# ---------------------------------------------------------------------------
# Changelog scraping with mocked HTTP
# ---------------------------------------------------------------------------

class TestChangelogScraping:

    @patch("httpx.Client")
    def test_parse_pypi_json(self, MockClient):
        from scripts.weekly_scraper import scrape_changelogs
        MockClient.return_value = _make_mock_client(
            _mock_httpx_response(200, json_data=SAMPLE_PYPI_JSON))

        chunks = scrape_changelogs(max_packages=5)
        assert len(chunks) >= 1
        assert "2.32.0" in chunks[0]["content"]

    @patch("httpx.Client")
    def test_changelog_http_error_skips(self, MockClient):
        from scripts.weekly_scraper import scrape_changelogs
        MockClient.return_value = _make_mock_client(
            _mock_httpx_response(404, ""))

        chunks = scrape_changelogs(max_packages=3)
        assert chunks == []


# ---------------------------------------------------------------------------
# Stack Exchange scraping with mocked HTTP
# ---------------------------------------------------------------------------

class TestSEScraping:

    @patch("httpx.Client")
    def test_parse_se_answers(self, MockClient):
        from scripts.weekly_scraper import scrape_se_answers
        MockClient.return_value = _make_mock_client(
            _mock_httpx_response(200, json_data=SAMPLE_SE_JSON))

        chunks = scrape_se_answers(min_score=5, max_per_site=10, days=7)
        # Only the long answer passes the len(body) >= 100 filter
        assert len(chunks) == 1
        assert "99999" in chunks[0]["source"]

    @patch("httpx.Client")
    def test_se_short_answers_skipped(self, MockClient):
        from scripts.weekly_scraper import scrape_se_answers
        short_only = {"items": [{"body": "<p>Too short</p>",
                                  "score": 50, "question_id": 1}]}
        MockClient.return_value = _make_mock_client(
            _mock_httpx_response(200, json_data=short_only))

        chunks = scrape_se_answers()
        assert chunks == []


# ---------------------------------------------------------------------------
# Deduplication logic
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_same_content_deduped(self, tmp_path):
        from scripts.weekly_scraper import ingest_chunks
        db = tmp_path / "dedup.db"
        chunk = [{"content": "duplicate me", "source": "t", "title": "T"}]
        assert ingest_chunks(chunk, str(db)) == 1
        assert ingest_chunks(chunk, str(db)) == 0
        assert ingest_chunks(chunk, str(db)) == 0

    def test_different_content_both_stored(self, tmp_path):
        from scripts.weekly_scraper import ingest_chunks
        db = tmp_path / "dedup.db"
        c1 = [{"content": "alpha", "source": "a", "title": "A"}]
        c2 = [{"content": "beta", "source": "b", "title": "B"}]
        assert ingest_chunks(c1, str(db)) == 1
        assert ingest_chunks(c2, str(db)) == 1

        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        assert count == 2

    def test_chunk_id_is_content_based(self):
        from scripts.weekly_scraper import _chunk_id
        id1 = _chunk_id("same text")
        id2 = _chunk_id("same text")
        id3 = _chunk_id("different text")
        assert id1 == id2
        assert id1 != id3
        assert len(id1) == 16  # sha256 hex truncated to 16


# ---------------------------------------------------------------------------
# Output file writing (report JSON)
# ---------------------------------------------------------------------------

class TestOutputFileWriting:

    def test_report_json_created(self, tmp_path):
        from scripts.weekly_scraper import run_weekly_scrape
        out = tmp_path / "reports"

        with patch("scripts.weekly_scraper.scrape_rss_feeds", return_value=[]), \
             patch("scripts.weekly_scraper.scrape_changelogs", return_value=[]), \
             patch("scripts.weekly_scraper.scrape_se_answers", return_value=[]):
            result = run_weekly_scrape(scrapers=["rss"], dry_run=True,
                                       output_dir=str(out))

        files = list(out.glob("scrape_*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["ingested"] == 0
        assert "rss" in data["scrapers"]

    def test_report_has_source_counts(self, tmp_path):
        from scripts.weekly_scraper import run_weekly_scrape
        out = tmp_path / "reports"

        with patch("scripts.weekly_scraper.scrape_rss_feeds") as mock_rss:
            mock_rss.return_value = [
                {"content": "c1", "source": "rss/Feed1", "title": "T1"},
                {"content": "c2", "source": "rss/Feed2", "title": "T2"},
            ]
            result = run_weekly_scrape(
                scrapers=["rss"], dry_run=True, output_dir=str(out))

        assert result["sources"].get("rss", 0) == 2

    def test_report_contains_elapsed_time(self, tmp_path):
        from scripts.weekly_scraper import run_weekly_scrape
        out = tmp_path / "reports"

        with patch("scripts.weekly_scraper.scrape_rss_feeds", return_value=[]):
            result = run_weekly_scrape(scrapers=["rss"], dry_run=True,
                                       output_dir=str(out))
        assert isinstance(result["elapsed_s"], float)
        assert result["elapsed_s"] >= 0


# ---------------------------------------------------------------------------
# Error handling (unreachable sources)
# ---------------------------------------------------------------------------

class TestErrorHandlingUnreachable:

    @patch("httpx.Client")
    def test_rss_connection_error_handled(self, MockClient):
        from scripts.weekly_scraper import scrape_rss_feeds
        MockClient.return_value = _make_mock_client(
            side_effect=ConnectionError("DNS failure"))

        chunks = scrape_rss_feeds()
        assert chunks == []

    @patch("httpx.Client")
    def test_changelog_timeout_handled(self, MockClient):
        from scripts.weekly_scraper import scrape_changelogs
        MockClient.return_value = _make_mock_client(
            side_effect=TimeoutError("read timed out"))

        chunks = scrape_changelogs(max_packages=2)
        assert chunks == []

    @patch("httpx.Client")
    def test_se_exception_handled(self, MockClient):
        from scripts.weekly_scraper import scrape_se_answers
        MockClient.return_value = _make_mock_client(
            side_effect=RuntimeError("unexpected"))

        chunks = scrape_se_answers()
        assert chunks == []


# ---------------------------------------------------------------------------
# Rate limiting / max-chunks cap
# ---------------------------------------------------------------------------

class TestRateLimiting:

    def test_max_chunks_caps_output(self, tmp_path):
        from scripts.weekly_scraper import run_weekly_scrape
        out = tmp_path / "reports"

        with patch("scripts.weekly_scraper.scrape_rss_feeds") as mock_rss:
            mock_rss.return_value = [
                {"content": f"chunk-{i}", "source": "rss/x", "title": f"T{i}"}
                for i in range(50)
            ]
            result = run_weekly_scrape(
                scrapers=["rss"], dry_run=True,
                max_chunks=10, output_dir=str(out))

        assert result["total_collected"] == 10

    @patch("httpx.Client")
    def test_max_per_feed_respected(self, MockClient):
        """max_per_feed limits items parsed from a single feed."""
        from scripts.weekly_scraper import scrape_rss_feeds, RSS_FEEDS

        big_rss = (
            '<?xml version="1.0"?><rss><channel><title>T</title>'
            + "".join(
                f"<item><title>Item {i}</title>"
                f"<description>Desc {i}</description></item>"
                for i in range(50)
            )
            + "</channel></rss>"
        )

        MockClient.return_value = _make_mock_client(
            _mock_httpx_response(200, big_rss))

        chunks = scrape_rss_feeds(max_per_feed=3)
        assert len(chunks) == 3 * len(RSS_FEEDS)

    def test_rotation_evicts_oldest(self, tmp_path):
        from scripts.weekly_scraper import ingest_chunks
        db = tmp_path / "rotate.db"

        # Insert 3 chunks
        for i in range(3):
            ingest_chunks(
                [{"content": f"old-{i}", "source": "s", "title": "T"}],
                str(db), max_chunks=3)

        # Insert 2 more -- should evict 2 oldest
        ingest_chunks(
            [{"content": "new-A", "source": "s", "title": "T"},
             {"content": "new-B", "source": "s", "title": "T"}],
            str(db), max_chunks=3)

        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT content FROM chunks").fetchall()
        conn.close()
        contents = {r[0] for r in rows}
        assert len(rows) <= 3
        assert "new-A" in contents
        assert "new-B" in contents
