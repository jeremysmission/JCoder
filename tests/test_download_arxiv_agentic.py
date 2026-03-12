from __future__ import annotations

import json

from scripts import download_arxiv_agentic as arxiv_script


class _FakeDownloader:
    def __init__(self, xml_text: str) -> None:
        self.xml_text = xml_text
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def fetch_text(self, url: str, *, headers: dict[str, str] | None = None) -> str:
        self.calls.append((url, headers))
        return self.xml_text


def test_fetch_arxiv_metadata_uses_download_manager_and_caches(monkeypatch, tmp_path):
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.12345v2</id>
    <published>2025-01-02T00:00:00Z</published>
    <title>  Tool   Learning   Paper  </title>
    <summary>  Abstract   with   extra spacing. </summary>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <category term="cs.AI" />
    <category term="cs.LG" />
  </entry>
</feed>
"""
    fake = _FakeDownloader(xml_text)

    monkeypatch.setattr(arxiv_script, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(arxiv_script, "_DOWNLOADER", None)
    monkeypatch.setattr(arxiv_script, "_get_downloader", lambda: fake)

    result = arxiv_script.fetch_arxiv_metadata("2501.12345")

    assert result == {
        "arxiv_id": "2501.12345",
        "title": "Tool Learning Paper",
        "abstract": "Abstract with extra spacing.",
        "authors": "Alice, Bob",
        "categories": "cs.AI, cs.LG",
        "published": "2025-01-02",
    }
    assert fake.calls == [
        (
            "http://export.arxiv.org/api/query?id_list=2501.12345",
            {"Accept": "application/atom+xml"},
        )
    ]
    assert json.loads((tmp_path / "2501.12345.json").read_text(encoding="utf-8")) == result


def test_fetch_arxiv_metadata_prefers_cache_over_network(monkeypatch, tmp_path):
    cached = {
        "arxiv_id": "2501.99999",
        "title": "Cached Title",
        "abstract": "Cached abstract",
        "authors": "Cached Author",
        "categories": "cs.AI",
        "published": "2025-01-03",
    }
    cache_path = tmp_path / "2501.99999.json"
    cache_path.write_text(json.dumps(cached), encoding="utf-8")

    monkeypatch.setattr(arxiv_script, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(arxiv_script, "_DOWNLOADER", None)

    def _unexpected_network() -> _FakeDownloader:
        raise AssertionError("network path should not run when cache exists")

    monkeypatch.setattr(arxiv_script, "_get_downloader", _unexpected_network)

    assert arxiv_script.fetch_arxiv_metadata("2501.99999") == cached
