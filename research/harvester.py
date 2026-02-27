"""
Research Harvester
------------------
Safe, auditable, cached harvester for external research sources.

Safety invariants:
- HARD requires network.mode == allowlist (enforced by caller)
- ALL requests go through NetworkGate.guard()
- Cached with ETag/Last-Modified conditional GET
- Deduped by SHA-256 content hash
- Rate-limited per host
- Writes markdown files for ingestion -- never executes fetched content

Sources: GitHub releases, PyPI metadata, HN stories, arXiv abstracts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx

from core.network_gate import NetworkGate


# ---------------------------------------------------------------------------
# Response cache (SQLite + disk bodies)
# ---------------------------------------------------------------------------

class ResponseCache:
    """HTTP response cache: headers in SQLite, bodies on disk keyed by SHA-256."""

    def __init__(self, cache_dir: str):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "cache.sqlite"
        self.body_dir = self.root / "bodies"
        self.body_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    url TEXT PRIMARY KEY,
                    etag TEXT,
                    last_modified TEXT,
                    status INTEGER,
                    fetched_ts REAL,
                    body_sha256 TEXT,
                    body_path TEXT
                )
                """
            )
            conn.commit()

    def get(self, url: str) -> Optional[Dict[str, str]]:
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.execute(
                "SELECT etag, last_modified, status, fetched_ts, body_sha256, body_path "
                "FROM responses WHERE url=?",
                (url,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "etag": row[0] or "",
            "last_modified": row[1] or "",
            "status": str(row[2] or 0),
            "fetched_ts": str(row[3] or 0.0),
            "body_sha256": row[4] or "",
            "body_path": row[5] or "",
        }

    def put(self, url: str, etag: str, last_modified: str, status: int, body: bytes) -> str:
        sha = hashlib.sha256(body).hexdigest()
        body_path = str(self.body_dir / f"{sha}.bin")
        if not os.path.exists(body_path):
            Path(body_path).write_bytes(body)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO responses
                (url, etag, last_modified, status, fetched_ts, body_sha256, body_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (url, etag, last_modified, int(status), time.time(), sha, body_path),
            )
            conn.commit()
        return body_path

    def load_body(self, body_path: str) -> bytes:
        return Path(body_path).read_bytes()


# ---------------------------------------------------------------------------
# Polite fetcher (rate-limited, retrying, conditional GET)
# ---------------------------------------------------------------------------

class PoliteFetcher:
    """HTTP client that respects rate limits, caching, and NetworkGate."""

    def __init__(
        self,
        gate: NetworkGate,
        cache: ResponseCache,
        user_agent: str,
        timeout_s: int,
        max_bytes: int,
        min_interval_s: float,
        retries: int,
        backoff_s: float,
    ):
        self.gate = gate
        self.cache = cache
        self.max_bytes = max_bytes
        self.min_interval_s = float(min_interval_s)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)
        self._last_by_host: Dict[str, float] = {}

        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout_s),
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def _rate_limit(self, url: str) -> None:
        host = httpx.URL(url).host or ""
        now = time.time()
        last = self._last_by_host.get(host, 0.0)
        wait = (last + self.min_interval_s) - now
        if wait > 0:
            time.sleep(wait)
        self._last_by_host[host] = time.time()

    def get_bytes(self, url: str) -> Tuple[int, bytes, Dict[str, str]]:
        """Fetch URL with conditional GET, retry, and cache."""
        self.gate.guard(url)
        cached = self.cache.get(url)

        headers = {}
        if cached:
            if cached.get("etag"):
                headers["If-None-Match"] = cached["etag"]
            if cached.get("last_modified"):
                headers["If-Modified-Since"] = cached["last_modified"]

        attempt = 0
        while True:
            attempt += 1
            self._rate_limit(url)
            try:
                r = self.client.get(url, headers=headers)
                status = r.status_code

                # Not modified: serve cached body
                if status == 304 and cached and cached.get("body_path"):
                    body = self.cache.load_body(cached["body_path"])
                    return 200, body, {"etag": cached.get("etag", ""), "url": url}

                # Retry on server errors / rate limit
                if status in (429, 500, 502, 503, 504):
                    if attempt <= self.retries:
                        time.sleep(self.backoff_s * (2 ** (attempt - 1)))
                        continue
                    r.raise_for_status()

                r.raise_for_status()
                body = r.content
                if len(body) > self.max_bytes:
                    raise ValueError(f"Response too large ({len(body)} bytes) for {url}")

                etag = r.headers.get("ETag", "")
                last_modified = r.headers.get("Last-Modified", "")
                self.cache.put(url, etag=etag, last_modified=last_modified,
                               status=status, body=body)
                return status, body, {"etag": etag, "url": url}
            except Exception:
                if attempt <= self.retries:
                    time.sleep(self.backoff_s * (2 ** (attempt - 1)))
                    continue
                raise


# ---------------------------------------------------------------------------
# Output writer (markdown)
# ---------------------------------------------------------------------------

def _safe_slug(s: str, max_len: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:max_len] if s else "item"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_markdown(out_dir: Path, title: str, meta: Dict[str, str], body: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    body_hash = hashlib.sha256(body.encode("utf-8", errors="ignore")).hexdigest()[:12]
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fname = f"{date}_{_safe_slug(title)}_{body_hash}.md"
    path = out_dir / fname

    front = ["---"]
    front.append(f'title: "{title.replace(chr(34), "")}"')
    for k, v in meta.items():
        v2 = (v or "").replace("\n", " ").replace("\r", " ").strip()
        front.append(f'{k}: "{v2.replace(chr(34), "")}"')
    front.append("---")
    front.append("")
    front.append(body.strip() + "\n")

    path.write_text("\n".join(front), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Source parsers
# ---------------------------------------------------------------------------

def _parse_github_releases(repo: str, payload: list) -> List[Tuple[str, str, str]]:
    out = []
    for rel in payload[:20]:
        name = rel.get("name") or rel.get("tag_name") or "release"
        html = rel.get("html_url") or ""
        body = rel.get("body") or ""
        published = rel.get("published_at") or ""
        md = f"# {repo} -- {name}\n\nPublished: {published}\n\n{body}\n"
        out.append((f"{repo} {name}", html, md))
    return out


def _parse_pypi(pkg: str, payload: dict) -> Tuple[str, str, str]:
    info = payload.get("info", {})
    releases = payload.get("releases", {})
    name = info.get("name", pkg)
    summary = info.get("summary", "")
    home = info.get("home_page") or info.get("project_url") or ""
    version = info.get("version", "")
    vers = sorted(releases.keys(), reverse=True)[:10]
    md = (f"# PyPI: {name}\n\nCurrent: {version}\n\nSummary: {summary}\n\n"
          f"Home: {home}\n\nRecent releases:\n" +
          "\n".join([f"- {v}" for v in vers]) + "\n")
    url = f"https://pypi.org/project/{pkg}/"
    return (f"pypi {name}", url, md)


def _parse_hn_results(query: str, payload: dict) -> List[Tuple[str, str, str]]:
    hits = payload.get("hits", [])
    out = []
    for h in hits[:25]:
        title = h.get("title") or h.get("story_title") or "HN item"
        url = h.get("url") or h.get("story_url") or ""
        created = h.get("created_at") or ""
        author = h.get("author") or ""
        points = h.get("points")
        obj_id = h.get("objectID") or ""
        item_url = url if url else f"https://news.ycombinator.com/item?id={obj_id}"
        md = (f"# HN: {title}\n\nCreated: {created}\nAuthor: {author}\n"
              f"Points: {points}\nLink: {item_url}\n")
        out.append((f"hn {title}", item_url, md))
    return out


def _parse_arxiv_atom(query: str, xml_bytes: bytes) -> List[Tuple[str, str, str]]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_bytes)
    entries = root.findall("a:entry", ns)
    out = []
    for e in entries[:25]:
        title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (e.findtext("a:summary", default="", namespaces=ns) or "").strip()
        published = (e.findtext("a:published", default="", namespaces=ns) or "").strip()
        link_el = e.find("a:link[@rel='alternate']", ns)
        link = link_el.attrib.get("href", "") if link_el is not None else ""
        md = (f"# arXiv: {title}\n\nPublished: {published}\nQuery: {query}\n\n"
              f"## Abstract\n\n{summary}\n")
        out.append((f"arxiv {title}", link, md))
    return out


# ---------------------------------------------------------------------------
# Main harvester
# ---------------------------------------------------------------------------

class ResearchHarvester:
    """
    Safe research harvester. HARD requires network.mode == allowlist.
    All requests go through NetworkGate.guard().
    Cached + deduped. Writes markdown for ingestion.
    """

    def __init__(self, gate: NetworkGate, cfg, repo_root: str):
        self.gate = gate
        self.cfg = cfg
        self.repo_root = Path(repo_root).resolve()
        self.out_dir = (self.repo_root / cfg.out_dir).resolve()
        self.cache = ResponseCache(cfg.cache_dir)
        self.fetcher = PoliteFetcher(
            gate=gate,
            cache=self.cache,
            user_agent=cfg.user_agent,
            timeout_s=cfg.timeout_s,
            max_bytes=cfg.max_bytes,
            min_interval_s=cfg.min_interval_s,
            retries=cfg.retries,
            backoff_s=cfg.backoff_s,
        )

    def close(self) -> None:
        self.fetcher.close()

    def run(self) -> Dict[str, object]:
        written: List[str] = []
        errors: List[str] = []
        started = _utc_now_iso()

        # GitHub releases
        for repo in (self.cfg.github_repos or []):
            try:
                url = f"https://api.github.com/repos/{repo}/releases?per_page=20"
                _status, body, _hdr = self.fetcher.get_bytes(url)
                payload = json.loads(body.decode("utf-8", errors="ignore"))
                items = _parse_github_releases(repo, payload)
                for title, item_url, md in items:
                    path = _write_markdown(
                        self.out_dir / "github", title=title,
                        meta={"source": "github", "source_url": item_url,
                              "fetched_at": started, "api_url": url},
                        body=md)
                    written.append(str(path))
            except Exception as e:
                errors.append(f"github {repo}: {e}")

        # PyPI
        for pkg in (self.cfg.pypi_packages or []):
            try:
                url = f"https://pypi.org/pypi/{pkg}/json"
                _status, body, _hdr = self.fetcher.get_bytes(url)
                payload = json.loads(body.decode("utf-8", errors="ignore"))
                title, item_url, md = _parse_pypi(pkg, payload)
                path = _write_markdown(
                    self.out_dir / "pypi", title=title,
                    meta={"source": "pypi", "source_url": item_url,
                          "fetched_at": started, "api_url": url},
                    body=md)
                written.append(str(path))
            except Exception as e:
                errors.append(f"pypi {pkg}: {e}")

        # HN Algolia
        for q in (self.cfg.hn_queries or []):
            try:
                params = {"query": q, "tags": "story", "hitsPerPage": 25}
                url = "https://hn.algolia.com/api/v1/search_by_date?" + urlencode(params)
                _status, body, _hdr = self.fetcher.get_bytes(url)
                payload = json.loads(body.decode("utf-8", errors="ignore"))
                items = _parse_hn_results(q, payload)
                for title, item_url, md in items:
                    path = _write_markdown(
                        self.out_dir / "hn", title=title,
                        meta={"source": "hn", "source_url": item_url,
                              "fetched_at": started, "query": q},
                        body=md)
                    written.append(str(path))
            except Exception as e:
                errors.append(f"hn {q}: {e}")

        # arXiv
        for q in (self.cfg.arxiv_queries or []):
            try:
                params = {
                    "search_query": "all:" + q,
                    "start": 0,
                    "max_results": 25,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
                url = "https://export.arxiv.org/api/query?" + urlencode(params)
                _status, body, _hdr = self.fetcher.get_bytes(url)
                items = _parse_arxiv_atom(q, body)
                for title, item_url, md in items:
                    path = _write_markdown(
                        self.out_dir / "arxiv", title=title,
                        meta={"source": "arxiv", "source_url": item_url,
                              "fetched_at": started, "query": q},
                        body=md)
                    written.append(str(path))
            except Exception as e:
                errors.append(f"arxiv {q}: {e}")

        # Manifest
        manifest = {
            "started_at": started,
            "ended_at": _utc_now_iso(),
            "written_count": len(written),
            "error_count": len(errors),
            "written_files": written,
            "errors": errors,
        }
        manifest_path = self.out_dir / "_harvest_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return manifest
