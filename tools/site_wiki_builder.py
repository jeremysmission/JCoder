"""
Site Wiki Builder -- crawls an entire drive, groups scattered files by site name,
and builds a Wikipedia-style knowledge base with one article per site.

Modes:
  discover  -- scan drive, suggest site names from folder/file names
  build     -- crawl drive, match files to sites, render wiki

Usage:
  python tools/site_wiki_builder.py discover "N:/" --output sites.yaml
  python tools/site_wiki_builder.py build "N:/" --sites sites.yaml --output wiki.html
  python tools/site_wiki_builder.py build "N:/" --sites sites.yaml --output wiki.html --fts5
"""

import io
import os
import re
import sqlite3
import sys
import time
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

# Import central extension registry from core
try:
    from ingestion.chunker import LANGUAGE_MAP
    _HAS_LANGUAGE_MAP = True
except ImportError:
    _HAS_LANGUAGE_MAP = False
    LANGUAGE_MAP = {}

# Document type classification by extension
# Note: Core language extensions (from LANGUAGE_MAP) are dynamically added to "Code"
_BASE_DOC_TYPES = {
    "Drawings": {".dwg", ".dxf", ".dgn", ".vsd", ".vsdx", ".stp", ".step", ".igs", ".iges"},
    "Documents": {".docx", ".doc", ".pdf", ".rtf", ".odt", ".txt", ".md"},
    "Spreadsheets": {".xlsx", ".xls", ".csv", ".ods"},
    "Presentations": {".pptx", ".ppt", ".odp"},
    "Photos": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".heic"},
    "Email": {".msg", ".eml", ".pst"},
    "Archives": {".zip", ".7z", ".rar", ".tar", ".gz"},
    "Video": {".mp4", ".avi", ".mkv", ".mov", ".wmv"},
    "Code": {".ps1", ".sh", ".bat", ".html", ".htm"},
}

# Build DOC_TYPES by starting with base and adding LANGUAGE_MAP extensions to Code
DOC_TYPES = dict(_BASE_DOC_TYPES)
if _HAS_LANGUAGE_MAP:
    DOC_TYPES["Code"].update(LANGUAGE_MAP.keys())

SKIP_DIRS = {
    "$recycle.bin", "system volume information", "windows",
    "program files", "program files (x86)", "programdata",
    "__pycache__", ".git", ".venv", "node_modules",
    "appdata", "recovery", "config.msi",
}

SKIP_FILES = {"thumbs.db", "desktop.ini", ".ds_store", "ntuser.dat"}


def _classify_file(ext):
    """Classify a file extension into a document type."""
    ext = ext.lower()
    for doc_type, exts in DOC_TYPES.items():
        if ext in exts:
            return doc_type
    return "Other"


def _human_size(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Site name matching
# ---------------------------------------------------------------------------

def load_sites_config(config_path):
    """Load site names and aliases from a YAML config file.

    Expected format:
      sites:
        Guam:
          aliases: [guam, GU, andersen, apra]
        Kwajalein:
          aliases: [kwaj, kwajalein, roi-namur]
        Japan:
          aliases: [japan, yokota, misawa, kadena]
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sites = {}
    for name, info in raw.get("sites", {}).items():
        aliases = info.get("aliases", []) if isinstance(info, dict) else []
        # Always include the site name itself as a search term
        terms = [name.lower()]
        terms.extend(a.lower() for a in aliases)
        # Build regex: match any alias as whole word (case-insensitive)
        pattern = re.compile(
            r"(?:^|[\\/_ \-.,])"
            r"(" + "|".join(re.escape(t) for t in terms) + r")"
            r"(?:$|[\\/_ \-.,])",
            re.IGNORECASE,
        )
        sites[name] = {"aliases": terms, "pattern": pattern}

    return sites


def match_sites(filepath, sites_config):
    """Return list of site names that match a file path."""
    path_lower = filepath.lower()
    matched = []
    for site_name, info in sites_config.items():
        if info["pattern"].search(path_lower):
            matched.append(site_name)
    return matched


# ---------------------------------------------------------------------------
# Discovery mode -- scan drive and suggest site names
# ---------------------------------------------------------------------------

def discover_sites(root_dir, min_occurrences=3, max_depth=4):
    """Scan folder names to discover potential site names.

    Returns a dict of {folder_name: count} for common folder names.
    """
    folder_counts = defaultdict(int)
    root = Path(root_dir)
    scanned = 0

    print(f"[OK] Scanning folders in {root_dir} (depth {max_depth})...")

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip system directories
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in SKIP_DIRS and not d.startswith("$")
        ]

        # Limit depth
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) >= max_depth:
            dirnames.clear()
            continue

        for d in dirnames:
            # Clean folder name
            clean = d.strip()
            if len(clean) >= 3 and not clean.startswith("."):
                folder_counts[clean] += 1

        scanned += 1
        if scanned % 500 == 0:
            print(f"     Scanned {scanned} directories...", end="\r")

    print(f"[OK] Scanned {scanned} directories, {len(folder_counts)} unique names")

    # Filter to names that appear multiple times (likely sites/projects)
    common = {
        name: count for name, count in folder_counts.items()
        if count >= min_occurrences
    }

    return dict(sorted(common.items(), key=lambda x: x[1], reverse=True))


def write_discovery_yaml(discovered, output_path):
    """Write discovered folder names as a sites.yaml template."""
    content = (
        "# Site Wiki Configuration\n"
        "# Edit this file: keep real site names, delete noise,\n"
        "# add aliases for each site.\n"
        "#\n"
        "# Format:\n"
        "#   sites:\n"
        "#     SiteName:\n"
        "#       aliases: [alias1, alias2, abbreviation]\n"
        "#\n\n"
        "sites:\n"
    )

    for name, count in discovered.items():
        content += f"  {name}:\n"
        content += f"    aliases: [{name.lower()}]  # seen {count} times\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] Template written to {output_path}")
    print(f"     Edit it: keep site names, delete noise, add aliases")


# ---------------------------------------------------------------------------
# Catalog crawler
# ---------------------------------------------------------------------------

@dataclass
class CatalogEntry:
    """A single file found on the drive."""
    path: str
    relative_path: str
    filename: str
    extension: str
    doc_type: str
    size: int
    modified: str
    modified_ts: float
    sites: List[str] = field(default_factory=list)


def crawl_drive(root_dir, sites_config):
    """Walk the entire drive and catalog every file, matching to sites."""
    root = Path(root_dir).resolve()
    entries = []
    scanned_dirs = 0
    scanned_files = 0
    matched_files = 0
    t0 = time.monotonic()

    print(f"[1/3] Crawling {root_dir}...")

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in SKIP_DIRS and not d.startswith("$")
        ]

        scanned_dirs += 1
        if scanned_dirs % 200 == 0:
            elapsed = time.monotonic() - t0
            print(f"     {scanned_dirs} dirs, {scanned_files} files, "
                  f"{matched_files} matched ({elapsed:.0f}s)...", end="\r")

        for filename in filenames:
            if filename.lower() in SKIP_FILES:
                continue

            filepath = Path(dirpath) / filename
            scanned_files += 1

            try:
                stat = filepath.stat()
                size = stat.st_size
                mtime = stat.st_mtime
                mdate = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            except OSError:
                continue

            ext = filepath.suffix.lower()
            rel = str(filepath.relative_to(root))

            # Match against site names
            sites = match_sites(rel, sites_config)
            if sites:
                matched_files += 1

            entries.append(CatalogEntry(
                path=str(filepath),
                relative_path=rel,
                filename=filename,
                extension=ext,
                doc_type=_classify_file(ext),
                size=size,
                modified=mdate,
                modified_ts=mtime,
                sites=sites,
            ))

    elapsed = time.monotonic() - t0
    print(f"[OK] Crawled {scanned_dirs} dirs, {scanned_files} files "
          f"in {elapsed:.0f}s                    ")
    print(f"     {matched_files} files matched to sites "
          f"({scanned_files - matched_files} unmatched)")

    return entries


# ---------------------------------------------------------------------------
# Build FTS5 index (optional)
# ---------------------------------------------------------------------------

def build_fts5(entries, db_path):
    """Build an FTS5 index from the catalog for full-text filename search."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("DROP TABLE IF EXISTS catalog")
    conn.execute(
        "CREATE VIRTUAL TABLE catalog "
        "USING fts5(filename, relative_path, doc_type, sites, modified)"
    )

    rows = [
        (
            e.filename,
            e.relative_path,
            e.doc_type,
            ", ".join(e.sites),
            e.modified,
        )
        for e in entries
    ]
    conn.executemany(
        "INSERT INTO catalog(filename, relative_path, doc_type, sites, modified) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    print(f"[OK] FTS5 catalog: {db_path} ({len(rows)} entries)")


# ---------------------------------------------------------------------------
# Build wiki articles per site
# ---------------------------------------------------------------------------

@dataclass
class SiteArticle:
    """A wiki article for one site."""
    name: str
    slug: str
    files: List[CatalogEntry] = field(default_factory=list)
    doc_types: Dict[str, List[CatalogEntry]] = field(default_factory=dict)
    total_size: int = 0
    first_date: str = ""
    last_date: str = ""
    file_count: int = 0
    related_sites: List[str] = field(default_factory=list)


def build_site_articles(entries, sites_config):
    """Group files by site and build one article per site."""
    site_files = defaultdict(list)
    unmatched = []

    for e in entries:
        if e.sites:
            for s in e.sites:
                site_files[s].append(e)
        else:
            unmatched.append(e)

    articles = []

    for site_name in sorted(sites_config.keys()):
        files = site_files.get(site_name, [])
        if not files:
            continue

        # Sort chronologically
        files.sort(key=lambda f: f.modified_ts)

        # Group by document type
        by_type = defaultdict(list)
        for f in files:
            by_type[f.doc_type].append(f)

        # Date range
        dates = [f.modified for f in files if f.modified]
        first = min(dates) if dates else ""
        last = max(dates) if dates else ""

        # Find related sites (files that match multiple sites)
        related = set()
        for f in files:
            for s in f.sites:
                if s != site_name:
                    related.add(s)

        article = SiteArticle(
            name=site_name,
            slug=re.sub(r"[^a-z0-9]+", "-", site_name.lower()).strip("-"),
            files=files,
            doc_types=dict(by_type),
            total_size=sum(f.size for f in files),
            first_date=first,
            last_date=last,
            file_count=len(files),
            related_sites=sorted(related),
        )
        articles.append(article)

    # Add unmatched as a special article
    if unmatched:
        by_type = defaultdict(list)
        for f in unmatched:
            by_type[f.doc_type].append(f)
        dates = [f.modified for f in unmatched if f.modified]
        articles.append(SiteArticle(
            name="Unclassified",
            slug="unclassified",
            files=sorted(unmatched, key=lambda f: f.modified_ts, reverse=True),
            doc_types=dict(by_type),
            total_size=sum(f.size for f in unmatched),
            first_date=min(dates) if dates else "",
            last_date=max(dates) if dates else "",
            file_count=len(unmatched),
        ))

    return articles


# ---------------------------------------------------------------------------
# Render Wikipedia-style HTML
# ---------------------------------------------------------------------------

def render_site_wiki(articles, source_dir, total_entries):
    total_sites = sum(1 for a in articles if a.name != "Unclassified")
    total_files = sum(a.file_count for a in articles)
    total_size = sum(a.total_size for a in articles)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Sidebar nav
    site_nav = ""
    for a in articles:
        badge = f' <span class="badge">{a.file_count}</span>'
        cls = " unclassified" if a.name == "Unclassified" else ""
        site_nav += (
            f'<a href="#" class="nav-link{cls}" data-slug="{a.slug}">'
            f'{escape(a.name)}{badge}</a>\n'
        )

    # Build article HTML
    article_html = []
    for a in articles:
        # Overview stats
        type_summary = ", ".join(
            f"{len(files)} {dtype}" for dtype, files in
            sorted(a.doc_types.items(), key=lambda x: len(x[1]), reverse=True)
        )

        # Date range
        date_range = ""
        if a.first_date and a.last_date:
            if a.first_date == a.last_date:
                date_range = a.first_date
            else:
                date_range = f"{a.first_date} to {a.last_date}"

        # Related sites
        related_html = ""
        if a.related_sites:
            links = ", ".join(
                f'<a href="#" class="site-link" data-slug="{re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")}">'
                f'{escape(s)}</a>'
                for s in a.related_sites
            )
            related_html = (
                f'<div class="related">'
                f'<b>Related sites:</b> {links}</div>'
            )

        # Document sections (one table per doc type)
        sections_html = ""
        type_order = [
            "Drawings", "Documents", "Reports", "Spreadsheets",
            "Presentations", "Photos", "Email", "Code", "Video",
            "Archives", "Other",
        ]
        for dtype in type_order:
            if dtype not in a.doc_types:
                continue
            files = sorted(a.doc_types[dtype], key=lambda f: f.modified_ts,
                           reverse=True)
            rows = ""
            shown = files[:50]
            for f in shown:
                rows += (
                    f'<tr>'
                    f'<td class="fn">{escape(f.filename)}</td>'
                    f'<td class="meta">{_human_size(f.size)}</td>'
                    f'<td class="meta">{escape(f.modified)}</td>'
                    f'<td class="meta path" title="{escape(f.relative_path)}">'
                    f'{escape(f.relative_path[:60])}</td>'
                    f'</tr>\n'
                )
            overflow = ""
            if len(files) > 50:
                overflow = (
                    f'<tr><td colspan="4" class="meta">'
                    f'...and {len(files) - 50} more files</td></tr>'
                )

            sections_html += f"""
<div class="doc-section">
  <h3>{escape(dtype)} ({len(files)})</h3>
  <table class="file-table">
    <thead><tr><th>File</th><th>Size</th><th>Date</th><th>Location</th></tr></thead>
    <tbody>{rows}{overflow}</tbody>
  </table>
</div>"""

        # Timeline (chronological summary)
        yearly = defaultdict(int)
        for f in a.files:
            year = f.modified[:4] if f.modified else "Unknown"
            yearly[year] += 1
        timeline_items = " ".join(
            f'<span class="tl-year">{y}: {c}</span>'
            for y, c in sorted(yearly.items())
        )
        timeline_html = (
            f'<div class="timeline"><b>Activity timeline:</b> '
            f'{timeline_items}</div>'
        ) if yearly else ""

        article_html.append(f"""
<article class="wiki-article" id="{a.slug}" style="display:none">
  <h2>{escape(a.name)}</h2>
  <div class="article-meta">
    <span><b>{a.file_count}</b> files</span>
    <span><b>{_human_size(a.total_size)}</b></span>
    <span>{escape(date_range)}</span>
  </div>
  <div class="overview">
    <p><b>{escape(a.name)}</b> contains {a.file_count} documents
    totaling {_human_size(a.total_size)}: {type_summary}.</p>
  </div>
  {timeline_html}
  {related_html}
  {sections_html}
</article>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Site Wiki -- {escape(os.path.basename(source_dir))}</title>
<style>
:root {{
  --bg:#f8f9fa; --bg2:#fff; --bg3:#f1f3f5; --text:#202122;
  --text2:#54595d; --link:#0645ad; --border:#a2a9b1;
  --accent:#3366cc; --cat-bg:#e8f0fe;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:"Linux Libertine","Georgia","Times",serif;
        background:var(--bg); color:var(--text); display:flex; min-height:100vh; }}
.sidebar {{
  width:240px; background:var(--bg2); border-right:1px solid var(--border);
  position:fixed; height:100vh; overflow-y:auto;
}}
.sb-header {{
  background:var(--accent); color:#fff; padding:16px; text-align:center;
}}
.sb-header h1 {{ font-size:1.2em; font-weight:normal; }}
.sb-header small {{ opacity:0.8; font-size:0.75em; }}
.sb-stats {{
  padding:10px 16px; background:var(--bg3); font-size:0.8em;
  color:var(--text2); border-bottom:1px solid var(--border);
}}
.sb-stats b {{ color:var(--text); }}
.sb-search {{ padding:10px 16px; border-bottom:1px solid var(--border); }}
.sb-search input {{
  width:100%; padding:6px 10px; border:1px solid var(--border);
  border-radius:3px; font-size:0.85em; font-family:sans-serif;
}}
.sb-search input:focus {{ outline:none; border-color:var(--accent); }}
.sb-label {{
  padding:10px 16px 4px; font-size:0.7em; color:var(--text2);
  text-transform:uppercase; letter-spacing:1px;
}}
.nav-link {{
  display:block; padding:5px 16px; color:var(--link); text-decoration:none;
  font-size:0.85em; font-family:sans-serif; cursor:pointer;
}}
.nav-link:hover {{ background:var(--bg3); }}
.nav-link.active {{ background:var(--cat-bg); font-weight:bold; }}
.nav-link.unclassified {{ color:var(--text2); font-style:italic; }}
.badge {{
  float:right; background:var(--bg3); color:var(--text2); border-radius:10px;
  padding:0 7px; font-size:0.8em;
}}
.nav-link.active .badge {{ background:var(--accent); color:#fff; }}

.main {{ margin-left:240px; flex:1; max-width:960px; padding:20px 30px; }}

/* Main page */
.main-page {{ margin-bottom:30px; }}
.main-page h2 {{
  font-size:1.4em; border-bottom:1px solid var(--border);
  padding-bottom:4px; margin-bottom:12px;
}}
.welcome {{
  background:var(--bg2); border:1px solid var(--border);
  border-radius:3px; padding:16px; margin-bottom:20px; font-size:0.95em;
}}
.site-grid {{
  display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
  gap:12px; margin-bottom:20px;
}}
.site-card {{
  background:var(--bg2); border:1px solid var(--border); border-radius:3px;
  padding:12px; cursor:pointer; text-decoration:none; color:var(--text);
}}
.site-card:hover {{ border-color:var(--accent); }}
.site-card h4 {{ font-size:1em; margin-bottom:4px; color:var(--link); }}
.site-card .sc-meta {{ font-size:0.8em; color:var(--text2); }}

/* Article */
.wiki-article {{
  background:var(--bg2); border:1px solid var(--border); border-radius:3px;
  padding:20px;
}}
.wiki-article h2 {{
  font-size:1.6em; font-weight:normal; border-bottom:1px solid var(--border);
  padding-bottom:6px; margin-bottom:10px;
}}
.article-meta {{
  display:flex; gap:16px; font-size:0.85em; color:var(--text2);
  margin-bottom:12px; font-family:sans-serif;
}}
.overview {{
  border-left:3px solid var(--accent); padding-left:12px;
  margin-bottom:14px; font-size:0.95em; line-height:1.6;
}}
.timeline {{
  background:var(--bg3); padding:10px 14px; border-radius:3px;
  margin-bottom:14px; font-family:sans-serif; font-size:0.85em;
}}
.tl-year {{
  display:inline-block; background:var(--cat-bg); padding:2px 8px;
  border-radius:3px; margin:2px; color:var(--accent);
}}
.related {{
  background:var(--bg3); padding:10px 14px; border-radius:3px;
  margin-bottom:14px; font-family:sans-serif; font-size:0.85em;
}}
.site-link {{ color:var(--link); text-decoration:none; cursor:pointer; }}
.site-link:hover {{ text-decoration:underline; }}
.doc-section {{ margin-bottom:18px; }}
.doc-section h3 {{
  font-size:1.05em; color:var(--text); border-bottom:1px solid #ddd;
  padding-bottom:4px; margin-bottom:6px;
}}
.file-table {{
  width:100%; border-collapse:collapse; font-family:sans-serif; font-size:0.82em;
}}
.file-table th {{
  text-align:left; padding:4px 8px; border-bottom:2px solid var(--border);
  font-size:0.8em; color:var(--text2);
}}
.file-table td {{ padding:3px 8px; border-bottom:1px solid #eee; }}
.fn {{ word-break:break-all; }}
.meta {{ color:var(--text2); white-space:nowrap; }}
.path {{ max-width:200px; overflow:hidden; text-overflow:ellipsis; }}
</style>
</head>
<body>

<div class="sidebar">
  <div class="sb-header">
    <h1>Site Wiki</h1>
    <small>{total_sites} sites | {_human_size(total_size)}</small>
  </div>
  <div class="sb-stats">
    <b>{total_files}</b> files across <b>{total_sites}</b> sites<br>
    Total: <b>{_human_size(total_size)}</b><br>
    Source: {escape(os.path.basename(source_dir))}<br>
    Built: {now}
  </div>
  <div class="sb-search">
    <input type="text" id="siteSearch" placeholder="Filter sites...">
  </div>
  <div class="sb-label">Sites</div>
  <a href="#" class="nav-link active" id="mainLink">Main Page</a>
  {site_nav}
</div>

<div class="main">
  <div class="main-page" id="mainPage">
    <h2>Site Knowledge Base</h2>
    <div class="welcome">
      Automatically cataloged from <b>{escape(source_dir)}</b>. Contains
      <b>{total_files}</b> files ({_human_size(total_size)}) across
      <b>{total_sites}</b> sites. Click any site in the sidebar or below
      to see all related documents sorted chronologically.
    </div>
    <h2>All Sites</h2>
    <div class="site-grid">
      {"".join(
          f'<a href="#" class="site-card" data-slug="{a.slug}">'
          f'<h4>{escape(a.name)}</h4>'
          f'<div class="sc-meta">{a.file_count} files | {_human_size(a.total_size)}</div>'
          f'<div class="sc-meta">{a.first_date} - {a.last_date}</div></a>'
          for a in articles if a.name != "Unclassified"
      )}
    </div>
  </div>

  {"".join(article_html)}
</div>

<script>
const navLinks = document.querySelectorAll('.nav-link[data-slug]');
const siteCards = document.querySelectorAll('.site-card');
const mainLink = document.getElementById('mainLink');
const mainPage = document.getElementById('mainPage');
const searchBox = document.getElementById('siteSearch');
const allArticles = document.querySelectorAll('.wiki-article');

function showMain() {{
  mainPage.style.display = 'block';
  allArticles.forEach(a => a.style.display = 'none');
  navLinks.forEach(l => l.classList.remove('active'));
  mainLink.classList.add('active');
}}

function showSite(slug) {{
  mainPage.style.display = 'none';
  allArticles.forEach(a => {{
    a.style.display = a.id === slug ? 'block' : 'none';
  }});
  navLinks.forEach(l => {{
    l.classList.toggle('active', l.dataset.slug === slug);
  }});
  mainLink.classList.remove('active');
  document.getElementById(slug)?.scrollIntoView({{block:'start'}});
}}

mainLink.addEventListener('click', e => {{ e.preventDefault(); showMain(); }});
navLinks.forEach(l => l.addEventListener('click', e => {{
  e.preventDefault(); showSite(l.dataset.slug);
}}));
siteCards.forEach(c => c.addEventListener('click', e => {{
  e.preventDefault(); showSite(c.dataset.slug);
}}));
document.addEventListener('click', e => {{
  const link = e.target.closest('.site-link');
  if (link) {{ e.preventDefault(); showSite(link.dataset.slug); }}
}});

searchBox.addEventListener('input', () => {{
  const q = searchBox.value.toLowerCase();
  navLinks.forEach(l => {{
    const text = l.textContent.toLowerCase();
    l.style.display = (!q || text.includes(q)) ? 'block' : 'none';
  }});
}});

showMain();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build a per-site Wikipedia-style wiki from a drive"
    )
    sub = parser.add_subparsers(dest="command")

    # discover
    disc = sub.add_parser("discover", help="Scan drive for site name candidates")
    disc.add_argument("source_dir", help="Drive or directory to scan")
    disc.add_argument("--output", default="sites.yaml",
                      help="Output YAML template (default: sites.yaml)")
    disc.add_argument("--min", type=int, default=3,
                      help="Minimum folder occurrences to include (default: 3)")
    disc.add_argument("--depth", type=int, default=4,
                      help="Max directory depth to scan (default: 4)")

    # build
    bld = sub.add_parser("build", help="Crawl drive and build site wiki")
    bld.add_argument("source_dir", help="Drive or directory to crawl")
    bld.add_argument("--sites", required=True,
                     help="Path to sites.yaml config")
    bld.add_argument("--output", default="site_wiki.html",
                     help="Output HTML path (default: site_wiki.html)")
    bld.add_argument("--fts5", action="store_true",
                     help="Also build FTS5 search index")

    args = parser.parse_args()

    if args.command == "discover":
        discovered = discover_sites(args.source_dir, min_occurrences=args.min,
                                     max_depth=args.depth)
        if not discovered:
            print("[WARN] No common folder names found")
            return

        print(f"\n[OK] Top folder names (seen >= {args.min} times):")
        for name, count in list(discovered.items())[:40]:
            print(f"     {count:4d}x  {name}")

        write_discovery_yaml(discovered, args.output)

    elif args.command == "build":
        sites_config = load_sites_config(args.sites)
        print(f"[OK] Loaded {len(sites_config)} sites from {args.sites}")

        entries = crawl_drive(args.source_dir, sites_config)

        print(f"\n[2/3] Building articles...")
        articles = build_site_articles(entries, sites_config)
        site_articles = [a for a in articles if a.name != "Unclassified"]
        print(f"[OK] {len(site_articles)} site articles, "
              f"{sum(a.file_count for a in site_articles)} matched files")

        if args.fts5:
            fts5_path = args.output.replace(".html", ".fts5.db")
            build_fts5(entries, fts5_path)

        print(f"\n[3/3] Rendering wiki...")
        html = render_site_wiki(articles, args.source_dir, len(entries))

        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"[OK] Wiki: {args.output}")
        print(f"     Open: file:///{os.path.abspath(args.output).replace(os.sep, '/')}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
