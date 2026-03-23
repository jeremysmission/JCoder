"""
Wikipedia-Style Knowledge Base Builder (v2)
---------------------------------------------
Turns a folder of files into a browsable Wikipedia-style knowledge base.
Groups files into topic articles, extracts content, and renders a
self-contained HTML wiki that anyone can navigate.

Usage:
    python tools/wiki_builder_v2.py D:/Docs
    python tools/wiki_builder_v2.py "\\server\share" --output wiki.html
    python tools/wiki_builder_v2.py D:/Docs --llm  (uses Ollama for summaries)
"""

import io
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Text extensions: base set + all code extensions from LANGUAGE_MAP
_BASE_TEXT_EXTENSIONS = {
    ".md", ".txt", ".html", ".htm", ".css",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".conf",
    ".sh", ".bash", ".ps1", ".bat", ".cmd", ".xml", ".csv",
    ".rst", ".log", ".dockerfile", ".sql", ".r",
}
TEXT_EXTENSIONS = _BASE_TEXT_EXTENSIONS.copy()
if _HAS_LANGUAGE_MAP:
    TEXT_EXTENSIONS.update(LANGUAGE_MAP.keys())

BINARY_LABELS = {
    ".docx": "Word Document", ".xlsx": "Excel Spreadsheet",
    ".pptx": "PowerPoint", ".pdf": "PDF Document",
    ".zip": "ZIP Archive", ".7z": "7-Zip Archive",
    ".tar": "TAR Archive", ".gz": "GZip Archive",
    ".jpg": "Image", ".jpeg": "Image", ".png": "Image",
    ".gif": "Image", ".svg": "SVG Image", ".bmp": "Image",
    ".dwg": "CAD Drawing", ".dxf": "CAD Drawing",
    ".mp3": "Audio", ".mp4": "Video", ".wav": "Audio",
    ".avi": "Video", ".mkv": "Video",
}

SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "node_modules", ".pytest_cache",
    ".mypy_cache", "dist", "build", ".cache", ".tmp",
}


@dataclass
class FileEntry:
    """A single file on disk."""
    path: str
    relative_path: str
    filename: str
    extension: str
    size: int
    modified: str
    modified_ts: float
    category: str
    subcategory: str
    content: str
    summary: str
    keywords: List[str] = field(default_factory=list)
    is_text: bool = True
    file_label: str = ""


@dataclass
class WikiArticle:
    """A wiki article synthesized from one or more files."""
    title: str
    slug: str
    category: str
    summary: str
    sections: List[Dict] = field(default_factory=list)
    files: List[FileEntry] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)
    last_modified: str = ""
    total_size: int = 0


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _human_size(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def _title_from_name(filename):
    name = Path(filename).stem
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)
    return name.strip()


def _extract_summary(content, max_len=250):
    lines = content.strip().splitlines()
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            if out:
                break
            continue
        if s.startswith("#") and not out:
            continue
        if s.startswith("```") or s == "---":
            if out:
                break
            continue
        out.append(s)
        if len(" ".join(out)) >= max_len:
            break
    text = " ".join(out)[:max_len]
    return text + "..." if len(text) == max_len else text


def _extract_headings(content):
    """Extract markdown headings as section titles."""
    headings = []
    for line in content.splitlines():
        m = re.match(r"^(#{1,4})\s+(.+)", line)
        if m:
            level = len(m.group(1))
            headings.append({"level": level, "title": m.group(2).strip()})
    return headings


def _extract_keywords(content, top_n=10):
    text = re.sub(r"[#*`\[\](){}|<>]", " ", content.lower())
    text = re.sub(r"https?://\S+", "", text)
    words = re.findall(r"[a-z][a-z0-9_]{3,}", text)
    stops = {
        "this", "that", "with", "from", "have", "been", "will",
        "your", "they", "their", "than", "then", "when", "what",
        "which", "where", "does", "into", "also", "each", "only",
        "should", "would", "could", "about", "more", "some", "other",
        "after", "before", "between", "through", "under", "over",
        "these", "those", "such", "just", "like", "very", "most",
        "make", "made", "need", "used", "using", "uses", "file",
        "line", "code", "note", "none", "true", "false", "self",
        "return", "import", "class", "function", "print",
    }
    filtered = [w for w in words if w not in stops and len(w) > 3]
    counts = {}
    for w in filtered:
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_n]]


def _llm_summary(content, filename, model="phi4-mini"):
    """Generate a wiki-style summary using a local LLM."""
    try:
        import httpx
    except ImportError:
        return None

    prompt = (
        f"Write a 2-3 sentence Wikipedia-style summary of this document. "
        f"Be factual and concise. File: {filename}\n\n"
        f"CONTENT:\n{content[:2000]}\n\n"
        f"SUMMARY:"
    )

    try:
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 150},
            },
            timeout=60.0,
        )
        return resp.json().get("response", "").strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Crawl
# ---------------------------------------------------------------------------

def crawl(root_dir, use_llm=False, llm_model="phi4-mini"):
    root = Path(root_dir).resolve()
    entries = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS
                       and not d.startswith(".")]
        # Skip web asset folders
        if os.path.basename(dirpath).endswith("_files"):
            dirnames.clear()
            continue

        for filename in sorted(filenames):
            fp = Path(dirpath) / filename
            ext = fp.suffix.lower()
            if filename.lower() in {"thumbs.db", ".ds_store", "desktop.ini"}:
                continue
            if ext in {".pyc", ".pyo", ".o", ".obj", ".dll", ".exe"}:
                continue

            try:
                stat = fp.stat()
                size = stat.st_size
                mtime = stat.st_mtime
                mdate = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            except OSError:
                continue

            rel = fp.relative_to(root)
            parts = rel.parts
            category = parts[0] if len(parts) > 1 else "General"
            subcategory = parts[1] if len(parts) > 2 else ""

            is_text = ext in TEXT_EXTENSIONS
            content = ""
            summary = ""
            keywords = []
            file_label = BINARY_LABELS.get(ext, ext.lstrip(".").upper() or "File")

            if is_text and size < 512_000:
                try:
                    content = fp.read_text(encoding="utf-8", errors="replace")
                    if use_llm:
                        llm_sum = _llm_summary(content, filename, llm_model)
                        summary = llm_sum or _extract_summary(content)
                    else:
                        summary = _extract_summary(content)
                    keywords = _extract_keywords(content)
                except Exception:
                    content = ""
                    summary = "(could not read)"
            elif not is_text:
                summary = file_label

            entries.append(FileEntry(
                path=str(fp),
                relative_path=str(rel),
                filename=filename,
                extension=ext,
                size=size,
                modified=mdate,
                modified_ts=mtime,
                category=category,
                subcategory=subcategory,
                content=content,
                summary=summary,
                keywords=keywords,
                is_text=is_text,
                file_label=file_label if not is_text else ext.lstrip(".").upper(),
            ))

    return entries


# ---------------------------------------------------------------------------
# Build articles from files
# ---------------------------------------------------------------------------

def build_articles(entries):
    """Group files into wiki articles by topic/folder."""
    # Group by category + subcategory
    groups = defaultdict(list)
    for e in entries:
        key = e.category
        if e.subcategory:
            key = f"{e.category}/{e.subcategory}"
        groups[key].append(e)

    articles = []
    all_keywords = defaultdict(int)

    for group_name, files in sorted(groups.items()):
        # Create one article per group, or one per file if few files
        if len(files) <= 3:
            for f in files:
                title = _title_from_name(f.filename)
                article = WikiArticle(
                    title=title,
                    slug=_slug(title),
                    category=f.category,
                    summary=f.summary,
                    keywords=f.keywords[:8],
                    last_modified=f.modified,
                    total_size=f.size,
                    files=[f],
                )
                # Build sections from headings
                if f.content and f.is_text:
                    headings = _extract_headings(f.content)
                    article.sections = headings[:15]
                articles.append(article)
                for kw in f.keywords[:5]:
                    all_keywords[kw] += 1
        else:
            # Group article
            title = group_name.replace("/", " - ")
            all_kw = []
            all_summaries = []
            latest_date = ""
            total_size = 0
            for f in files:
                all_kw.extend(f.keywords[:5])
                if f.summary and f.summary not in BINARY_LABELS.values():
                    all_summaries.append(f.summary)
                if f.modified > latest_date:
                    latest_date = f.modified
                total_size += f.size

            # Deduplicate keywords
            kw_counts = {}
            for k in all_kw:
                kw_counts[k] = kw_counts.get(k, 0) + 1
            top_kw = [k for k, _ in sorted(kw_counts.items(),
                      key=lambda x: x[1], reverse=True)][:10]

            summary = all_summaries[0] if all_summaries else (
                f"Collection of {len(files)} files covering {title}."
            )

            article = WikiArticle(
                title=title,
                slug=_slug(title),
                category=files[0].category,
                summary=summary,
                keywords=top_kw,
                last_modified=latest_date,
                total_size=total_size,
                files=files,
            )
            articles.append(article)
            for kw in top_kw[:5]:
                all_keywords[kw] += 1

    # Build see_also links (shared keywords)
    for article in articles:
        related = []
        for other in articles:
            if other.slug == article.slug:
                continue
            shared = set(article.keywords) & set(other.keywords)
            if len(shared) >= 2:
                related.append((len(shared), other.title, other.slug))
        related.sort(reverse=True)
        article.see_also = [slug for _, _, slug in related[:5]]

    return articles


# ---------------------------------------------------------------------------
# Render Wikipedia-style HTML
# ---------------------------------------------------------------------------

def render_wiki(articles, source_dir, entries):
    categories = sorted(set(a.category for a in articles))
    total_files = len(entries)
    total_size = sum(e.size for e in entries)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build article HTML
    article_html_list = []
    for a in sorted(articles, key=lambda x: x.title.lower()):
        # Table of contents from sections
        toc = ""
        if a.sections:
            toc_items = "".join(
                f'<li style="margin-left:{(s["level"]-1)*16}px">'
                f'{escape(s["title"])}</li>'
                for s in a.sections[:12]
            )
            toc = f'<div class="toc"><b>Contents</b><ol>{toc_items}</ol></div>'

        # File listing
        file_rows = ""
        for f in a.files[:20]:
            icon = "doc" if f.is_text else "bin"
            file_rows += (
                f'<tr><td class="f-icon {icon}"></td>'
                f'<td>{escape(f.filename)}</td>'
                f'<td class="f-meta">{escape(f.file_label)}</td>'
                f'<td class="f-meta">{_human_size(f.size)}</td>'
                f'<td class="f-meta">{escape(f.modified)}</td></tr>\n'
            )
        file_count_note = ""
        if len(a.files) > 20:
            file_count_note = (
                f'<tr><td colspan="5" class="f-meta">'
                f'...and {len(a.files) - 20} more files</td></tr>'
            )

        # See also
        see_also_html = ""
        if a.see_also:
            links = ", ".join(
                f'<a href="#" class="wiki-link" data-target="{s}">{s.replace("-"," ").title()}</a>'
                for s in a.see_also
            )
            see_also_html = f'<div class="see-also"><b>See also:</b> {links}</div>'

        # Keywords as categories
        cats_html = " ".join(
            f'<a href="#" class="cat-tag" data-kw="{escape(k)}">{escape(k)}</a>'
            for k in a.keywords[:8]
        )

        # Content preview (first 1500 chars for text files)
        content_preview = ""
        if a.files and a.files[0].is_text and a.files[0].content:
            preview_text = escape(a.files[0].content[:1500])
            if len(a.files[0].content) > 1500:
                preview_text += "\n..."
            content_preview = (
                f'<details class="content-block">'
                f'<summary>Full content</summary>'
                f'<pre>{preview_text}</pre></details>'
            )

        article_html_list.append(f"""
<article class="wiki-article" id="article-{a.slug}"
         data-cat="{escape(a.category)}"
         data-search="{escape((a.title + ' ' + a.summary + ' ' + ' '.join(a.keywords)).lower())}"
         data-slug="{a.slug}">
  <h2 class="article-title">{escape(a.title)}</h2>
  <div class="article-meta">
    <span class="cat-badge">{escape(a.category)}</span>
    <span>{len(a.files)} file{"s" if len(a.files) != 1 else ""}</span>
    <span>{_human_size(a.total_size)}</span>
    <span>Last modified: {escape(a.last_modified)}</span>
  </div>
  <p class="article-summary">{escape(a.summary)}</p>
  {toc}
  <table class="file-table">
    <thead><tr><th></th><th>File</th><th>Type</th><th>Size</th><th>Modified</th></tr></thead>
    <tbody>{file_rows}{file_count_note}</tbody>
  </table>
  {content_preview}
  {see_also_html}
  <div class="article-cats">{cats_html}</div>
</article>""")

    # Category nav items
    cat_nav = '<a href="#" class="nav-link active" data-cat="__all__">All articles</a>\n'
    for c in categories:
        count = sum(1 for a in articles if a.category == c)
        cat_nav += (
            f'<a href="#" class="nav-link" data-cat="{escape(c)}">'
            f'{escape(c)} ({count})</a>\n'
        )

    # Recent articles for main page
    recent = sorted(articles, key=lambda a: a.last_modified, reverse=True)[:8]
    recent_links = "".join(
        f'<li><a href="#" class="wiki-link" data-target="{a.slug}">'
        f'{escape(a.title)}</a> <small>({a.last_modified})</small></li>'
        for a in recent
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Knowledge Wiki</title>
<style>
:root {{
  --bg: #f8f9fa; --bg2: #ffffff; --bg3: #f1f3f5;
  --text: #202122; --text2: #54595d; --link: #0645ad;
  --border: #a2a9b1; --accent: #3366cc; --highlight: #fee7e6;
  --cat-bg: #e8f0fe;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: "Linux Libertine","Georgia","Times",serif;
        background:var(--bg); color:var(--text); display:flex; min-height:100vh; }}

/* Sidebar */
.sidebar {{
  width:220px; background:var(--bg2); border-right:1px solid var(--border);
  position:fixed; height:100vh; overflow-y:auto; padding:0;
}}
.sidebar-header {{
  background:var(--accent); color:#fff; padding:16px;
  text-align:center;
}}
.sidebar-header h1 {{ font-size:1.2em; font-weight:normal; }}
.sidebar-header small {{ opacity:0.8; font-size:0.75em; }}
.sidebar-stats {{
  padding:10px 16px; background:var(--bg3); font-size:0.8em; color:var(--text2);
  border-bottom:1px solid var(--border);
}}
.sidebar-stats b {{ color:var(--text); }}
.search-wrap {{ padding:10px 16px; border-bottom:1px solid var(--border); }}
.search-wrap input {{
  width:100%; padding:6px 10px; border:1px solid var(--border);
  border-radius:3px; font-size:0.85em; font-family:sans-serif;
}}
.search-wrap input:focus {{ outline:none; border-color:var(--accent); }}
.nav-section {{ padding:10px 16px 4px; font-size:0.7em; color:var(--text2);
  text-transform:uppercase; letter-spacing:1px; }}
.nav-link {{
  display:block; padding:4px 16px; color:var(--link); text-decoration:none;
  font-size:0.85em; font-family:sans-serif;
}}
.nav-link:hover {{ background:var(--bg3); }}
.nav-link.active {{ background:var(--cat-bg); font-weight:bold; }}

/* Main */
.main {{ margin-left:220px; flex:1; max-width:960px; padding:20px 30px; }}

/* Main page */
.main-page {{ margin-bottom:30px; }}
.main-page h2 {{
  font-size:1.4em; border-bottom:1px solid var(--border);
  padding-bottom:4px; margin-bottom:12px;
}}
.main-page .welcome {{
  background:var(--bg2); border:1px solid var(--border);
  border-radius:3px; padding:16px; margin-bottom:20px; font-size:0.95em;
}}
.main-page .recent {{ columns:2; column-gap:20px; }}
.main-page .recent li {{
  font-size:0.9em; margin:4px 0; break-inside:avoid; list-style:disc inside;
}}
.main-page .recent a {{ color:var(--link); text-decoration:none; }}
.main-page .recent a:hover {{ text-decoration:underline; }}
.main-page .recent small {{ color:var(--text2); }}

/* Articles */
.wiki-article {{
  background:var(--bg2); border:1px solid var(--border); border-radius:3px;
  padding:20px; margin-bottom:20px; display:none;
}}
.wiki-article.visible {{ display:block; }}
.article-title {{
  font-size:1.6em; font-weight:normal; border-bottom:1px solid var(--border);
  padding-bottom:6px; margin-bottom:10px;
}}
.article-meta {{
  display:flex; gap:12px; font-size:0.8em; color:var(--text2);
  margin-bottom:12px; font-family:sans-serif; flex-wrap:wrap;
}}
.cat-badge {{
  background:var(--cat-bg); padding:2px 8px; border-radius:3px;
  color:var(--accent); font-weight:bold;
}}
.article-summary {{
  font-size:1.0em; line-height:1.6; margin-bottom:14px;
  border-left:3px solid var(--accent); padding-left:12px;
}}
.toc {{
  background:var(--bg3); border:1px solid var(--border); border-radius:3px;
  padding:10px 16px; margin-bottom:14px; display:inline-block;
  font-family:sans-serif; font-size:0.85em;
}}
.toc ol {{ margin:6px 0 0 20px; }}
.toc li {{ margin:2px 0; }}
.file-table {{
  width:100%; border-collapse:collapse; font-family:sans-serif;
  font-size:0.85em; margin-bottom:14px;
}}
.file-table th {{
  text-align:left; padding:6px 8px; border-bottom:2px solid var(--border);
  font-size:0.8em; color:var(--text2);
}}
.file-table td {{ padding:4px 8px; border-bottom:1px solid #eee; }}
.f-meta {{ color:var(--text2); }}
.f-icon {{ width:20px; }}
.f-icon.doc::before {{ content:"\\1F4C4"; }}
.f-icon.bin::before {{ content:"\\1F4CE"; }}
.see-also {{
  margin:12px 0; padding:10px; background:var(--bg3);
  border-radius:3px; font-size:0.9em; font-family:sans-serif;
}}
.wiki-link {{ color:var(--link); text-decoration:none; cursor:pointer; }}
.wiki-link:hover {{ text-decoration:underline; }}
.article-cats {{
  margin-top:14px; padding-top:10px; border-top:1px solid var(--border);
}}
.cat-tag {{
  display:inline-block; font-size:0.75em; font-family:sans-serif;
  background:var(--cat-bg); color:var(--accent); padding:2px 8px;
  border-radius:3px; margin:2px; text-decoration:none;
}}
.cat-tag:hover {{ background:var(--accent); color:#fff; }}
.content-block {{ margin:10px 0; font-family:sans-serif; }}
.content-block summary {{
  cursor:pointer; color:var(--link); font-size:0.85em;
}}
.content-block pre {{
  margin-top:8px; background:#f6f8fa; padding:12px; border-radius:3px;
  font-size:0.8em; overflow-x:auto; white-space:pre-wrap;
  word-wrap:break-word; max-height:500px; overflow-y:auto;
  border:1px solid var(--border); font-family:"Consolas","Monaco",monospace;
}}
.no-results {{
  text-align:center; padding:60px; color:var(--text2); font-size:1.1em;
  display:none;
}}
</style>
</head>
<body>

<div class="sidebar">
  <div class="sidebar-header">
    <h1>Knowledge Wiki</h1>
    <small>Personal Knowledge Base</small>
  </div>
  <div class="sidebar-stats">
    <b>{len(articles)}</b> articles |
    <b>{total_files}</b> files |
    <b>{_human_size(total_size)}</b><br>
    Source: {escape(os.path.basename(source_dir))}<br>
    Built: {now}
  </div>
  <div class="search-wrap">
    <input type="text" id="searchBox" placeholder="Search wiki...">
  </div>
  <div class="nav-section">Navigation</div>
  <a href="#" class="nav-link" id="mainPageLink" style="font-weight:bold">Main Page</a>
  <div class="nav-section">Categories</div>
  {cat_nav}
</div>

<div class="main" id="mainContent">
  <div class="main-page" id="mainPage">
    <h2>Welcome to Your Knowledge Wiki</h2>
    <div class="welcome">
      This wiki was automatically generated from <b>{total_files}</b> files
      ({_human_size(total_size)}) in <b>{escape(source_dir)}</b>.
      Browse by category in the sidebar, or use the search box to find
      specific topics. Each article contains file listings, content previews,
      and cross-references to related articles.
    </div>
    <h2>Recent Articles</h2>
    <ul class="recent">{recent_links}</ul>
  </div>

  {"".join(article_html_list)}

  <div class="no-results" id="noResults">
    No articles match your search.
  </div>
</div>

<script>
const articles = document.querySelectorAll('.wiki-article');
const catLinks = document.querySelectorAll('.nav-link[data-cat]');
const searchBox = document.getElementById('searchBox');
const noResults = document.getElementById('noResults');
const mainPage = document.getElementById('mainPage');
const mainPageLink = document.getElementById('mainPageLink');
let activeCat = '__all__';
let activeView = 'main'; // 'main' | 'browse' | 'article'

function showMain() {{
  activeView = 'main';
  mainPage.style.display = 'block';
  articles.forEach(a => a.classList.remove('visible'));
  noResults.style.display = 'none';
  catLinks.forEach(l => l.classList.remove('active'));
}}

function showBrowse() {{
  activeView = 'browse';
  mainPage.style.display = 'none';
  filterArticles();
}}

function showArticle(slug) {{
  activeView = 'article';
  mainPage.style.display = 'none';
  noResults.style.display = 'none';
  articles.forEach(a => {{
    if (a.dataset.slug === slug) {{
      a.classList.add('visible');
      a.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }} else {{
      a.classList.remove('visible');
    }}
  }});
}}

function filterArticles() {{
  const q = searchBox.value.toLowerCase().trim();
  let visible = 0;
  articles.forEach(a => {{
    const matchCat = activeCat === '__all__' || a.dataset.cat === activeCat;
    const matchSearch = !q || a.dataset.search.includes(q);
    if (matchCat && matchSearch) {{
      a.classList.add('visible');
      visible++;
    }} else {{
      a.classList.remove('visible');
    }}
  }});
  noResults.style.display = visible === 0 ? 'block' : 'none';
}}

mainPageLink.addEventListener('click', e => {{ e.preventDefault(); showMain(); }});

catLinks.forEach(link => {{
  link.addEventListener('click', e => {{
    e.preventDefault();
    catLinks.forEach(l => l.classList.remove('active'));
    link.classList.add('active');
    activeCat = link.dataset.cat;
    showBrowse();
  }});
}});

searchBox.addEventListener('input', () => {{
  if (searchBox.value.trim()) {{
    activeCat = '__all__';
    catLinks.forEach(l => l.classList.remove('active'));
    catLinks[0].classList.add('active');
    showBrowse();
  }}
}});

// Wiki internal links
document.addEventListener('click', e => {{
  const link = e.target.closest('.wiki-link');
  if (link) {{
    e.preventDefault();
    showArticle(link.dataset.target);
  }}
}});

// Keyword filter
document.addEventListener('click', e => {{
  const tag = e.target.closest('.cat-tag');
  if (tag) {{
    e.preventDefault();
    searchBox.value = tag.dataset.kw;
    activeCat = '__all__';
    showBrowse();
  }}
}});

// Start on main page
showMain();
</script>

</body>
</html>"""


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build a Wikipedia-style knowledge base from a folder"
    )
    parser.add_argument("source_dir", help="Directory to index")
    parser.add_argument("--output", default="",
                        help="Output HTML path (default: <source>/wiki.html)")
    parser.add_argument("--llm", action="store_true",
                        help="Use Ollama LLM for better summaries (slower)")
    parser.add_argument("--model", default="phi4-mini",
                        help="Ollama model for summaries (default: phi4-mini)")
    args = parser.parse_args()

    source = Path(args.source_dir).resolve()
    if not source.is_dir():
        print(f"[FAIL] Not a directory: {source}")
        sys.exit(1)

    output = Path(args.output) if args.output else source / "wiki.html"

    print(f"[OK] Crawling: {source}")
    entries = crawl(str(source), use_llm=args.llm, llm_model=args.model)
    print(f"[OK] Found {len(entries)} files")

    print("[OK] Building articles...")
    articles = build_articles(entries)
    print(f"[OK] Created {len(articles)} wiki articles")

    html = render_wiki(articles, str(source), entries)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Wiki: {output}")
    print(f"     Open: file:///{str(output).replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
