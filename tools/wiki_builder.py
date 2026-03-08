"""
Wiki Builder -- generates a Wikipedia-style HTML index from a folder.

Usage:
    python tools/wiki_builder.py D:/Docs --output D:/Docs/wiki_index.html
    python tools/wiki_builder.py /path/to/folder --output wiki.html
"""

import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

# File types we can extract text from natively
TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".js", ".ts", ".html", ".htm", ".css",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".conf",
    ".sh", ".bash", ".ps1", ".bat", ".cmd", ".xml", ".csv",
    ".rst", ".log", ".env", ".gitignore", ".dockerfile",
}

# Extensions we recognize but can't parse without extra deps
BINARY_EXTENSIONS = {
    ".docx": "Word Document",
    ".xlsx": "Excel Spreadsheet",
    ".pptx": "PowerPoint Presentation",
    ".pdf": "PDF Document",
    ".zip": "ZIP Archive",
    ".7z": "7-Zip Archive",
    ".tar": "TAR Archive",
    ".gz": "GZip Archive",
    ".jpg": "JPEG Image", ".jpeg": "JPEG Image",
    ".png": "PNG Image",
    ".gif": "GIF Image",
    ".svg": "SVG Image",
    ".mp3": "Audio File",
    ".mp4": "Video File",
    ".wav": "Audio File",
}

# Skip patterns
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "node_modules", ".cache",
    ".tmp", "thumbs.db", ".DS_Store",
}

SKIP_SUFFIXES = {
    ".pyc", ".pyo", ".class", ".o", ".obj", ".dll", ".so",
    ".exe", ".bin", ".dat",
}


@dataclass
class Article:
    """A single wiki article derived from a file."""
    title: str
    file_path: str
    relative_path: str
    category: str
    file_type: str
    file_size: int
    modified_date: str
    summary: str
    content: str  # full text (empty for binary)
    keywords: List[str] = field(default_factory=list)
    is_text: bool = True


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _title_from_filename(filename: str) -> str:
    """Convert filename to readable title."""
    name = Path(filename).stem
    # Replace underscores and hyphens with spaces
    name = name.replace("_", " ").replace("-", " ")
    # Remove common noise patterns
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)  # remove "(1)" suffixes
    return name.strip()


def _extract_summary(content: str, max_lines: int = 5) -> str:
    """Extract first meaningful paragraph as summary."""
    lines = content.strip().splitlines()
    summary_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if summary_lines:
                break
            continue
        # Skip markdown headers, code fences, dividers
        if stripped.startswith("#") and len(summary_lines) == 0:
            continue
        if stripped.startswith("```") or stripped.startswith("---"):
            if summary_lines:
                break
            continue
        summary_lines.append(stripped)
        if len(summary_lines) >= max_lines:
            break
    return " ".join(summary_lines)[:300]


def _extract_keywords(content: str, top_n: int = 8) -> List[str]:
    """Extract most frequent meaningful words as keywords."""
    # Strip markdown/HTML syntax
    text = re.sub(r"[#*`\[\](){}|<>]", " ", content.lower())
    text = re.sub(r"https?://\S+", "", text)
    words = re.findall(r"[a-z][a-z0-9_]{3,}", text)

    # Remove common stop words
    stops = {
        "this", "that", "with", "from", "have", "been", "will",
        "your", "they", "their", "than", "then", "when", "what",
        "which", "where", "does", "into", "also", "each", "only",
        "should", "would", "could", "about", "more", "some", "other",
        "after", "before", "between", "through", "under", "over",
        "these", "those", "such", "just", "like", "very", "most",
        "make", "made", "need", "used", "using", "uses", "file",
        "line", "code", "note", "none", "true", "false",
    }
    filtered = [w for w in words if w not in stops and len(w) > 3]

    # Count and rank
    counts: Dict[str, int] = {}
    for w in filtered:
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_n]]


def crawl_directory(root_dir: str) -> List[Article]:
    """Walk a directory tree and build article list."""
    root = Path(root_dir).resolve()
    articles: List[Article] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out skip directories
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in SKIP_DIRS
            and not d.startswith(".")
        ]

        for filename in sorted(filenames):
            filepath = Path(dirpath) / filename
            ext = filepath.suffix.lower()

            # Skip junk files
            if ext in SKIP_SUFFIXES:
                continue
            if filename.lower() in {"thumbs.db", ".ds_store", "desktop.ini"}:
                continue

            # Determine category from relative path
            rel = filepath.relative_to(root)
            parts = rel.parts
            category = parts[0] if len(parts) > 1 else "Root"

            # Skip deep web asset directories (favicon files, etc.)
            if any("_files" in p for p in parts[:-1]) and ext not in TEXT_EXTENSIONS:
                continue

            # File metadata
            try:
                stat = filepath.stat()
                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
            except OSError:
                size = 0
                mtime = "unknown"

            title = _title_from_filename(filename)

            if ext in TEXT_EXTENSIONS:
                try:
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                    # Skip huge files (> 500 KB of text)
                    if len(content) > 512_000:
                        content = content[:512_000] + "\n... [truncated]"
                    summary = _extract_summary(content)
                    keywords = _extract_keywords(content)
                    is_text = True
                except Exception:
                    content = ""
                    summary = "(could not read file)"
                    keywords = []
                    is_text = False
            elif ext in BINARY_EXTENSIONS:
                content = ""
                summary = BINARY_EXTENSIONS[ext]
                keywords = []
                is_text = False
            else:
                content = ""
                summary = f"Binary file ({ext or 'unknown type'})"
                keywords = []
                is_text = False

            articles.append(Article(
                title=title,
                file_path=str(filepath),
                relative_path=str(rel),
                category=category,
                file_type=ext.lstrip(".").upper() or "FILE",
                file_size=size,
                modified_date=mtime,
                summary=summary,
                content=content,
                keywords=keywords,
                is_text=is_text,
            ))

    return articles


def render_wiki_html(articles: List[Article], source_dir: str) -> str:
    """Render articles into a self-contained HTML wiki page."""
    # Group by category
    categories: Dict[str, List[Article]] = {}
    for a in articles:
        categories.setdefault(a.category, []).append(a)

    total_size = sum(a.file_size for a in articles)
    text_count = sum(1 for a in articles if a.is_text)
    binary_count = sum(1 for a in articles if not a.is_text)

    # Build category sidebar items
    cat_items = []
    for cat in sorted(categories.keys()):
        count = len(categories[cat])
        cat_items.append(
            f'<a href="#" class="cat-link" data-cat="{escape(cat)}">'
            f'{escape(cat)} <span class="badge">{count}</span></a>'
        )

    # Build article cards
    article_cards = []
    for cat in sorted(categories.keys()):
        for a in sorted(categories[cat], key=lambda x: x.title.lower()):
            kw_html = " ".join(
                f'<span class="kw">{escape(k)}</span>' for k in a.keywords[:6]
            )
            type_class = "text" if a.is_text else "binary"
            content_preview = ""
            if a.is_text and a.content:
                # Show first 800 chars of content in collapsible
                preview = escape(a.content[:800])
                if len(a.content) > 800:
                    preview += "..."
                content_preview = (
                    f'<details class="content-preview">'
                    f'<summary>View content</summary>'
                    f'<pre>{preview}</pre></details>'
                )

            article_cards.append(f"""
<div class="article" data-cat="{escape(a.category)}"
     data-search="{escape((a.title + ' ' + a.summary + ' ' + ' '.join(a.keywords)).lower())}">
  <div class="article-header">
    <span class="type-badge {type_class}">{escape(a.file_type)}</span>
    <h3>{escape(a.title)}</h3>
  </div>
  <div class="meta">
    <span>{escape(a.relative_path)}</span>
    <span>{_human_size(a.file_size)}</span>
    <span>{escape(a.modified_date)}</span>
  </div>
  <p class="summary">{escape(a.summary)}</p>
  <div class="keywords">{kw_html}</div>
  {content_preview}
</div>""")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Knowledge Wiki -- {escape(source_dir)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #1a1a2e; color: #e0e0e0; display: flex; min-height: 100vh;
}}
.sidebar {{
  width: 260px; background: #16213e; padding: 20px;
  border-right: 1px solid #0f3460; position: fixed; height: 100vh;
  overflow-y: auto;
}}
.sidebar h1 {{ font-size: 1.3em; color: #e94560; margin-bottom: 6px; }}
.sidebar .subtitle {{ font-size: 0.8em; color: #888; margin-bottom: 20px; }}
.sidebar .stats {{
  background: #0f3460; border-radius: 8px; padding: 12px;
  margin-bottom: 20px; font-size: 0.85em;
}}
.sidebar .stats div {{ margin: 4px 0; }}
.sidebar .stats .num {{ color: #e94560; font-weight: bold; }}
.search-box {{
  width: 100%; padding: 8px 12px; border: 1px solid #0f3460;
  border-radius: 6px; background: #1a1a2e; color: #e0e0e0;
  font-size: 0.9em; margin-bottom: 16px;
}}
.search-box:focus {{ outline: none; border-color: #e94560; }}
.cat-section {{ font-size: 0.75em; color: #888; text-transform: uppercase;
  letter-spacing: 1px; margin: 16px 0 8px; }}
.cat-link {{
  display: block; padding: 6px 10px; color: #c0c0c0; text-decoration: none;
  border-radius: 4px; margin: 2px 0; font-size: 0.9em;
}}
.cat-link:hover, .cat-link.active {{ background: #0f3460; color: #e94560; }}
.badge {{
  float: right; background: #0f3460; color: #e94560; border-radius: 10px;
  padding: 1px 8px; font-size: 0.8em;
}}
.cat-link.active .badge {{ background: #e94560; color: #fff; }}
.main {{ margin-left: 260px; padding: 30px; flex: 1; max-width: 900px; }}
.main h2 {{ color: #e94560; margin-bottom: 20px; font-size: 1.6em; }}
.article {{
  background: #16213e; border-radius: 8px; padding: 18px;
  margin-bottom: 16px; border: 1px solid #0f3460;
  transition: border-color 0.2s;
}}
.article:hover {{ border-color: #e94560; }}
.article.hidden {{ display: none; }}
.article-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
.article-header h3 {{ font-size: 1.1em; color: #fff; }}
.type-badge {{
  font-size: 0.7em; font-weight: bold; padding: 2px 8px;
  border-radius: 4px; text-transform: uppercase;
}}
.type-badge.text {{ background: #0a6640; color: #8ef; }}
.type-badge.binary {{ background: #5c3d0a; color: #fc9; }}
.meta {{
  display: flex; gap: 16px; font-size: 0.8em; color: #888; margin-bottom: 8px;
}}
.summary {{ font-size: 0.9em; color: #bbb; line-height: 1.5; margin-bottom: 8px; }}
.keywords {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.kw {{
  font-size: 0.75em; background: #0f3460; color: #8ef; padding: 2px 8px;
  border-radius: 10px;
}}
.content-preview {{ margin-top: 10px; }}
.content-preview summary {{
  cursor: pointer; color: #e94560; font-size: 0.85em;
}}
.content-preview pre {{
  margin-top: 8px; background: #0d1117; padding: 12px; border-radius: 6px;
  font-size: 0.8em; overflow-x: auto; white-space: pre-wrap;
  word-wrap: break-word; max-height: 400px; overflow-y: auto;
  color: #c9d1d9; line-height: 1.4;
}}
.no-results {{
  text-align: center; padding: 40px; color: #666; font-size: 1.1em;
  display: none;
}}
</style>
</head>
<body>

<div class="sidebar">
  <h1>Knowledge Wiki</h1>
  <div class="subtitle">{escape(source_dir)}</div>
  <div class="stats">
    <div><span class="num">{len(articles)}</span> articles</div>
    <div><span class="num">{text_count}</span> text / <span class="num">{binary_count}</span> binary</div>
    <div><span class="num">{len(categories)}</span> categories</div>
    <div><span class="num">{_human_size(total_size)}</span> total</div>
    <div style="color:#666;margin-top:6px;">Built {now}</div>
  </div>
  <input type="text" class="search-box" placeholder="Search articles..." id="searchBox">
  <div class="cat-section">Categories</div>
  <a href="#" class="cat-link active" data-cat="__all__">
    All Articles <span class="badge">{len(articles)}</span></a>
  {"".join(cat_items)}
</div>

<div class="main">
  <h2 id="mainTitle">All Articles</h2>
  {"".join(article_cards)}
  <div class="no-results" id="noResults">No articles match your search.</div>
</div>

<script>
const articles = document.querySelectorAll('.article');
const catLinks = document.querySelectorAll('.cat-link');
const searchBox = document.getElementById('searchBox');
const noResults = document.getElementById('noResults');
const mainTitle = document.getElementById('mainTitle');
let activeCat = '__all__';

function filterArticles() {{
  const q = searchBox.value.toLowerCase().trim();
  let visible = 0;
  articles.forEach(a => {{
    const matchCat = activeCat === '__all__' || a.dataset.cat === activeCat;
    const matchSearch = !q || a.dataset.search.includes(q);
    if (matchCat && matchSearch) {{
      a.classList.remove('hidden');
      visible++;
    }} else {{
      a.classList.add('hidden');
    }}
  }});
  noResults.style.display = visible === 0 ? 'block' : 'none';
}}

catLinks.forEach(link => {{
  link.addEventListener('click', e => {{
    e.preventDefault();
    catLinks.forEach(l => l.classList.remove('active'));
    link.classList.add('active');
    activeCat = link.dataset.cat;
    mainTitle.textContent = activeCat === '__all__' ? 'All Articles' : activeCat;
    filterArticles();
  }});
}});

searchBox.addEventListener('input', filterArticles);
</script>

</body>
</html>"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build a wiki-style HTML index from a folder")
    parser.add_argument("source_dir", help="Directory to index")
    parser.add_argument("--output", default="", help="Output HTML path (default: <source>/wiki_index.html)")
    parser.add_argument("--skip-assets", action="store_true", default=True,
                        help="Skip web asset directories (*_files/)")
    args = parser.parse_args()

    source = Path(args.source_dir).resolve()
    if not source.is_dir():
        print(f"[FAIL] Not a directory: {source}")
        sys.exit(1)

    output = Path(args.output) if args.output else source / "wiki_index.html"

    print(f"[OK] Crawling: {source}")
    articles = crawl_directory(str(source))
    print(f"[OK] Found {len(articles)} articles in {len(set(a.category for a in articles))} categories")

    html = render_wiki_html(articles, str(source))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Wiki written to: {output}")
    print(f"     Open in browser: file:///{str(output).replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
