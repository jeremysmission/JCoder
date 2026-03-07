"""
Agentic Research Source Tracker
--------------------------------
Weekly scraper that records the latest status for the top 10 agentic research sources.
It reads `data/agentic_sources_watchlist.json`, sends a lightweight request to each URL,
and appends a timestamped snapshot to `data/agentic_sources_history.jsonl`.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCHLIST_PATH = os.path.join(BASE_DIR, "data", "agentic_sources_watchlist.json")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "agentic_sources_history.jsonl")


def load_watchlist() -> List[Dict]:
    with open(WATCHLIST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_request(url: str):
    request = urllib.request.Request(url, method="HEAD")
    request.add_header("User-Agent", "JCoder-AgenticScraper/1.0")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return {
                "status": response.getcode(),
                "final_url": response.geturl(),
                "content_length": response.getheader("Content-Length"),
                "last_modified": response.getheader("Last-Modified"),
            }
    except urllib.error.HTTPError as exc:
        try:
            alt = urllib.request.urlopen(url, timeout=15)
            return {
                "status": alt.getcode(),
                "final_url": alt.geturl(),
                "content_length": alt.getheader("Content-Length"),
                "last_modified": alt.getheader("Last-Modified"),
            }
        except Exception as nested:
            return {"status": exc.code, "error": str(nested)}
    except Exception as exc:
        return {"status": None, "error": str(exc)}


def append_history(entry: Dict):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    entry["checked_at"] = timestamp
    with open(HISTORY_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def main():
    watchlist = load_watchlist()
    summary = []
    for idx, source in enumerate(watchlist, start=1):
        meta = _safe_request(source["url"])
        payload = {
            "rank": idx,
            "name": source["name"],
            "url": source["url"],
            "category": source.get("category", "general"),
            "notes": source.get("notes", ""),
            **meta,
        }
        summary.append(payload)
        append_history(payload)
        print(f"[{idx}/{len(watchlist)}] {source['name']} -> {payload.get('status')}")

    print("\nDaily Summary")
    print("=============")
    for entry in summary:
        print(f"{entry['rank']:2d}. {entry['name']} ({entry['status']})")
    return summary


if __name__ == "__main__":
    main()
