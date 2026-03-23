"""Knowledge, memory, and web operation tools mixin for ToolRegistry."""

from __future__ import annotations

from typing import Any, List, Optional


class KnowledgeOpsMixin:
    """Mixin providing RAG, memory, web, and task-complete tool implementations."""

    # These attributes are provided by ToolRegistry (the concrete class).
    _rag_callback: Any
    _memory: Any
    _web: Any

    # -- RAG tool ----------------------------------------------------------

    def _rag_query(self, query: str) -> Any:
        from agent.tools import ToolResult

        if not self._rag_callback:
            return ToolResult(
                False, "",
                "RAG not configured. Set rag_callback in ToolRegistry.",
            )
        try:
            answer = self._rag_callback(query)
            return ToolResult(True, answer)
        except Exception as e:
            return ToolResult(False, "", f"RAG query failed: {e}")

    # -- Memory tools ------------------------------------------------------

    def _memory_search(self, query: str, top_k: int = 5) -> Any:
        from agent.tools import ToolResult

        if not self._memory:
            return ToolResult(
                False, "",
                "Agent memory not configured. Memory module is unavailable.",
            )
        try:
            results = self._memory.search(query, top_k=top_k)
            if not results:
                return ToolResult(True, "No relevant memories found.")
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] (score={r.get('score', 0):.3f}, "
                    f"confidence={r.get('confidence', 0):.2f}) "
                    f"{r.get('content', '')[:500]}"
                )
                if r.get("source_task"):
                    lines.append(f"    Source: {r['source_task'][:200]}")
            return ToolResult(True, "\n".join(lines))
        except Exception as e:
            return ToolResult(False, "", f"Memory search failed: {e}")

    def _memory_store(self, content: str, tags: Optional[List[str]] = None) -> Any:
        from agent.tools import ToolResult

        if not self._memory:
            return ToolResult(
                False, "",
                "Agent memory not configured. Memory module is unavailable.",
            )
        try:
            entry = self._memory.ingest(
                content=content,
                source_task="agent_tool_store",
                tags=tags or [],
                confidence=0.8,
                tokens_used=0,
            )
            return ToolResult(
                True,
                f"Stored in memory (id={getattr(entry, 'id', 'unknown')})",
            )
        except Exception as e:
            return ToolResult(False, "", f"Memory store failed: {e}")

    # -- Web tools ---------------------------------------------------------

    def _web_search(self, query: str, max_results: int = 5) -> Any:
        from agent.tools import ToolResult

        if not self._web:
            return ToolResult(
                False, "",
                "Web search not available. WebSearcher is not configured.",
            )
        try:
            results = self._web.search_duckduckgo(query, max_results=max_results)
            if not results:
                return ToolResult(True, "No results found.")
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r['title']}\n"
                    f"    URL: {r['url']}\n"
                    f"    {r['snippet']}"
                )
            return ToolResult(True, "\n\n".join(lines))
        except PermissionError as e:
            return ToolResult(False, "", str(e))
        except Exception as e:
            return ToolResult(False, "", f"Web search failed: {e}")

    def _web_fetch(self, url: str, max_chars: int = 50_000) -> Any:
        from agent.tools import ToolResult

        if not self._web:
            return ToolResult(
                False, "",
                "Web fetch not available. WebSearcher is not configured.",
            )
        try:
            text = self._web.fetch_page(url, max_chars=max_chars)
            if not text:
                return ToolResult(True, "[Page returned no text content]")
            return ToolResult(True, text)
        except PermissionError as e:
            return ToolResult(False, "", str(e))
        except Exception as e:
            return ToolResult(False, "", f"Web fetch failed: {e}")

    # -- Control tool ------------------------------------------------------

    def _task_complete(self, summary: str) -> Any:
        from agent.tools import ToolResult

        return ToolResult(True, f"TASK_COMPLETE: {summary}")
