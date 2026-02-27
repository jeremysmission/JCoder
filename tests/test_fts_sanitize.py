"""FTS5 query sanitization must never throw, regardless of input."""

import pytest

from core.index_engine import IndexEngine


class TestFTS5Sanitize:
    """_sanitize_fts5_query must handle any string without raising."""

    @pytest.mark.parametrize("query", [
        "normal query",
        "Where is Chunker defined?",
        "function(arg1, arg2)",
        "SELECT * FROM users; DROP TABLE;",
        '{"json": "injection"}',
        "",
        "   ",
        "a" * 10000,
        "def __init__(self):",
        "C:\\Users\\path\\file.py",
        "file:///etc/passwd",
        "hello 'world' \"test\"",
        "emoji test",
        "tab\there",
        "newline\nhere",
        "mixed?!@#$%^&*()special",
        "OR AND NOT NEAR",
        '"already quoted"',
    ])
    def test_sanitize_never_throws(self, query):
        """No input should cause _sanitize_fts5_query to raise."""
        result = IndexEngine._sanitize_fts5_query(query)
        assert isinstance(result, str)
        assert len(result) > 0  # Always produces something (at minimum '""')
