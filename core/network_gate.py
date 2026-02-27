"""
Network Gate
------------
Enforces network access policy for all outbound HTTP.

Modes:
  offline     -- block everything
  localhost   -- allow only 127.0.0.1 / localhost
  allowlist   -- allow only URLs matching the allowlist
"""

from urllib.parse import urlparse
from typing import List, Optional, Set


class NetworkGate:
    """Single allow(url) method that enforces the active network policy."""

    MODES = ("offline", "localhost", "allowlist")

    def __init__(
        self,
        mode: str = "localhost",
        allowlist: Optional[List[str]] = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {self.MODES}")
        self.mode = mode
        self._allowlist: Set[str] = set(allowlist or [])

    def allow(self, url: str) -> bool:
        """Return True if the URL is permitted under the current policy."""
        if self.mode == "offline":
            return False

        parsed = urlparse(url)
        host = parsed.hostname or ""

        if self.mode == "localhost":
            return host in ("localhost", "127.0.0.1", "::1")

        if self.mode == "allowlist":
            return host in self._allowlist

        return False

    def guard(self, url: str):
        """Raise if the URL is not permitted."""
        if not self.allow(url):
            raise PermissionError(
                f"NetworkGate blocked request to {url} (mode={self.mode})"
            )
