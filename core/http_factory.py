"""Shared HTTP client factory for JCoder modules."""

from __future__ import annotations

import os
from typing import Any

import httpx

_DEFAULT_USER_AGENT = "JCoder/1.0"


def _resolve_verify_setting(explicit_verify: Any) -> Any:
    """Resolve CA bundle settings from common environment variables."""
    if explicit_verify is not None:
        return explicit_verify

    for env_name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        bundle_path = os.environ.get(env_name, "").strip()
        if bundle_path and os.path.exists(bundle_path):
            return bundle_path
    return True


def make_client(
    timeout_s: float = 30.0,
    read_timeout_s: float | None = None,
    retries: int = 2,
    follow_redirects: bool = True,
    trust_env: bool = True,
    verify: Any = None,
    headers: dict[str, str] | None = None,
    **kwargs,
) -> httpx.Client:
    """Create a pre-configured httpx.Client with standard retry and timeout settings."""
    merged_headers = {"User-Agent": _DEFAULT_USER_AGENT}
    if headers:
        merged_headers.update(headers)

    return httpx.Client(
        timeout=httpx.Timeout(timeout_s, read=read_timeout_s or timeout_s),
        transport=httpx.HTTPTransport(retries=retries),
        follow_redirects=follow_redirects,
        trust_env=trust_env,
        verify=_resolve_verify_setting(verify),
        headers=merged_headers,
        **kwargs,
    )
