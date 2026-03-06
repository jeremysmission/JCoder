"""Shared HTTP client factory for JCoder modules."""
import httpx


def make_client(
    timeout_s: float = 30.0,
    retries: int = 2,
    **kwargs,
) -> httpx.Client:
    """Create a pre-configured httpx.Client with standard retry and timeout settings."""
    return httpx.Client(
        timeout=httpx.Timeout(timeout_s),
        transport=httpx.HTTPTransport(retries=retries),
        **kwargs,
    )
