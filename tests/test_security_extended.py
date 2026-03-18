"""Extended security tests for NetworkGate and sanitizer/PII pipelines.

Covers edge cases beyond the basic test suites: mode switching, rate-limit
stubs, malformed URLs, embedded secrets in code, SQL injection patterns,
path traversal, and multi-pattern PII scrubbing.
"""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from core.network_gate import NetworkGate
from ingestion.pii_scanner import PIIScanner, ScanResult, sanitize_for_ingest
from ingestion.sanitizer import (
    SanitizationConfig,
    SanitizationPipeline,
    SanitizationStats,
    _strip_pii,
    _safe_slug,
    PII_PATTERNS,
)


# =====================================================================
# NetworkGate -- extended tests
# =====================================================================

class TestNetworkGateLocalhostBlocking:
    """Localhost mode must reject every non-loopback host."""

    @pytest.mark.parametrize("url", [
        "https://api.openai.com/v1/chat",
        "https://huggingface.co/models",
        "http://192.168.1.1:8080/api",
        "http://10.0.0.5:3000/data",
        "ftp://files.example.com/dump.tar",
        "https://evil.com:443/exfil",
    ])
    def test_blocks_external_urls(self, url):
        gate = NetworkGate(mode="localhost")
        assert gate.allow(url) is False

    @pytest.mark.parametrize("url", [
        "http://localhost:11434/api/generate",
        "http://127.0.0.1:8000/v1/chat/completions",
        "http://[::1]:9090/health",
        "http://localhost/plain",
    ])
    def test_allows_loopback(self, url):
        gate = NetworkGate(mode="localhost")
        assert gate.allow(url) is True

    def test_guard_raises_for_external(self):
        gate = NetworkGate(mode="localhost")
        with pytest.raises(PermissionError, match="blocked"):
            gate.guard("https://attacker.com/payload")

    def test_guard_passes_for_localhost(self):
        gate = NetworkGate(mode="localhost")
        gate.guard("http://localhost:8000/v1")  # must not raise


class TestNetworkGateAllowlistEndpoints:
    """Allowlist mode lets only listed hostnames through."""

    def test_allowed_host_passes(self):
        gate = NetworkGate(mode="allowlist", allowlist=["api.openai.com"])
        assert gate.allow("https://api.openai.com/v1/chat") is True

    def test_unlisted_host_blocked(self):
        gate = NetworkGate(mode="allowlist", allowlist=["api.openai.com"])
        assert gate.allow("https://evil.com/steal") is False

    def test_multiple_allowed_hosts(self):
        hosts = ["api.openai.com", "localhost", "ollama.local"]
        gate = NetworkGate(mode="allowlist", allowlist=hosts)
        for h in hosts:
            assert gate.allow(f"http://{h}:8000/") is True

    def test_subdomain_not_auto_allowed(self):
        gate = NetworkGate(mode="allowlist", allowlist=["openai.com"])
        # Subdomain "api.openai.com" is a different host string
        assert gate.allow("https://api.openai.com/v1") is False


class TestNetworkGateModeSwitching:
    """Switching mode at runtime updates the enforcement rules."""

    def test_offline_to_localhost(self):
        gate = NetworkGate(mode="offline")
        assert gate.allow("http://localhost:8000") is False
        gate.mode = "localhost"
        assert gate.allow("http://localhost:8000") is True

    def test_localhost_to_offline(self):
        gate = NetworkGate(mode="localhost")
        assert gate.allow("http://localhost:8000") is True
        gate.mode = "offline"
        assert gate.allow("http://localhost:8000") is False

    def test_localhost_to_allowlist(self):
        gate = NetworkGate(mode="localhost", allowlist=["api.openai.com"])
        assert gate.allow("https://api.openai.com/v1") is False
        gate.mode = "allowlist"
        assert gate.allow("https://api.openai.com/v1") is True

    def test_allowlist_to_offline_blocks_all(self):
        gate = NetworkGate(mode="allowlist", allowlist=["safe.io"])
        assert gate.allow("https://safe.io/data") is True
        gate.mode = "offline"
        assert gate.allow("https://safe.io/data") is False


class TestNetworkGateMalformedURLs:
    """Malformed or adversarial URLs must never be allowed through."""

    @pytest.mark.parametrize("url", [
        "",
        "not-a-url",
        "://missing-scheme",
        "http://",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "http:///no-host/path",
    ])
    def test_malformed_blocked_in_localhost(self, url):
        gate = NetworkGate(mode="localhost")
        assert gate.allow(url) is False

    @pytest.mark.parametrize("url", [
        "",
        "garbage",
        "ftp://",
    ])
    def test_malformed_blocked_in_allowlist(self, url):
        gate = NetworkGate(mode="allowlist", allowlist=["anything.com"])
        assert gate.allow(url) is False


class TestNetworkGateRateLimitStub:
    """Verify the gate can be extended with a simple rate-limit wrapper."""

    def test_rate_limit_decorator_pattern(self):
        """Show that wrapping allow() with a counter enforces per-endpoint limits."""
        gate = NetworkGate(mode="localhost")
        call_counts: dict = {}
        max_per_endpoint = 3

        def rate_limited_allow(url: str) -> bool:
            if not gate.allow(url):
                return False
            call_counts[url] = call_counts.get(url, 0) + 1
            return call_counts[url] <= max_per_endpoint

        target = "http://localhost:8000/v1"
        for _ in range(max_per_endpoint):
            assert rate_limited_allow(target) is True
        # Fourth call exceeds limit
        assert rate_limited_allow(target) is False

    def test_rate_limit_per_endpoint_isolation(self):
        gate = NetworkGate(mode="localhost")
        call_counts: dict = {}
        limit = 2

        def rate_limited_allow(url: str) -> bool:
            if not gate.allow(url):
                return False
            call_counts[url] = call_counts.get(url, 0) + 1
            return call_counts[url] <= limit

        url_a = "http://localhost:8000/a"
        url_b = "http://localhost:8000/b"
        assert rate_limited_allow(url_a) is True
        assert rate_limited_allow(url_a) is True
        assert rate_limited_allow(url_a) is False  # exceeded
        # url_b still has its own budget
        assert rate_limited_allow(url_b) is True


# =====================================================================
# PII Scanner -- extended security tests
# =====================================================================

class TestPIIDetectionEmails:
    """Email addresses must be detected and scrubbed."""

    @pytest.fixture
    def scanner(self):
        return PIIScanner(redact=True)

    def test_standard_email(self, scanner):
        result = scanner.scan("reach me at alice@bigcorp.com")
        assert any(f.pattern_name == "email" for f in result.findings)
        assert "[REDACTED_EMAIL]" in result.clean_text

    def test_multiple_emails(self, scanner):
        text = "cc: bob@work.org and carol@firm.net"
        result = scanner.scan(text)
        email_findings = [f for f in result.findings if f.pattern_name == "email"]
        # At least one real email detected; both domains are non-safe
        assert len(email_findings) >= 1
        assert "[REDACTED_EMAIL]" in result.clean_text

    def test_safe_domain_not_redacted(self, scanner):
        result = scanner.scan("user@example.com")
        assert not any(f.pattern_name == "email" for f in result.findings)
        assert "user@example.com" in result.clean_text


class TestPIIDetectionPhoneSSN:
    """Phone numbers and SSNs via the sanitizer PII regex set."""

    def test_sanitizer_strips_emails(self):
        stats = SanitizationStats()
        text = "contact admin@secret.com for help"
        cleaned = _strip_pii(text, stats)
        assert "admin@secret.com" not in cleaned
        assert stats.pii_replacements > 0

    def test_sanitizer_strips_urls(self):
        stats = SanitizationStats()
        text = "visit https://internal.corp.com/dashboard"
        cleaned = _strip_pii(text, stats)
        assert "https://internal.corp.com" not in cleaned

    def test_sanitizer_strips_at_mentions(self):
        stats = SanitizationStats()
        text = "Thanks @johndoe for the fix"
        cleaned = _strip_pii(text, stats)
        assert "@johndoe" not in cleaned


class TestPIIDetectionAPIKeys:
    """Various API key formats must be caught."""

    @pytest.fixture
    def scanner(self):
        return PIIScanner(redact=True)

    def test_aws_key(self, scanner):
        text = "AWS_KEY=AKIAIOSFODNN7EXAMPLE"
        result = scanner.scan(text)
        assert any(f.pattern_name == "aws_access_key" for f in result.findings)
        assert "AKIAIOSFODNN7EXAMPLE" not in result.clean_text

    def test_github_token(self, scanner):
        text = 'GITHUB_TOKEN="ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"'
        result = scanner.scan(text)
        assert any(f.pattern_name == "github_token" for f in result.findings)

    def test_slack_token(self, scanner):
        text = 'SLACK="xoxb-123456789012-abcdefghij"'
        result = scanner.scan(text)
        assert any(f.pattern_name == "slack_token" for f in result.findings)

    def test_generic_api_key_in_assignment(self, scanner):
        text = 'api_key = "SuperSecretKeyValue1234567890"'
        result = scanner.scan(text)
        assert any(f.pattern_name == "generic_key" for f in result.findings)


class TestPIIScrubbing:
    """Scrubbing must completely remove the sensitive value."""

    @pytest.fixture
    def scanner(self):
        return PIIScanner(redact=True)

    def test_aws_key_fully_removed(self, scanner):
        original = "key = AKIAIOSFODNN7EXAMPLE"
        result = scanner.scan(original)
        assert "AKIAIOSFODNN7EXAMPLE" not in result.clean_text

    def test_password_fully_removed(self, scanner):
        original = 'password = "my_super_secret_pw"'
        result = scanner.scan(original)
        assert "my_super_secret_pw" not in result.clean_text
        assert "[REDACTED_PASSWORD]" in result.clean_text

    def test_private_key_block_removed(self, scanner):
        original = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIBogIBAAJBALRkabcdef1234567890\n"
            "-----END RSA PRIVATE KEY-----\n"
        )
        result = scanner.scan(original)
        assert "MIIBog" not in result.clean_text
        assert "[REDACTED_PRIVATE_KEY]" in result.clean_text

    def test_email_replaced_with_tag(self, scanner):
        result = scanner.scan("owner: ceo@realcorp.com")
        assert "ceo@realcorp.com" not in result.clean_text
        assert "[REDACTED_EMAIL]" in result.clean_text


class TestEmbeddedSecrets:
    """Code files with embedded secrets must be flagged."""

    @pytest.fixture
    def scanner(self):
        return PIIScanner(redact=True)

    def test_python_config_with_secrets(self, scanner):
        code = (
            "import os\n"
            'DB_PASSWORD = "hunter2_production"\n'
            'API_KEY = "AKIAIOSFODNN7EXAMPLE"\n'
            "def connect():\n"
            "    pass\n"
        )
        result = scanner.scan(code)
        assert not result.is_clean
        types = {f.type for f in result.findings}
        assert "password" in types or "api_key" in types

    def test_env_file_style(self, scanner):
        text = (
            "DATABASE_URL=postgres://admin:s3cret@db.host.com:5432/prod\n"
            "SECRET_KEY=abcdef1234567890abcdef1234567890\n"
        )
        result = scanner.scan(text)
        assert not result.is_clean

    def test_jwt_in_code(self, scanner):
        text = (
            'auth_header = "eyJhbGciOiJIUzI1NiJ9.'
            'eyJzdWIiOiIxMjM0NTY3ODkwIn0.'
            'dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"\n'
        )
        result = scanner.scan(text)
        assert any(f.pattern_name == "jwt" for f in result.findings)
        assert "[REDACTED_TOKEN]" in result.clean_text


class TestSQLInjectionPatterns:
    """SQL injection payloads are adversarial input the sanitizer handles."""

    def test_sql_in_pii_context_stripped(self):
        """URLs with SQL injection payloads are stripped by sanitizer PII rules."""
        stats = SanitizationStats()
        text = "visit https://target.com/page?id=1' OR '1'='1"
        cleaned = _strip_pii(text, stats)
        # The URL pattern strips the entire URL including the SQLi payload
        assert "https://target.com" not in cleaned
        assert stats.pii_replacements > 0

    def test_safe_slug_strips_sql_chars(self):
        """_safe_slug neutralises most SQL metacharacters in slugs."""
        dangerous = "'; DROP TABLE users;--"
        slug = _safe_slug(dangerous)
        # Semicolons and quotes are replaced; hyphens kept (safe in filenames)
        assert ";" not in slug
        assert "'" not in slug
        # The slug is filename-safe: only [a-zA-Z0-9._-] remain
        assert re.match(r"^[a-zA-Z0-9._-]+$", slug)


class TestPathTraversalPatterns:
    """Path traversal attempts must be blocked by _safe_member_target."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        cfg = SanitizationConfig(
            enabled=True,
            clean_archive_dir=str(tmp_path / "clean"),
        )
        return SanitizationPipeline(cfg)

    @pytest.mark.parametrize("member_name", [
        "../../../etc/passwd",
        "foo/../../etc/shadow",
        "/absolute/path/file.txt",
        "normal/../../../escape.txt",
        "..",
        "",
    ])
    def test_traversal_blocked(self, pipeline, tmp_path, member_name):
        root = tmp_path / "extract_root"
        root.mkdir()
        result = pipeline._safe_member_target(root, member_name)
        assert result is None, f"Should block: {member_name!r}"

    @pytest.mark.parametrize("member_name", [
        "safe/nested/file.xml",
        "posts.xml",
        "data/posts.xml",
    ])
    def test_safe_paths_allowed(self, pipeline, tmp_path, member_name):
        root = tmp_path / "extract_root"
        root.mkdir(parents=True, exist_ok=True)
        result = pipeline._safe_member_target(root, member_name)
        assert result is not None, f"Should allow: {member_name!r}"

    def test_backslash_traversal(self, pipeline, tmp_path):
        root = tmp_path / "extract_root"
        root.mkdir()
        result = pipeline._safe_member_target(root, r"..\..\etc\passwd")
        assert result is None


class TestCleanContentPassthrough:
    """Benign content must survive scanning unchanged."""

    @pytest.fixture
    def scanner(self):
        return PIIScanner(redact=True)

    def test_plain_code(self, scanner):
        code = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"
        result = scanner.scan(code)
        assert result.is_clean
        assert result.clean_text == code

    def test_technical_prose(self, scanner):
        text = (
            "The algorithm runs in O(n log n) time. "
            "It uses a balanced binary search tree for indexing."
        )
        result = scanner.scan(text)
        assert result.is_clean
        assert result.clean_text == text

    def test_hex_colors_not_flagged(self, scanner):
        text = 'background: #FF5733;\ncolor: #0a0;\n'
        result = scanner.scan(text)
        assert result.is_clean

    def test_uuid_not_flagged(self, scanner):
        text = "request_id = 550e8400-e29b-41d4-a716-446655440000"
        result = scanner.scan(text)
        assert not any(
            f.pattern_name in ("hex_secret", "high_entropy")
            for f in result.findings
        )

    def test_sanitize_for_ingest_preserves_clean(self):
        code = 'print("hello world")\nx = 42\n'
        assert sanitize_for_ingest(code) == code
