"""Tests for ingestion.pii_scanner -- PII/secret detection and redaction."""

import tempfile
from pathlib import Path

import pytest

from ingestion.pii_scanner import Finding, PIIScanner, ScanResult, sanitize_for_ingest


@pytest.fixture
def scanner():
    return PIIScanner(redact=True)


class TestPIIScanner:
    """Core detection and redaction tests."""

    def test_clean_code(self, scanner):
        code = "def hello():\n    return 42\n"
        result = scanner.scan(code)
        assert result.is_clean
        assert result.clean_text == code

    def test_aws_key(self, scanner):
        text = "aws_key = AKIAIOSFODNN7EXAMPLE"
        result = scanner.scan(text)
        assert not result.is_clean
        assert any(f.pattern_name == "aws_access_key" for f in result.findings)
        assert "[REDACTED_API_KEY]" in result.clean_text

    def test_github_token(self, scanner):
        text = 'token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"'
        result = scanner.scan(text)
        assert any(f.pattern_name == "github_token" for f in result.findings)

    def test_slack_token(self, scanner):
        text = 'SLACK_TOKEN = "xoxb-1234567890-abcdefghij"'
        result = scanner.scan(text)
        assert any(f.pattern_name == "slack_token" for f in result.findings)

    def test_generic_api_key(self, scanner):
        text = 'api_key = "aBcDeFgHiJkLmNoPqRsTuVwX"'
        result = scanner.scan(text)
        assert any(f.pattern_name == "generic_key" for f in result.findings)

    def test_password_assignment(self, scanner):
        text = 'password = "super_secret_123"'
        result = scanner.scan(text)
        assert any(f.pattern_name == "password_assign" for f in result.findings)
        assert "[REDACTED_PASSWORD]" in result.clean_text

    def test_basic_auth_url(self, scanner):
        text = "url = https://admin:hunter2@example.com/api"
        result = scanner.scan(text)
        assert any(f.pattern_name == "basic_auth_url" for f in result.findings)
        assert "[REDACTED_PASSWORD]" in result.clean_text

    def test_private_key_block(self, scanner):
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIBogIBAAJBALRk...\n"
            "-----END RSA PRIVATE KEY-----\n"
        )
        result = scanner.scan(text)
        assert any(f.pattern_name == "private_key_block" for f in result.findings)
        assert "[REDACTED_PRIVATE_KEY]" in result.clean_text

    def test_jwt_token(self, scanner):
        text = 'token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"'
        result = scanner.scan(text)
        assert any(f.pattern_name == "jwt" for f in result.findings)
        assert "[REDACTED_TOKEN]" in result.clean_text

    def test_bearer_header(self, scanner):
        text = 'Authorization: Bearer eyAbCdEfGhIjKlMnOpQrSt.long_token_value_here'
        result = scanner.scan(text)
        assert any(f.pattern_name == "bearer_token" for f in result.findings)

    def test_email_detected(self, scanner):
        text = "contact john.doe@realcompany.com for details"
        result = scanner.scan(text)
        assert any(f.pattern_name == "email" for f in result.findings)
        assert "[REDACTED_EMAIL]" in result.clean_text

    def test_email_safe_domain_skipped(self, scanner):
        text = "send to test@example.com"
        result = scanner.scan(text)
        assert not any(f.pattern_name == "email" for f in result.findings)
        assert "test@example.com" in result.clean_text

    def test_ip_address(self, scanner):
        text = "server at 203.0.113.42"
        result = scanner.scan(text)
        assert any(f.pattern_name == "ipv4" for f in result.findings)
        assert "[REDACTED_IP]" in result.clean_text

    def test_ip_localhost_skipped(self, scanner):
        text = "bind to 127.0.0.1"
        result = scanner.scan(text)
        assert not any(f.pattern_name == "ipv4" for f in result.findings)
        assert "127.0.0.1" in result.clean_text

    def test_ip_private_skipped(self, scanner):
        text = "host at 192.168.1.1 and 10.0.0.1"
        result = scanner.scan(text)
        assert not any(f.pattern_name == "ipv4" for f in result.findings)

    def test_windows_path(self, scanner):
        r"""C:\Users\john\... detected, C:\Windows skipped."""
        text = r"log at C:\Users\john\Documents\secret.txt"
        result = scanner.scan(text)
        assert any(f.pattern_name == "windows_home_path" for f in result.findings)
        assert "[REDACTED_PATH]" in result.clean_text

    def test_windows_safe_path_not_flagged(self, scanner):
        text = r"installed at C:\Windows\System32\cmd.exe"
        result = scanner.scan(text)
        assert not any(f.pattern_name == "windows_home_path" for f in result.findings)

    def test_unix_path(self, scanner):
        text = "config at /home/user/app/config.yaml"
        result = scanner.scan(text)
        assert any(f.pattern_name == "unix_home_path" for f in result.findings)
        assert "[REDACTED_PATH]" in result.clean_text

    def test_unix_safe_path_not_flagged(self, scanner):
        text = "binary at /usr/bin/python3"
        result = scanner.scan(text)
        assert not any(f.pattern_name == "unix_home_path" for f in result.findings)
        assert "/usr/bin/python3" in result.clean_text

    def test_entropy_detection(self, scanner):
        # 30-char high-entropy random string in assignment context
        text = 'secret_val = "aZ9$xK!mR^pL3qW&vN8jT#cY5bF2gH"'
        result = scanner.scan(text)
        has_entropy = any(f.pattern_name == "high_entropy" for f in result.findings)
        has_generic = any(f.pattern_name == "generic_key" for f in result.findings)
        # Either entropy or generic key should catch it
        assert has_entropy or has_generic

    def test_hex_color_not_flagged(self, scanner):
        text = 'color = "#FF0000"\nbg = "#0a0"\n'
        result = scanner.scan(text)
        assert result.is_clean

    def test_uuid_not_flagged(self, scanner):
        text = "id = 550e8400-e29b-41d4-a716-446655440000"
        result = scanner.scan(text)
        # UUID itself should not be flagged as a secret
        assert not any(f.pattern_name in ("hex_secret", "high_entropy")
                       for f in result.findings)

    def test_redaction_format(self, scanner):
        text = 'key = AKIAIOSFODNN7EXAMPLE\npwd = "hunter2"'
        result = scanner.scan(text)
        assert "[REDACTED_API_KEY]" in result.clean_text
        # Redaction markers follow [REDACTED_TYPE] pattern
        for tag in result.clean_text.split():
            if tag.startswith("[REDACTED_"):
                assert tag.endswith("]")

    def test_multiple_findings(self, scanner):
        text = (
            'aws = AKIAIOSFODNN7EXAMPLE\n'
            'password = "hunter2"\n'
            'email = john@realcorp.com\n'
        )
        result = scanner.scan(text)
        types_found = {f.type for f in result.findings}
        assert "api_key" in types_found
        assert "password" in types_found
        assert "email" in types_found
        assert len(result.findings) >= 3

    def test_scan_file(self, scanner, tmp_path):
        content = 'token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"\n'
        f = tmp_path / "sample.py"
        f.write_text(content, encoding="utf-8")
        result = scanner.scan_file(str(f))
        assert not result.is_clean
        assert any(f.pattern_name == "github_token" for f in result.findings)

    def test_sanitize_for_ingest(self):
        text = 'key = AKIAIOSFODNN7EXAMPLE\nprint("hello")\n'
        clean = sanitize_for_ingest(text)
        assert "AKIA" not in clean
        assert "[REDACTED_API_KEY]" in clean
        assert 'print("hello")' in clean

    def test_no_redact_mode(self):
        scanner = PIIScanner(redact=False)
        text = "key = AKIAIOSFODNN7EXAMPLE"
        result = scanner.scan(text)
        assert not result.is_clean
        assert len(result.findings) >= 1
        # Original text preserved -- no redaction markers
        assert result.clean_text == text
        assert "[REDACTED_" not in result.clean_text
