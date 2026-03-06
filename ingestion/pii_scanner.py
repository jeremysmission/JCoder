"""
PII and secret scanner for code files.

Pre-filter used before corpus ingestion to strip sensitive data.
Called by ingestion/corpus_pipeline.py before chunking.

Performance: all regexes compiled at module level (runs on 953K files).
False-positive conservative: better to leave harmless strings than redact code.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A single detected PII or secret occurrence."""
    type: str           # "api_key", "email", "password", "private_key", etc.
    value: str          # redacted preview (first 4 chars + "..." for keys)
    line: int           # 1-based line number
    pattern_name: str   # which regex or heuristic matched


@dataclass
class ScanResult:
    """Output of a scan: clean text plus a list of findings."""
    clean_text: str
    findings: List[Finding] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.findings) == 0


# ---------------------------------------------------------------------------
# Redaction tags
# ---------------------------------------------------------------------------

_REDACT = {
    "api_key":      "[REDACTED_API_KEY]",
    "email":        "[REDACTED_EMAIL]",
    "password":     "[REDACTED_PASSWORD]",
    "private_key":  "[REDACTED_PRIVATE_KEY]",
    "ip_address":   "[REDACTED_IP]",
    "path":         "[REDACTED_PATH]",
    "token":        "[REDACTED_TOKEN]",
}


# ---------------------------------------------------------------------------
# Compiled patterns -- order matters (first match wins per position)
# ---------------------------------------------------------------------------

# Helper: assignment context -- key/value after = or : in code
_ASSIGN = r"""(?:=|:)\s*["'`]"""

# --- Private keys (multi-line, checked first) ---
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH|PGP)\s+PRIVATE\s+KEY(?:\s+BLOCK)?-----"
    r".*?"
    r"-----END\s+(?:RSA|DSA|EC|OPENSSH|PGP)\s+PRIVATE\s+KEY(?:\s+BLOCK)?-----",
    re.DOTALL,
)

# --- JWT tokens ---
_JWT_RE = re.compile(
    r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
)

# --- Bearer tokens ---
_BEARER_RE = re.compile(
    r"(?:Authorization[\"']?\s*[:=]\s*[\"']?\s*Bearer\s+)([A-Za-z0-9._~+/=-]{20,})",
    re.IGNORECASE,
)

# --- AWS access keys ---
_AWS_RE = re.compile(r"AKIA[0-9A-Z]{16}")

# --- GitHub tokens ---
_GITHUB_RE = re.compile(r"gh[porus]_[A-Za-z0-9]{36}")

# --- Slack tokens ---
_SLACK_RE = re.compile(r"xox[bpoa]-[A-Za-z0-9-]{10,}")

# --- Generic API key/secret/token in assignment ---
_GENERIC_KEY_RE = re.compile(
    r"(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token|auth[_-]?token"
    r"|token|secret|password|passwd|pwd)"
    r"""\s*(?:=|:)\s*["'`]([A-Za-z0-9._~+/=-]{16,})["'`]""",
    re.IGNORECASE,
)

# --- Passwords in assignment ---
_PASSWORD_RE = re.compile(
    r"""(?:password|passwd|pwd)\s*(?:=|:)\s*["'`]([^"'`]{1,200})["'`]""",
    re.IGNORECASE,
)

# --- Basic auth URLs ---
_BASIC_AUTH_RE = re.compile(
    r"https?://[A-Za-z0-9._~%-]+:[A-Za-z0-9._~%-]+@[A-Za-z0-9.-]+",
)

# --- Generic hex secrets (32+ hex chars in assignment context) ---
_HEX_SECRET_RE = re.compile(
    r"""(?:key|secret|token|hash|signature)\s*(?:=|:)\s*["'`]([0-9a-fA-F]{32,})["'`]""",
    re.IGNORECASE,
)

# --- Email addresses ---
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
_EMAIL_SAFE_DOMAINS = {"example.com", "test.com", "localhost", "example.org",
                       "test.org", "noreply.com", "placeholder.com"}

# --- IPv4 addresses ---
_IPV4_RE = re.compile(
    r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"
)

# --- User home paths ---
_WIN_PATH_RE = re.compile(
    r"""[A-Za-z]:\\Users\\[A-Za-z0-9._-]+\\[^\s"'`<>|*?]{1,200}""",
)
_UNIX_PATH_RE = re.compile(
    r"""/home/[A-Za-z0-9._-]+/[^\s"'`<>|*?]{1,200}""",
)
_SAFE_UNIX_PREFIXES = ("/usr/bin", "/usr/lib", "/usr/local", "/usr/share",
                       "/etc/", "/tmp/", "/var/", "/dev/", "/opt/",
                       "/proc/", "/sys/")


# ---------------------------------------------------------------------------
# False-positive filters
# ---------------------------------------------------------------------------

# Hex color codes (#RRGGBB or #RGB)
_HEX_COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{3}){1,2}\b")

# UUIDs
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# Git commit hashes (40 hex chars, standalone)
_COMMIT_HASH_RE = re.compile(r"\b[0-9a-f]{40}\b")

# Common version strings that look like IPs
_VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+\.\d+\b")


def _is_private_ip(ip: str) -> bool:
    """Return True for localhost, link-local, and RFC-1918 private ranges."""
    parts = ip.split(".")
    if len(parts) != 4:
        return True  # malformed -> skip
    try:
        octets = [int(p) for p in parts]
    except ValueError:
        return True
    if any(o < 0 or o > 255 for o in octets):
        return True  # invalid
    a, b = octets[0], octets[1]
    # 127.x.x.x, 0.0.0.0, 10.x.x.x, 192.168.x.x, 172.16-31.x.x, 169.254.x.x
    if a == 127 or (a == 0 and octets == [0, 0, 0, 0]):
        return True
    if a == 10:
        return True
    if a == 192 and b == 168:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 169 and b == 254:
        return True
    return False


def _safe_preview(value: str, max_show: int = 4) -> str:
    """Return a redacted preview: first N chars + '...'"""
    if len(value) <= max_show:
        return value
    return value[:max_show] + "..."


def _shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


# Assignment context for entropy check
_ENTROPY_ASSIGN_RE = re.compile(
    r"""(?:=|:)\s*["'`]([A-Za-z0-9._~+/=!@#$^&*-]{20,})["'`]""",
)

# Set of spans known to be false positives (built per-scan)
def _build_fp_spans(text: str) -> List[Tuple[int, int]]:
    """Collect spans of known false positives to skip."""
    spans = []
    for pat in (_HEX_COLOR_RE, _UUID_RE, _COMMIT_HASH_RE):
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def _in_fp_span(pos: int, end: int, fp_spans: List[Tuple[int, int]]) -> bool:
    """Check if a match overlaps a known false-positive span."""
    for s, e in fp_spans:
        if pos < e and end > s:
            return True
    return False


def _line_number(text: str, pos: int) -> int:
    """1-based line number for a character position."""
    return text.count("\n", 0, pos) + 1


# ---------------------------------------------------------------------------
# Scanner class
# ---------------------------------------------------------------------------

class PIIScanner:
    """Regex-based scanner for secrets and PII in source code."""

    def __init__(
        self,
        redact: bool = True,
        custom_patterns: Optional[Dict[str, re.Pattern]] = None,
    ):
        self._redact = redact
        self._custom: Dict[str, re.Pattern] = custom_patterns or {}

    def scan(self, text: str) -> ScanResult:
        """Scan text for PII/secrets, optionally redact them."""
        findings: List[Finding] = []
        fp_spans = _build_fp_spans(text)

        # --- Multi-line: private keys (handle before line-based work) ---
        for m in _PRIVATE_KEY_RE.finditer(text):
            findings.append(Finding(
                type="private_key",
                value=_safe_preview(m.group(), 20),
                line=_line_number(text, m.start()),
                pattern_name="private_key_block",
            ))

        # --- Single-match patterns: (regex, type, pattern_name, group) ---
        _patterns: List[Tuple[re.Pattern, str, str, int]] = [
            (_JWT_RE,           "token",    "jwt",              0),
            (_AWS_RE,           "api_key",  "aws_access_key",   0),
            (_GITHUB_RE,        "api_key",  "github_token",     0),
            (_SLACK_RE,         "api_key",  "slack_token",      0),
            (_BEARER_RE,        "token",    "bearer_token",     1),
            (_GENERIC_KEY_RE,   "api_key",  "generic_key",      1),
            (_HEX_SECRET_RE,    "api_key",  "hex_secret",       1),
            (_PASSWORD_RE,      "password", "password_assign",  1),
            (_BASIC_AUTH_RE,    "password", "basic_auth_url",   0),
        ]

        for pat, ftype, pname, grp in _patterns:
            for m in pat.finditer(text):
                if _in_fp_span(m.start(), m.end(), fp_spans):
                    continue
                val = m.group(grp)
                findings.append(Finding(
                    type=ftype,
                    value=_safe_preview(val),
                    line=_line_number(text, m.start()),
                    pattern_name=pname,
                ))

        # --- Emails (skip safe domains) ---
        for m in _EMAIL_RE.finditer(text):
            addr = m.group()
            domain = addr.rsplit("@", 1)[-1].lower()
            if domain in _EMAIL_SAFE_DOMAINS:
                continue
            if _in_fp_span(m.start(), m.end(), fp_spans):
                continue
            findings.append(Finding(
                type="email",
                value=_safe_preview(addr),
                line=_line_number(text, m.start()),
                pattern_name="email",
            ))

        # --- IPv4 (skip private/localhost) ---
        for m in _IPV4_RE.finditer(text):
            ip = m.group(1)
            if _is_private_ip(ip):
                continue
            if _in_fp_span(m.start(), m.end(), fp_spans):
                continue
            findings.append(Finding(
                type="ip_address",
                value=ip,
                line=_line_number(text, m.start()),
                pattern_name="ipv4",
            ))

        # --- User paths ---
        for pat, pname in [(_WIN_PATH_RE, "windows_home_path"),
                           (_UNIX_PATH_RE, "unix_home_path")]:
            for m in pat.finditer(text):
                p = m.group()
                if any(p.startswith(sp) for sp in _SAFE_UNIX_PREFIXES):
                    continue
                findings.append(Finding(
                    type="path",
                    value=_safe_preview(p, 12),
                    line=_line_number(text, m.start()),
                    pattern_name=pname,
                ))

        # --- Custom patterns ---
        for pname, pat in self._custom.items():
            for m in pat.finditer(text):
                if _in_fp_span(m.start(), m.end(), fp_spans):
                    continue
                findings.append(Finding(
                    type="api_key",
                    value=_safe_preview(m.group()),
                    line=_line_number(text, m.start()),
                    pattern_name=f"custom_{pname}",
                ))

        # --- High-entropy strings in assignment context ---
        for m in _ENTROPY_ASSIGN_RE.finditer(text):
            val = m.group(1)
            if len(val) < 20:
                continue
            # Skip if already caught by an earlier pattern
            if any(f.line == _line_number(text, m.start()) for f in findings):
                continue
            if _in_fp_span(m.start(), m.end(), fp_spans):
                continue
            ent = _shannon_entropy(val)
            if ent > 4.5:
                findings.append(Finding(
                    type="api_key",
                    value=_safe_preview(val),
                    line=_line_number(text, m.start()),
                    pattern_name="high_entropy",
                ))

        # --- Deduplicate by (type, line, pattern_name) ---
        seen = set()
        unique: List[Finding] = []
        for f in findings:
            key = (f.type, f.line, f.pattern_name)
            if key not in seen:
                seen.add(key)
                unique.append(f)
        findings = unique

        # --- Redaction ---
        if self._redact and findings:
            clean = self._apply_redactions(text)
        else:
            clean = text

        return ScanResult(clean_text=clean, findings=findings)

    def scan_file(self, path: str) -> ScanResult:
        """Scan a file for PII/secrets."""
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        result = self.scan(text)
        return result

    def is_clean(self, text: str) -> bool:
        """Quick boolean check -- True if no secrets found."""
        # Run scan without redaction for speed
        saved = self._redact
        self._redact = False
        try:
            result = self.scan(text)
            return result.is_clean
        finally:
            self._redact = saved

    def _apply_redactions(self, text: str) -> str:
        """Replace all detected patterns with redaction tags."""
        out = text

        # Private keys first (multi-line)
        out = _PRIVATE_KEY_RE.sub(_REDACT["private_key"], out)

        # Token patterns
        out = _JWT_RE.sub(_REDACT["token"], out)
        out = _BEARER_RE.sub(lambda m: m.group().replace(m.group(1), _REDACT["token"]),
                             out)

        # API keys / secrets
        for pat in (_AWS_RE, _GITHUB_RE, _SLACK_RE):
            out = pat.sub(_REDACT["api_key"], out)

        def _redact_group(pat: re.Pattern, tag: str, grp: int, txt: str) -> str:
            def _repl(m: re.Match) -> str:
                full = m.group()
                secret = m.group(grp)
                return full.replace(secret, tag)
            return pat.sub(_repl, txt)

        out = _redact_group(_GENERIC_KEY_RE, _REDACT["api_key"], 1, out)
        out = _redact_group(_HEX_SECRET_RE, _REDACT["api_key"], 1, out)

        # Passwords
        out = _redact_group(_PASSWORD_RE, _REDACT["password"], 1, out)
        out = _BASIC_AUTH_RE.sub(_REDACT["password"], out)

        # Emails (skip safe domains)
        def _email_repl(m: re.Match) -> str:
            domain = m.group().rsplit("@", 1)[-1].lower()
            if domain in _EMAIL_SAFE_DOMAINS:
                return m.group()
            return _REDACT["email"]
        out = _EMAIL_RE.sub(_email_repl, out)

        # IPs (skip private)
        def _ip_repl(m: re.Match) -> str:
            if _is_private_ip(m.group(1)):
                return m.group()
            return _REDACT["ip_address"]
        out = _IPV4_RE.sub(_ip_repl, out)

        # Paths
        out = _WIN_PATH_RE.sub(_REDACT["path"], out)
        def _unix_path_repl(m: re.Match) -> str:
            if any(m.group().startswith(sp) for sp in _SAFE_UNIX_PREFIXES):
                return m.group()
            return _REDACT["path"]
        out = _UNIX_PATH_RE.sub(_unix_path_repl, out)

        # Custom patterns
        for _pname, pat in self._custom.items():
            out = pat.sub(_REDACT["api_key"], out)

        return out


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def sanitize_for_ingest(text: str) -> str:
    """One-liner: scan and return clean text. Drops the findings."""
    scanner = PIIScanner(redact=True)
    return scanner.scan(text).clean_text
