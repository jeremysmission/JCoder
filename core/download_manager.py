from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from core.download_ledger import DownloadLedger
from core.download_staging import DownloadStaging
from core.http_factory import make_client

log = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _to_relative_path(path: str | Path) -> str:
    return Path(path).as_posix().lstrip("/")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _huggingface_token() -> str:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(env_name, "").strip()
        if token:
            return token
    return ""


def _is_huggingface_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == "huggingface.co" or host.endswith(".huggingface.co")


def fetch_huggingface_parquet_urls(
    downloader: "DownloadManager",
    dataset_id: str,
    *,
    config: str = "default",
    split: str = "train",
    fallback_to_first_config: bool = False,
) -> list[str]:
    """Fetch Parquet shard URLs for a HuggingFace dataset."""
    url = f"https://huggingface.co/api/datasets/{dataset_id}/parquet"
    payload = downloader.fetch_json(url)
    if not isinstance(payload, dict):
        return []

    configs: list[Any] = []
    if config in payload:
        configs.append(payload[config])
    if fallback_to_first_config:
        configs.extend(value for key, value in payload.items() if key != config)

    for config_data in configs:
        if isinstance(config_data, dict) and split in config_data:
            split_value = config_data[split]
            if isinstance(split_value, list):
                return [str(item) for item in split_value]
        if isinstance(config_data, list):
            return [str(item) for item in config_data]
    return []


@dataclass
class DownloadResult:
    url: str
    path: Path
    ok: bool
    status: str
    attempts: int
    resumed_from: int = 0
    bytes_written: int = 0
    expected_size: int = 0
    sha256: str = ""
    etag: str = ""
    last_modified: str = ""
    error: str = ""


class DownloadManager:
    """Single download path for staged JCoder corpus acquisition."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        user_agent: str = "JCoder/1.0",
        timeout_s: float = 30.0,
        read_timeout_s: float = 600.0,
        max_retries: int = 3,
        retry_delay_s: float = 2.0,
        ledger_name: str = "_download_ledger.sqlite3",
        staging_dir_name: str = "_staging",
    ) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.read_timeout_s = read_timeout_s
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self._client = make_client(
            timeout_s=timeout_s,
            read_timeout_s=read_timeout_s,
            retries=0,
            headers={"User-Agent": user_agent},
        )
        self._staging = DownloadStaging(
            str(self.cache_root / staging_dir_name),
            final_root=str(self.cache_root),
        )
        self._ledger = DownloadLedger(str(self.cache_root / ledger_name))
        self._run_id = f"download-{_utc_now_slug()}"
        self._ledger.start_run(
            self._run_id,
            str(self.cache_root),
            {
                "timeout_s": timeout_s,
                "read_timeout_s": read_timeout_s,
                "max_retries": max_retries,
                "retry_delay_s": retry_delay_s,
                "user_agent": user_agent,
            },
        )

    def __enter__(self) -> "DownloadManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._ledger.finish_run(self._run_id)
        self._ledger.close()
        self._client.close()

    def _request_headers(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, str] | None:
        merged = dict(headers or {})
        if _is_huggingface_url(url):
            token = _huggingface_token()
            if token and "Authorization" not in merged:
                merged["Authorization"] = f"Bearer {token}"
        return merged or None

    def fetch_json(self, url: str) -> Any:
        text = self.fetch_text(url, headers={"Accept": "application/json"})
        return json.loads(text)

    def fetch_text(self, url: str, *, headers: dict[str, str] | None = None) -> str:
        response = self._request_with_retry("GET", url, headers=headers)
        return response.text

    def download_file(
        self,
        url: str,
        relative_path: str | Path,
        *,
        min_existing_bytes: int = 1,
        expected_sha256: str = "",
        overwrite: bool = False,
        chunk_size: int = 256 * 1024,
        progress_label: str = "",
        progress_every_bytes: int = 0,
    ) -> DownloadResult:
        dest_rel = _to_relative_path(relative_path)
        final_path = self.cache_root / dest_rel

        if (
            not overwrite
            and final_path.exists()
            and final_path.stat().st_size >= min_existing_bytes
        ):
            # Verify integrity of cached file if hash was provided
            if expected_sha256:
                cached_sha = _sha256_file(final_path)
                if cached_sha.lower() != expected_sha256.lower():
                    log.warning(
                        "Cached file %s has sha256 mismatch (expected %s, got %s); re-downloading",
                        dest_rel, expected_sha256[:12], cached_sha[:12],
                    )
                    final_path.unlink()
                    # Fall through to download path
                else:
                    result = DownloadResult(
                        url=url,
                        path=final_path,
                        ok=True,
                        status="cached",
                        attempts=0,
                        bytes_written=final_path.stat().st_size,
                        sha256=cached_sha,
                    )
                    self._ledger.record(
                        self._run_id,
                        dest_rel,
                        url,
                        "success",
                        bytes_written=result.bytes_written,
                        sha256=cached_sha,
                        expected_sha256=expected_sha256,
                        extra={"cached": True},
                    )
                    return result
            else:
                result = DownloadResult(
                    url=url,
                    path=final_path,
                    ok=True,
                    status="cached",
                    attempts=0,
                    bytes_written=final_path.stat().st_size,
                )
                self._ledger.record(
                    self._run_id,
                    dest_rel,
                    url,
                    "success",
                    bytes_written=result.bytes_written,
                    extra={"cached": True},
                )
                return result

        temp_path = self._staging.incoming_path(dest_rel)
        resumed_from = temp_path.stat().st_size if temp_path.exists() else 0
        delay = self.retry_delay_s
        last_error = ""

        for attempt in range(1, self.max_retries + 2):
            current_resume = temp_path.stat().st_size if temp_path.exists() else 0
            headers: dict[str, str] = {}
            if current_resume > 0:
                headers["Range"] = f"bytes={current_resume}-"
            request_headers = self._request_headers(url, headers)

            try:
                with self._client.stream(
                    "GET",
                    url,
                    headers=request_headers,
                    timeout=httpx.Timeout(self.timeout_s, read=self.read_timeout_s),
                ) as response:
                    if current_resume > 0 and response.status_code == 200 and temp_path.exists():
                        temp_path.unlink()
                        current_resume = 0
                    if current_resume > 0 and response.status_code == 416 and temp_path.exists():
                        sha416 = _sha256_file(temp_path)
                        if expected_sha256 and sha416.lower() != expected_sha256.lower():
                            # Partial is corrupt -- delete and retry fresh
                            log.warning(
                                "HTTP 416 but sha256 mismatch for %s (expected %s, got %s); "
                                "deleting partial and retrying",
                                dest_rel, expected_sha256[:12], sha416[:12],
                            )
                            temp_path.unlink()
                            continue
                        final_path = self._staging.promote(temp_path, dest_rel)
                        return DownloadResult(
                            url=url,
                            path=final_path,
                            ok=True,
                            status="success",
                            attempts=attempt,
                            resumed_from=current_resume,
                            bytes_written=final_path.stat().st_size,
                            sha256=sha416,
                        )

                    response.raise_for_status()
                    mode = "ab" if current_resume > 0 and response.status_code == 206 else "wb"
                    expected_size = int(response.headers.get("content-length", 0))
                    if current_resume > 0 and response.status_code == 206:
                        expected_size += current_resume

                    if mode == "wb" and temp_path.exists():
                        temp_path.unlink()

                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    next_report = progress_every_bytes if progress_every_bytes > 0 else 0
                    with temp_path.open(mode) as handle:
                        written = current_resume
                        for chunk in response.iter_bytes(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            written += len(chunk)
                            if progress_every_bytes > 0 and written >= next_report:
                                if progress_label:
                                    print(f"       {progress_label}: {written / 1e6:.1f} MB", flush=True)
                                next_report += progress_every_bytes

                    final_path = self._staging.promote(temp_path, dest_rel)
                    sha256 = _sha256_file(final_path)
                    if expected_sha256 and sha256.lower() != expected_sha256.lower():
                        self._staging.quarantine_file(
                            final_path,
                            dest_rel,
                            reason=f"sha256 mismatch: expected {expected_sha256}, got {sha256}",
                        )
                        raise ValueError(
                            f"sha256 mismatch for {dest_rel}: expected {expected_sha256}, got {sha256}"
                        )

                    result = DownloadResult(
                        url=url,
                        path=final_path,
                        ok=True,
                        status="success",
                        attempts=attempt,
                        resumed_from=current_resume,
                        bytes_written=final_path.stat().st_size,
                        expected_size=expected_size,
                        sha256=sha256,
                        etag=response.headers.get("etag", ""),
                        last_modified=response.headers.get("last-modified", ""),
                    )
                    self._ledger.record(
                        self._run_id,
                        dest_rel,
                        url,
                        "success",
                        attempts=attempt,
                        resumed_from=current_resume,
                        bytes_written=result.bytes_written,
                        expected_size=result.expected_size,
                        sha256=result.sha256,
                        expected_sha256=expected_sha256,
                        etag=result.etag,
                        last_modified=result.last_modified,
                    )
                    return result
            except (httpx.HTTPError, OSError, ValueError) as exc:
                last_error = str(exc)
                if attempt <= self.max_retries and self._should_retry(exc):
                    time.sleep(delay)
                    delay *= 2
                    continue

        result = DownloadResult(
            url=url,
            path=final_path,
            ok=False,
            status="failed",
            attempts=self.max_retries + 1,
            resumed_from=resumed_from,
            bytes_written=temp_path.stat().st_size if temp_path.exists() else 0,
            error=last_error,
        )
        self._ledger.record(
            self._run_id,
            dest_rel,
            url,
            "failed",
            attempts=result.attempts,
            resumed_from=result.resumed_from,
            bytes_written=result.bytes_written,
            expected_sha256=expected_sha256,
            error=result.error,
        )
        return result

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        delay = self.retry_delay_s
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 2):
            try:
                request_headers = self._request_headers(url, headers)
                response = self._client.request(
                    method,
                    url,
                    headers=request_headers,
                    timeout=httpx.Timeout(self.timeout_s, read=self.read_timeout_s),
                )
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt <= self.max_retries:
                    response.close()
                    time.sleep(delay)
                    delay *= 2
                    continue
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status_code = exc.response.status_code if exc.response is not None else 0
                if status_code in _RETRYABLE_STATUS_CODES and attempt <= self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt <= self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

        raise RuntimeError(f"Request retries exhausted for {url}") from last_exc

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            response = exc.response
            return response is not None and response.status_code in _RETRYABLE_STATUS_CODES
        return isinstance(exc, (httpx.HTTPError, OSError))
