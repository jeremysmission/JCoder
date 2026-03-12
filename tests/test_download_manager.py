from __future__ import annotations

import json

import httpx

from core.download_manager import DownloadManager, fetch_huggingface_parquet_urls


def test_download_file_records_success(tmp_path):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-length": "11", "etag": '"abc"'},
            content=b"hello world",
            request=request,
        )
    )

    with DownloadManager(tmp_path, max_retries=0) as downloader:
        downloader._client = httpx.Client(transport=transport)
        result = downloader.download_file("https://example.com/file.txt", "nested/file.txt")

        assert result.ok is True
        assert result.status == "success"
        assert result.path.read_bytes() == b"hello world"
        record = downloader._ledger.latest_success("nested/file.txt")
        assert record is not None
        assert record.bytes_written == 11
        assert record.etag == '"abc"'


def test_download_file_resumes_partial_content(tmp_path):
    requests_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request.headers.get("Range", ""))
        return httpx.Response(
            206,
            headers={"content-length": "6"},
            content=b" world",
            request=request,
        )

    with DownloadManager(tmp_path, max_retries=0) as downloader:
        temp_path = downloader._staging.incoming_path("resume/file.txt")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(b"hello")
        downloader._client = httpx.Client(transport=httpx.MockTransport(handler))

        result = downloader.download_file("https://example.com/file.txt", "resume/file.txt")

        assert result.ok is True
        assert result.resumed_from == 5
        assert result.path.read_bytes() == b"hello world"
        assert requests_seen == ["bytes=5-"]


def test_fetch_huggingface_parquet_urls_uses_fallback_config(tmp_path):
    payload = {
        "python": {"train": ["https://example.com/python.parquet"]},
        "default": {"validation": ["https://example.com/ignore.parquet"]},
    }

    with DownloadManager(tmp_path, max_retries=0) as downloader:
        downloader._client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=json.dumps(payload).encode("utf-8"),
                    headers={"content-type": "application/json"},
                    request=request,
                )
            )
        )

        urls = fetch_huggingface_parquet_urls(
            downloader,
            "org/dataset",
            config="default",
            fallback_to_first_config=True,
        )

        assert urls == ["https://example.com/python.parquet"]
