from __future__ import annotations

from scripts import download_expansion_tier1, download_instruction_corpora


def test_instruction_main_returns_nonzero_when_processor_reports_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        download_instruction_corpora,
        "ALL_PROCESSORS",
        {"tiny_codes": lambda: False},
    )
    monkeypatch.setattr(download_instruction_corpora, "INDEX_DIR", tmp_path)

    assert download_instruction_corpora.main(["--only", "tiny_codes"]) == 1


def test_expansion_main_returns_nonzero_when_processor_reports_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        download_expansion_tier1,
        "ALL_PROCESSORS",
        {"learn_rust": lambda: False},
    )
    monkeypatch.setattr(download_expansion_tier1, "INDEX_DIR", tmp_path)

    assert download_expansion_tier1.main(["--only", "learn_rust"]) == 1


def test_process_learn_rust_builds_index_from_raw_text_assets(monkeypatch, tmp_path):
    index_dir = tmp_path / "indexes"
    download_dir = tmp_path / "downloads"
    index_dir.mkdir()
    download_dir.mkdir()

    def fake_download(url: str, local_path):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.name == "rust-books.txt":
            local_path.write_text(
                "Chapter one: ownership and borrowing in Rust.\n\n"
                "Borrow checker basics with longer explanatory text for indexing.",
                encoding="utf-8",
            )
        else:
            local_path.write_text(
                "Q: What is ownership in Rust and why does it matter?\n\n"
                "A: Ownership tracks values, enforces move semantics, and prevents many memory bugs.",
                encoding="utf-8",
            )
        return True

    monkeypatch.setattr(download_expansion_tier1, "INDEX_DIR", index_dir)
    monkeypatch.setattr(download_expansion_tier1, "DOWNLOAD_DIR", download_dir)
    monkeypatch.setattr(download_expansion_tier1, "_download_text_asset", fake_download)

    assert download_expansion_tier1.process_learn_rust() is True
    assert (index_dir / "learn_rust.fts5.db").exists()
