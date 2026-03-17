from __future__ import annotations

from scripts import build_fts5_indexes


def test_build_fts5_index_logs_all_parse_errors(tmp_path, monkeypatch):
    clean_dir = tmp_path / "clean_source"
    index_dir = tmp_path / "indexes"
    source_dir = clean_dir / "docs_src"
    source_dir.mkdir(parents=True)

    for idx in range(20):
        (source_dir / f"doc_{idx:02d}.md").write_text(
            "## Heading\nBody content\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(build_fts5_indexes, "CLEAN_DIR", clean_dir)
    monkeypatch.setattr(build_fts5_indexes, "INDEX_DIR", index_dir)

    def explode(_path, max_chars=4000):
        raise ValueError("parse boom")

    monkeypatch.setattr(build_fts5_indexes, "_chunk_docs_file", explode)

    stats = build_fts5_indexes.build_fts5_index(
        "docs_src",
        {"index": "docs_idx", "type": "docs"},
    )

    error_log = index_dir / "docs_idx.errors.log"
    lines = error_log.read_text(encoding="utf-8").splitlines()

    assert stats["errors"] == 20
    assert len(lines) == 20
    assert all("ValueError: parse boom" in line for line in lines)
