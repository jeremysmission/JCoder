from __future__ import annotations

from pathlib import Path

import pytest

from scripts import download_subject_repos


def test_selected_sources_returns_catalog_order_when_unfiltered():
    assert [source.key for source in download_subject_repos.selected_sources()] == [
        source.key for source in download_subject_repos.REPO_SOURCES
    ]


def test_selected_sources_preserves_requested_order_without_duplicates():
    selected = download_subject_repos.selected_sources([
        "mdn_content",
        "openjdk_jdk",
        "mdn_content",
    ])

    assert [source.key for source in selected] == ["mdn_content", "openjdk_jdk"]


def test_selected_sources_rejects_unknown_key():
    with pytest.raises(KeyError):
        download_subject_repos.selected_sources(["missing_repo"])


def test_build_clone_command_uses_depth_one_clone():
    source = download_subject_repos.SOURCE_BY_KEY["mdn_content"]
    destination = Path("D:/tmp/mdn_content")

    assert download_subject_repos.build_clone_command(source, destination) == [
        "git",
        "clone",
        "--depth",
        "1",
        source.url,
        str(destination),
    ]


def test_build_update_command_uses_fast_forward_pull():
    destination = Path("D:/tmp/mdn_content")

    assert download_subject_repos.build_update_command(destination) == [
        "git",
        "-C",
        str(destination),
        "pull",
        "--ff-only",
    ]
