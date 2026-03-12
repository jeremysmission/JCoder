from scripts import reacquire_se_archives


def test_select_archives_filters_coding_sites():
    rows = [
        {"path": r"D:\se\android.stackexchange.com.7z", "size": 100},
        {"path": r"D:\se\codereview.stackexchange.com.7z", "size": 100},
        {"path": r"D:\se\softwarerecs.stackexchange.com.7z", "size": 100},
    ]

    selected = reacquire_se_archives.select_archives(rows, coding_only=True)

    assert [reacquire_se_archives.archive_site(row) for row in selected] == [
        "codereview.stackexchange.com"
    ]


def test_select_archives_filters_specific_sites_after_coding_filter():
    rows = [
        {"path": r"D:\se\android.stackexchange.com.7z", "size": 100},
        {"path": r"D:\se\codereview.stackexchange.com.7z", "size": 100},
        {"path": r"D:\se\softwarerecs.stackexchange.com.7z", "size": 100},
    ]

    selected = reacquire_se_archives.select_archives(
        rows,
        sites=["android.stackexchange.com", "softwarerecs.stackexchange.com"],
    )

    assert [reacquire_se_archives.archive_site(row) for row in selected] == [
        "android.stackexchange.com",
        "softwarerecs.stackexchange.com",
    ]


def test_select_archives_combines_coding_and_site_filters():
    rows = [
        {"path": r"D:\se\android.stackexchange.com.7z", "size": 100},
        {"path": r"D:\se\codereview.stackexchange.com.7z", "size": 100},
        {"path": r"D:\se\serverfault.com.7z", "size": 100},
    ]

    selected = reacquire_se_archives.select_archives(
        rows,
        coding_only=True,
        sites=["serverfault.com", "android.stackexchange.com"],
    )

    assert [reacquire_se_archives.archive_site(row) for row in selected] == [
        "serverfault.com"
    ]


def test_select_archives_can_exclude_coding_sites():
    rows = [
        {"path": r"D:\se\academia.meta.stackexchange.com.7z", "size": 100},
        {"path": r"D:\se\codereview.stackexchange.com.7z", "size": 100},
        {"path": r"D:\se\serverfault.com.7z", "size": 100},
    ]

    selected = reacquire_se_archives.select_archives(
        rows,
        exclude_coding=True,
    )

    assert [reacquire_se_archives.archive_site(row) for row in selected] == [
        "academia.meta.stackexchange.com"
    ]


def test_resolve_download_target_uses_real_destination_parent():
    cache_root, relative_path = reacquire_se_archives.resolve_download_target(
        r"D:\Projects\KnowledgeBase\stackexchange_20251231\biology.stackexchange.com.7z"
    )

    assert str(cache_root) == r"D:\Projects\KnowledgeBase\stackexchange_20251231"
    assert str(relative_path) == "biology.stackexchange.com.7z"
