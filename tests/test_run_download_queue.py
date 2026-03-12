from __future__ import annotations

import sys

from scripts import run_download_queue


def test_build_command_uses_module_execution_for_repo_python_scripts():
    entry = {
        "id": "tiny_codes",
        "command": ["scripts/download_instruction_corpora.py", "--only", "tiny_codes"],
    }

    assert run_download_queue.build_command(entry) == [
        sys.executable,
        "-m",
        "scripts.download_instruction_corpora",
        "--only",
        "tiny_codes",
    ]


def test_build_command_falls_back_to_script_path_for_non_python_targets():
    entry = {
        "id": "custom_tool",
        "command": ["tools/custom_runner", "--flag"],
    }

    assert run_download_queue.build_command(entry) == [
        sys.executable,
        str(run_download_queue.PROJECT_ROOT / "tools" / "custom_runner"),
        "--flag",
    ]


def test_filter_queue_skips_manual_auth_optional_and_giant_by_default():
    queue = [
        {"id": "normal", "command": ["scripts/download_python_docs.py"], "tags": ["docs"]},
        {"id": "auth", "command": ["scripts/download_instruction_corpora.py"], "tags": ["manual-auth"]},
        {"id": "optional", "command": ["scripts/download_phase6_datasets.py"], "tags": ["optional"]},
        {"id": "giant", "command": ["scripts/download_github_code.py"], "tags": ["giant"]},
    ]

    runnable, skipped = run_download_queue.filter_queue(queue)

    assert [entry["id"] for entry in runnable] == ["normal"]
    assert [(entry["id"], reason) for entry, reason in skipped] == [
        ("auth", "manual-auth"),
        ("optional", "optional"),
        ("giant", "giant"),
    ]


def test_filter_queue_includes_tagged_jobs_when_flags_enabled():
    queue = [
        {
            "id": "combo",
            "command": ["scripts/download_phase6_datasets.py"],
            "tags": ["manual-auth", "optional", "giant"],
        }
    ]

    runnable, skipped = run_download_queue.filter_queue(
        queue,
        include_manual_auth=True,
        include_optional=True,
        include_giant=True,
    )

    assert [entry["id"] for entry in runnable] == ["combo"]
    assert skipped == []


def test_default_service_flags_include_optional_and_giant(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("JCODER_DOWNLOAD_QUEUE_INCLUDE_MANUAL_AUTH", raising=False)
    monkeypatch.delenv("JCODER_DOWNLOAD_QUEUE_INCLUDE_OPTIONAL", raising=False)
    monkeypatch.delenv("JCODER_DOWNLOAD_QUEUE_INCLUDE_GIANT", raising=False)

    assert run_download_queue.default_service_flags() == {
        "include_manual_auth": False,
        "include_optional": True,
        "include_giant": True,
    }


def test_default_service_flags_enable_manual_auth_when_hf_token_present(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("JCODER_DOWNLOAD_QUEUE_INCLUDE_MANUAL_AUTH", raising=False)

    flags = run_download_queue.default_service_flags()

    assert flags["include_manual_auth"] is True


def test_build_service_command_includes_all_requested_flags():
    assert run_download_queue.build_service_command(
        include_manual_auth=True,
        include_optional=True,
        include_giant=True,
    ) == [
        sys.executable,
        "-m",
        "scripts.run_download_queue",
        "--continue-on-error",
        "--service",
        "--include-manual-auth",
        "--include-optional",
        "--include-giant",
    ]


def test_should_launch_queue_service_skips_clean_completed_same_signature():
    status = {
        "status": "completed",
        "queue_signature": "abc123",
        "runnable_ids": ["job1", "job2"],
        "failures": [],
    }

    should_launch = run_download_queue.should_launch_queue_service(
        status,
        queue_signature="abc123",
        runnable_queue_ids=["job1", "job2"],
    )

    assert should_launch is False


def test_should_launch_queue_service_retries_failed_queue():
    status = {
        "status": "failed",
        "queue_signature": "abc123",
        "runnable_ids": ["job1", "job2"],
        "failures": [{"id": "job2", "exit_code": 1}],
    }

    should_launch = run_download_queue.should_launch_queue_service(
        status,
        queue_signature="abc123",
        runnable_queue_ids=["job1", "job2"],
    )

    assert should_launch is True
