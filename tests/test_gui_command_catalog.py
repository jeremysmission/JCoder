"""Tests for the JCoder GUI command catalog and CLI wiring."""

import time
import tkinter as tk
from pathlib import Path

from click.testing import CliRunner
import pytest

from cli.commands import cli
from gui.app import JCoderGuiApp
from gui.command_catalog import build_cli_args, build_command_catalog, get_command_spec


def test_gui_command_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "gui" in result.output


def test_gui_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["gui", "--help"])
    assert result.exit_code == 0
    assert "Launch the JCoder command-center GUI." in result.output


def test_catalog_contains_leaf_commands():
    catalog = build_command_catalog(cli)
    command_ids = {spec.command_id for spec in catalog}
    assert "ask" in command_ids
    assert "agent.run" in command_ids
    assert "doctor.check" in command_ids
    assert "ingest-corpus.stackoverflow" in command_ids
    assert "research.sprint" in command_ids
    assert "interactive" in command_ids


def test_feedback_signal_collapses_to_single_choice():
    feedback = get_command_spec(build_command_catalog(cli), "agent.feedback")
    signal = next(param for param in feedback.params if param.name == "signal")
    assert signal.kind == "flag_choice"
    assert signal.choices == ("good", "bad")


def test_interactive_uses_external_console_mode():
    interactive = get_command_spec(build_command_catalog(cli), "interactive")
    assert interactive.launch_mode == "external_console"


def test_build_cli_args_handles_root_options_and_dual_flags():
    spec = get_command_spec(build_command_catalog(cli), "ingest-corpus.stackoverflow")
    args = build_cli_args(
        spec,
        {"config_dir": "D:/cfg", "mock": True},
        {
            "source_dir": "D:/data/stackoverflow",
            "index_name": "so",
            "max_files": "25",
            "resume": False,
            "fts5_only": True,
            "batch_size": "32",
            "dedup": False,
            "dedup_threshold": "0.9",
        },
    )
    assert args[:3] == ["--config-dir", "D:/cfg", "--mock"]
    assert args[3:5] == ["ingest-corpus", "stackoverflow"]
    assert "--no-resume" in args
    assert "--fts5-only" in args
    assert "--no-dedup" in args
    assert args[-2:] == ["--dedup-threshold", "0.9"]


def test_build_cli_args_handles_flag_choice():
    spec = get_command_spec(build_command_catalog(cli), "agent.feedback")
    args = build_cli_args(
        spec,
        {"config_dir": "", "mock": False},
        {"query_id": "abc123", "signal": "bad", "note": "hallucinated file path"},
    )
    assert args == ["agent", "feedback", "abc123", "--bad", "--note", "hallucinated file path"]


def test_gui_runs_mock_ingest_then_ask_end_to_end(tmp_path):
    """Drive the real Tk app through mock ingest -> ask with a temp config root."""
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    sample_repo = tmp_path / "sample_repo"
    config_dir.mkdir()
    sample_repo.mkdir()
    (sample_repo / "sample.py").write_text(
        "def greet(name):\n"
        "    \"\"\"Return a greeting for the supplied user.\"\"\"\n"
        "    return f\"hello {name}\"\n",
        encoding="utf-8",
    )
    (config_dir / "default.yaml").write_text(
        "storage:\n"
        f"  data_dir: \"{data_dir.as_posix()}\"\n"
        f"  index_dir: \"{(data_dir / 'indexes').as_posix()}\"\n",
        encoding="utf-8",
    )

    try:
        app = JCoderGuiApp(repo_root=Path(__file__).resolve().parent.parent)
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable for GUI smoke: {exc}")
    app.withdraw()
    app.update()

    def select_command(command_id: str) -> None:
        app.selected_command = app.command_lookup[command_id]
        app._render_selected_command()
        app.update()

    def set_field(name: str, value) -> None:
        app.field_handles[name].setter(value)
        app.update()

    def wait_for_finish(timeout: float = 60.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            status = app.status_var.get()
            if status.startswith("Completed") or status.startswith("Exited") or status == "Execution failed":
                return status
            time.sleep(0.05)
        raise AssertionError(f"GUI command timed out; last status was {app.status_var.get()!r}")

    try:
        app.config_dir_var.set(str(config_dir))
        app.mock_var.set(True)
        app.update()

        select_command("ingest")
        set_field("path", str(sample_repo))
        set_field("index_name", "gui_smoke")
        app._run_selected_command()
        assert wait_for_finish() == "Completed"
        ingest_output = app.output_text.get("1.0", "end-1c")
        assert "[OK] Ingested" in ingest_output or "Ingested " in ingest_output

        app._clear_output()
        app.update()

        select_command("ask")
        set_field("question", "How does greet work?")
        set_field("index_name", "gui_smoke")
        app._run_selected_command()
        assert wait_for_finish() == "Completed"
        ask_output = app.output_text.get("1.0", "end-1c")
        assert "Question: How does greet work?" in ask_output
        assert "greet" in ask_output.lower()
    finally:
        app.destroy()
