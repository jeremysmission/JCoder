"""Tests for the GUI Click bridge."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from cli.commands import cli
from gui.cli_bridge import build_command_argv, discover_gui_schema


def _command(schema, *path: str):
    return schema.command_for_path(path)


def test_discover_gui_schema_captures_root_and_nested_commands() -> None:
    schema = discover_gui_schema(cli)

    assert any(param.name == "config_dir" for param in schema.root_params)
    assert any(param.name == "mock" for param in schema.root_params)
    assert any(command.path == ("agent", "run") for command in schema.commands)
    assert any(command.path == ("doctor", "fix") for command in schema.commands)
    assert any(command.path == ("ingest-corpus", "stackoverflow") for command in schema.commands)


def test_discover_gui_schema_groups_flag_choice_options() -> None:
    schema = discover_gui_schema(cli)
    feedback = _command(schema, "agent", "feedback")
    signal = next(param for param in feedback.params if param.name == "signal")

    assert signal.kind == "flag_choice"
    assert {choice.value for choice in signal.flag_choices} == {"good", "bad"}
    assert {choice.flag for choice in signal.flag_choices} == {"--good", "--bad"}


def test_build_command_argv_handles_bool_pairs_and_root_flags() -> None:
    schema = discover_gui_schema(cli)
    command = _command(schema, "ingest-corpus", "stackoverflow")

    argv = build_command_argv(
        schema.root_params,
        {"config_dir": "D:/portable_cfg", "mock": True},
        command,
        {
            "source_dir": "D:/datasets/stackoverflow",
            "resume": False,
            "batch_size": "32",
            "dedup": False,
        },
        strict=True,
    )

    assert argv == [
        "--config-dir",
        "D:/portable_cfg",
        "--mock",
        "ingest-corpus",
        "stackoverflow",
        "D:/datasets/stackoverflow",
        "--no-resume",
        "--batch-size",
        "32",
        "--no-dedup",
    ]


def test_build_command_argv_emits_flag_choice_values() -> None:
    schema = discover_gui_schema(cli)
    command = _command(schema, "agent", "feedback")

    argv = build_command_argv(
        schema.root_params,
        {"config_dir": "", "mock": False},
        command,
        {
            "query_id": "query-17",
            "signal": "bad",
            "note": "retrieval missed the right file",
        },
        strict=True,
    )

    assert argv == [
        "agent",
        "feedback",
        "query-17",
        "--bad",
        "--note",
        "retrieval missed the right file",
    ]


def test_build_command_argv_requires_missing_required_values() -> None:
    schema = discover_gui_schema(cli)
    command = _command(schema, "agent", "run")

    with pytest.raises(ValueError, match="Task"):
        build_command_argv(
            schema.root_params,
            {"config_dir": "", "mock": False},
            command,
            {"mode": "agent"},
            strict=True,
        )


def test_gui_command_help_is_available_without_launching_tk() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["gui", "--help"])

    assert result.exit_code == 0
    assert "Launch the JCoder command-center GUI" in result.output
