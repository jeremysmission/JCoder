"""Smoke tests for all CLI commands (R18).

Verifies that all registered CLI commands are importable, have help text,
and the Click group is properly structured. Does NOT invoke real backends.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from cli.commands import cli


runner = CliRunner()


# ---------------------------------------------------------------------------
# CLI group structure
# ---------------------------------------------------------------------------

class TestCLIStructure:

    def test_cli_is_click_group(self):
        import click
        assert isinstance(cli, click.Group)

    def test_help_runs(self):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "JCoder" in result.output or "Usage" in result.output

    def test_all_expected_commands_registered(self):
        names = set(cli.commands.keys())
        expected = {"ask", "doctor", "agent", "ingest-corpus"}
        # At least these core commands should exist
        for cmd in expected:
            assert cmd in names, f"Missing CLI command: {cmd}"


# ---------------------------------------------------------------------------
# Doctor command
# ---------------------------------------------------------------------------

class TestDoctorCLI:

    def test_doctor_help(self):
        result = runner.invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "check" in result.output.lower() or "health" in result.output.lower()

    def test_doctor_check_subcommand_exists(self):
        from cli.doctor_cmd import doctor_cmd
        assert "check" in doctor_cmd.commands

    def test_doctor_fix_subcommand_exists(self):
        from cli.doctor_cmd import doctor_cmd
        assert "fix" in doctor_cmd.commands


# ---------------------------------------------------------------------------
# Individual command help (no backend needed)
# ---------------------------------------------------------------------------

class TestCommandHelp:

    @pytest.mark.parametrize("cmd", [
        "ask",
        "agent",
        "ingest-corpus",
        "evolve",
        "harvest",
        "research",
        "interactive",
        "bench-search",
    ])
    def test_command_help(self, cmd):
        result = runner.invoke(cli, [cmd, "--help"])
        # Some commands may not exist -- skip gracefully
        if result.exit_code == 2 and "No such command" in result.output:
            pytest.skip(f"Command '{cmd}' not registered")
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"


# ---------------------------------------------------------------------------
# Config loading (mock mode)
# ---------------------------------------------------------------------------

class TestMockMode:

    def test_ask_with_mock_and_no_question_shows_error(self):
        result = runner.invoke(cli, ["--mock", "ask"])
        # Should fail because no question argument provided
        assert result.exit_code != 0
