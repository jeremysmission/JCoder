"""Tests for the GUI logic layer (command catalog, CLI bridge, runner).

Sprint 18 -- No tkinter needed; all UI interactions are mocked.
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest

from gui.command_catalog import (
    CommandSpec,
    ParamSpec,
    build_cli_args,
    build_command_catalog,
    get_command_spec,
    _infer_browse_kind,
    _infer_kind,
)
from gui.cli_bridge import (
    GuiChoice,
    GuiCommandSpec,
    GuiParam,
    GuiSchema,
    build_command_argv,
    default_value_for_param,
    discover_gui_schema,
    find_missing_required,
    _as_bool,
    _is_blank,
)
from gui.runner import CommandRunner
from gui.theme import DARK, apply_ttk_styles


# ---------------------------------------------------------------------------
# Fixtures -- synthetic Click trees
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_cli():
    """A minimal Click group with two leaf commands."""

    @click.group()
    @click.option("--config-dir", default="")
    @click.option("--mock/--no-mock", default=False)
    def root(config_dir, mock):
        pass

    @root.command()
    @click.argument("question")
    @click.option("--top-k", type=int, default=5, help="Number of results")
    @click.option("--verbose", is_flag=True, help="Enable verbose output")
    def ask(question, top_k, verbose):
        """Ask a question."""

    @root.command()
    @click.option("--path", type=click.Path(), help="Target directory")
    @click.option("--format", "fmt", type=click.Choice(["json", "csv"]), default="json")
    def export(path, fmt):
        """Export data."""

    return root


@pytest.fixture()
def nested_cli():
    """Click group with a nested sub-group."""

    @click.group()
    def root():
        pass

    @root.group()
    def sub():
        pass

    @sub.command()
    @click.argument("name")
    def hello(name):
        """Say hello."""

    @root.command()
    def ping():
        """Health check."""

    return root


# ---------------------------------------------------------------------------
# Command catalog generation
# ---------------------------------------------------------------------------

class TestBuildCommandCatalog:

    def test_discovers_leaf_commands(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        ids = [spec.command_id for spec in catalog]
        assert "ask" in ids
        assert "export" in ids

    def test_catalog_sorted_by_id(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        ids = [spec.command_id for spec in catalog]
        assert ids == sorted(ids)

    def test_nested_commands_use_dot_notation(self, nested_cli):
        catalog = build_command_catalog(nested_cli)
        ids = [spec.command_id for spec in catalog]
        assert "sub.hello" in ids
        assert "ping" in ids

    def test_display_name_formatting(self, nested_cli):
        catalog = build_command_catalog(nested_cli)
        spec = get_command_spec(catalog, "sub.hello")
        assert spec.display_name == "Sub / Hello"

    def test_help_text_preserved(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        assert "Ask a question" in spec.help_text

    def test_get_command_spec_missing_raises(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        with pytest.raises(KeyError):
            get_command_spec(catalog, "nonexistent")

    def test_launch_mode_defaults_to_capture(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        for spec in catalog:
            assert spec.launch_mode == "capture"


# ---------------------------------------------------------------------------
# Form field extraction (ParamSpec)
# ---------------------------------------------------------------------------

class TestParamSpecExtraction:

    def test_argument_detected(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        question_param = next(p for p in spec.params if p.name == "question")
        assert question_param.is_argument is True
        assert question_param.required is True

    def test_integer_option(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        top_k = next(p for p in spec.params if p.name == "top_k")
        assert top_k.kind == "int"
        assert top_k.default == 5

    def test_flag_option(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        verbose = next(p for p in spec.params if p.name == "verbose")
        assert verbose.kind == "bool"

    def test_choice_option(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "export")
        fmt = next(p for p in spec.params if p.name == "fmt")
        assert fmt.kind == "choice"
        assert "json" in fmt.choices
        assert "csv" in fmt.choices

    def test_browse_kind_inference(self):
        assert _infer_browse_kind("config_dir") == "dir"
        assert _infer_browse_kind("repo_root") == "dir"
        assert _infer_browse_kind("out_dir") == "dir"
        assert _infer_browse_kind("file_path") == "file"
        assert _infer_browse_kind("benchmark") == "file"
        assert _infer_browse_kind("question") is None

    def test_multiline_fields_detected(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        question = next(p for p in spec.params if p.name == "question")
        assert question.multiline is True

    def test_infer_kind_for_types(self):
        opt_flag = MagicMock(spec=click.Option)
        opt_flag.is_flag = True
        assert _infer_kind(opt_flag) == "bool"

        opt_choice = MagicMock(spec=click.Argument)
        opt_choice.is_flag = False
        opt_choice.type = click.Choice(["a", "b"])
        assert _infer_kind(opt_choice) == "choice"


# ---------------------------------------------------------------------------
# build_cli_args
# ---------------------------------------------------------------------------

class TestBuildCliArgs:

    def test_basic_argument(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        args = build_cli_args(spec, {}, {"question": "hello world", "top_k": 5, "verbose": False})
        assert "ask" in args
        assert "hello world" in args

    def test_global_config_dir_injected(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        args = build_cli_args(
            spec,
            {"config_dir": "/tmp/cfg", "mock": False},
            {"question": "test", "top_k": 5, "verbose": False},
        )
        assert "--config-dir" in args
        assert "/tmp/cfg" in args

    def test_global_mock_flag(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        args = build_cli_args(
            spec,
            {"config_dir": "", "mock": True},
            {"question": "test", "top_k": 5, "verbose": False},
        )
        assert "--mock" in args

    def test_flag_included_when_true(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        args = build_cli_args(spec, {}, {"question": "test", "top_k": 5, "verbose": True})
        assert "--verbose" in args

    def test_flag_excluded_when_false(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        args = build_cli_args(spec, {}, {"question": "test", "top_k": 5, "verbose": False})
        assert "--verbose" not in args

    def test_required_argument_missing_raises(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        with pytest.raises(ValueError, match="required"):
            build_cli_args(spec, {}, {"question": "", "top_k": 5, "verbose": False})

    def test_choice_option_included(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "export")
        args = build_cli_args(spec, {}, {"path": "/out", "fmt": "csv"})
        assert "--format" in args
        assert "csv" in args

    def test_integer_coercion_error(self, simple_cli):
        catalog = build_command_catalog(simple_cli)
        spec = get_command_spec(catalog, "ask")
        with pytest.raises(ValueError, match="integer"):
            build_cli_args(spec, {}, {"question": "test", "top_k": "notanumber", "verbose": False})


# ---------------------------------------------------------------------------
# CLI bridge (gui/cli_bridge.py)
# ---------------------------------------------------------------------------

class TestCliBridge:

    def test_discover_gui_schema(self, simple_cli):
        schema = discover_gui_schema(simple_cli)
        assert isinstance(schema, GuiSchema)
        assert len(schema.commands) == 2
        paths = [cmd.path for cmd in schema.commands]
        assert ("ask",) in paths
        assert ("export",) in paths

    def test_root_params_discovered(self, simple_cli):
        schema = discover_gui_schema(simple_cli)
        names = [p.name for p in schema.root_params]
        assert "config_dir" in names
        assert "mock" in names

    def test_command_for_path(self, simple_cli):
        schema = discover_gui_schema(simple_cli)
        cmd = schema.command_for_path(("ask",))
        assert cmd.path == ("ask",)

    def test_command_for_path_missing(self, simple_cli):
        schema = discover_gui_schema(simple_cli)
        with pytest.raises(KeyError):
            schema.command_for_path(("nope",))

    def test_build_command_argv(self, simple_cli):
        schema = discover_gui_schema(simple_cli)
        cmd = schema.command_for_path(("ask",))
        argv = build_command_argv(
            schema.root_params,
            {"config_dir": "", "mock": False},
            cmd,
            {"question": "why?", "top_k": "3", "verbose": False},
        )
        assert "ask" in argv
        assert "why?" in argv

    def test_find_missing_required(self):
        params = [
            GuiParam(name="q", kind="text", label="Question", help_text="", required=True),
            GuiParam(name="k", kind="integer", label="Top K", help_text="", required=False, default="5"),
        ]
        missing = find_missing_required(params, {"q": "", "k": "5"})
        assert "Question" in missing
        missing2 = find_missing_required(params, {"q": "hello", "k": ""})
        assert missing2 == []

    def test_default_value_for_param(self):
        flag_param = GuiParam(name="v", kind="flag", label="V", help_text="", required=False, default=False)
        assert default_value_for_param(flag_param) is False

        text_param = GuiParam(name="q", kind="text", label="Q", help_text="", required=True, default=None)
        assert default_value_for_param(text_param) == ""

    def test_as_bool_conversions(self):
        assert _as_bool(True, default=False) is True
        assert _as_bool("yes", default=False) is True
        assert _as_bool("off", default=True) is False
        assert _as_bool(None, default=True) is True
        assert _as_bool("", default=False) is False

    def test_is_blank(self):
        assert _is_blank(None) is True
        assert _is_blank("") is True
        assert _is_blank("  ") is True
        assert _is_blank("hello") is False
        assert _is_blank(0) is False


# ---------------------------------------------------------------------------
# CommandRunner (mocked subprocess)
# ---------------------------------------------------------------------------

class TestCommandRunner:

    def test_start_capture_posts_events(self, tmp_path):
        runner = CommandRunner(tmp_path)

        fake_process = MagicMock()
        fake_process.pid = 42
        fake_process.stdout = iter(["line 1\n", "line 2\n"])
        fake_process.wait.return_value = 0
        fake_process.poll.return_value = None

        with patch("gui.runner.subprocess.Popen", return_value=fake_process):
            runner.start(["ask", "hello"])
            # Wait for the daemon thread to finish
            runner._thread.join(timeout=5)

        events = []
        while not runner.events.empty():
            events.append(runner.events.get_nowait())

        types = [e["type"] for e in events]
        assert "started" in types
        assert "output" in types
        assert "finished" in types
        finished = next(e for e in events if e["type"] == "finished")
        assert finished["return_code"] == 0

    def test_start_when_busy_raises(self, tmp_path):
        runner = CommandRunner(tmp_path)
        fake_process = MagicMock()
        fake_process.poll.return_value = None  # still running
        runner._process = fake_process

        with pytest.raises(RuntimeError, match="already running"):
            runner.start(["ask", "test"])

    def test_stop_terminates_process(self, tmp_path):
        runner = CommandRunner(tmp_path)
        fake_process = MagicMock()
        fake_process.poll.return_value = None
        runner._process = fake_process

        runner.stop()
        fake_process.terminate.assert_called_once()

    def test_stop_noop_when_idle(self, tmp_path):
        runner = CommandRunner(tmp_path)
        runner.stop()  # should not raise

    def test_is_running_false_when_idle(self, tmp_path):
        runner = CommandRunner(tmp_path)
        assert runner.is_running is False

    def test_capture_error_posts_error_event(self, tmp_path):
        runner = CommandRunner(tmp_path)

        with patch("gui.runner.subprocess.Popen", side_effect=OSError("boom")):
            runner.start(["bad-cmd"])
            runner._thread.join(timeout=5)

        events = []
        while not runner.events.empty():
            events.append(runner.events.get_nowait())

        types = [e["type"] for e in events]
        assert "error" in types
        error_evt = next(e for e in events if e["type"] == "error")
        assert "boom" in error_evt["message"]

    def test_nonzero_exit_code(self, tmp_path):
        runner = CommandRunner(tmp_path)
        fake_process = MagicMock()
        fake_process.pid = 99
        fake_process.stdout = iter([])
        fake_process.wait.return_value = 1
        fake_process.poll.return_value = None

        with patch("gui.runner.subprocess.Popen", return_value=fake_process):
            runner.start(["fail-cmd"])
            runner._thread.join(timeout=5)

        events = []
        while not runner.events.empty():
            events.append(runner.events.get_nowait())

        finished = next(e for e in events if e["type"] == "finished")
        assert finished["return_code"] == 1


# ---------------------------------------------------------------------------
# Output parsing (ANSI stripping)
# ---------------------------------------------------------------------------

class TestOutputParsing:

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def test_ansi_codes_stripped(self):
        raw = "\x1b[32mSuccess\x1b[0m: done"
        clean = self._ANSI_RE.sub("", raw)
        assert clean == "Success: done"

    def test_plain_text_unchanged(self):
        text = "No special chars here"
        assert self._ANSI_RE.sub("", text) == text

    def test_multiline_ansi_stripping(self):
        raw = "\x1b[1mBold\x1b[0m\n\x1b[31mRed\x1b[0m"
        clean = self._ANSI_RE.sub("", raw)
        assert clean == "Bold\nRed"


# ---------------------------------------------------------------------------
# Theme structure
# ---------------------------------------------------------------------------

class TestTheme:

    def test_dark_theme_has_required_keys(self):
        required = {"bg", "panel_bg", "fg", "input_bg", "accent", "accent_fg", "border"}
        assert required.issubset(set(DARK.keys()))

    def test_dark_colors_are_hex(self):
        hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
        for key, value in DARK.items():
            if key == "name":
                continue
            assert hex_re.match(value), f"DARK['{key}'] = {value!r} is not valid hex"

    def test_apply_ttk_styles_does_not_crash(self):
        """apply_ttk_styles needs a Tk root; skip if display unavailable."""
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pytest.skip("No display available")
        try:
            apply_ttk_styles(DARK)
        finally:
            root.destroy()


# ---------------------------------------------------------------------------
# Command history tracking (simulated)
# ---------------------------------------------------------------------------

class TestCommandHistory:

    def test_history_accumulates(self):
        """Verify a simple list-based history accumulator works."""
        history: list[str] = []
        for cmd in ["ask hello", "export --path /out", "ask again"]:
            history.append(cmd)
        assert len(history) == 3
        assert history[-1] == "ask again"

    def test_history_dedup_last(self):
        """Consecutive duplicates should be skippable."""
        history: list[str] = []
        for cmd in ["ask hello", "ask hello", "export"]:
            if not history or history[-1] != cmd:
                history.append(cmd)
        assert len(history) == 2


# ---------------------------------------------------------------------------
# Invalid command handling
# ---------------------------------------------------------------------------

class TestInvalidCommandHandling:

    def test_plain_command_produces_single_entry(self):
        """A non-group command yields one entry with an empty path."""

        @click.command()
        def solo():
            pass

        catalog = build_command_catalog(solo)
        assert len(catalog) == 1
        assert catalog[0].path == ()
        assert catalog[0].command_id == ""

    def test_build_cli_args_missing_required_option(self):
        spec = CommandSpec(
            command_id="test",
            path=("test",),
            display_name="Test",
            help_text="",
            params=(
                ParamSpec(
                    name="target",
                    label="Target",
                    kind="string",
                    option_strings=("--target",),
                    secondary_option_strings=(),
                    required=True,
                    default=None,
                    help_text="",
                    is_argument=False,
                ),
            ),
        )
        with pytest.raises(ValueError, match="required"):
            build_cli_args(spec, {}, {"target": ""})

    def test_build_cli_args_invalid_int_option(self):
        spec = CommandSpec(
            command_id="test",
            path=("test",),
            display_name="Test",
            help_text="",
            params=(
                ParamSpec(
                    name="count",
                    label="Count",
                    kind="int",
                    option_strings=("--count",),
                    secondary_option_strings=(),
                    required=False,
                    default=None,
                    help_text="",
                    is_argument=False,
                ),
            ),
        )
        with pytest.raises(ValueError, match="integer"):
            build_cli_args(spec, {}, {"count": "abc"})

    def test_build_cli_args_invalid_float_option(self):
        spec = CommandSpec(
            command_id="test",
            path=("test",),
            display_name="Test",
            help_text="",
            params=(
                ParamSpec(
                    name="rate",
                    label="Rate",
                    kind="float",
                    option_strings=("--rate",),
                    secondary_option_strings=(),
                    required=False,
                    default=None,
                    help_text="",
                    is_argument=False,
                ),
            ),
        )
        with pytest.raises(ValueError, match="number"):
            build_cli_args(spec, {}, {"rate": "notfloat"})
