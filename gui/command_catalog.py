"""Build a GUI-safe catalog from the Click command tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import click


_MULTILINE_FIELDS = {"question", "task", "topic", "claim", "followup", "note"}
_DIRECTORY_FIELDS = {
    "path",
    "source_dir",
    "config_dir",
    "working_dir",
    "repo_root",
    "out_dir",
    "index_dir",
    "eval_dir",
}
_FILE_FIELDS = {"file_path", "benchmark"}


@dataclass(frozen=True)
class ParamSpec:
    """Normalized parameter metadata for GUI rendering."""

    name: str
    label: str
    kind: str
    option_strings: tuple[str, ...]
    secondary_option_strings: tuple[str, ...]
    required: bool
    default: Any
    help_text: str
    is_argument: bool
    choices: tuple[str, ...] = ()
    multiline: bool = False
    browse_kind: str | None = None
    flag_values: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CommandSpec:
    """A single executable CLI command."""

    command_id: str
    path: tuple[str, ...]
    display_name: str
    help_text: str
    params: tuple[ParamSpec, ...]
    launch_mode: str = "capture"


def build_command_catalog(root: click.BaseCommand) -> list[CommandSpec]:
    """Collect executable leaf commands from the Click tree."""
    specs: list[CommandSpec] = []
    _walk_commands(root, (), specs)
    return sorted(specs, key=lambda item: item.command_id)


def get_command_spec(specs: Iterable[CommandSpec], command_id: str) -> CommandSpec:
    """Return a specific command spec by ID."""
    for spec in specs:
        if spec.command_id == command_id:
            return spec
    raise KeyError(command_id)


def build_cli_args(
    spec: CommandSpec,
    global_values: dict[str, Any],
    command_values: dict[str, Any],
) -> list[str]:
    """Convert GUI field values into CLI args."""
    args: list[str] = []
    config_dir = str(global_values.get("config_dir", "") or "").strip()
    if config_dir:
        args.extend(["--config-dir", config_dir])
    if bool(global_values.get("mock", False)):
        args.append("--mock")

    args.extend(spec.path)
    for param in spec.params:
        raw_value = command_values.get(param.name, param.default)
        args.extend(_build_param_args(param, raw_value))
    return args


def _walk_commands(
    command: click.BaseCommand,
    prefix: tuple[str, ...],
    out: list[CommandSpec],
) -> None:
    if isinstance(command, click.Group):
        for name, subcommand in command.commands.items():
            _walk_commands(subcommand, prefix + (name,), out)
        return

    command_id = ".".join(prefix)
    display_name = " / ".join(part.replace("-", " ").title() for part in prefix)
    help_text = (command.help or command.short_help or "").strip()
    out.append(
        CommandSpec(
            command_id=command_id,
            path=prefix,
            display_name=display_name,
            help_text=help_text,
            params=tuple(_normalize_params(command.params)),
            launch_mode="external_console" if command_id == "interactive" else "capture",
        )
    )


def _normalize_params(params: list[click.Parameter]) -> list[ParamSpec]:
    normalized: list[ParamSpec] = []
    consumed_flag_groups: set[str] = set()

    flag_value_groups: dict[str, list[click.Option]] = {}
    for param in params:
        if isinstance(param, click.Option) and getattr(param, "flag_value", None) is not None:
            flag_value_groups.setdefault(param.name, []).append(param)

    for param in params:
        if isinstance(param, click.Option) and getattr(param, "flag_value", None) is not None:
            if len(flag_value_groups.get(param.name, ())) <= 1:
                normalized.append(_build_param_spec(param))
                continue
            if param.name in consumed_flag_groups:
                continue
            consumed_flag_groups.add(param.name)
            normalized.append(_build_flag_choice_param(param.name, flag_value_groups[param.name]))
            continue
        normalized.append(_build_param_spec(param))
    return normalized


def _build_flag_choice_param(name: str, options: list[click.Option]) -> ParamSpec:
    label = name.replace("_", " ").title()
    help_text = " / ".join(opt.help or opt.opts[0] for opt in options)
    flag_values = tuple((str(opt.flag_value), opt.opts[0]) for opt in options)
    return ParamSpec(
        name=name,
        label=label,
        kind="flag_choice",
        option_strings=tuple(opt.opts[0] for opt in options),
        secondary_option_strings=(),
        required=False,
        default="",
        help_text=help_text,
        is_argument=False,
        choices=tuple(value for value, _ in flag_values),
        multiline=False,
        browse_kind=None,
        flag_values=flag_values,
    )


def _build_param_spec(param: click.Parameter) -> ParamSpec:
    label = param.name.replace("_", " ").title()
    option_strings = tuple(getattr(param, "opts", ()))
    secondary_option_strings = tuple(getattr(param, "secondary_opts", ()))
    help_text = getattr(param, "help", "") or ""
    default = getattr(param, "default", None)
    if repr(default) == "Sentinel.UNSET":
        default = None
    kind = _infer_kind(param)
    browse_kind = _infer_browse_kind(param.name)
    return ParamSpec(
        name=param.name,
        label=label,
        kind=kind,
        option_strings=option_strings,
        secondary_option_strings=secondary_option_strings,
        required=getattr(param, "required", False),
        default=default,
        help_text=help_text,
        is_argument=isinstance(param, click.Argument),
        choices=tuple(getattr(getattr(param, "type", None), "choices", ()) or ()),
        multiline=param.name in _MULTILINE_FIELDS,
        browse_kind=browse_kind,
    )


def _infer_kind(param: click.Parameter) -> str:
    if isinstance(param, click.Option) and getattr(param, "is_flag", False):
        return "bool"
    param_type = getattr(param, "type", None)
    if isinstance(param_type, click.Choice):
        return "choice"
    type_name = type(param_type).__name__
    if type_name == "IntParamType":
        return "int"
    if type_name == "FloatParamType":
        return "float"
    return "string"


def _infer_browse_kind(name: str) -> str | None:
    if name in _DIRECTORY_FIELDS or name.endswith("_dir") or name.endswith("_root"):
        return "dir"
    if name in _FILE_FIELDS or name.endswith("_file"):
        return "file"
    return None


def _build_param_args(param: ParamSpec, raw_value: Any) -> list[str]:
    if param.kind == "flag_choice":
        selected = str(raw_value or "").strip()
        if not selected:
            return []
        for value, option_string in param.flag_values:
            if value == selected:
                return [option_string]
        raise ValueError(f"{param.label} must be one of: {', '.join(param.choices)}")

    if param.kind == "bool":
        value = bool(raw_value)
        if param.secondary_option_strings:
            if value != bool(param.default):
                return [param.option_strings[0] if value else param.secondary_option_strings[0]]
            return []
        return [param.option_strings[0]] if value else []

    if param.kind == "int":
        value = _coerce_number(raw_value, int, param)
    elif param.kind == "float":
        value = _coerce_number(raw_value, float, param)
    else:
        value = "" if raw_value is None else str(raw_value)

    if param.is_argument:
        if not value and param.required:
            raise ValueError(f"{param.label} is required")
        return [value] if value else []

    if value == "" and not param.required:
        return []
    if value == "" and param.required:
        raise ValueError(f"{param.label} is required")
    return [param.option_strings[0], str(value)]


def _coerce_number(raw_value: Any, caster: type[int] | type[float], param: ParamSpec) -> str | int | float:
    if raw_value in ("", None):
        if param.required:
            raise ValueError(f"{param.label} is required")
        return ""
    try:
        return caster(raw_value)
    except (TypeError, ValueError) as exc:
        kind = "integer" if caster is int else "number"
        raise ValueError(f"{param.label} must be a valid {kind}") from exc
