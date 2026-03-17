"""Bridge helpers between Click metadata and the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import click


@dataclass(frozen=True)
class GuiChoice:
    """Single choice entry for a GUI-controlled flag or enum."""

    label: str
    value: str
    flag: str


@dataclass(frozen=True)
class GuiParam:
    """GUI-friendly description of a Click parameter."""

    name: str
    kind: str
    label: str
    help_text: str
    required: bool
    default: Any = None
    option_flag: str | None = None
    false_flag: str | None = None
    choices: tuple[str, ...] = ()
    flag_choices: tuple[GuiChoice, ...] = ()


@dataclass(frozen=True)
class GuiCommandSpec:
    """GUI-friendly description of a concrete CLI command."""

    path: tuple[str, ...]
    help_text: str
    short_help: str
    params: tuple[GuiParam, ...]


@dataclass(frozen=True)
class GuiSchema:
    """Discovered GUI schema for the whole CLI tree."""

    root_params: tuple[GuiParam, ...]
    commands: tuple[GuiCommandSpec, ...]

    def command_for_path(self, path: Iterable[str]) -> GuiCommandSpec:
        """Return the command matching a path tuple."""
        wanted = tuple(path)
        for command in self.commands:
            if command.path == wanted:
                return command
        raise KeyError(f"Unknown command path: {' '.join(wanted)}")


def resolve_repo_root() -> Path:
    """Return the JCoder repository root."""
    return Path(__file__).resolve().parents[1]


def resolve_main_script() -> Path:
    """Return the main CLI entry script used by the GUI."""
    return resolve_repo_root() / "main.py"


def default_value_for_param(param: GuiParam) -> Any:
    """Return a GUI-friendly default value for a parameter."""
    if param.kind in {"flag", "bool_pair"}:
        return bool(param.default)
    if param.kind == "flag_choice":
        return "" if param.default is None else str(param.default)
    if param.default is None:
        return ""
    return str(param.default)


def find_missing_required(params: Iterable[GuiParam], values: Mapping[str, Any]) -> list[str]:
    """Return user-facing labels for required parameters that are still blank."""
    missing: list[str] = []
    for param in params:
        if not param.required:
            continue
        value = values.get(param.name)
        if _is_blank(value):
            missing.append(param.label)
    return missing


def discover_gui_schema(root_command: click.Group) -> GuiSchema:
    """Discover the full leaf-command surface from a Click group."""
    root_params = tuple(_convert_params(root_command.params))
    commands: list[GuiCommandSpec] = []
    for path, command in _walk_leaf_commands(root_command, ()):
        commands.append(
            GuiCommandSpec(
                path=path,
                help_text=(command.help or "").strip(),
                short_help=(command.short_help or "").strip(),
                params=tuple(_convert_params(command.params)),
            )
        )
    commands.sort(key=lambda item: item.path)
    return GuiSchema(root_params=root_params, commands=tuple(commands))


def build_command_argv(
    root_params: Iterable[GuiParam],
    root_values: Mapping[str, Any],
    command_spec: GuiCommandSpec,
    command_values: Mapping[str, Any],
    *,
    strict: bool = False,
) -> list[str]:
    """Build the CLI argv list for a command and its current GUI values."""
    argv: list[str] = []
    for param in root_params:
        _append_param_argv(argv, param, root_values.get(param.name), strict=strict)
    argv.extend(command_spec.path)
    for param in command_spec.params:
        _append_param_argv(argv, param, command_values.get(param.name), strict=strict)
    return argv


def command_preview(
    root_params: Iterable[GuiParam],
    root_values: Mapping[str, Any],
    command_spec: GuiCommandSpec,
    command_values: Mapping[str, Any],
) -> str:
    """Return a human-readable command preview string."""
    command = ["python", str(resolve_main_script())]
    command.extend(build_command_argv(root_params, root_values, command_spec, command_values, strict=False))
    return " ".join(_quote_part(part) for part in command)


def _walk_leaf_commands(
    group: click.Group,
    prefix: tuple[str, ...],
) -> Iterable[tuple[tuple[str, ...], click.Command]]:
    for name, command in group.commands.items():
        path = prefix + (name,)
        if isinstance(command, click.Group):
            yield from _walk_leaf_commands(command, path)
        else:
            yield path, command


def _convert_params(params: list[click.Parameter]) -> list[GuiParam]:
    converted: list[GuiParam] = []
    index = 0
    while index < len(params):
        param = params[index]
        if _is_flag_choice_start(params, index):
            grouped: list[click.Option] = []
            while index < len(params):
                next_param = params[index]
                if not isinstance(next_param, click.Option):
                    break
                if next_param.name != param.name or not _is_flag_choice_option(next_param):
                    break
                grouped.append(next_param)
                index += 1
            converted.append(_convert_flag_choice(grouped))
            continue
        converted.append(_convert_param(param))
        index += 1
    return converted


def _is_flag_choice_start(params: list[click.Parameter], index: int) -> bool:
    param = params[index]
    if not isinstance(param, click.Option) or not _is_flag_choice_option(param):
        return False
    if index + 1 >= len(params):
        return False
    next_param = params[index + 1]
    return isinstance(next_param, click.Option) and next_param.name == param.name and _is_flag_choice_option(next_param)


def _is_flag_choice_option(param: click.Option) -> bool:
    return bool(param.is_flag and param.flag_value not in (None, True, False))


def _convert_flag_choice(grouped: list[click.Option]) -> GuiParam:
    first = grouped[0]
    choices = []
    for option in grouped:
        value = str(option.flag_value)
        flag = option.opts[0]
        choices.append(GuiChoice(label=value.replace("_", " ").title(), value=value, flag=flag))
    return GuiParam(
        name=first.name,
        kind="flag_choice",
        label=_label_for_name(first.name),
        help_text=(first.help or "").strip(),
        required=first.required,
        default=None if _is_missing_default(first.default) else first.default,
        flag_choices=tuple(choices),
    )


def _convert_param(param: click.Parameter) -> GuiParam:
    default = None if _is_missing_default(getattr(param, "default", None)) else getattr(param, "default", None)
    help_text = getattr(param, "help", "") or ""
    if isinstance(param, click.Argument):
        return GuiParam(
            name=param.name,
            kind=_kind_for_type(param.type),
            label=_label_for_name(param.name),
            help_text=help_text.strip(),
            required=param.required,
            default=default,
        )
    if isinstance(param, click.Option):
        if isinstance(param.type, click.Choice):
            return GuiParam(
                name=param.name,
                kind="choice",
                label=_label_for_name(param.name),
                help_text=help_text.strip(),
                required=param.required,
                default=default,
                option_flag=param.opts[0],
                choices=tuple(param.type.choices),
            )
        if param.is_flag and param.secondary_opts:
            return GuiParam(
                name=param.name,
                kind="bool_pair",
                label=_label_for_name(param.name),
                help_text=help_text.strip(),
                required=param.required,
                default=default,
                option_flag=param.opts[0],
                false_flag=param.secondary_opts[0],
            )
        if param.is_flag:
            return GuiParam(
                name=param.name,
                kind="flag",
                label=_label_for_name(param.name),
                help_text=help_text.strip(),
                required=param.required,
                default=default,
                option_flag=param.opts[0],
            )
        return GuiParam(
            name=param.name,
            kind=_kind_for_type(param.type),
            label=_label_for_name(param.name),
            help_text=help_text.strip(),
            required=param.required,
            default=default,
            option_flag=param.opts[0],
        )
    raise TypeError(f"Unsupported Click parameter type: {type(param)!r}")


def _kind_for_type(param_type: click.ParamType) -> str:
    if isinstance(param_type, click.Choice):
        return "choice"
    name = param_type.__class__.__name__
    if name == "IntParamType":
        return "integer"
    if name == "FloatParamType":
        return "float"
    if name == "BoolParamType":
        return "flag"
    return "text"


def _append_param_argv(argv: list[str], param: GuiParam, value: Any, *, strict: bool) -> None:
    if param.kind == "flag":
        if _as_bool(value, default=bool(param.default)):
            argv.append(param.option_flag or "")
        return
    if param.kind == "bool_pair":
        current = _as_bool(value, default=bool(param.default))
        default = bool(param.default)
        if current == default:
            return
        argv.append((param.option_flag if current else param.false_flag) or "")
        return
    if param.kind == "flag_choice":
        if _is_blank(value):
            if strict and param.required:
                raise ValueError(f"Missing required value for {param.label}")
            return
        for choice in param.flag_choices:
            if choice.value == str(value):
                argv.append(choice.flag)
                return
        raise ValueError(f"Unsupported choice for {param.label}: {value}")

    if _is_blank(value):
        if strict and param.required:
            raise ValueError(f"Missing required value for {param.label}")
        return

    string_value = str(value)
    if param.option_flag is None:
        argv.append(string_value)
        return
    if not param.required and param.default is not None and string_value == str(param.default):
        return
    argv.extend([param.option_flag, string_value])


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _is_missing_default(value: Any) -> bool:
    return value is None or value.__class__.__name__ == "_Missing"


def _label_for_name(name: str) -> str:
    return name.replace("_", " ").title()


def _quote_part(part: str) -> str:
    if not part or any(char.isspace() for char in part):
        escaped = part.replace("\"", "\\\"")
        return f"\"{escaped}\""
    return part
