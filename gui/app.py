"""Tkinter command center for JCoder."""

from __future__ import annotations

import queue
import re
import shlex
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from cli.commands import cli
from gui.command_catalog import CommandSpec, ParamSpec, build_cli_args, build_command_catalog
from gui.runner import CommandRunner
from gui.theme import DARK, FONT_MONO, FONT_SMALL, apply_ttk_styles

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass
class _FieldHandle:
    spec: ParamSpec
    widget: Any
    getter: Callable[[], object]
    setter: Callable[[object], None]


class JCoderGuiApp(tk.Tk):
    """Main JCoder GUI window."""

    def __init__(self, repo_root: Path | None = None):
        super().__init__()
        self.repo_root = Path(repo_root or Path(__file__).resolve().parent.parent)
        self.theme = DARK
        self.catalog = build_command_catalog(cli)
        self.runner = CommandRunner(self.repo_root)
        self.command_lookup = {spec.command_id: spec for spec in self.catalog}
        self.filtered_ids = [spec.command_id for spec in self.catalog]
        self.selected_command: CommandSpec | None = None
        self.field_handles: dict[str, _FieldHandle] = {}
        self.output_text: tk.Text | None = None
        self.preview_text: tk.Text | None = None

        self.search_var = tk.StringVar()
        self.config_dir_var = tk.StringVar()
        self.mock_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Idle")

        self.title("JCoder Command Center")
        self.geometry("1440x920")
        self.minsize(1180, 760)
        self.configure(bg=self.theme["bg"])
        apply_ttk_styles(self.theme)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()
        self._bind_global_updates()
        self._refresh_command_list()
        self._select_initial_command()
        self.after(120, self._poll_runner)

    def _build_layout(self) -> None:
        header = ttk.Frame(self, padding=16)
        header.pack(fill=tk.X)
        ttk.Label(header, text="JCoder Command Center", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Generated from the live Click command tree. Standard commands stream output below.",
            style="Subtle.TLabel",
        ).pack(anchor=tk.W, pady=(4, 0))

        toolbar = ttk.Frame(self, style="Panel.TFrame", padding=16)
        toolbar.pack(fill=tk.X, padx=16, pady=(0, 12))
        ttk.Label(toolbar, text="Global Options", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(toolbar, text="Config Dir", style="PanelSubtle.TLabel").grid(row=1, column=0, sticky="w", pady=(14, 4))
        config_entry = ttk.Entry(toolbar, textvariable=self.config_dir_var, width=64)
        config_entry.grid(row=2, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(toolbar, text="Browse", command=self._browse_config_dir).grid(row=2, column=1, padx=(0, 12))
        ttk.Checkbutton(toolbar, text="Use Mock Backends", variable=self.mock_var).grid(row=2, column=2, sticky="w")
        toolbar.columnconfigure(0, weight=1)

        body = ttk.Frame(self, padding=(16, 0, 16, 16))
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_nav_panel(body)
        self._build_main_panel(body)

    def _build_nav_panel(self, parent: ttk.Frame) -> None:
        nav = ttk.Frame(parent, style="Panel.TFrame", padding=16)
        nav.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        nav.rowconfigure(2, weight=1)
        nav.columnconfigure(0, weight=1)

        ttk.Label(nav, text="Commands", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        search_entry = ttk.Entry(nav, textvariable=self.search_var)
        search_entry.grid(row=1, column=0, sticky="ew", pady=(14, 10))

        list_frame = tk.Frame(nav, bg=self.theme["panel_bg"], highlightbackground=self.theme["border"], highlightthickness=1)
        list_frame.grid(row=2, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.command_list = tk.Listbox(
            list_frame,
            bg=self.theme["panel_bg"],
            fg=self.theme["fg"],
            selectbackground=self.theme["accent"],
            selectforeground=self.theme["accent_fg"],
            activestyle="none",
            relief="flat",
            borderwidth=0,
            font=FONT_SMALL,
            yscrollcommand=scrollbar.set,
        )
        self.command_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.configure(command=self.command_list.yview)
        self.command_list.bind("<<ListboxSelect>>", self._on_command_select)

    def _build_main_panel(self, parent: ttk.Frame) -> None:
        main = ttk.Frame(parent)
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        main.columnconfigure(0, weight=1)

        form_card = ttk.Frame(main, style="Panel.TFrame", padding=16)
        form_card.grid(row=0, column=0, sticky="nsew")
        form_card.columnconfigure(0, weight=1)
        form_card.rowconfigure(5, weight=1)
        ttk.Label(form_card, text="Command Details", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.command_title = ttk.Label(form_card, text="", style="PanelTitle.TLabel")
        self.command_title.grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.command_help = ttk.Label(form_card, text="", style="PanelSubtle.TLabel", wraplength=900, justify=tk.LEFT)
        self.command_help.grid(row=2, column=0, sticky="w", pady=(2, 12))

        ttk.Label(form_card, text="Command Preview", style="PanelSubtle.TLabel").grid(row=3, column=0, sticky="w", pady=(0, 4))
        self.preview_text = tk.Text(
            form_card,
            height=3,
            bg=self.theme["input_bg"],
            fg=self.theme["fg"],
            insertbackground=self.theme["fg"],
            relief="flat",
            borderwidth=0,
            font=FONT_MONO,
            wrap="word",
        )
        self.preview_text.grid(row=4, column=0, sticky="ew")
        self.preview_text.configure(state="disabled")

        form_container = ttk.Frame(form_card, style="Panel.TFrame")
        form_container.grid(row=5, column=0, sticky="nsew", pady=(12, 0))
        form_container.rowconfigure(0, weight=1)
        form_container.columnconfigure(0, weight=1)

        self.form_canvas = tk.Canvas(
            form_container,
            bg=self.theme["panel_bg"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.form_canvas.grid(row=0, column=0, sticky="nsew")
        form_scroll = ttk.Scrollbar(form_container, orient=tk.VERTICAL, command=self.form_canvas.yview)
        form_scroll.grid(row=0, column=1, sticky="ns")
        self.form_canvas.configure(yscrollcommand=form_scroll.set)
        self.form_frame = ttk.Frame(self.form_canvas, style="Panel.TFrame")
        self.form_frame.columnconfigure(0, weight=1)
        self.form_window = self.form_canvas.create_window((0, 0), window=self.form_frame, anchor="nw")
        self.form_frame.bind("<Configure>", self._sync_form_scrollregion)
        self.form_canvas.bind("<Configure>", self._sync_form_width)

        action_bar = ttk.Frame(main, style="Panel.TFrame", padding=16)
        action_bar.grid(row=1, column=0, sticky="ew", pady=(12, 12))
        self.run_button = ttk.Button(action_bar, text="Run Command", style="Accent.TButton", command=self._run_selected_command)
        self.run_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(action_bar, text="Stop", style="Tertiary.TButton", command=self._stop_command, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(action_bar, text="Clear Output", style="Tertiary.TButton", command=self._clear_output).pack(side=tk.LEFT, padx=(8, 0))
        self.status_label = ttk.Label(action_bar, textvariable=self.status_var, style="PanelSubtle.TLabel")
        self.status_label.pack(side=tk.RIGHT)

        output_card = ttk.Frame(main, style="Panel.TFrame", padding=16)
        output_card.grid(row=2, column=0, sticky="nsew")
        output_card.rowconfigure(1, weight=1)
        output_card.columnconfigure(0, weight=1)
        ttk.Label(output_card, text="Command Output", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.output_text = tk.Text(
            output_card,
            bg=self.theme["input_bg"],
            fg=self.theme["fg"],
            insertbackground=self.theme["fg"],
            relief="flat",
            borderwidth=0,
            wrap="word",
            font=FONT_MONO,
        )
        self.output_text.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        output_scroll = ttk.Scrollbar(output_card, orient=tk.VERTICAL, command=self.output_text.yview)
        output_scroll.grid(row=1, column=1, sticky="ns", pady=(12, 0))
        self.output_text.configure(yscrollcommand=output_scroll.set)

    def _bind_global_updates(self) -> None:
        self.search_var.trace_add("write", lambda *_: self._refresh_command_list())
        self.config_dir_var.trace_add("write", lambda *_: self._update_preview())
        self.mock_var.trace_add("write", lambda *_: self._update_preview())

    def _sync_form_scrollregion(self, event=None) -> None:
        self.form_canvas.configure(scrollregion=self.form_canvas.bbox("all"))

    def _sync_form_width(self, event) -> None:
        self.form_canvas.itemconfigure(self.form_window, width=event.width)

    def _refresh_command_list(self) -> None:
        search_term = self.search_var.get().strip().lower()
        selected_command_id = self.selected_command.command_id if self.selected_command else None
        self.filtered_ids = []
        self.command_list.delete(0, tk.END)
        selected_index = None
        for spec in self.catalog:
            haystack = f"{spec.display_name} {spec.help_text} {spec.command_id}".lower()
            if search_term and search_term not in haystack:
                continue
            self.filtered_ids.append(spec.command_id)
            self.command_list.insert(tk.END, spec.display_name)
            if spec.command_id == selected_command_id:
                selected_index = len(self.filtered_ids) - 1
        if selected_index is not None:
            self.command_list.selection_set(selected_index)

    def _select_initial_command(self) -> None:
        if not self.filtered_ids:
            return
        initial_index = 0
        for index, command_id in enumerate(self.filtered_ids):
            if command_id == "ask":
                initial_index = index
                break
        self.command_list.selection_set(initial_index)
        self.command_list.event_generate("<<ListboxSelect>>")

    def _on_command_select(self, event=None) -> None:
        selection = self.command_list.curselection()
        if not selection:
            return
        command_id = self.filtered_ids[selection[0]]
        self.selected_command = self.command_lookup[command_id]
        self._render_selected_command()

    def _render_selected_command(self) -> None:
        if self.selected_command is None:
            return
        self.command_title.configure(text=self.selected_command.display_name)
        self.command_help.configure(text=self.selected_command.help_text or "No help text available.")
        for child in self.form_frame.winfo_children():
            child.destroy()
        self.field_handles.clear()

        for row, param in enumerate(self.selected_command.params):
            container = ttk.Frame(self.form_frame, style="Panel.TFrame")
            container.grid(row=row, column=0, sticky="ew", pady=(0, 12))
            container.columnconfigure(0, weight=1)

            label_text = param.label + (" *" if param.required else "")
            ttk.Label(container, text=label_text, style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
            widget_holder = ttk.Frame(container, style="Panel.TFrame")
            widget_holder.grid(row=1, column=0, sticky="ew", pady=(4, 0))
            widget_holder.columnconfigure(0, weight=1)
            handle = self._build_field(widget_holder, param)
            self.field_handles[param.name] = handle

            help_parts = []
            if param.help_text:
                help_parts.append(param.help_text)
            if param.option_strings and not param.is_argument:
                help_parts.append(", ".join(param.option_strings + param.secondary_option_strings))
            if help_parts:
                ttk.Label(
                    container,
                    text="  ".join(help_parts),
                    style="PanelSubtle.TLabel",
                    wraplength=880,
                    justify=tk.LEFT,
                ).grid(row=2, column=0, sticky="w", pady=(4, 0))

        action_text = "Launch Interactive Console" if self.selected_command.launch_mode == "external_console" else "Run Command"
        self.run_button.configure(text=action_text)
        self._update_preview()

    def _build_field(self, parent: ttk.Frame, param: ParamSpec) -> _FieldHandle:
        default_value = "" if param.default in (None, "") else param.default

        if param.kind == "bool":
            variable = tk.BooleanVar(value=bool(param.default))
            widget = ttk.Checkbutton(parent, text=param.label, variable=variable)
            widget.grid(row=0, column=0, sticky="w")
            variable.trace_add("write", lambda *_: self._update_preview())
            return _FieldHandle(param, widget, variable.get, variable.set)

        if param.kind in {"choice", "flag_choice"}:
            initial = str(default_value)
            values = param.choices
            if param.kind == "flag_choice":
                values = ("",) + values
                initial = ""
            variable = tk.StringVar(value=initial)
            widget = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
            widget.grid(row=0, column=0, sticky="ew")
            variable.trace_add("write", lambda *_: self._update_preview())
            return _FieldHandle(param, widget, variable.get, variable.set)

        if param.multiline:
            widget = tk.Text(
                parent,
                height=4,
                bg=self.theme["input_bg"],
                fg=self.theme["fg"],
                insertbackground=self.theme["fg"],
                relief="flat",
                borderwidth=0,
                wrap="word",
                font=FONT_MONO,
            )
            widget.grid(row=0, column=0, sticky="ew")
            if default_value:
                widget.insert("1.0", str(default_value))
            widget.bind("<KeyRelease>", lambda *_: self._update_preview())
            return _FieldHandle(
                param,
                widget,
                lambda w=widget: w.get("1.0", "end-1c").strip(),
                lambda value, w=widget: self._set_text_widget(w, value),
            )

        variable = tk.StringVar(value=str(default_value))
        widget = ttk.Entry(parent, textvariable=variable)
        widget.grid(row=0, column=0, sticky="ew")
        variable.trace_add("write", lambda *_: self._update_preview())
        if param.browse_kind:
            ttk.Button(
                parent,
                text="Browse",
                style="Tertiary.TButton",
                command=lambda p=param, v=variable: self._browse_for_param(p, v),
            ).grid(row=0, column=1, padx=(8, 0))
        return _FieldHandle(param, widget, variable.get, variable.set)

    def _set_text_widget(self, widget: tk.Text, value: object) -> None:
        widget.delete("1.0", tk.END)
        if value:
            widget.insert("1.0", str(value))
        self._update_preview()

    def _browse_config_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self._browse_start_dir(self.config_dir_var.get()))
        if chosen:
            self.config_dir_var.set(chosen)

    def _browse_for_param(self, param: ParamSpec, variable: tk.StringVar) -> None:
        initial_dir = self._browse_start_dir(variable.get())
        if param.browse_kind == "file":
            chosen = filedialog.askopenfilename(initialdir=initial_dir)
        else:
            chosen = filedialog.askdirectory(initialdir=initial_dir)
        if chosen:
            variable.set(chosen)

    def _browse_start_dir(self, current_value: str) -> str:
        if current_value:
            path = Path(current_value)
            if path.exists():
                return str(path if path.is_dir() else path.parent)
        return str(self.repo_root)

    def _collect_command_values(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for name, handle in self.field_handles.items():
            values[name] = handle.getter()
        return values

    def _update_preview(self) -> None:
        if self.selected_command is None or self.preview_text is None:
            return
        try:
            cli_args = build_cli_args(
                self.selected_command,
                {"config_dir": self.config_dir_var.get(), "mock": self.mock_var.get()},
                self._collect_command_values(),
            )
            preview = " ".join(shlex.quote(part) for part in ["jcoder", *cli_args])
        except ValueError as exc:
            preview = f"Validation pending: {exc}"
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert("1.0", preview)
        self.preview_text.configure(state="disabled")

    def _run_selected_command(self) -> None:
        if self.selected_command is None:
            return
        try:
            cli_args = build_cli_args(
                self.selected_command,
                {"config_dir": self.config_dir_var.get(), "mock": self.mock_var.get()},
                self._collect_command_values(),
            )
        except ValueError as exc:
            messagebox.showerror("Invalid Command", str(exc), parent=self)
            return

        self._append_output("\n" + "=" * 80 + "\n")
        self._append_output("Launching: " + " ".join(shlex.quote(part) for part in ["jcoder", *cli_args]) + "\n")
        self.status_var.set("Running")
        self.stop_button.configure(state="normal" if self.selected_command.launch_mode == "capture" else "disabled")
        try:
            self.runner.start(cli_args, external_console=self.selected_command.launch_mode == "external_console")
            if self.selected_command.launch_mode == "external_console":
                self.status_var.set("Interactive console launched")
        except RuntimeError as exc:
            messagebox.showwarning("Runner Busy", str(exc), parent=self)
        except Exception as exc:
            self.status_var.set("Launch failed")
            messagebox.showerror("Launch Failed", str(exc), parent=self)

    def _stop_command(self) -> None:
        self.runner.stop()
        self.status_var.set("Stopping")

    def _clear_output(self) -> None:
        if self.output_text is None:
            return
        self.output_text.delete("1.0", tk.END)

    def _append_output(self, text: str) -> None:
        if self.output_text is None:
            return
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)

    def _poll_runner(self) -> None:
        if not self.winfo_exists():
            return
        while True:
            try:
                event = self.runner.events.get_nowait()
            except queue.Empty:
                break
            self._handle_runner_event(event)
        self.after(120, self._poll_runner)

    def _handle_runner_event(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "started":
            self.status_var.set(f"Running PID {event['pid']}")
            return
        if event_type == "launched_external":
            self._append_output("Interactive session launched in a new console window.\n")
            self.stop_button.configure(state="disabled")
            return
        if event_type == "output":
            self._append_output(_ANSI_RE.sub("", event.get("text", "")))
            return
        if event_type == "finished":
            code = event.get("return_code", 1)
            self.status_var.set("Completed" if code == 0 else f"Exited with code {code}")
            self.stop_button.configure(state="disabled")
            self._append_output(f"\n[process exited with code {code}]\n")
            return
        if event_type == "error":
            self.status_var.set("Execution failed")
            self.stop_button.configure(state="disabled")
            self._append_output(f"\n[runner error] {event.get('message', 'unknown error')}\n")

    def _on_close(self) -> None:
        if self.runner.is_running:
            if not messagebox.askyesno("Exit JCoder GUI", "A command is still running. Close anyway?", parent=self):
                return
            self.runner.stop()
        self.destroy()


def main() -> None:
    """Launch the JCoder GUI."""
    app = JCoderGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
