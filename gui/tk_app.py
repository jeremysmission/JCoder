"""Tk desktop shell for the JCoder CLI."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk
from typing import Any

import click

from gui.cli_bridge import (
    GuiCommandSpec,
    command_preview,
    build_command_argv,
    default_value_for_param,
    discover_gui_schema,
    find_missing_required,
    resolve_main_script,
    resolve_repo_root,
)
from gui.theme import (
    FONT,
    FONT_BOLD,
    FONT_MONO,
    apply_ttk_theme,
    configure_entry_widget,
    configure_text_widget,
    palette,
)


class JCoderGuiApp:
    """Operator-friendly GUI wrapper over the full Click command tree."""

    def __init__(self, root: tk.Tk, root_command: click.Group) -> None:
        self.root = root
        self.repo_root = resolve_repo_root()
        self.main_script = resolve_main_script()
        self.schema = discover_gui_schema(root_command)
        self.command_index = {" ".join(spec.path): spec for spec in self.schema.commands}
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._process: subprocess.Popen[str] | None = None
        self._process_thread: threading.Thread | None = None
        self._active_command_key = ""
        self._global_vars: dict[str, tk.Variable] = {}
        self._global_choice_maps: dict[str, dict[str, str]] = {}
        self._command_vars: dict[str, tk.Variable] = {}
        self._command_choice_maps: dict[str, dict[str, str]] = {}
        self._command_title = tk.StringVar(value="Select a command")
        self._command_help = tk.StringVar(value="Pick a command from the left to generate its form.")
        self._status_text = tk.StringVar(value="Ready.")
        self._validation_text = tk.StringVar(value="")

        colors = palette()
        self.root.title("JCoder Control Room")
        self.root.geometry("1540x940")
        self.root.minsize(1220, 760)
        self.root.configure(bg=colors["bg"])
        apply_ttk_theme(self.root)

        self._build_layout()
        self._populate_command_tree()
        self._refresh_preview()
        self.root.after(60, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        colors = palette()
        self.root.grid_columnconfigure(0, weight=0, minsize=280)
        self.root.grid_columnconfigure(1, weight=3, minsize=640)
        self.root.grid_columnconfigure(2, weight=2, minsize=360)
        self.root.grid_rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", padding=(18, 18, 18, 18))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(1, weight=1)

        ttk.Label(sidebar, text="CLI Navigator", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        nav_note = tk.Label(
            sidebar,
            text="Every leaf mirrors a real Click command. No separate shadow workflow.",
            bg=colors["sidebar"],
            fg=colors["muted_fg"],
            font=FONT,
            justify=tk.LEFT,
            wraplength=220,
        )
        nav_note.grid(row=1, column=0, sticky="nw", pady=(6, 10))

        self.command_tree = ttk.Treeview(sidebar, show="tree", style="Nav.Treeview", selectmode="browse")
        self.command_tree.grid(row=2, column=0, sticky="nsew")
        self.command_tree.bind("<<TreeviewSelect>>", self._on_tree_selected)
        sidebar.grid_rowconfigure(2, weight=1)

        content = ttk.Frame(self.root, padding=(16, 16, 12, 16))
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(2, weight=1)

        hero = ttk.Frame(content, style="Card.TFrame", padding=(18, 18, 18, 16))
        hero.grid(row=0, column=0, sticky="ew")
        hero.grid_columnconfigure(0, weight=1)
        ttk.Label(hero, text="JCoder Control Room", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            hero,
            text="HybridRAG-style dark desktop layout: command sidebar, structured forms, live console, and operator rail.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        badge_row = tk.Frame(hero, bg=colors["panel_bg"])
        badge_row.grid(row=2, column=0, sticky="w", pady=(12, 0))
        for index, badge in enumerate(("Desktop parity", "Live subprocess", "Interactive stdin")):
            pill = tk.Label(
                badge_row,
                text=badge,
                bg=colors["accent"] if index == 0 else colors["panel_alt"],
                fg=colors["accent_fg"] if index == 0 else colors["fg"],
                font=FONT_BOLD if index == 0 else FONT,
                padx=10,
                pady=4,
            )
            pill.grid(row=0, column=index, padx=(0, 8))

        form_card = ttk.Frame(content, style="Card.TFrame", padding=(18, 18, 18, 16))
        form_card.grid(row=1, column=0, sticky="ew", pady=(14, 14))
        form_card.grid_columnconfigure(0, weight=1)
        ttk.Label(form_card, text="Global Options", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self._global_frame = tk.Frame(form_card, bg=colors["panel_bg"])
        self._global_frame.grid(row=1, column=0, sticky="ew", pady=(8, 16))
        self._global_frame.grid_columnconfigure(1, weight=1)
        self._render_global_options()

        ttk.Label(form_card, textvariable=self._command_title, style="Section.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(form_card, textvariable=self._command_help, style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(6, 10))
        self._command_form = tk.Frame(form_card, bg=colors["panel_bg"])
        self._command_form.grid(row=4, column=0, sticky="ew")
        self._command_form.grid_columnconfigure(1, weight=1)

        controls = tk.Frame(form_card, bg=colors["panel_bg"])
        controls.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        controls.grid_columnconfigure(0, weight=1)
        self.run_button = ttk.Button(controls, text="Run Command", style="Primary.TButton", command=self._run_selected_command)
        self.run_button.grid(row=0, column=0, sticky="w")
        self.stop_button = ttk.Button(controls, text="Stop", style="Secondary.TButton", command=self._stop_process)
        self.stop_button.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.stop_button.state(["disabled"])
        ttk.Button(controls, text="Clear Output", style="Secondary.TButton", command=self._clear_output).grid(row=0, column=2, sticky="w", padx=(10, 0))

        output_card = ttk.Frame(content, style="Card.TFrame", padding=(18, 18, 18, 18))
        output_card.grid(row=2, column=0, sticky="nsew")
        output_card.grid_columnconfigure(0, weight=1)
        output_card.grid_rowconfigure(1, weight=1)
        ttk.Label(output_card, text="Live Console", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        output_shell = tk.Frame(output_card, bg=colors["panel_bg"])
        output_shell.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        output_shell.grid_columnconfigure(0, weight=1)
        output_shell.grid_rowconfigure(0, weight=1)
        self.output_text = tk.Text(output_shell, height=18)
        configure_text_widget(self.output_text)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        self.output_text.configure(state=tk.DISABLED)
        output_scroll = ttk.Scrollbar(output_shell, orient=tk.VERTICAL, command=self.output_text.yview)
        output_scroll.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=output_scroll.set)
        self.output_text.tag_configure("command", foreground=colors["accent"])
        self.output_text.tag_configure("status_ok", foreground=colors["success"])
        self.output_text.tag_configure("status_fail", foreground=colors["danger"])
        self.output_text.tag_configure("stdin", foreground=colors["warning"])

        rail = ttk.Frame(self.root, padding=(4, 16, 16, 16))
        rail.grid(row=0, column=2, sticky="nsew")
        rail.grid_columnconfigure(0, weight=1)
        rail.grid_rowconfigure(1, weight=1)

        preview_card = ttk.Frame(rail, style="Card.TFrame", padding=(16, 16, 16, 16))
        preview_card.grid(row=0, column=0, sticky="ew")
        preview_card.grid_columnconfigure(0, weight=1)
        ttk.Label(preview_card, text="Command Preview", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(preview_card, text="Preview updates as you type. The GUI still executes the real CLI.", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 8))
        self.preview_text = tk.Text(preview_card, height=6)
        configure_text_widget(self.preview_text)
        self.preview_text.grid(row=2, column=0, sticky="ew")
        self.preview_text.configure(state=tk.DISABLED)
        preview_actions = tk.Frame(preview_card, bg=colors["panel_bg"])
        preview_actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(preview_actions, text="Copy Preview", style="Secondary.TButton", command=self._copy_preview).grid(row=0, column=0, sticky="w")

        history_card = ttk.Frame(rail, style="Card.TFrame", padding=(16, 16, 16, 16))
        history_card.grid(row=1, column=0, sticky="nsew", pady=(14, 14))
        history_card.grid_columnconfigure(0, weight=1)
        history_card.grid_rowconfigure(1, weight=1)
        ttk.Label(history_card, text="Operator Rail", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        history_shell = tk.Frame(history_card, bg=colors["panel_bg"])
        history_shell.grid(row=1, column=0, sticky="nsew", pady=(10, 14))
        history_shell.grid_columnconfigure(0, weight=1)
        history_shell.grid_rowconfigure(1, weight=1)
        stdin_row = tk.Frame(history_shell, bg=colors["panel_bg"])
        stdin_row.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        stdin_row.grid_columnconfigure(0, weight=1)
        self.stdin_entry = tk.Entry(stdin_row)
        configure_entry_widget(self.stdin_entry)
        self.stdin_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(stdin_row, text="Send Input", style="Secondary.TButton", command=self._send_stdin).grid(row=0, column=1, padx=(8, 0))

        self.history_tree = ttk.Treeview(
            history_shell,
            columns=("time", "result", "command"),
            show="headings",
            style="History.Treeview",
            height=8,
        )
        self.history_tree.heading("time", text="Time")
        self.history_tree.heading("result", text="Result")
        self.history_tree.heading("command", text="Command")
        self.history_tree.column("time", width=78, anchor=tk.W, stretch=False)
        self.history_tree.column("result", width=62, anchor=tk.CENTER, stretch=False)
        self.history_tree.column("command", width=220, anchor=tk.W)
        self.history_tree.grid(row=1, column=0, sticky="nsew")
        history_scroll = ttk.Scrollbar(history_shell, orient=tk.VERTICAL, command=self.history_tree.yview)
        history_scroll.grid(row=1, column=1, sticky="ns")
        self.history_tree.configure(yscrollcommand=history_scroll.set)

        status_card = ttk.Frame(rail, style="Card.TFrame", padding=(16, 16, 16, 16))
        status_card.grid(row=2, column=0, sticky="ew")
        status_card.grid_columnconfigure(0, weight=1)
        ttk.Label(status_card, text="Run Status", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self._status_label = tk.Label(status_card, textvariable=self._status_text, bg=colors["panel_bg"], fg=colors["fg"], font=FONT, justify=tk.LEFT, anchor="w", wraplength=300)
        self._status_label.grid(row=1, column=0, sticky="ew", pady=(8, 6))
        self._validation_label = tk.Label(status_card, textvariable=self._validation_text, bg=colors["panel_bg"], fg=colors["warning"], font=FONT, justify=tk.LEFT, anchor="w", wraplength=300)
        self._validation_label.grid(row=2, column=0, sticky="ew")

    def _render_global_options(self) -> None:
        colors = palette()
        for child in self._global_frame.winfo_children():
            child.destroy()
        self._global_vars.clear()
        self._global_choice_maps.clear()
        row_cursor = 0
        for param in self.schema.root_params:
            label = tk.Label(self._global_frame, text=param.label, bg=colors["panel_bg"], fg=colors["fg"], font=FONT_BOLD, anchor="w")
            label.grid(row=row_cursor, column=0, sticky="w", padx=(0, 12), pady=(0, 8))
            var, widget = self._build_input_widget(self._global_frame, param, self._global_vars, self._global_choice_maps)
            widget.grid(row=row_cursor, column=1, sticky="ew", pady=(0, 8))
            row_cursor += 1
            if param.help_text:
                help_label = tk.Label(self._global_frame, text=param.help_text, bg=colors["panel_bg"], fg=colors["muted_fg"], font=FONT, anchor="w", justify=tk.LEFT)
                help_label.grid(row=row_cursor, column=1, sticky="w", pady=(0, 10))
                row_cursor += 1
            self._attach_refresh(var)

    def _populate_command_tree(self) -> None:
        leaf_keys = {" ".join(spec.path) for spec in self.schema.commands}
        inserted: set[str] = set()
        for spec in self.schema.commands:
            trail: list[str] = []
            for part in spec.path:
                trail.append(part)
                key = " ".join(trail)
                if key in inserted:
                    continue
                parent = " ".join(trail[:-1])
                text = part.replace("-", " ").title()
                self.command_tree.insert(parent, "end", iid=key, text=text, open=True)
                inserted.add(key)
        if self.schema.commands:
            first_key = " ".join(self.schema.commands[0].path)
            self.command_tree.selection_set(first_key)
            self.command_tree.focus(first_key)
            self.command_tree.see(first_key)
            self._load_command(first_key)

    def _on_tree_selected(self, _event: object) -> None:
        selected = self.command_tree.selection()
        if not selected:
            return
        key = selected[0]
        if key not in self.command_index:
            return
        self._load_command(key)

    def _load_command(self, key: str) -> None:
        colors = palette()
        spec = self.command_index[key]
        self._active_command_key = key
        self._command_title.set(" / ".join(spec.path))
        self._command_help.set(spec.help_text or spec.short_help or "No command help available.")
        self._command_vars.clear()
        self._command_choice_maps.clear()
        for child in self._command_form.winfo_children():
            child.destroy()

        for row, param in enumerate(spec.params):
            label = tk.Label(self._command_form, text=param.label, bg=colors["panel_bg"], fg=colors["fg"], font=FONT_BOLD, anchor="w")
            label.grid(row=row * 2, column=0, sticky="nw", padx=(0, 12), pady=(0, 4))
            var, widget = self._build_input_widget(self._command_form, param, self._command_vars, self._command_choice_maps)
            widget.grid(row=row * 2, column=1, sticky="ew", pady=(0, 4))
            help_parts = []
            if param.help_text:
                help_parts.append(param.help_text)
            if param.kind == "bool_pair" and param.false_flag:
                help_parts.append(f"Unchecked sends {param.false_flag}.")
            if param.kind == "flag_choice":
                help_parts.append("Pick one flag value.")
            if param.required:
                help_parts.append("Required.")
            if help_parts:
                help_label = tk.Label(
                    self._command_form,
                    text=" ".join(help_parts),
                    bg=colors["panel_bg"],
                    fg=colors["muted_fg"],
                    font=FONT,
                    anchor="w",
                    justify=tk.LEFT,
                    wraplength=540,
                )
                help_label.grid(row=row * 2 + 1, column=1, sticky="w", pady=(0, 10))
            self._attach_refresh(var)
        self._refresh_preview()

    def _build_input_widget(
        self,
        parent: tk.Misc,
        param,
        var_store: dict[str, tk.Variable],
        choice_maps: dict[str, dict[str, str]],
    ) -> tuple[tk.Variable, tk.Widget]:
        if param.kind in {"flag", "bool_pair"}:
            var = tk.BooleanVar(value=default_value_for_param(param))
            widget = ttk.Checkbutton(parent, variable=var, text="Enabled")
        elif param.kind == "choice":
            default = default_value_for_param(param)
            var = tk.StringVar(value=str(default))
            widget = ttk.Combobox(parent, textvariable=var, state="readonly", values=list(param.choices))
        elif param.kind == "flag_choice":
            mapping = {choice.label: choice.value for choice in param.flag_choices}
            reverse = {value: label for label, value in mapping.items()}
            default_value = default_value_for_param(param)
            display = reverse.get(str(default_value), "")
            var = tk.StringVar(value=display)
            widget = ttk.Combobox(parent, textvariable=var, state="readonly", values=list(mapping.keys()))
            choice_maps[param.name] = mapping
        else:
            var = tk.StringVar(value=str(default_value_for_param(param)))
            widget = tk.Entry(parent)
            configure_entry_widget(widget)
            widget.configure(textvariable=var)
        var_store[param.name] = var
        return var, widget

    def _attach_refresh(self, var: tk.Variable) -> None:
        var.trace_add("write", lambda *_: self._refresh_preview())

    def _collect_values(self, values: dict[str, tk.Variable], choice_maps: dict[str, dict[str, str]]) -> dict[str, Any]:
        collected: dict[str, Any] = {}
        for name, var in values.items():
            value = var.get()
            mapping = choice_maps.get(name)
            if mapping:
                collected[name] = mapping.get(str(value), "")
            else:
                collected[name] = value
        return collected

    def _refresh_preview(self) -> None:
        if not self._active_command_key:
            return
        spec = self.command_index[self._active_command_key]
        root_values = self._collect_values(self._global_vars, self._global_choice_maps)
        command_values = self._collect_values(self._command_vars, self._command_choice_maps)
        preview = command_preview(self.schema.root_params, root_values, spec, command_values)
        self._set_text(self.preview_text, preview)
        missing = find_missing_required(spec.params, command_values)
        if missing:
            self._validation_text.set("Missing required fields: " + ", ".join(missing))
        else:
            self._validation_text.set("Ready to run.")

    def _run_selected_command(self) -> None:
        if self._process and self._process.poll() is None:
            self._set_status("A command is already running. Stop it before launching another one.", tone="warning")
            return
        if not self._active_command_key:
            self._set_status("Select a command first.", tone="warning")
            return

        spec = self.command_index[self._active_command_key]
        root_values = self._collect_values(self._global_vars, self._global_choice_maps)
        command_values = self._collect_values(self._command_vars, self._command_choice_maps)
        try:
            argv = build_command_argv(self.schema.root_params, root_values, spec, command_values, strict=True)
        except ValueError as exc:
            self._set_status(str(exc), tone="danger")
            return

        command = [sys.executable, "-u", str(self.main_script), *argv]
        preview = command_preview(self.schema.root_params, root_values, spec, command_values)
        self._append_output("$ " + preview + "\n", tag="command")
        self._set_status(f"Running {' '.join(spec.path)}", tone="normal")

        self.run_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self._process = subprocess.Popen(
            command,
            cwd=str(self.repo_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=0,
        )
        self._process_thread = threading.Thread(target=self._stream_process, args=(self._process, spec, preview), daemon=True)
        self._process_thread.start()

    def _stream_process(self, process: subprocess.Popen[str], spec: GuiCommandSpec, preview: str) -> None:
        assert process.stdout is not None
        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    self._queue.put(("output", line))
                elif process.poll() is not None:
                    break
        except (OSError, ValueError):
            pass  # pipe closed
        code = process.wait()
        self._queue.put(("exit", {"code": code, "spec": spec, "preview": preview}))

    def _poll_queue(self) -> None:
        while True:
            try:
                kind, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "output":
                self._append_output(str(payload))
            elif kind == "exit":
                self._handle_exit(payload["code"], payload["spec"], payload["preview"])
        self.root.after(60, self._poll_queue)

    def _handle_exit(self, code: int, spec: GuiCommandSpec, preview: str) -> None:
        self._process = None
        self.run_button.state(["!disabled"])
        self.stop_button.state(["disabled"])
        timestamp = datetime.now().strftime("%H:%M:%S")
        result = "OK" if code == 0 else f"FAIL {code}"
        self.history_tree.insert("", 0, values=(timestamp, result, " ".join(spec.path)))
        while len(self.history_tree.get_children()) > 20:
            last = self.history_tree.get_children()[-1]
            self.history_tree.delete(last)
        tone = "success" if code == 0 else "danger"
        self._append_output(f"\n[process exited with code {code}]\n", tag="status_ok" if code == 0 else "status_fail")
        self._set_status(f"{' '.join(spec.path)} finished with code {code}.", tone=tone)
        self._set_text(self.preview_text, preview)

    def _send_stdin(self) -> None:
        text = self.stdin_entry.get().strip()
        if not text:
            return
        if not self._process or self._process.poll() is not None or self._process.stdin is None:
            self._set_status("No interactive process is waiting for input.", tone="warning")
            return
        self._process.stdin.write(text + "\n")
        self._process.stdin.flush()
        self.stdin_entry.delete(0, tk.END)
        self._append_output(f"\n> {text}\n", tag="stdin")

    def _stop_process(self) -> None:
        if not self._process or self._process.poll() is not None:
            return
        self._set_status("Stopping running command...", tone="warning")
        self._process.terminate()
        self.root.after(1200, self._force_kill_if_needed)

    def _force_kill_if_needed(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.kill()

    def _clear_output(self) -> None:
        self._set_text(self.output_text, "")

    def _copy_preview(self) -> None:
        preview = self.preview_text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(preview)
        self._set_status("Preview copied to clipboard.", tone="success")

    def _append_output(self, text: str, tag: str | None = None) -> None:
        self.output_text.configure(state=tk.NORMAL)
        if tag:
            self.output_text.insert(tk.END, text, tag)
        else:
            self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.configure(state=tk.DISABLED)

    def _set_status(self, message: str, *, tone: str) -> None:
        colors = palette()
        color = colors["fg"]
        if tone == "success":
            color = colors["success"]
        elif tone == "warning":
            color = colors["warning"]
        elif tone == "danger":
            color = colors["danger"]
        self._status_text.set(message)
        self._status_label.configure(fg=color)

    def _on_close(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.kill()
        self.root.destroy()


def launch_gui(root_command: click.Group) -> None:
    """Launch the desktop GUI for the given Click command tree."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("Tk is unavailable in this environment, so the desktop GUI cannot be launched.") from exc
    JCoderGuiApp(root, root_command)
    root.mainloop()
