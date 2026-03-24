"""HybridRAG3-derived dark theme for the JCoder GUI."""

from __future__ import annotations

from tkinter import ttk

FONT_FAMILY = "Segoe UI"
FONT = (FONT_FAMILY, 11)
FONT_BOLD = (FONT_FAMILY, 11, "bold")
FONT_TITLE = (FONT_FAMILY, 16, "bold")
FONT_SMALL = (FONT_FAMILY, 10)
FONT_MONO = ("Consolas", 10)

DARK = {
    "name": "dark",
    "bg": "#1e1e1e",
    "panel_bg": "#2d2d2d",
    "fg": "#ffffff",
    "input_bg": "#3c3c3c",
    "input_fg": "#ffffff",
    "accent": "#0078d4",
    "accent_fg": "#ffffff",
    "accent_hover": "#106ebe",
    "border": "#555555",
    "label_fg": "#a0a0a0",
    "disabled_fg": "#777777",
    "green": "#4caf50",
    "red": "#f44336",
    "orange": "#ff9800",
    "gray": "#a0a0a0",
    "scrollbar_bg": "#3c3c3c",
    "scrollbar_fg": "#666666",
}


def apply_ttk_styles(theme: dict | None = None) -> None:
    """Apply ttk styles for the JCoder GUI."""
    t = theme or DARK
    style = ttk.Style()
    style.theme_use("clam")

    style.configure(".", background=t["bg"], foreground=t["fg"], font=FONT)
    style.configure("TFrame", background=t["bg"])
    style.configure("Panel.TFrame", background=t["panel_bg"])
    style.configure("Header.TLabel", background=t["bg"], foreground=t["fg"], font=FONT_TITLE)
    style.configure("Subtle.TLabel", background=t["bg"], foreground=t["label_fg"], font=FONT_SMALL)
    style.configure("Panel.TLabel", background=t["panel_bg"], foreground=t["fg"], font=FONT)
    style.configure("PanelSubtle.TLabel", background=t["panel_bg"], foreground=t["label_fg"], font=FONT_SMALL)
    style.configure("PanelTitle.TLabel", background=t["panel_bg"], foreground=t["fg"], font=FONT_BOLD)

    style.configure(
        "TEntry",
        fieldbackground=t["input_bg"],
        foreground=t["input_fg"],
        insertcolor=t["fg"],
        bordercolor=t["border"],
        relief="flat",
    )
    style.configure(
        "TCombobox",
        fieldbackground=t["input_bg"],
        background=t["input_bg"],
        foreground=t["input_fg"],
        arrowcolor=t["fg"],
        bordercolor=t["border"],
        selectbackground=t["accent"],
        selectforeground=t["accent_fg"],
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", t["input_bg"])],
        foreground=[("readonly", t["input_fg"])],
        selectbackground=[("readonly", t["accent"])],
        selectforeground=[("readonly", t["accent_fg"])],
    )

    style.configure(
        "TButton",
        background=t["accent"],
        foreground=t["accent_fg"],
        font=FONT,
        padding=(16, 8),
        relief="flat",
        borderwidth=0,
    )
    style.map(
        "TButton",
        background=[("active", t["accent_hover"]), ("disabled", t["disabled_fg"])],
        foreground=[("disabled", t["bg"])],
    )
    style.configure(
        "Accent.TButton",
        background=t["accent"],
        foreground=t["accent_fg"],
        font=FONT_BOLD,
        padding=(24, 10),
        relief="flat",
    )
    style.map("Accent.TButton", background=[("active", t["accent_hover"])])
    style.configure(
        "Tertiary.TButton",
        background=t["input_bg"],
        foreground=t["fg"],
        font=FONT,
        padding=(12, 6),
        relief="flat",
        borderwidth=0,
    )
    style.map(
        "Tertiary.TButton",
        background=[("active", t["border"]), ("disabled", t["disabled_fg"])],
        foreground=[("disabled", t["bg"])],
    )

    style.configure("TCheckbutton", background=t["panel_bg"], foreground=t["fg"], font=FONT)
    style.map(
        "TCheckbutton",
        background=[("active", t["panel_bg"])],
        indicatorcolor=[("selected", t["accent"])],
    )
    style.configure(
        "TScrollbar",
        background=t["scrollbar_bg"],
        troughcolor=t["bg"],
        arrowcolor=t["fg"],
        bordercolor=t["border"],
    )
    style.map("TScrollbar", background=[("active", t["scrollbar_fg"])])


# Public dict used by gui.tk_app and external consumers.
THEME = DARK


def palette(name: str | None = None) -> dict:
    """Return the active colour palette dictionary."""
    return DARK


def apply_ttk_theme(root=None, theme: dict | None = None) -> None:
    """Alias for apply_ttk_styles (used by tk_app)."""
    apply_ttk_styles(theme)


def configure_text_widget(widget, theme: dict | None = None) -> None:
    """Apply dark theme colours to a tk.Text widget."""
    t = theme or DARK
    widget.configure(
        bg=t["input_bg"],
        fg=t["input_fg"],
        insertbackground=t["fg"],
        selectbackground=t["accent"],
        selectforeground=t["accent_fg"],
        relief="flat",
        borderwidth=1,
        highlightbackground=t["border"],
        highlightcolor=t["accent"],
        highlightthickness=1,
        font=FONT_MONO,
    )


def configure_entry_widget(widget, theme: dict | None = None) -> None:
    """Apply dark theme colours to a tk.Entry widget."""
    t = theme or DARK
    widget.configure(
        bg=t["input_bg"],
        fg=t["input_fg"],
        insertbackground=t["fg"],
        selectbackground=t["accent"],
        selectforeground=t["accent_fg"],
        relief="flat",
        borderwidth=1,
        highlightbackground=t["border"],
        highlightcolor=t["accent"],
        highlightthickness=1,
    )
