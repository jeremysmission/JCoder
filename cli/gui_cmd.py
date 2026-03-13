"""CLI entrypoint for launching the JCoder GUI."""

from __future__ import annotations

import click


@click.command("gui")
def gui_cmd():
    """Launch the JCoder command-center GUI."""
    from gui.app import main

    main()
