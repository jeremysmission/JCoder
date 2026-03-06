"""Interactive chat REPL for the JCoder agent."""

from __future__ import annotations

import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

# -- Slash-command help text ------------------------------------------------

_HELP_TEXT = """
[bold]Available commands:[/bold]
  /help               Show this help message
  /quit, /exit        Save session and exit
  /save               Save current session
  /load <session_id>  Load a previous session
  /sessions           List recent sessions
  /clear              Clear conversation history
  /tools              List available tools
  /model              Show current model info
"""

_AUTOSAVE_INTERVAL = 5


# -- Build stack (mirrors agent_cmd._build_stack) ---------------------------

def _build_stack(
    backend: str = "",
    model: str = "",
    endpoint: str = "",
    working_dir: str = ".",
):
    """Build the full agent stack via config_loader, with CLI overrides."""
    from agent.config_loader import load_agent_config, build_agent_from_config

    cfg = load_agent_config()
    if backend:
        cfg.backend = backend
    if model:
        cfg.model = model
    if endpoint:
        cfg.endpoint = endpoint
    cfg.working_dir = working_dir
    cfg.mode = "agent"
    stack = build_agent_from_config(cfg)
    return stack


# -- Session helpers --------------------------------------------------------

def _save_session(session_store, session_id, agent, task_label):
    """Persist the current session to disk."""
    if not session_store:
        console.print("[yellow][WARN][/yellow] No session store configured")
        return
    total = agent._total_input_tokens + agent._total_output_tokens
    session_store.save(
        session_id=session_id,
        task=task_label,
        history=agent._history,
        status="active",
        iterations=len(agent._steps),
        tokens=total,
    )
    console.print(f"[dim]Session saved: {session_id}[/dim]")


def _load_session(session_store, session_id, agent):
    """Load a previous session into the agent's state."""
    if not session_store:
        console.print("[yellow][WARN][/yellow] No session store configured")
        return False
    try:
        data = session_store.load(session_id)
    except FileNotFoundError:
        console.print(f"[red][FAIL][/red] Session not found: {session_id}")
        return False
    agent._history = data["history"]
    agent._session_id = session_id
    agent._total_input_tokens = 0
    agent._total_output_tokens = 0
    agent._steps = []
    msg_count = data.get("message_count", len(data["history"]))
    console.print(
        f"[green][OK][/green] Loaded session {session_id} "
        f"({msg_count} messages, status={data['status']})"
    )
    return True


def _list_sessions(session_store):
    """Print a table of recent sessions."""
    if not session_store:
        console.print("[yellow][WARN][/yellow] No session store configured")
        return
    items = session_store.list_sessions(limit=15)
    if not items:
        console.print("No sessions found.")
        return
    table = Table(title=f"Recent Sessions ({len(items)})")
    table.add_column("ID", style="dim", width=14)
    table.add_column("Task", max_width=50)
    table.add_column("Status", width=10)
    table.add_column("Iters", width=5)
    table.add_column("Updated")
    for s in items:
        table.add_row(
            s.session_id[:12],
            s.task[:50],
            s.status,
            str(s.iterations),
            s.updated_at[:19] if s.updated_at else "",
        )
    console.print(table)


def _list_tools(tools):
    """Print a table of available tools."""
    table = Table(title="Available Tools")
    table.add_column("#", style="dim", width=4)
    table.add_column("Tool Name")
    names = sorted(tools._dispatch.keys())
    for i, name in enumerate(names, 1):
        table.add_row(str(i), name)
    console.print(table)


def _show_model(backend, config):
    """Print current model info."""
    model_name = getattr(backend, "model", "(unknown)")
    endpoint = getattr(backend, "endpoint", getattr(config, "endpoint", ""))
    backend_type = type(backend).__name__
    console.print(f"[bold]Backend:[/bold]  {backend_type}")
    console.print(f"[bold]Model:[/bold]    {model_name}")
    if endpoint:
        console.print(f"[bold]Endpoint:[/bold] {endpoint}")


def _print_banner(model_name, working_dir):
    """Print a welcome banner."""
    console.print()
    console.print("[bold cyan]JCoder Interactive REPL[/bold cyan]")
    console.print(f"  Model:       {model_name}")
    console.print(f"  Working dir: {working_dir}")
    console.print(
        "  Type [bold]/help[/bold] for commands, "
        "[bold]/quit[/bold] to exit."
    )
    console.print()


# -- Main REPL loop --------------------------------------------------------

def _repl_loop(agent, stack, session_id):
    """Core read-eval-print loop."""
    session_store = stack.get("session_store")
    backend = stack["backend"]
    tools = stack["tools"]
    config = stack["config"]
    model_name = getattr(backend, "model", "(unknown)")
    working_dir = getattr(config, "working_dir", ".")

    _print_banner(model_name, working_dir)

    task_count = 0
    task_label = "interactive-repl"

    while True:
        try:
            user_input = console.input("[bold green]>>> [/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            _save_session(session_store, session_id, agent, task_label)
            console.print("[dim]Session saved. Goodbye.[/dim]")
            return

        if not user_input:
            continue

        # -- Slash commands ------------------------------------------------

        cmd_lower = user_input.lower()

        if cmd_lower in ("/quit", "/exit"):
            _save_session(session_store, session_id, agent, task_label)
            console.print("[dim]Session saved. Goodbye.[/dim]")
            return

        if cmd_lower == "/save":
            _save_session(session_store, session_id, agent, task_label)
            continue

        if cmd_lower.startswith("/load"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[red]Usage: /load <session_id>[/red]")
                continue
            target_id = parts[1].strip()
            if _load_session(session_store, target_id, agent):
                session_id = target_id
            continue

        if cmd_lower == "/sessions":
            _list_sessions(session_store)
            continue

        if cmd_lower == "/clear":
            agent._history = []
            agent._steps = []
            agent._total_input_tokens = 0
            agent._total_output_tokens = 0
            task_count = 0
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if cmd_lower == "/tools":
            _list_tools(tools)
            continue

        if cmd_lower == "/model":
            _show_model(backend, config)
            continue

        if cmd_lower == "/help":
            console.print(_HELP_TEXT)
            continue

        if user_input.startswith("/"):
            console.print(
                f"[yellow][WARN][/yellow] Unknown command: {user_input.split()[0]}  "
                f"(type /help for available commands)"
            )
            continue

        # -- Run the agent on user input -----------------------------------

        task_label = user_input[:80]
        try:
            result = agent.run(user_input)
        except Exception as exc:
            console.print(f"[red][FAIL][/red] Agent error: {exc}")
            continue

        # Print the response
        console.print()
        console.print(result.summary)
        console.print()

        # Print summary line
        tag = "[bold green][OK][/bold green]" if result.success else "[bold red][FAIL][/bold red]"
        console.print(
            f"{tag} {result.iterations} iterations, "
            f"{result.tokens:,} tokens"
        )
        console.print()

        # Auto-save every N tasks
        task_count += 1
        if task_count % _AUTOSAVE_INTERVAL == 0:
            _save_session(session_store, session_id, agent, task_label)


# -- Click command ---------------------------------------------------------

@click.command("interactive")
@click.option(
    "--backend", default="",
    help="LLM backend: openai, ollama, anthropic (default from config)",
)
@click.option("--model", default="", help="Model name (default from config)")
@click.option("--endpoint", default="", help="API endpoint URL")
@click.option(
    "--working-dir", default=".",
    help="Working directory for the agent",
)
@click.option(
    "--session", "resume_session", default=None,
    help="Session ID to resume",
)
def interactive(backend, model, endpoint, working_dir, resume_session):
    """Start an interactive chat REPL with the JCoder agent."""
    working_dir = str(Path(working_dir).resolve())

    stack = _build_stack(
        backend=backend,
        model=model,
        endpoint=endpoint,
        working_dir=working_dir,
    )

    agent = stack["agent"]
    session_store = stack.get("session_store")

    # Determine session ID: resume existing or generate new
    if resume_session:
        if session_store:
            try:
                data = session_store.load(resume_session)
                agent._history = data["history"]
                agent._session_id = resume_session
                session_id = resume_session
                console.print(
                    f"[green][OK][/green] Resumed session: {session_id}"
                )
            except FileNotFoundError:
                console.print(
                    f"[red][FAIL][/red] Session not found: {resume_session}"
                )
                raise SystemExit(1)
        else:
            console.print(
                "[red][FAIL][/red] Cannot resume: no session store configured"
            )
            raise SystemExit(1)
    else:
        session_id = uuid.uuid4().hex[:12]
        console.print(f"[dim]New session: {session_id}[/dim]")

    _repl_loop(agent, stack, session_id)
