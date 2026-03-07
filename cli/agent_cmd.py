"""Click CLI commands for the autonomous agent."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _get_agent_config():
    """Load agent config once for persistence path lookups."""
    from agent.config_loader import load_agent_config
    return load_agent_config()


def _goals_path() -> str:
    """Config-driven goals persistence path."""
    try:
        return _get_agent_config().goals_path
    except Exception:
        return str(_REPO_ROOT / "data" / "agent_goals.json")


def _session_dir() -> str:
    """Config-driven session directory."""
    try:
        return _get_agent_config().session_dir
    except Exception:
        return str(_REPO_ROOT / "data" / "agent_sessions")


def _build_stack(backend: str = "", model: str = "", endpoint: str = "",
                 max_iterations: int = 50, working_dir: str = ".",
                 mode: str = "agent", profile: str = ""):
    """Build the full agent stack via config_loader, with CLI overrides."""
    from agent.config_loader import (
        load_agent_config, build_agent_from_config, apply_profile,
    )
    cfg = load_agent_config()
    if profile:
        apply_profile(cfg, profile)
    if backend:
        cfg.backend = backend
    if model:
        cfg.model = model
    if endpoint:
        cfg.endpoint = endpoint
    if max_iterations != 50:  # only override if explicitly changed
        cfg.max_iterations = max_iterations
    cfg.working_dir = working_dir
    if mode != "agent" or not profile:  # explicit --mode overrides profile
        cfg.mode = mode
    stack = build_agent_from_config(cfg)
    return stack


def _goal_table(title: str, goal_list):
    """Print a Rich table of goals."""
    table = Table(title=title)
    table.add_column("#", style="dim", width=4)
    table.add_column("Priority", width=8)
    table.add_column("Title")
    table.add_column("Status", width=10)
    for i, g in enumerate(goal_list, 1):
        table.add_row(str(i), str(getattr(g, "priority", "-")),
                      str(getattr(g, "title", g)),
                      str(getattr(g, "status", "pending")))
    console.print(table)


@click.group("agent")
def agent_cmd():
    """Autonomous agent commands."""


@agent_cmd.command("run")
@click.argument("task")
@click.option("--profile", default="",
              help="Query profile: code, debug, review, explain, refactor, quick, deep")
@click.option("--mode", default="agent",
              type=click.Choice(["agent", "code", "qa", "review", "explain", "debug", "refactor"]),
              help="Prompt mode (overrides profile mode)")
@click.option("--backend", default="",
              help="LLM backend: openai, ollama, anthropic (default from config)")
@click.option("--model", default="", help="Model name (default from config)")
@click.option("--endpoint", default="", help="API endpoint URL")
@click.option("--max-iterations", default=50, type=int)
@click.option("--working-dir", default=".", help="Working directory for agent")
def run(task, profile, mode, backend, model, endpoint, max_iterations, working_dir):
    """Run the agent on a single TASK."""
    stack = _build_stack(backend, model, endpoint, max_iterations,
                         working_dir, mode, profile)
    cfg = stack["config"]
    agent = stack["agent"]
    console.print(f"[bold]Agent starting:[/bold] {task[:80]}")
    label = f"profile={profile}" if profile else f"mode={cfg.mode}"
    console.print(f"  {label}  backend={backend or '(config)'}  "
                  f"model={model or '(auto)'}  max_iter={cfg.max_iterations}\n")
    result = agent.run(task)
    tag = "[bold green][OK]" if result.success else "[bold red][FAIL]"
    console.print(f"\n{tag} {result.iterations} iterations, "
                  f"{result.tokens} tokens")
    console.print(f"[bold]Summary:[/bold] {result.summary}")
    if result.steps:
        table = Table(title="Steps")
        table.add_column("#", style="dim", width=4)
        table.add_column("Action")
        for i, step in enumerate(result.steps, 1):
            table.add_row(str(i), str(step)[:120])
        console.print(table)


@agent_cmd.command("study")
@click.argument("topic")
@click.option("--backend", default="",
              help="LLM backend: openai, ollama, anthropic (default from config)")
@click.option("--model", default="", help="Model name (default from config)")
@click.option("--endpoint", default="", help="API endpoint URL")
def study(topic, backend, model, endpoint):
    """Queue a self-study goal on TOPIC."""
    from agent.goals import GoalQueue, StudyEngine
    stack = _build_stack(backend, model, endpoint)
    goals = GoalQueue(persist_path=_goals_path())
    engine = StudyEngine(agent=stack["agent"], goals=goals,
                         llm_backend=stack["backend"])
    console.print(f"[bold]Generating study goals for:[/bold] {topic}\n")
    engine.study(topic)
    queued = goals.list()
    if not queued:
        console.print("[WARN] No goals were queued.")
        return
    _goal_table(f"Queued Goals ({len(queued)})", queued)


@agent_cmd.command("goals")
@click.option("--status", default=None,
              help="Filter by status: pending, done, failed")
def goals(status):
    """List goals in the queue."""
    from agent.goals import GoalQueue
    queue = GoalQueue(persist_path=_goals_path())
    items = queue.list()
    if status:
        items = [g for g in items if getattr(g, "status", "pending") == status]
    if not items:
        console.print("No goals found." +
                      (f" (filter: status={status})" if status else ""))
        return
    _goal_table(f"Goal Queue ({len(items)})", items)


@agent_cmd.command("autopilot")
@click.option("--max-goals", default=10, type=int,
              help="Max goals to process before stopping")
@click.option("--backend", default="",
              help="LLM backend: openai, ollama, anthropic (default from config)")
@click.option("--model", default="", help="Model name (default from config)")
@click.option("--endpoint", default="", help="API endpoint URL")
def autopilot(max_goals, backend, model, endpoint):
    """Run the agent in autonomous study mode."""
    from agent.goals import GoalQueue, StudyEngine
    stack = _build_stack(backend, model, endpoint)
    goals_q = GoalQueue(persist_path=_goals_path())
    engine = StudyEngine(agent=stack["agent"], goals=goals_q,
                         llm_backend=stack["backend"])
    console.print(f"[bold]Autopilot:[/bold] processing up to "
                  f"{max_goals} goals\n")
    engine.run_loop(max_goals=max_goals)
    all_g = goals_q.list()
    done = sum(1 for g in all_g if getattr(g, "status", "") == "done")
    errs = sum(1 for g in all_g if getattr(g, "status", "") == "failed")
    pend = sum(1 for g in all_g if getattr(g, "status", "") == "pending")
    console.print(f"\n[bold]Autopilot finished[/bold]")
    console.print(f"  Completed: {done}  Failed: {errs}  Remaining: {pend}")


@agent_cmd.command("resume")
@click.argument("session_id")
@click.option("--followup", default="", help="Additional instruction to append")
@click.option("--backend", default="",
              help="LLM backend: openai, ollama, anthropic (default from config)")
@click.option("--model", default="", help="Model name (default from config)")
@click.option("--endpoint", default="", help="API endpoint URL")
def resume(session_id, followup, backend, model, endpoint):
    """Resume a previous agent session."""
    stack = _build_stack(backend, model, endpoint)
    agent = stack["agent"]
    if followup:
        # Inject follow-up into session history before resuming
        session_store = stack["session_store"]
        if session_store:
            data = session_store.load(session_id)
            data["history"].append({"role": "user", "content": followup})
            session_store.save(session_id, data["task"], data["history"],
                               status="active",
                               iterations=data.get("iterations", 0),
                               tokens=data.get("total_tokens", 0))
    console.print(f"[bold]Resuming session:[/bold] {session_id}")
    if followup:
        console.print(f"  Follow-up: {followup[:80]}")
    result = agent.resume(session_id)
    tag = "[bold green][OK]" if result.success else "[bold red][FAIL]"
    console.print(f"\n{tag} {result.iterations} iterations, "
                  f"{result.tokens} tokens")
    console.print(f"[bold]Summary:[/bold] {result.summary}")


@agent_cmd.command("sessions")
@click.option("--status", default=None, help="Filter: active, completed, failed")
@click.option("--search", default=None, help="Search task descriptions")
@click.option("--delete", default=None, help="Delete a session by ID")
@click.option("--cleanup-days", default=None, type=int,
              help="Delete sessions older than N days")
def sessions(status, search, delete, cleanup_days):
    """List, search, or manage agent sessions."""
    from agent.session import SessionStore
    store = SessionStore(store_dir=_session_dir())
    if delete:
        ok = store.delete(delete)
        console.print(f"[bold green][OK][/bold green] Deleted {delete}" if ok
                      else f"[bold red][FAIL][/bold red] Not found: {delete}")
        return
    if cleanup_days is not None:
        n = store.cleanup(max_age_days=cleanup_days)
        console.print(f"[OK] Cleaned up {n} sessions older than {cleanup_days} days")
        return
    if search:
        items = store.search(search)
    else:
        items = store.list_sessions(status=status)
    if not items:
        console.print("No sessions found.")
        return
    table = Table(title=f"Sessions ({len(items)})")
    table.add_column("ID", style="dim", width=14)
    table.add_column("Task", max_width=50)
    table.add_column("Status", width=10)
    table.add_column("Iters", width=5)
    table.add_column("Updated")
    for s in items:
        table.add_row(s.session_id[:12], s.task[:50],
                      s.status, str(s.iterations),
                      s.updated_at[:19] if s.updated_at else "")
    console.print(table)


@agent_cmd.command("complete")
@click.argument("file_path")
@click.option("--line", required=True, type=int, help="Line number for completion")
@click.option("--model-format", default="devstral", help="FIM format")
@click.option("--backend", default="ollama")
@click.option("--model", default="", help="Model name (default from config)")
@click.option("--endpoint", default="", help="API endpoint URL")
def complete(file_path, line, model_format, backend, model, endpoint):
    """Complete code at a specific line in a file."""
    from agent.prompts import FIMFormatter

    target = Path(file_path)
    if not target.is_file():
        console.print(f"[bold red][FAIL][/bold red] File not found: {file_path}")
        raise SystemExit(1)

    source = target.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)

    if line < 1 or line > len(lines) + 1:
        console.print(
            f"[bold red][FAIL][/bold red] Line {line} out of range "
            f"(file has {len(lines)} lines)")
        raise SystemExit(1)

    prefix = "".join(lines[:line - 1])
    suffix = "".join(lines[line - 1:])

    formatter = FIMFormatter(model_format)
    prompt = formatter.format_completion(prefix, suffix)

    stack = _build_stack(backend, model, endpoint)
    llm = stack["backend"]
    console.print(f"[bold]FIM completion:[/bold] {file_path}:{line}  "
                  f"format={model_format}")

    response = llm.chat(
        [{"role": "user", "content": prompt}],
        tools=[],
    )

    completion = formatter.extract_completion(response.content)
    console.print(f"\n[bold green]Completion:[/bold green]\n{completion}")


# -----------------------------------------------------------------------
# Wire into main CLI by adding to the bottom of cli/commands.py:
#
#   from cli.agent_cmd import agent_cmd      # noqa: E402
#   cli.add_command(agent_cmd)
# -----------------------------------------------------------------------
