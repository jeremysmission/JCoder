"""CLI command: evolve -- self-improvement loop (config/prompt tuning only)."""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from core.ledger import ExperimentLedger
from core.evolver import Evolver

console = Console()


@click.command(name="evolve")
@click.option("--benchmark", default=None, help="Benchmark JSON file")
@click.option("--index-name", default="default", help="Index to search")
@click.option("--iters", default=12, type=int, help="Max evolution iterations")
@click.option("--cands", default=6, type=int, help="Candidates per iteration")
@click.option("--max-regress", default=5.0, type=float, help="Max regression % before stop")
@click.option("--out-dir", default="_runs", help="Output directory for run artifacts")
@click.pass_context
def evolve_cmd(ctx, benchmark: Optional[str], index_name: str, iters: int,
               cands: int, max_regress: float, out_dir: str):
    """
    Run self-improvement loop (safe: config/prompt tuning only).

    Mutates retrieval and chunking config knobs, evaluates each candidate
    against the benchmark, and picks the best configuration. All runs are
    logged to an SQLite ledger for audit.
    """
    from cli.commands import _build_pipeline
    config = ctx.obj["config"]
    mock = ctx.obj.get("mock", False)

    if benchmark is None:
        repo_root = Path(__file__).resolve().parent.parent
        benchmark = str(repo_root / "evaluation" / "benchmark_set.json")

    seed = getattr(config, "evolver_seed", 1337)
    ledger = ExperimentLedger(db_path=str(Path(out_dir) / "ledger.sqlite"))
    ev = Evolver(ledger=ledger, out_dir=out_dir, seed=seed)

    console.print(f"[bold]Evolver: {iters} iters x {cands} candidates")
    console.print(f"Benchmark: {benchmark}")
    console.print(f"Index: {index_name}")
    console.print(f"Mock: {mock}")
    console.print()

    summary = ev.run(
        config=config,
        build_pipeline_fn=_build_pipeline,
        benchmark_path=benchmark,
        index_name=index_name,
        max_iters=iters,
        candidates_per_iter=cands,
        max_regression_pct=max_regress,
        mock=mock,
        label="evolver",
    )

    console.print()
    console.print("[bold green][OK] Evolver run complete.")
    console.print(f"Run ID:          {summary['run_id']}")
    console.print(f"Iterations:      {summary['iterations_completed']}")
    baseline = summary["baseline"]
    best = summary["best"]
    console.print(f"Baseline R/A:    {baseline['retrieval_score_pct']:.1f}% / {baseline['answer_score_pct']:.1f}%")
    console.print(f"Best     R/A:    {best['retrieval_score_pct']:.1f}% / {best['answer_score_pct']:.1f}%")
    console.print(f"Best config:     {summary['best_config']}")
    console.print(f"Artifacts:       {Path(out_dir) / summary['run_id']}")
