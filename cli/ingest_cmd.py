"""Click CLI commands for corpus ingestion pipeline."""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _make_pipeline(fts5_only: bool, batch_size: int, config,
                   dedup_enabled: bool = True, dedup_threshold: float = 0.8,
                   index_name: str = ""):
    """Create CorpusPipeline with optional embedder and dedup.
    When --fts5-only is set, no embedding server is needed.
    """
    import os

    from core.config import _ms_to_seconds
    from core.embedding_engine import EmbeddingEngine
    from core.network_gate import NetworkGate
    from ingestion.chunker import Chunker
    from ingestion.corpus_pipeline import CorpusPipeline

    embedder = None
    if not fts5_only:
        gate = NetworkGate(mode=config.network.mode, allowlist=config.network.allowlist)
        embedder = EmbeddingEngine(
            config.embedder, _ms_to_seconds(config.policies.timeout_embed_ms), gate=gate)

    dedup = None
    if dedup_enabled:
        from ingestion.dedup import MinHashDedup
        persist_path = os.path.join(
            config.storage.data_dir, "checkpoints",
            f"{index_name}_dedup.json") if index_name else ""
        dedup = MinHashDedup(threshold=dedup_threshold, persist_path=persist_path)

    return CorpusPipeline(
        embedding_engine=embedder, storage_config=config.storage,
        chunker=Chunker(max_chars=config.chunking.max_chars),
        batch_size=batch_size, dimension=config.embedder.dimension or 768,
        progress_callback=lambda total, msg: console.print(f"  {msg}"),
        dedup=dedup,
    ), dedup


def _print_stats(stats, dedup=None):
    """Print final ingestion stats as a Rich table."""
    rate = stats.files_processed / stats.elapsed_s if stats.elapsed_s > 0 else 0
    table = Table(title="Ingestion Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for label, val in [("Source", stats.source),
                       ("Files processed", stats.files_processed),
                       ("Files skipped", stats.files_skipped),
                       ("Chunks created", stats.chunks_created),
                       ("Chunks embedded", stats.chunks_embedded),
                       ("Errors", len(stats.errors)),
                       ("Elapsed", f"{stats.elapsed_s:.1f}s"),
                       ("Throughput", f"{rate:.0f} files/s")]:
        table.add_row(label, str(val))
    if dedup:
        ds = dedup.stats()
        table.add_row("Dedup: exact dupes", str(ds.exact_dupes))
        table.add_row("Dedup: near dupes", str(ds.near_dupes))
        table.add_row("Dedup: unique chunks", str(ds.unique))
    console.print(table)
    if stats.errors:
        console.print(f"\n[bold yellow][WARN] {len(stats.errors)} errors:")
        for err in stats.errors[:10]:
            console.print(f"  {err}")
        if len(stats.errors) > 10:
            console.print(f"  ... and {len(stats.errors) - 10} more")


def _parse_languages(languages: str) -> Optional[List[str]]:
    """Split comma-separated language string, or None if empty."""
    parts = [l.strip() for l in languages.split(",") if l.strip()] if languages else []
    return parts or None


@click.group("ingest-corpus")
@click.pass_context
def ingest_corpus_cmd(ctx):
    """Corpus ingestion commands."""


@ingest_corpus_cmd.command()
@click.argument("source_dir")
@click.option("--index-name", default="stackoverflow")
@click.option("--max-files", default=0, help="0 = all files")
@click.option("--resume/--no-resume", default=True)
@click.option("--fts5-only", is_flag=True, help="Skip embeddings, build FTS5 index only")
@click.option("--batch-size", default=64)
@click.option("--dedup/--no-dedup", default=True, help="Enable near-duplicate detection")
@click.option("--dedup-threshold", default=0.8, help="Jaccard similarity threshold")
@click.pass_context
def stackoverflow(ctx, source_dir, index_name, max_files, resume, fts5_only, batch_size,
                  dedup, dedup_threshold):
    """Ingest Stack Overflow markdown files."""
    config = ctx.obj["config"]
    mode = "FTS5-only" if fts5_only else "hybrid (FAISS + FTS5)"
    console.print(f"[bold]Ingesting Stack Overflow:[/bold] {source_dir}")
    console.print(f"  index={index_name}  mode={mode}  batch={batch_size}  "
                  f"resume={resume}  max_files={max_files or 'all'}  dedup={dedup}\n")

    pipeline, dedup_obj = _make_pipeline(
        fts5_only, batch_size, config,
        dedup_enabled=dedup, dedup_threshold=dedup_threshold,
        index_name=index_name)
    with console.status("Ingesting Stack Overflow files..."):
        stats = pipeline.ingest_stackoverflow(
            source_dir, index_name=index_name,
            max_files=max_files, resume=resume,
        )
    _print_stats(stats, dedup=dedup_obj)


@ingest_corpus_cmd.command()
@click.argument("source_dir")
@click.option("--index-name", default="codesearchnet")
@click.option("--languages", default="", help="Comma-separated language filter")
@click.option("--max-files", default=0, help="0 = all files")
@click.option("--resume/--no-resume", default=True)
@click.option("--fts5-only", is_flag=True, help="Skip embeddings, build FTS5 index only")
@click.option("--batch-size", default=64)
@click.option("--dedup/--no-dedup", default=True, help="Enable near-duplicate detection")
@click.option("--dedup-threshold", default=0.8, help="Jaccard similarity threshold")
@click.pass_context
def codesearchnet(ctx, source_dir, index_name, languages, max_files, resume,
                  fts5_only, batch_size, dedup, dedup_threshold):
    """Ingest CodeSearchNet files."""
    config = ctx.obj["config"]
    lang_list = _parse_languages(languages)
    mode = "FTS5-only" if fts5_only else "hybrid (FAISS + FTS5)"
    console.print(f"[bold]Ingesting CodeSearchNet:[/bold] {source_dir}")
    console.print(f"  index={index_name}  mode={mode}  batch={batch_size}  "
                  f"resume={resume}  max_files={max_files or 'all'}  "
                  f"languages={lang_list or 'all'}  dedup={dedup}\n")

    pipeline, dedup_obj = _make_pipeline(
        fts5_only, batch_size, config,
        dedup_enabled=dedup, dedup_threshold=dedup_threshold,
        index_name=index_name)
    with console.status("Ingesting CodeSearchNet files..."):
        stats = pipeline.ingest_codesearchnet(
            source_dir, index_name=index_name,
            languages=lang_list,
            max_files=max_files, resume=resume,
        )
    _print_stats(stats, dedup=dedup_obj)


@ingest_corpus_cmd.command()
@click.argument("source_dir")
@click.option("--index-name", default="docs")
@click.option("--max-files", default=0, help="0 = all files")
@click.option("--fts5-only", is_flag=True, help="Skip embeddings, build FTS5 index only")
@click.option("--batch-size", default=64)
@click.option("--dedup/--no-dedup", default=True, help="Enable near-duplicate detection")
@click.option("--dedup-threshold", default=0.8, help="Jaccard similarity threshold")
@click.pass_context
def docs(ctx, source_dir, index_name, max_files, fts5_only, batch_size,
         dedup, dedup_threshold):
    """Ingest documentation markdown files."""
    config = ctx.obj["config"]
    mode = "FTS5-only" if fts5_only else "hybrid (FAISS + FTS5)"
    console.print(f"[bold]Ingesting documentation:[/bold] {source_dir}")
    console.print(f"  index={index_name}  mode={mode}  batch={batch_size}  "
                  f"max_files={max_files or 'all'}  dedup={dedup}\n")

    pipeline, dedup_obj = _make_pipeline(
        fts5_only, batch_size, config,
        dedup_enabled=dedup, dedup_threshold=dedup_threshold,
        index_name=index_name)
    with console.status("Ingesting documentation files..."):
        stats = pipeline.ingest_markdown_docs(
            source_dir, index_name=index_name,
            max_files=max_files, resume=True,
        )
    _print_stats(stats, dedup=dedup_obj)


@ingest_corpus_cmd.command()
@click.argument("source_dir")
@click.option("--index-name", default="code")
@click.option("--languages", default="", help="Comma-separated: python,javascript,go")
@click.option("--max-files", default=0, help="0 = all files")
@click.option("--fts5-only", is_flag=True, help="Skip embeddings, build FTS5 index only")
@click.option("--batch-size", default=64)
@click.option("--dedup/--no-dedup", default=True, help="Enable near-duplicate detection")
@click.option("--dedup-threshold", default=0.8, help="Jaccard similarity threshold")
@click.pass_context
def code(ctx, source_dir, index_name, languages, max_files, fts5_only, batch_size,
         dedup, dedup_threshold):
    """Ingest raw source code files with AST chunking."""
    config = ctx.obj["config"]
    lang_list = _parse_languages(languages)
    mode = "FTS5-only" if fts5_only else "hybrid (FAISS + FTS5)"
    console.print(f"[bold]Ingesting source code:[/bold] {source_dir}")
    console.print(f"  index={index_name}  mode={mode}  batch={batch_size}  "
                  f"max_files={max_files or 'all'}  "
                  f"languages={lang_list or 'all'}  dedup={dedup}\n")

    pipeline, dedup_obj = _make_pipeline(
        fts5_only, batch_size, config,
        dedup_enabled=dedup, dedup_threshold=dedup_threshold,
        index_name=index_name)
    with console.status("Ingesting source code files..."):
        stats = pipeline.ingest_code_files(
            source_dir, index_name=index_name,
            languages=lang_list,
            max_files=max_files, resume=True,
        )
    _print_stats(stats, dedup=dedup_obj)


@ingest_corpus_cmd.command()
@click.pass_context
def status(ctx):
    """Show ingestion status: existing indexes, checkpoint state, disk usage."""
    config = ctx.obj["config"]
    index_dir = Path(config.storage.index_dir)
    data_dir = Path(config.storage.data_dir)
    checkpoint_dir = data_dir / "checkpoints"

    # -- Indexes --
    console.print("[bold]Indexes:[/bold]")
    if index_dir.exists() and any(index_dir.iterdir()):
        table = Table()
        table.add_column("Name", style="bold")
        table.add_column("Type")
        table.add_column("Size", justify="right")
        seen: set = set()
        for entry in sorted(index_dir.iterdir()):
            stem = entry.stem.split(".")[0]
            if stem in seen:
                continue
            seen.add(stem)
            size_bytes = sum(
                f.stat().st_size for f in index_dir.glob(f"{stem}*") if f.is_file()
            )
            files = [f.suffix for f in index_dir.glob(f"{stem}*") if f.is_file()]
            idx_type = "hybrid" if ".faiss" in str(files) else "FTS5"
            table.add_row(stem, idx_type, f"{size_bytes / (1024 * 1024):.1f} MB")
        console.print(table)
    else:
        console.print("  (none)")

    # -- Checkpoints --
    console.print("\n[bold]Checkpoints:[/bold]")
    cp_files = sorted(checkpoint_dir.glob("*_checkpoint.json")) if checkpoint_dir.exists() else []
    if cp_files:
        table = Table()
        table.add_column("Index", style="bold")
        table.add_column("Files tracked", justify="right")
        table.add_column("Updated")
        for cp in cp_files:
            name = cp.stem.replace("_checkpoint", "")
            try:
                import json
                with open(cp, "r", encoding="utf-8") as f:
                    count = len(json.load(f))
            except Exception:
                count = -1
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(cp.stat().st_mtime))
            table.add_row(name, str(count), mtime)
        console.print(table)
    else:
        console.print("  (none)")

    # -- Disk usage --
    console.print("\n[bold]Disk usage:[/bold]")
    for label, dpath in [("Data", data_dir), ("Indexes", index_dir)]:
        if dpath.exists():
            total = sum(f.stat().st_size for f in dpath.rglob("*") if f.is_file())
            console.print(f"  {label}: {total / (1024 * 1024):.1f} MB ({dpath})")
        else:
            console.print(f"  {label}: (not found)")

# In cli/commands.py, add:
#   from cli.ingest_cmd import ingest_corpus_cmd  # noqa: E402
#   cli.add_command(ingest_corpus_cmd)
