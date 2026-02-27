"""Click CLI commands for JCoder."""

import json
import os
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from core.config import load_config, _ms_to_seconds, JCoderConfig
from core.embedding_engine import EmbeddingEngine
from core.eval_guard import save_hashes, verify_hashes
from core.index_engine import IndexEngine
from core.mock_backend import MockEmbedder, MockReranker, MockLLM
from core.reranker import Reranker
from core.retrieval_engine import RetrievalEngine
from core.runtime import Runtime
from core.orchestrator import Orchestrator
from ingestion.chunker import Chunker
from ingestion.repo_loader import RepoLoader

console = Console()


def _build_pipeline(config: JCoderConfig, mock: bool = False):
    """Wire up the full retrieval + generation pipeline from config."""
    p = config.policies
    dim = config.embedder.dimension or 768

    if mock:
        embedder = MockEmbedder(dimension=dim)
        reranker = MockReranker()
        runtime = MockLLM()
    else:
        embedder = EmbeddingEngine(config.embedder, _ms_to_seconds(p.timeout_embed_ms))
        reranker = Reranker(config.reranker, _ms_to_seconds(p.timeout_rerank_ms))
        runtime = Runtime(config.llm, _ms_to_seconds(p.timeout_generate_ms))

    index = IndexEngine(
        dim, config.storage, config.retrieval.rrf_k,
        gpu_safety_margin_mb=p.gpu_memory_safety_margin_mb,
    )
    retriever = RetrievalEngine(
        embedder, index, reranker,
        top_k=min(config.retrieval.top_k, p.max_chunks_retrieved),
        rerank_top_n=min(config.retrieval.rerank_top_n, p.max_rerank_n),
    )
    orchestrator = Orchestrator(retriever, runtime)
    return embedder, index, retriever, runtime, orchestrator


@click.group()
@click.option("--config-dir", default=None, help="Path to config directory")
@click.option("--mock", is_flag=True, default=False, help="Use mock backends (no vLLM needed)")
@click.pass_context
def cli(ctx, config_dir: Optional[str], mock: bool):
    """JCoder -- Local AI coding assistant."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_dir)
    ctx.obj["mock"] = mock


@cli.command()
@click.pass_context
def doctor(ctx):
    """Check environment readiness."""
    from cli.doctor import run_doctor
    run_doctor(ctx.obj["config"])


@cli.command()
@click.argument("path")
@click.option("--index-name", default="default", help="Name for the saved index")
@click.pass_context
def ingest(ctx, path: str, index_name: str):
    """Ingest a repository into the vector index."""
    config = ctx.obj["config"]

    with console.status("Chunking files..."):
        chunker = Chunker(max_chars=config.chunking.max_chars)
        loader = RepoLoader(chunker)
        chunks = loader.load(path)

    if not chunks:
        console.print("[bold red]No chunks produced. Check path and supported extensions.")
        return

    mock = ctx.obj.get("mock", False)
    dim = config.embedder.dimension or 768

    if mock:
        console.print("Using mock embedder (no vLLM)")
        embedder = MockEmbedder(dimension=dim)
    else:
        embedder = EmbeddingEngine(config.embedder, _ms_to_seconds(config.policies.timeout_embed_ms))

    console.print(f"Embedding {len(chunks)} chunks...")

    try:
        batch_size = config.policies.embed_batch_size
        all_vectors = []
        for i in range(0, len(chunks), batch_size):
            batch = [c["content"] for c in chunks[i:i + batch_size]]
            vectors = embedder.embed(batch)
            all_vectors.append(vectors)
            console.print(f"  Embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")

        import numpy as np
        all_vectors = np.concatenate(all_vectors, axis=0)
    finally:
        embedder.close()

    console.print("Building index...")
    p = config.policies
    index = IndexEngine(dim, config.storage, config.retrieval.rrf_k,
                        gpu_safety_margin_mb=p.gpu_memory_safety_margin_mb)
    index.add_vectors(all_vectors, chunks)
    index.save(index_name)

    console.print(f"[bold green][OK] Ingested {len(chunks)} chunks into index '{index_name}'")


@cli.command()
@click.argument("question")
@click.option("--index-name", default="default", help="Index to search")
@click.pass_context
def ask(ctx, question: str, index_name: str):
    """Ask a question about ingested code."""
    config = ctx.obj["config"]
    mock = ctx.obj.get("mock", False)
    embedder, index, retriever, runtime, orchestrator = _build_pipeline(config, mock=mock)

    try:
        index.load(index_name)
        console.print(f"Loaded index '{index_name}' ({index.count} chunks)")

        with console.status("Thinking..."):
            result = orchestrator.answer(question)

        console.print()
        console.print(result.answer)
        console.print()

        if result.sources:
            table = Table(title="Sources")
            table.add_column("File")
            for src in result.sources:
                table.add_row(src)
            console.print(table)
    finally:
        embedder.close()
        runtime.close()


def _score_retrieval(item: dict, result) -> dict:
    """Score A: Retrieval quality (LLM-independent).

    Checks source files and chunk excerpt text, NOT the generated answer.
    Returns dict with file_hit, symbol_hit, pass, failure_mode.
    """
    expected_file = item.get("expected_file_contains", "")
    expected_symbols = item.get("expected_symbols", [])
    expected_phrase = item.get("expected_phrase", "")
    min_rank = item.get("min_sources_hit_rank", 0)

    if not expected_file and not expected_symbols and not expected_phrase:
        return {"pass": True, "file_hit": True, "symbol_hit": True, "failure_mode": None}

    # File check: expected file in top-K sources (optionally rank-limited)
    sources = result.sources or []
    if min_rank > 0:
        sources = sources[:min_rank]
    norm_expected = expected_file.replace("\\", "/") if expected_file else ""
    file_hit = (not expected_file) or any(
        norm_expected in src.replace("\\", "/") for src in sources
    )

    # Symbol/phrase check: look in retrieved CHUNK TEXT, not LLM answer
    chunk_text = ""
    if result.chunks:
        chunk_text = " ".join(c.get("content", "") for c in result.chunks).lower()

    symbol_hit = True
    if expected_symbols:
        symbol_hit = any(sym.lower() in chunk_text for sym in expected_symbols)
    if expected_phrase:
        symbol_hit = symbol_hit and (expected_phrase.lower() in chunk_text)

    passed = file_hit and symbol_hit
    failure_mode = None
    if not passed:
        if not file_hit and not symbol_hit:
            failure_mode = "wrong_file+missing_symbol"
        elif not file_hit:
            failure_mode = "wrong_file"
        elif not symbol_hit:
            failure_mode = "right_file_missing_symbol"

    return {"pass": passed, "file_hit": file_hit, "symbol_hit": symbol_hit, "failure_mode": failure_mode}


def _score_answer(item: dict, result) -> dict:
    """Score B: Answer quality (LLM-dependent).

    Checks the generated answer text for expected symbols/keywords.
    Returns dict with pass, failure_mode.
    """
    expected_symbols = item.get("expected_symbols", [])
    expected_phrase = item.get("expected_phrase", "")
    expected_kw = item.get("expected_keyword", "")

    if not expected_symbols and not expected_phrase and not expected_kw:
        return {"pass": True, "failure_mode": None}

    answer_lower = result.answer.lower()

    symbol_hit = True
    if expected_symbols:
        symbol_hit = any(sym.lower() in answer_lower for sym in expected_symbols)
    if expected_phrase:
        symbol_hit = symbol_hit and (expected_phrase.lower() in answer_lower)
    if expected_kw:
        symbol_hit = symbol_hit and (expected_kw.lower() in answer_lower)

    failure_mode = None if symbol_hit else "answer_missing_symbol"
    return {"pass": symbol_hit, "failure_mode": failure_mode}


@cli.command(name="seal-benchmarks")
@click.option("--eval-dir", default=None, help="Evaluation directory")
@click.pass_context
def seal_benchmarks(ctx, eval_dir: Optional[str]):
    """Generate SHA-256 hashes for benchmark files (run after authoring)."""
    if eval_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        eval_dir = str(repo_root / "evaluation")
    save_hashes(eval_dir)
    console.print(f"[bold green][OK] Benchmark hashes written to config/benchmark_hashes.json")


@cli.command(name="eval")
@click.option("--benchmark", default=None, help="Benchmark JSON file")
@click.option("--index-name", default="default", help="Index to search")
@click.pass_context
def evaluate(ctx, benchmark: Optional[str], index_name: str):
    """Run evaluation benchmark with hash verification."""
    config = ctx.obj["config"]

    # Default benchmark path
    if benchmark is None:
        repo_root = Path(__file__).resolve().parent.parent
        benchmark = str(repo_root / "evaluation" / "benchmark_set.json")

    eval_dir = str(Path(benchmark).parent)

    # Hash verification -- refuse to run on tampered benchmarks
    # Manifest path from config, resolved relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    manifest = str(repo_root / config.policies.benchmark_hash_manifest)

    if config.policies.benchmark_hash_verify:
        error = verify_hashes(eval_dir, hash_path=manifest)
        if error:
            console.print(f"[bold red][FAIL] {error}")
            console.print("Benchmarks may have been tampered with. Aborting.")
            return

    mock = ctx.obj.get("mock", False)
    embedder, index, retriever, runtime, orchestrator = _build_pipeline(config, mock=mock)

    try:
        index.load(index_name)

        with open(benchmark, "r", encoding="utf-8") as f:
            data = json.load(f)

        console.print(f"Running {len(data)} benchmark questions...")

        retrieval_passes = 0
        answer_passes = 0
        failure_modes = {}
        start = time.time()

        for item in data:
            q = item["question"]
            result = orchestrator.answer(q)

            r_score = _score_retrieval(item, result)
            a_score = _score_answer(item, result)

            if r_score["pass"]:
                retrieval_passes += 1
            if a_score["pass"]:
                answer_passes += 1

            # Track failure modes
            for fm in [r_score["failure_mode"], a_score["failure_mode"]]:
                if fm:
                    failure_modes[fm] = failure_modes.get(fm, 0) + 1

            r_tag = "[green]R:P[/green]" if r_score["pass"] else "[red]R:F[/red]"
            a_tag = "[green]A:P[/green]" if a_score["pass"] else "[yellow]A:F[/yellow]"
            console.print(f"  {item.get('id', '?')}: {r_tag} {a_tag} -- {q[:55]}")

        elapsed = time.time() - start
        n = len(data)

        console.print()
        console.print(f"[bold]RetrievalScore: {retrieval_passes}/{n} ({100 * retrieval_passes / n:.1f}%)")
        console.print(f"[bold]AnswerScore:    {answer_passes}/{n} ({100 * answer_passes / n:.1f}%)")
        console.print(f"Time: {elapsed:.1f}s ({elapsed / n:.1f}s/question)")

        if failure_modes:
            console.print()
            console.print("[bold]Failure modes:")
            for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1])[:5]:
                console.print(f"  {mode}: {count}")
    finally:
        embedder.close()
        runtime.close()
