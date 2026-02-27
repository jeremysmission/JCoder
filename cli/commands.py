"""Click CLI commands for JCoder."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

import subprocess

from core.config import load_config, _ms_to_seconds, JCoderConfig
from core.embedding_engine import EmbeddingEngine
from core.eval_guard import save_hashes, verify_hashes
from core.index_engine import IndexEngine
from core.mock_backend import MockEmbedder, MockReranker, MockLLM
from core.network_gate import NetworkGate
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
    gate = NetworkGate(mode="localhost")

    if mock:
        embedder = MockEmbedder(dimension=dim)
        reranker = MockReranker()
        runtime = MockLLM()
    else:
        embedder = EmbeddingEngine(config.embedder, _ms_to_seconds(p.timeout_embed_ms), gate=gate)
        reranker = Reranker(config.reranker, _ms_to_seconds(p.timeout_rerank_ms), gate=gate)
        runtime = Runtime(config.llm, _ms_to_seconds(p.timeout_generate_ms), gate=gate)

    index = IndexEngine(
        dim, config.storage, config.retrieval.rrf_k,
        gpu_safety_margin_mb=p.gpu_memory_safety_margin_mb,
        sparse_only=mock,
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
@click.option("--diagnose-retrieval", is_flag=True, default=False,
              help="Print detailed source ranking for retrieval failures")
@click.pass_context
def evaluate(ctx, benchmark: Optional[str], index_name: str, diagnose_retrieval: bool):
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

            if diagnose_retrieval and not r_score["pass"]:
                from core.index_engine import IndexEngine, _normalize_for_search
                exp = item.get("expected_file_contains", "")
                console.print(f"    expected: {exp}")
                sources = result.sources or []
                for rank, src in enumerate(sources[:5]):
                    norm = src.replace("\\", "/")
                    boost = IndexEngine._path_prior_boost(q, src)
                    console.print(f"    [{rank}] {norm}  boost={boost:.2f}")

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


def _probe_endpoint(url: str, timeout: float = 5.0) -> tuple:
    """Probe a vLLM/OpenAI endpoint.

    Tries GET {base}/v1/models first, falls back to GET {base}/models.
    Returns (ok: bool, latency_ms: float | None, path_used: str | None).
    """
    import httpx
    base = url.rstrip("/")
    # Strip trailing /v1 so we can try both paths cleanly
    if base.endswith("/v1"):
        base = base[:-3]
    paths = ["/v1/models", "/models"]
    for path in paths:
        try:
            t0 = time.time()
            with httpx.Client(timeout=timeout) as client:
                r = client.get(f"{base}{path}")
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            if r.status_code == 200:
                return (True, elapsed_ms, path)
        except Exception:
            continue
    return (False, None, None)


def _safe_elapsed(t0: float) -> float:
    """Return elapsed seconds since t0, minimum 1e-6 to prevent ZeroDivisionError."""
    return max(time.time() - t0, 1e-6)


def _bench_tok_per_s(llm_endpoint: str, timeout: float) -> float:
    """Single chat completion, return tokens/sec or raise."""
    import httpx
    base = llm_endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Return exactly 200 tokens of plain ASCII text about indexing."}],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    t0 = time.time()
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{base}/chat/completions", json=payload)
    elapsed = _safe_elapsed(t0)
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens and completion_tokens > 0:
        return round(completion_tokens / elapsed, 2)
    # Fallback: estimate from response text length divided by 4 chars per token
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    estimated = max(1, len(text) // 4)
    return round(estimated / elapsed, 2)


def _bench_p95_latency(llm_endpoint: str, timeout: float) -> float:
    """Fire 4 parallel chat completions, return p95 latency in ms."""
    import httpx
    base = llm_endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
        "temperature": 0.0,
    }

    def _one_request():
        t0 = time.time()
        with httpx.Client(timeout=timeout) as client:
            r = client.post(f"{base}/chat/completions", json=payload)
        r.raise_for_status()
        return max((time.time() - t0) * 1000, 0.01)

    latencies = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_one_request) for _ in range(4)]
        for f in as_completed(futures):
            latencies.append(f.result())
    latencies.sort()
    # p95 of 4 samples = max (index 3)
    return round(latencies[-1], 2)


def _bench_embed_throughput(embed_endpoint: str, batch_size: int, timeout: float) -> float:
    """Embed batch_size short strings, return items/sec."""
    import httpx
    base = embed_endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    inputs = [f"sample text number {i}" for i in range(batch_size)]
    payload = {"model": "default", "input": inputs}
    t0 = time.time()
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{base}/embeddings", json=payload)
    elapsed = _safe_elapsed(t0)
    r.raise_for_status()
    return round(batch_size / elapsed, 2)


def _bench_rerank_throughput(rerank_endpoint: str, n: int, timeout: float) -> float:
    """Rerank n document pairs, return pairs/sec."""
    import httpx
    base = rerank_endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    payload = {
        "model": "default",
        "query": "What does this code do?",
        "documents": [f"Document content number {i}" for i in range(n)],
    }
    t0 = time.time()
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{base}/score", json=payload)
    elapsed = _safe_elapsed(t0)
    r.raise_for_status()
    return round(n / elapsed, 2)


def _run_bench_probes(config, performance: dict, allow_wait: bool) -> None:
    """Fill performance dict in-place. Each probe is best-effort; failures stay null."""
    timeout = _ms_to_seconds(config.policies.timeout_generate_ms)

    try:
        performance["tok_per_s_single"] = _bench_tok_per_s(config.llm.endpoint, timeout)
    except Exception:
        pass

    try:
        performance["p95_latency_ms_4_parallel"] = _bench_p95_latency(config.llm.endpoint, timeout)
    except Exception:
        pass

    try:
        performance["embed_items_per_s_batch16"] = _bench_embed_throughput(config.embedder.endpoint, 16, timeout)
    except Exception:
        pass

    try:
        performance["embed_items_per_s_batch32"] = _bench_embed_throughput(config.embedder.endpoint, 32, timeout)
    except Exception:
        pass

    try:
        performance["rerank_pairs_per_s_n50"] = _bench_rerank_throughput(config.reranker.endpoint, 50, timeout)
    except Exception:
        pass

    if allow_wait and performance["tok_per_s_single"] is not None:
        first_tok_s = performance["tok_per_s_single"]
        console.print("[WARN] Sleeping 10 minutes for drift measurement...")
        time.sleep(600)
        try:
            second_tok_s = _bench_tok_per_s(config.llm.endpoint, timeout)
            performance["drift_tok_per_s_delta_10min"] = round(second_tok_s - first_tok_s, 2)
        except Exception:
            pass


@cli.command()
@click.option("--run-bench", is_flag=True, default=False, help="Run performance benchmarks against live endpoints")
@click.option("--allow-wait", is_flag=True, default=False, help="Allow 10-minute sleep for drift measurement")
@click.pass_context
def measure(ctx, run_bench: bool, allow_wait: bool):
    """Measure GPU environment and endpoint readiness."""
    config = ctx.obj["config"]

    # -- system --
    system = {
        "torch_version": "",
        "torch_cuda_compiled": "",
        "cuda_available": False,
        "cuda_device_count": 0,
        "gpus": [],
        "driver_version": "",
        "cuda_runtime_version": "",
    }

    try:
        import torch
        system["torch_version"] = torch.__version__
        system["torch_cuda_compiled"] = torch.version.cuda or ""
        system["cuda_available"] = torch.cuda.is_available()
        system["cuda_device_count"] = torch.cuda.device_count()
        if system["cuda_available"] and system["cuda_device_count"] > 0:
            system["cuda_runtime_version"] = str(torch.version.cuda or "")
    except ImportError:
        console.print("[WARN] torch not installed")

    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    system["gpus"].append({
                        "name": parts[0],
                        "total_mb": int(parts[2]),
                        "free_mb": int(parts[3]),
                    })
                    if not system["driver_version"]:
                        system["driver_version"] = parts[1]
    except FileNotFoundError:
        console.print("[WARN] nvidia-smi not found")
    except Exception:
        pass

    # -- endpoints --
    llm_ok, llm_ms, llm_path = _probe_endpoint(config.llm.endpoint)
    embed_ok, embed_ms, embed_path = _probe_endpoint(config.embedder.endpoint)
    rerank_ok, rerank_ms, rerank_path = _probe_endpoint(config.reranker.endpoint)

    endpoints = {
        "llm_models_ok": llm_ok,
        "embed_models_ok": embed_ok,
        "rerank_models_ok": rerank_ok,
        "llm_response_ms": llm_ms,
        "embed_response_ms": embed_ms,
        "rerank_response_ms": rerank_ms,
    }

    # -- performance --
    performance = {
        "tok_per_s_single": None,
        "p95_latency_ms_4_parallel": None,
        "embed_items_per_s_batch16": None,
        "embed_items_per_s_batch32": None,
        "rerank_pairs_per_s_n50": None,
        "drift_tok_per_s_delta_10min": None,
    }

    if run_bench:
        all_up = llm_ok and embed_ok and rerank_ok
        if all_up:
            console.print("Running performance benchmarks...")
            _run_bench_probes(config, performance, allow_wait)
        else:
            console.print("[WARN] --run-bench: not all endpoints reachable, skipping perf probes")

    measurements = {
        "system": system,
        "endpoints": endpoints,
        "performance": performance,
    }

    # -- pretty table --
    table = Table(title="JCoder Measurements")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("torch", system["torch_version"] or "not installed")
    table.add_row("torch_cuda_compiled", system["torch_cuda_compiled"] or "n/a")
    table.add_row("cuda_available", str(system["cuda_available"]))
    table.add_row("device_count", str(system["cuda_device_count"]))
    table.add_row("driver_version", system["driver_version"] or "n/a")
    table.add_row("cuda_runtime", system["cuda_runtime_version"] or "n/a")

    for i, g in enumerate(system["gpus"]):
        table.add_row(f"GPU {i}", f"{g['name']} ({g['total_mb']} MB total, {g['free_mb']} MB free)")
    if not system["gpus"]:
        table.add_row("GPUs", "None detected")

    table.add_row("", "")
    table.add_row("llm_models_ok", f"{endpoints['llm_models_ok']} (via {llm_path})" if llm_path else str(endpoints["llm_models_ok"]))
    table.add_row("embed_models_ok", f"{endpoints['embed_models_ok']} (via {embed_path})" if embed_path else str(endpoints["embed_models_ok"]))
    table.add_row("rerank_models_ok", f"{endpoints['rerank_models_ok']} (via {rerank_path})" if rerank_path else str(endpoints["rerank_models_ok"]))
    table.add_row("llm_response_ms", str(endpoints["llm_response_ms"]) if endpoints["llm_response_ms"] else "n/a")
    table.add_row("embed_response_ms", str(endpoints["embed_response_ms"]) if endpoints["embed_response_ms"] else "n/a")
    table.add_row("rerank_response_ms", str(endpoints["rerank_response_ms"]) if endpoints["rerank_response_ms"] else "n/a")

    table.add_row("", "")
    for k, v in performance.items():
        table.add_row(k, str(v) if v is not None else "pending")

    console.print(table)

    # -- write JSON --
    metrics_dir = Path(config.storage.data_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "measurements.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(measurements, f, indent=2)
    console.print(f"[bold green][OK] Written to {out_path}")
