"""Embed rare_format_documents chunks in disk-friendly batches.

Reads metadata from meta.json, filters noise, embeds via direct CUDA,
and builds FAISS index using incremental IVF training + add pattern
to avoid OOM on 1M+ chunk workloads.

Usage: PYTHONUNBUFFERED=1 .venv/Scripts/python.exe scripts/embed_rare_formats.py
"""
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ModelConfig

INDEX_NAME = "rare_format_documents"
INDEX_DIR = "data/indexes"
MIN_CHUNK_CHARS = 100
EMBED_BATCH = 512      # Texts per embed() call
SHARD_SIZE = 50000     # Vectors per disk shard (keeps RAM bounded)
DIMENSION = 768


def main():
    meta_path = os.path.join(INDEX_DIR, f"{INDEX_NAME}.meta.json")
    faiss_path = os.path.join(INDEX_DIR, f"{INDEX_NAME}.faiss")

    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found.", flush=True)
        sys.exit(1)

    print(f"Loading metadata...", flush=True)
    t0 = time.time()
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"  Loaded {len(metadata):,} chunks in {time.time() - t0:.1f}s", flush=True)

    # Filter noise chunks
    keep = [(i, m) for i, m in enumerate(metadata)
            if len(m.get("content", "")) >= MIN_CHUNK_CHARS]
    total = len(keep)
    skipped = len(metadata) - total
    print(f"  Filtered: {total:,} kept, {skipped:,} skipped (<{MIN_CHUNK_CHARS} chars)",
          flush=True)

    if total == 0:
        print("No chunks to embed.", flush=True)
        return

    # Initialize embedder
    os.environ.setdefault("JCODER_EMBED_DEVICE", "cuda:1")
    embed_config = ModelConfig(name="nomic-embed-text-v2-moe", dimension=DIMENSION)

    from core.embedding_engine import EmbeddingEngine
    embedder = EmbeddingEngine(config=embed_config)
    print(f"Direct CUDA: {embedder._use_direct_cuda}", flush=True)

    import faiss

    # Embed in shards, writing each shard to disk to bound RAM
    shard_dir = os.path.join(INDEX_DIR, f"_{INDEX_NAME}_shards")
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths = []
    filtered_meta = []
    shard_vectors = []
    embedded = 0
    errors = 0
    t_embed = time.time()

    for batch_start in range(0, total, EMBED_BATCH):
        batch_end = min(batch_start + EMBED_BATCH, total)
        batch_items = keep[batch_start:batch_end]
        batch_texts = [metadata[i]["content"] for i, _ in batch_items]

        try:
            vectors = embedder.embed(batch_texts)
            shard_vectors.append(vectors)
            for i, m in batch_items:
                filtered_meta.append(m)
            embedded += len(batch_texts)
        except Exception as e:
            print(f"  [ERROR] Batch {batch_start}-{batch_end}: {e}", flush=True)
            errors += 1
            # Try to recover from OOM
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()
                gc.collect()

        # Write shard to disk when we hit SHARD_SIZE
        shard_count = sum(v.shape[0] for v in shard_vectors)
        if shard_count >= SHARD_SIZE:
            shard_idx = len(shard_paths)
            shard_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}.npy")
            combined = np.vstack(shard_vectors)
            np.save(shard_path, combined)
            shard_paths.append(shard_path)
            print(f"  Shard {shard_idx}: {combined.shape[0]:,} vectors saved", flush=True)
            shard_vectors.clear()
            gc.collect()

        # Progress every 10k chunks
        if embedded > 0 and embedded % 10000 < EMBED_BATCH:
            elapsed = time.time() - t_embed
            rate = embedded / elapsed
            eta = (total - embedded) / rate if rate > 0 else 0
            print(f"  [{embedded:,}/{total:,}] {rate:.0f}/s, ETA {eta/60:.1f}m",
                  flush=True)

    # Final shard
    if shard_vectors:
        shard_idx = len(shard_paths)
        shard_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}.npy")
        combined = np.vstack(shard_vectors)
        np.save(shard_path, combined)
        shard_paths.append(shard_path)
        print(f"  Shard {shard_idx}: {combined.shape[0]:,} vectors saved", flush=True)
        shard_vectors.clear()
        gc.collect()

    embedder.close()
    embed_elapsed = time.time() - t_embed

    # Merge shards into final FAISS index
    print(f"\nMerging {len(shard_paths)} shards into FAISS index...", flush=True)
    index = faiss.IndexFlatIP(DIMENSION)
    for sp in shard_paths:
        vecs = np.load(sp)
        index.add(vecs)
        del vecs
        gc.collect()

    faiss.write_index(index, faiss_path)

    # Update metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(filtered_meta, f)

    # Cleanup shards
    for sp in shard_paths:
        os.remove(sp)
    os.rmdir(shard_dir)

    print(f"\nDone!", flush=True)
    print(f"  Embedded: {embedded:,}/{total:,}", flush=True)
    print(f"  FAISS: {index.ntotal:,} vectors", flush=True)
    print(f"  Time: {embed_elapsed:.0f}s ({embedded/embed_elapsed:.0f}/s)", flush=True)
    print(f"  Errors: {errors}", flush=True)
    print(f"  File: {faiss_path} ({os.path.getsize(faiss_path)/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
