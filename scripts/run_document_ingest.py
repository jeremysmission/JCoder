"""Run ingest_documents() on all rare-format downloads.

Ingests EPUBs, XLS/XLSX, SVGs, .dia, .drawio, .rst files using
the parser registry and HybridRAG-compatible chunking (1200/200/768).
"""
import os
import sys
import time

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import StorageConfig, ModelConfig
from core.embedding_engine import EmbeddingEngine
from ingestion.corpus_pipeline import CorpusPipeline


def main():
    storage = StorageConfig()
    print(f"Data dir: {storage.data_dir}")

    # Initialize embedding engine with nomic-embed-text-v2-moe on CUDA:1
    embed_config = ModelConfig(
        name="nomic-embed-text-v2-moe",
        dimension=768,
    )
    embedder = EmbeddingEngine(config=embed_config)

    pipeline = CorpusPipeline(
        embedding_engine=embedder,
        storage_config=storage,
        batch_size=64,
        dimension=768,
    )

    source_dir = os.path.join(storage.data_dir, "raw_downloads", "rare_formats")
    if not os.path.isdir(source_dir):
        print(f"ERROR: Source dir not found: {source_dir}")
        sys.exit(1)

    print(f"Source: {source_dir}")
    print("Starting document ingest (HybridRAG-compatible: 1200/200/768)...")
    t0 = time.time()

    stats = pipeline.ingest_documents(
        source_dir=source_dir,
        index_name="rare_format_documents",
        hybridrag_compat=True,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Files processed: {stats.files_processed}")
    print(f"  Chunks created:  {stats.chunks_created}")
    print(f"  Chunks embedded: {stats.chunks_embedded}")
    if stats.errors:
        print(f"  Errors ({len(stats.errors)}):")
        for e in stats.errors[:10]:
            print(f"    - {e}")


if __name__ == "__main__":
    main()
