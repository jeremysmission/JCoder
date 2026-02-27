"""
Self-Benchmark Generator (SimRAG Pattern)
------------------------------------------
Automatically generates QA pairs from the ingested codebase,
then filters them via round-trip consistency. This creates a
continuously growing evaluation set that tracks the system's
actual knowledge, not just hand-written test cases.

Pipeline:
1. Sample chunks from the index
2. Ask the LLM to generate questions about each chunk
3. Round-trip consistency filter:
   - Run each generated question through retrieval
   - Keep only pairs where the original chunk appears in top-K results
   - This ensures the question is answerable by the retrieval system
4. Difficulty scoring:
   - Easy: original chunk is rank 1
   - Medium: original chunk is rank 2-5
   - Hard: original chunk is rank 6-K
5. Write filtered QA pairs to evaluation JSON

This is the SimRAG self-training pattern adapted for code retrieval.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.runtime import Runtime
from core.retrieval_engine import RetrievalEngine


@dataclass
class SyntheticQA:
    """A self-generated question-answer pair with metadata."""
    qa_id: str
    question: str
    expected_file: str
    expected_chunk_id: str
    source_chunk_snippet: str
    difficulty: str  # "easy" | "medium" | "hard"
    retrieval_rank: int  # rank of original chunk in results
    round_trip_pass: bool
    generated_at: float = 0.0


_QA_GEN_PROMPT = (
    "You are a code QA generator. Given this code snippet, generate "
    "exactly 3 specific technical questions that can ONLY be answered "
    "by reading this code. Each question should require understanding "
    "the code's logic, not just surface-level details.\n\n"
    "Output format (one per line, no numbering):\n"
    "Q: <question>\n"
    "Q: <question>\n"
    "Q: <question>\n\n"
    "Code from {source_path}:\n```\n{code}\n```"
)


class SelfBenchGenerator:
    """
    Generates and validates QA benchmarks from the ingested codebase.
    Uses round-trip consistency filtering for quality control.
    """

    def __init__(
        self,
        runtime: Runtime,
        retriever: RetrievalEngine,
        out_dir: str = "_self_bench",
        seed: int = 42,
    ):
        self.runtime = runtime
        self.retriever = retriever
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)

    def generate(
        self,
        chunks: List[Dict],
        sample_size: int = 50,
        top_k_check: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate self-benchmark from sampled chunks.

        Args:
            chunks: All indexed chunks (list of metadata dicts)
            sample_size: How many chunks to sample for QA generation
            top_k_check: How many results to check for round-trip consistency

        Returns:
            Manifest dict with counts and file paths.
        """
        if not chunks:
            return {"error": "No chunks provided", "qa_count": 0}

        # Sample chunks (prefer longer, more interesting ones)
        eligible = [c for c in chunks
                    if len(c.get("content", "")) > 200
                    and c.get("source_path", "")]
        if len(eligible) > sample_size:
            eligible = self.rng.sample(eligible, sample_size)

        qa_pairs: List[SyntheticQA] = []
        errors: List[str] = []

        for chunk in eligible:
            try:
                questions = self._generate_questions(chunk)
                for q_text in questions:
                    qa = self._round_trip_check(
                        q_text, chunk, top_k_check)
                    if qa.round_trip_pass:
                        qa_pairs.append(qa)
            except Exception as e:
                errors.append(f"{chunk.get('source_path', '?')}: {e}")

        # Write benchmark JSON (compatible with eval format)
        benchmark = self._to_eval_format(qa_pairs)
        ts = int(time.time())
        bench_path = self.out_dir / f"self_bench_{ts}.json"
        bench_path.write_text(
            json.dumps(benchmark, indent=2), encoding="utf-8")

        # Write full QA details
        detail_path = self.out_dir / f"self_bench_{ts}_detail.json"
        detail_path.write_text(
            json.dumps([asdict(qa) for qa in qa_pairs], indent=2),
            encoding="utf-8")

        manifest = {
            "generated_at": ts,
            "chunks_sampled": len(eligible),
            "qa_generated": len(qa_pairs),
            "qa_easy": sum(1 for q in qa_pairs if q.difficulty == "easy"),
            "qa_medium": sum(1 for q in qa_pairs if q.difficulty == "medium"),
            "qa_hard": sum(1 for q in qa_pairs if q.difficulty == "hard"),
            "errors": len(errors),
            "benchmark_path": str(bench_path),
            "detail_path": str(detail_path),
        }
        manifest_path = self.out_dir / f"self_bench_{ts}_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

        return manifest

    def _generate_questions(self, chunk: Dict) -> List[str]:
        """Ask the LLM to generate questions about a code chunk."""
        code = chunk.get("content", "")[:1500]
        source = chunk.get("source_path", "unknown")
        prompt = _QA_GEN_PROMPT.format(source_path=source, code=code)

        raw = self.runtime.generate(
            question=prompt,
            context_chunks=[],
            system_prompt="Generate technical questions about code.",
            temperature=0.7,
            max_tokens=256,
        )

        # Parse "Q: ..." lines
        questions = []
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("Q:"):
                q = line[2:].strip()
                if len(q) > 15:
                    questions.append(q)
        return questions[:3]

    def _round_trip_check(self, question: str, original_chunk: Dict,
                           top_k: int) -> SyntheticQA:
        """
        Run the question through retrieval and check if the original
        chunk appears in the results. This is the quality filter.
        """
        results = self.retriever.retrieve(question)
        original_id = original_chunk.get("id", "")
        original_file = original_chunk.get("source_path", "")

        rank = -1
        for i, r in enumerate(results[:top_k]):
            if r.get("id", "") == original_id:
                rank = i + 1
                break

        round_trip_pass = rank > 0
        if rank == 1:
            difficulty = "easy"
        elif 1 < rank <= 5:
            difficulty = "medium"
        elif rank > 5:
            difficulty = "hard"
        else:
            difficulty = "hard"  # not found = hardest

        qa_id = hashlib.sha256(
            f"{question}:{original_id}".encode()).hexdigest()[:12]

        return SyntheticQA(
            qa_id=qa_id,
            question=question,
            expected_file=original_file,
            expected_chunk_id=original_id,
            source_chunk_snippet=original_chunk.get("content", "")[:200],
            difficulty=difficulty,
            retrieval_rank=rank,
            round_trip_pass=round_trip_pass,
            generated_at=time.time(),
        )

    @staticmethod
    def _to_eval_format(qa_pairs: List[SyntheticQA]) -> List[Dict]:
        """Convert to the same format as evaluation/benchmark_set.json."""
        out = []
        for qa in qa_pairs:
            out.append({
                "id": qa.qa_id,
                "question": qa.question,
                "expected_file_contains": qa.expected_file,
                "expected_symbols": [],
                "difficulty": qa.difficulty,
                "source": "self_generated",
            })
        return out
