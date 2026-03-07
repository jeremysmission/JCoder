from typing import Callable, Dict, Iterable

Config = Dict[str, object]


class GEPAEvaluator:
    """
    Wraps HybridRAG's RAG pipeline to evaluate a config on a benchmark dataset.
    """

    def __init__(self, rag_runner: Callable[[str, Config], str], dataset: Iterable[Dict[str, str]]):
        self.rag_runner = rag_runner
        self.dataset = list(dataset)

    def evaluate(self, config: Config) -> float:
        successful = 0
        for pair in self.dataset:
            question = pair.get("question", "")
            expected = pair.get("answer", "").strip().lower()
            result = self.rag_runner(question, config).strip().lower()
            if expected and expected in result:
                successful += 1
        return successful / max(1, len(self.dataset))
