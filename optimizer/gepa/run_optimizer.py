import argparse
import json
import os
from typing import Iterable, Dict

from optimizer.gepa.gepa_engine import GEPAEngine
from optimizer.gepa.evaluation import GEPAEvaluator
from optimizer.gepa.mutation import GEPAConfigMutator

Config = Dict[str, object]


def load_benchmark(dataset_dir: str) -> Iterable[Dict[str, str]]:
    questions_path = os.path.join(dataset_dir, "questions.json")
    answers_path = os.path.join(dataset_dir, "answers.json")
    if not os.path.exists(questions_path) or not os.path.exists(answers_path):
        raise FileNotFoundError("Benchmark dataset missing questions/answers files")

    with open(questions_path, "r", encoding="utf-8") as qf:
        questions = json.load(qf)
    with open(answers_path, "r", encoding="utf-8") as af:
        answers = json.load(af)

    for q, a in zip(questions, answers):
        yield {"question": q, "answer": a}


def rag_runner_stub(question: str, config: Config) -> str:
    """Stub runner until a proper RAG evaluation proxy is implemented."""
    return f"stub answer for {question} on config {config.get('top_k')}"


def main(
    benchmark_dir: str,
    generations: int,
    population: int,
    output: str,
):
    dataset = load_benchmark(benchmark_dir)
    evaluator = GEPAEvaluator(rag_runner_stub, dataset)
    mutator = GEPAConfigMutator()
    engine = GEPAEngine(population_size=population, evaluator=evaluator.evaluate, mutator=mutator.mutate)

    initial_population = [mutator.mutate({"top_k": 4, "chunk_size": 384, "temperature": 0.2, "prompt_template": "cot_prompt_v3", "reranker": "none"}) for _ in range(population)]
    population = initial_population

    for generation in range(generations):
        population = engine.evolve(population)
        best = population[0]
        score = evaluator.evaluate(best)
        print(f"Generation {generation+1}: best score {score:.4f} top_k={best['top_k']} chunk_size={best['chunk_size']} reranker={best['reranker']} prompt={best['prompt_template']} temp={best['temperature']}")

    best_config = population[0]
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(best_config, fh, indent=2)
    print(f"Best config saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GEPA optimizer")
    parser.add_argument("--benchmark", default="experiments/rag_benchmark_dataset", help="Benchmark dataset directory")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--population", type=int, default=30, help="Population size per generation")
    parser.add_argument("--output", default="config/best_rag_config.json", help="Output path for best config")
    args = parser.parse_args()
    main(args.benchmark, args.generations, args.population, args.output)
