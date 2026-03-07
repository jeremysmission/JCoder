import random
from typing import Callable, Dict, Iterable, List, Tuple

Config = Dict[str, object]
Score = float


class GEPAEngine:
    """Evolutionary engine that mutates and scores RAG configuration candidates."""

    def __init__(
        self,
        population_size: int,
        evaluator: Callable[[Config], Score],
        mutator: Callable[[Config], Config],
        seed: int = 42,
    ):
        self.population_size = population_size
        self.evaluator = evaluator
        self.mutator = mutator
        self.random = random.Random(seed)

    def evolve(self, population: List[Config]) -> List[Config]:
        scored: List[Tuple[Config, Score]] = []
        for candidate in population:
            score = self.evaluator(candidate)
            scored.append((candidate, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        elites = [config for config, _ in scored[: max(1, self.population_size // 5)]]

        new_population: List[Config] = elites.copy()
        while len(new_population) < self.population_size:
            parent = self.random.choice(elites)
            mutated = self.mutator(parent.copy())
            new_population.append(mutated)

        return new_population
