from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, TypeVar

from algorithm.optimizer_common import hamming_distance

NestT = TypeVar("NestT")
ScoredNest = Tuple[NestT, float]


@lru_cache(maxsize=16)
def levy_sigma(beta: float) -> float:
    beta = float(beta)
    return (
        math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
        / (
            math.gamma((1.0 + beta) / 2.0)
            * beta
            * (2.0 ** ((beta - 1.0) / 2.0))
        )
    ) ** (1.0 / beta)


def normalized_levy(beta: float, rng: Any) -> float:
    sigma = levy_sigma(float(beta))
    u = rng.gauss(0.0, sigma)
    v = rng.gauss(0.0, 1.0)
    value = abs(u / (abs(v) ** (1.0 / float(beta))))
    return value / (1.0 + value)


def levy_step_length(hamming: int, dimension: int, beta: float, rng: Any) -> int:
    if int(hamming) <= 0 or int(dimension) <= 0:
        return 0
    length = round(2 * int(hamming) * normalized_levy(float(beta), rng))
    return max(0, min(int(length), 2 * int(dimension)))


def mutate_provider_map(
    task_order: Sequence[int],
    source_provider_map: Mapping[int, int],
    best_provider_map: Mapping[int, int] | None,
    *,
    levy_lambda: float,
    rng: Any,
    domain_for_task: Callable[[int], Sequence[int]],
    mode_for_task_provider: Callable[[int, int], str],
) -> Dict[int, int]:
    """Canonical Procedure 3 shared by runtime and paper benchmark."""
    tasks = [int(task_id) for task_id in task_order]
    if not tasks:
        return {}

    source = {task_id: int(source_provider_map[task_id]) for task_id in tasks}
    best = (
        {task_id: int(best_provider_map[task_id]) for task_id in tasks}
        if best_provider_map is not None
        else None
    )
    new_map = dict(source)
    dimension = len(tasks)
    hamming = dimension if best is None else hamming_distance(source, best)
    length = levy_step_length(hamming, dimension, levy_lambda, rng)
    quotient, remainder = divmod(length, 2)
    transformed: set[int] = set()

    def choose_same_mode(task_id: int, current: int, target_mode: str) -> int:
        domain = [int(provider_id) for provider_id in domain_for_task(task_id)]
        candidates = [
            provider_id
            for provider_id in domain
            if provider_id != current
            and mode_for_task_provider(task_id, provider_id) == target_mode
        ]
        if not candidates:
            candidates = [
                provider_id
                for provider_id in domain
                if mode_for_task_provider(task_id, provider_id) == target_mode
            ]
        if not candidates:
            candidates = [provider_id for provider_id in domain if provider_id != current]
        return int(rng.choice(candidates)) if candidates else int(current)

    if best is not None and hamming != dimension:
        different = [task_id for task_id in tasks if new_map[task_id] != best[task_id]]
        if quotient and different:
            for task_id in rng.sample(different, min(quotient, len(different))):
                new_map[task_id] = best[task_id]
                transformed.add(task_id)
        if remainder:
            remaining = [
                task_id
                for task_id in different
                if task_id not in transformed and new_map[task_id] != best[task_id]
            ]
            if remaining:
                task_id = int(rng.choice(remaining))
                target_mode = mode_for_task_provider(task_id, best[task_id])
                new_map[task_id] = choose_same_mode(
                    task_id, new_map[task_id], target_mode
                )
    else:
        if quotient:
            for task_id in rng.sample(tasks, min(quotient, len(tasks))):
                domain = [int(provider_id) for provider_id in domain_for_task(task_id)]
                new_map[task_id] = int(rng.choice(domain))
                transformed.add(task_id)
        if remainder:
            remaining = [task_id for task_id in tasks if task_id not in transformed]
            if remaining:
                task_id = int(rng.choice(remaining))
                current_mode = mode_for_task_provider(task_id, new_map[task_id])
                new_map[task_id] = choose_same_mode(
                    task_id, new_map[task_id], current_mode
                )
    return new_map


def run_population_search(
    initial_population: Sequence[NestT],
    *,
    population_size: int,
    tmax: int,
    initial_discard_probability: float,
    rng: Any,
    generate_new_solution: Callable[[NestT, NestT | None], NestT],
    evaluate_population: Callable[[Sequence[NestT]], List[ScoredNest]],
    on_iteration: Callable[[int, List[ScoredNest]], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    evaluation_budget_exhausted: Callable[[], bool] | None = None,
) -> Tuple[List[NestT], NestT, List[ScoredNest]]:
    """Canonical generation loop from paper Algorithm 2."""
    size = int(population_size)
    if size < 2:
        raise ValueError("population_size must be at least 2")

    evaluated = list(evaluate_population(list(initial_population)))
    if len(evaluated) < size:
        raise ValueError("Initial population is smaller than population_size")
    evaluated = evaluated[:size]
    population = [row[0] for row in evaluated]
    best_nest = population[0]
    if on_iteration is not None:
        on_iteration(0, evaluated)

    discard_probability = float(initial_discard_probability)
    t = 1
    while t < max(1, int(tmax)):
        if cancel_check is not None:
            cancel_check()

        previous_evaluated = list(evaluated)
        new_population: List[NestT] = [best_nest]
        for index in range(1, size):
            if cancel_check is not None:
                cancel_check()
            new_population.append(generate_new_solution(population[index], best_nest))

        random_cuckoo = rng.choice(new_population)
        random_walk: List[NestT] = []
        for _ in range(size):
            if cancel_check is not None:
                cancel_check()
            random_walk.append(generate_new_solution(random_cuckoo, None))

        candidate_evaluated = list(
            evaluate_population(new_population + random_walk)
        )
        if evaluation_budget_exhausted is not None and evaluation_budget_exhausted():
            # At the exact NFE boundary the final batch may be partial. Keep
            # every evaluated candidate and supplement it with already-scored
            # parents so the returned population remains valid and elitist.
            if len(candidate_evaluated) < size:
                candidate_evaluated.extend(previous_evaluated)
            candidate_evaluated.sort(key=lambda row: row[1], reverse=True)
            evaluated = list(candidate_evaluated[:size])
            population = [row[0] for row in evaluated]
            best_nest = population[0]
            if on_iteration is not None:
                on_iteration(t, evaluated)
            break

        evaluated = candidate_evaluated
        discard_probability = min(1.0, (2.0 * discard_probability) / max(t, 1))

        if rng.random() <= discard_probability:
            worst_nest = evaluated[-1][0]
            worst_walk: List[NestT] = []
            for _ in range(size):
                if cancel_check is not None:
                    cancel_check()
                worst_walk.append(generate_new_solution(worst_nest, None))
            before_discard = list(evaluated)
            discard_evaluated = list(evaluate_population(
                [row[0] for row in evaluated[:-1]] + worst_walk
            ))
            if evaluation_budget_exhausted is not None and evaluation_budget_exhausted():
                if len(discard_evaluated) < size:
                    discard_evaluated.extend(before_discard)
                discard_evaluated.sort(key=lambda row: row[1], reverse=True)
                evaluated = list(discard_evaluated[:size])
                population = [row[0] for row in evaluated]
                best_nest = population[0]
                if on_iteration is not None:
                    on_iteration(t, evaluated)
                break
            evaluated = discard_evaluated

        evaluated = list(evaluated[:size])
        population = [row[0] for row in evaluated]
        best_nest = population[0]
        if on_iteration is not None:
            on_iteration(t, evaluated)
        t += 1

    return population, best_nest, evaluated
