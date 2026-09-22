from __future__ import annotations

import random


GREEDY_RATIO = 0.75


def create_initial_population(context, size: int, rng=None):
    """Build CPO porcupines from greedy schedules and discrete coverage."""
    rng = rng or random.Random()
    size = max(2, int(size))
    tasks = [int(task) for task in context.task_order]
    if not tasks:
        raise ValueError("CPO task order is empty")

    greedy_count = min(size, max(1, int(round(size * GREEDY_RATIO))))
    greedy = [
        context.repair_solution(solution)
        for solution in context.greedy_population(greedy_count)
    ]
    greedy = [solution for solution in greedy if solution]
    if not greedy:
        raise RuntimeError("CPO greedy initialization returned no solution")

    diverse_count = size - greedy_count
    provider_plans = {}
    for task in tasks:
        providers = list(dict.fromkeys(context.valid_provider(task)))
        if not providers:
            raise ValueError(f"Task {task} has no valid provider")
        quotient, remainder = divmod(diverse_count, len(providers))
        choices = [provider for provider in providers for _ in range(quotient)]
        extra = list(providers)
        rng.shuffle(extra)
        choices.extend(extra[:remainder])
        rng.shuffle(choices)
        provider_plans[task] = choices

    diverse = [
        context.repair_solution(
            [
                (task, int(provider_plans[task][row]), position)
                for position, task in enumerate(tasks)
            ]
        )
        for row in range(diverse_count)
    ]

    population = []
    seen = set()

    def append(candidate):
        key = tuple((int(gene[0]), int(gene[1])) for gene in candidate)
        if key and key not in seen:
            seen.add(key)
            population.append(candidate)

    for index in range(max(len(greedy), len(diverse))):
        if index < len(greedy):
            append(greedy[index])
        if index < len(diverse):
            append(diverse[index])

    attempts = 0
    while len(population) < size and attempts < size * 100:
        attempts += 1
        append(
            context.repair_solution(
                [
                    (task, int(rng.choice(context.valid_provider(task))), position)
                    for position, task in enumerate(tasks)
                ]
            )
        )
    if len(population) != size:
        raise RuntimeError(
            f"CPO created {len(population)} unique porcupines; expected {size}"
        )
    return population
