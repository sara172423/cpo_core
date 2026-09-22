from __future__ import annotations

import random


GREEDY_RATIO = 0.0


def _key(solution):
    return tuple((int(gene[0]), int(gene[1])) for gene in solution)


def create_initial_population(context, size: int, rng=None):
    """Create a feasible, mixed-quality population for discrete PO.

    The original continuous PO samples its population across the search box.
    Here, balanced categorical samples cover every feasible provider without treating provider identifiers as
    continuous coordinates.
    """
    rng = rng or random.Random()
    size = max(2, int(size))
    tasks = [int(task) for task in context.task_order]
    if not tasks:
        raise ValueError("Puma task order is empty")

    domains = {
        task: list(dict.fromkeys(int(value) for value in context.valid_provider(task)))
        for task in tasks
    }
    missing = [task for task, values in domains.items() if not values]
    if missing:
        raise ValueError(f"Tasks have no valid provider: {missing}")

    greedy_count = min(size, max(1, int(round(size * GREEDY_RATIO))))
    population = []
    seen = set()

    def append(raw):
        candidate = context.repair_solution(raw)
        key = _key(candidate)
        if key and key not in seen:
            seen.add(key)
            population.append(candidate)

    if greedy_count > 0:
        for candidate in context.greedy_population(greedy_count):
            append(candidate)

    # Balanced categorical coverage is the discrete counterpart of uniform
    # sampling inside continuous bounds.
    diverse_count = size - len(population)
    plans = {}
    for task, providers in domains.items():
        quotient, remainder = divmod(max(1, diverse_count), len(providers))
        choices = [provider for provider in providers for _ in range(quotient)]
        extras = list(providers)
        rng.shuffle(extras)
        choices.extend(extras[:remainder])
        rng.shuffle(choices)
        plans[task] = choices

    for row in range(diverse_count):
        append([
            (task, int(plans[task][row]), position)
            for position, task in enumerate(tasks)
        ])

    attempts = 0
    while len(population) < size and attempts < size * 150:
        attempts += 1
        append([
            (task, int(rng.choice(domains[task])), position)
            for position, task in enumerate(tasks)
        ])

    if len(population) != size:
        raise RuntimeError(
            f"Puma created {len(population)} unique solutions; expected {size}"
        )
    return population
