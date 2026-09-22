from __future__ import annotations

import math


def solution_key(solution):
    return tuple((int(gene[0]), int(gene[1])) for gene in solution)


def hamming_distance(left, right):
    left_map = {int(gene[0]): int(gene[1]) for gene in left}
    right_map = {int(gene[0]): int(gene[1]) for gene in right}
    return sum(
        left_map.get(task) != right_map.get(task)
        for task in set(left_map) | set(right_map)
    )


def _provider_map(solution):
    return {int(gene[0]): int(gene[1]) for gene in solution}


def _weighted_sample(tasks, weights, count, rng):
    remaining = list(tasks)
    selected = []
    while remaining and len(selected) < count:
        local_weights = [max(1e-9, float(weights.get(task, 1.0))) for task in remaining]
        target = rng.random() * sum(local_weights)
        cumulative = 0.0
        choice = remaining[-1]
        for task, weight in zip(remaining, local_weights):
            cumulative += weight
            if cumulative >= target:
                choice = task
                break
        selected.append(choice)
        remaining.remove(choice)
    return selected


def _task_weights(context, tasks):
    # Uniform task selection for a fair Puma baseline.
    # Problem-specific priorities are intentionally not injected into the optimizer.
    return {
        int(task): 1.0
        for task in tasks
    }


def _build_solution(context, mapping):
    return context.repair_solution([
        (int(task), int(mapping[int(task)]), position)
        for position, task in enumerate(context.task_order)
    ])


def exploration_candidate(parent, population, context, pcr, rng):
    """Categorical projection of PO exploration.

    It retains PO's random-global and six-peer differential branches plus the
    crossover probability, while every proposed coordinate remains inside the
    task-specific provider domain.
    """
    tasks = [int(task) for task in context.task_order]
    dimension = len(tasks)
    weights = _task_weights(context, tasks)
    parent_map = _provider_map(parent)
    candidate = dict(parent_map)

    change_count = max(1, min(
        dimension,
        int(round(max(1.0, float(pcr) * math.sqrt(max(1, dimension)) * 2.0))),
    ))
    selected = _weighted_sample(tasks, weights, change_count, rng)

    maps = [_provider_map(solution) for solution in population]
    if not maps:
        maps = [parent_map]
    peers = [rng.choice(maps) for _ in range(6)]
    random_global = rng.random() < 0.5

    for task in selected:
        domain = list(dict.fromkeys(int(value) for value in context.valid_provider(task)))
        current = parent_map[task]
        if random_global:
            proposals = [value for value in domain if value != current]
        else:
            a, b, c, d, e, f = (peer.get(task, current) for peer in peers)
            proposals = []
            # A categorical differential is active only when the paired peers
            # disagree; the leading member supplies the feasible direction.
            for positive, negative in ((a, b), (c, d), (e, f)):
                if positive != negative and positive in domain and positive != current:
                    proposals.append(int(positive))
            if not proposals:
                proposals = [value for value in domain if value != current]
        if proposals:
            candidate[task] = int(rng.choice(proposals))

    return _build_solution(context, candidate)


def exploitation_candidate(parent, best, population, context, progress, q, beta, rng):
    """Categorical projection of PO exploitation around the current best."""
    tasks = [int(task) for task in context.task_order]
    dimension = len(tasks)
    weights = _task_weights(context, tasks)
    parent_map = _provider_map(parent)
    best_map = _provider_map(best)
    population_maps = [_provider_map(solution) for solution in population]
    candidate = dict(parent_map)

    # PO's exploitation contracts over time. Beta controls the contraction and
    # Q separates the two principal attack behaviours used by the source code.
    radius = max(1, int(math.ceil(
        1.0 + math.sqrt(max(1, dimension)) * ((1.0 - progress) ** max(1.0, beta))
    )))
    selected = _weighted_sample(tasks, weights, min(dimension, radius), rng)
    mode = rng.random()

    for task in selected:
        domain = list(dict.fromkeys(int(value) for value in context.valid_provider(task)))
        current = parent_map[task]
        best_provider = best_map.get(task, current)
        peer_provider = rng.choice(population_maps).get(task, current)

        counts = {}
        for mapping in population_maps:
            provider = mapping.get(task)
            if provider in domain:
                counts[provider] = counts.get(provider, 0) + 1
        consensus = max(counts, key=lambda value: (counts[value], value)) if counts else best_provider

        if mode < q:
            proposals = [best_provider, best_provider, peer_provider]
        elif mode < (q + (1.0 - q) * 0.5):
            proposals = [best_provider, consensus, peer_provider]
        else:
            # Rare close-range evasion keeps a route out of a local optimum.
            proposals = [value for value in domain if value != best_provider]
            proposals.extend([best_provider, consensus])

        proposals = [int(value) for value in proposals if value in domain]
        if proposals:
            candidate[task] = int(rng.choice(proposals))

    return _build_solution(context, candidate)
