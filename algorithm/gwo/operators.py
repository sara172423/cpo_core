from __future__ import annotations

import copy
import math
import random
from typing import Sequence

from .predictive_cache import (
    DEFAULT_PROVIDER_GUIDANCE_WEIGHT,
    build_provider_future_index,
    normalized_provider_reuse_guidance,
)

Gene = tuple[int, int, int]


def _context_flag(context, key: str, default=False) -> bool:
    """Read flags from either dict-like or attribute-only context adapters."""
    if hasattr(context, key):
        return bool(getattr(context, key))
    if isinstance(context, dict) and key in context:
        return bool(context[key])
    return bool(context.get(key, default))


def _provider_guidance_weight(context) -> float:
    if not _context_flag(context, "gwo_cache_guidance", False):
        return 0.0
    try:
        return max(
            0.0,
            float(
                context.get(
                    "predictive_provider_guidance_weight",
                    DEFAULT_PROVIDER_GUIDANCE_WEIGHT,
                )
            ),
        )
    except (TypeError, ValueError):
        return float(DEFAULT_PROVIDER_GUIDANCE_WEIGHT)


def _valid_providers(context, task: int) -> list[int]:
    return list(dict.fromkeys(int(value) for value in context.valid_provider(int(task))))


def _provider_map(solution: Sequence) -> dict[int, int]:
    return {
        int(gene[0]): int(gene[1])
        for gene in solution
        if isinstance(gene, (tuple, list)) and len(gene) >= 2
    }


def _get_task_ranks(context) -> dict[int, float]:
    """Use the canonical rank provided by build_context."""
    if not _context_flag(context, "gwo_rank_guidance", False):
        return {}
    for key in ("task_rank", "global_ranks"):
        rank_map = context.get(key)
        if isinstance(rank_map, dict):
            return {int(k): float(v) for k, v in rank_map.items()}
    return {}


def _rank_weight(task, task_ranks):
    """Normalize the shared rank to [0, 1].  Imported by benchmark/search.py."""
    if not task_ranks:
        return 0.0
    values = [float(v) for v in task_ranks.values()]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return 0.5
    return max(
        0.0,
        min(1.0, (float(task_ranks.get(int(task), lo)) - lo) / (hi - lo)),
    )


def hamming_distance(solution_a: Sequence, solution_b: Sequence) -> int:
    a = _provider_map(solution_a)
    b = _provider_map(solution_b)
    tasks = set(a) | set(b)
    return sum(a.get(task) != b.get(task) for task in tasks)


def _leader_provider(leader: Sequence, task: int) -> int | None:
    for gene in leader or []:
        if isinstance(gene, (tuple, list)) and len(gene) >= 2 and int(gene[0]) == int(task):
            return int(gene[1])
    return None


def _weighted_choice(weights: dict[int, float], rng: random.Random) -> int:
    candidates = list(weights)
    if not candidates:
        raise ValueError("Weighted choice received no candidates")
    total = sum(max(0.0, float(weights[p])) for p in candidates)
    if total <= 0.0:
        return int(rng.choice(candidates))
    threshold = rng.random() * total
    cumulative = 0.0
    for provider in candidates:
        cumulative += max(0.0, float(weights[provider]))
        if threshold <= cumulative:
            return int(provider)
    return int(candidates[-1])


def _normalized_pheromone(pheromone, task: int, providers: Sequence[int]) -> dict[int, float]:
    providers = list(dict.fromkeys(int(p) for p in providers))
    if not providers:
        return {}
    if not pheromone:
        return {provider: 0.0 for provider in providers}
    values = [float(pheromone.get((int(task), provider), 1.0)) for provider in providers]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return {provider: 0.5 for provider in providers}
    return {
        provider: (float(pheromone.get((int(task), provider), 1.0)) - lo) / (hi - lo)
        for provider in providers
    }


def _search_phase(a: float) -> float:
    """1 at the exploratory start, 0 near the exploitative end."""
    return max(0.0, min(1.0, float(a) / 2.0))


def _leader_base_weights(a: float) -> tuple[float, float, float]:
    """Trust beta/delta more early and let alpha dominate progressively."""
    phase = _search_phase(a)
    exploitation = 1.0 - phase
    return (
        1.00 + 1.20 * exploitation,
        0.90 - 0.30 * exploitation,
        0.80 - 0.35 * exploitation,
    )


def _change_budget(source, alpha, beta, delta, *, a: float, rng: random.Random) -> int:
    dimension = max(1, len(source))
    mean_distance = (
        hamming_distance(source, alpha)
        + hamming_distance(source, beta)
        + hamming_distance(source, delta)
    ) / 3.0
    phase = _search_phase(a)
    fraction = 0.15 + 0.85 * phase
    budget = int(math.ceil(max(1.0, mean_distance) * fraction))
    # A continuous GWO updates every coordinate, but provider identifiers are
    # categorical and this schedule has hundreds of coordinates. Bound the
    # discrete move to an O(sqrt D) trust region so leader guidance does not
    # destroy an otherwise strong feasible schedule in one step.
    trust_region = max(1, int(math.ceil(math.sqrt(dimension) * (0.50 + phase))))
    budget = min(budget, trust_region)
    if phase > 0.5 and rng.random() < 0.10:
        budget += 1
    return max(1, min(dimension, budget))


def generate_alpha_neighborhood_children(
    alpha,
    context,
    *,
    count: int,
    rng: random.Random,
    pheromone=None,
    beta=None,
    delta=None,
    a: float = 0.5,
) -> list:
    """Sparse evidence-guided one-coordinate exploitation around alpha."""
    if not alpha or count <= 0:
        return []
    beta, delta = beta or alpha, delta or alpha
    alpha_map = _provider_map(alpha)
    future_index = (
        build_provider_future_index(context, alpha_map)
        if _context_flag(context, "gwo_cache_guidance", False)
        else {}
    )
    task_ranks = _get_task_ranks(context)
    rank_values = list(task_ranks.values())
    rank_lo, rank_hi = (min(rank_values), max(rank_values)) if rank_values else (0.0, 0.0)
    rank_span = max(1e-12, rank_hi - rank_lo)
    mutable = []
    for index, gene in enumerate(alpha):
        task, provider = int(gene[0]), int(gene[1])
        alternatives = [int(p) for p in _valid_providers(context, task) if int(p) != provider]
        if not alternatives:
            continue
        importance = (float(task_ranks.get(task, rank_lo)) - rank_lo) / rank_span if rank_values else 0.5
        disagreement = int(_leader_provider(beta, task) != provider) + int(_leader_provider(delta, task) != provider)
        mutable.append((index, alternatives, 0.30 + 0.55 * (1.0 - importance) + 0.45 * disagreement))
    if not mutable:
        return []
    children = []
    for _ in range(int(count)):
        child = copy.deepcopy(alpha)
        threshold, cumulative = rng.random() * sum(x[2] for x in mutable), 0.0
        index, alternatives, _ = mutable[-1]
        for item in mutable:
            cumulative += item[2]
            if threshold <= cumulative:
                index, alternatives, _ = item
                break
        task, current, _position = child[index]
        provider = int(current)
        # Keep a phase-dependent exploration floor: learned leader/cache
        # evidence guides most trials, but cannot make unseen providers
        # unreachable in a large V2V domain.
        exploration = 0.15 + 0.20 * _search_phase(float(a))
        if rng.random() < exploration:
            provider = int(rng.choice(alternatives))
        else:
            for _attempt in range(3):
                provider = _choose_provider(
                    int(task), int(current), alpha, beta, delta, pheromone or {},
                    context, alpha_map, future_index,
                    a=max(0.10, min(0.75, float(a))), rng=rng,
                )
                if provider != int(current):
                    break
        if provider == int(current):
            provider = rng.choice(alternatives)
        child[index] = (int(task), int(provider), int(index))
        child = context.repair_solution(child)
        if child:
            children.append(child)
    return children


def _select_tasks(source, alpha, beta, delta, budget: int, task_ranks, rng) -> list[int]:
    source_map = _provider_map(source)
    alpha_map = _provider_map(alpha)
    beta_map = _provider_map(beta)
    delta_map = _provider_map(delta)

    weights: dict[int, float] = {}
    for task, current in source_map.items():
        leaders = [alpha_map.get(task), beta_map.get(task), delta_map.get(task)]
        present = [p for p in leaders if p is not None]
        current_disagreement = sum(p != current for p in present)
        leader_disagreement = max(0, len(set(present)) - 1)
        weights[int(task)] = (
            1.0
            + 1.20 * float(current_disagreement)
            + 0.80 * float(leader_disagreement)
            + 0.75 * _rank_weight(int(task), task_ranks)
        )

    selected = []
    pool = dict(weights)
    for _ in range(min(int(budget), len(pool))):
        task = _weighted_choice(pool, rng)
        selected.append(int(task))
        pool.pop(task, None)
    return selected


def _candidate_subset(
    domain: Sequence[int],
    current: int,
    leader_providers: Sequence[int | None],
    *,
    a: float,
    rng: random.Random,
) -> list[int]:
    """Keep provider scoring cheap even when a task has a large V2V domain."""
    domain = list(dict.fromkeys(int(p) for p in domain))
    candidates = {int(current)} if int(current) in domain else set()
    for provider in leader_providers:
        if provider is not None and int(provider) in domain:
            candidates.add(int(provider))

    alternatives = [p for p in domain if p not in candidates]
    rng.shuffle(alternatives)
    phase = _search_phase(a)
    extra_count = 2 + int(round(2.0 * phase))  # 2 late, at most 4 early.
    candidates.update(alternatives[:extra_count])

    if not candidates and domain:
        candidates.add(int(rng.choice(domain)))
    return list(candidates)


def _choose_provider(
    task: int,
    current: int,
    alpha,
    beta,
    delta,
    pheromone,
    context,
    provider_map: dict[int, int],
    future_index,
    *,
    a: float,
    rng: random.Random,
) -> int:
    domain = _valid_providers(context, int(task))
    if not domain:
        return int(current)

    leader_providers = [
        _leader_provider(alpha, task),
        _leader_provider(beta, task),
        _leader_provider(delta, task),
    ]
    candidates = _candidate_subset(
        domain,
        int(current),
        leader_providers,
        a=float(a),
        rng=rng,
    )
    if not candidates:
        return int(current)

    tau = _normalized_pheromone(pheromone, int(task), candidates)
    guidance = normalized_provider_reuse_guidance(
        context,
        int(task),
        candidates,
        provider_map,
        future_index=future_index,
    )

    phase = _search_phase(a)
    leader_set = {int(p) for p in leader_providers if p is not None}
    weights = {}
    for provider in candidates:
        # Small inertia avoids gratuitous churn late in search.
        inertia = (0.12 + 0.22 * (1.0 - phase)) if int(provider) == int(current) else 0.0
        novelty = 0.18 * phase if int(provider) not in leader_set else 0.0
        weights[int(provider)] = 0.05 + inertia + novelty + 0.30 * tau.get(int(provider), 0.5)

    # Preserve the A and C mechanics of GWO as categorical leader influence.
    for base, provider in zip(_leader_base_weights(a), leader_providers):
        if provider is None or int(provider) not in weights:
            continue
        r1, r2 = rng.random(), rng.random()
        A = 2.0 * float(a) * r1 - float(a)
        C = 2.0 * r2
        follow = max(0.0, 1.0 - min(abs(A), 2.0) / 2.0)
        c_scale = 0.75 + 0.25 * C
        weights[int(provider)] += float(base) * follow * c_scale

    guidance_weight = _provider_guidance_weight(context)
    if guidance_weight > 0.0:
        for provider in candidates:
            weights[int(provider)] += guidance_weight * guidance.get(int(provider), 0.0)

    return _weighted_choice(weights, rng)


def discrete_adaptive_move(
    current,
    alpha,
    beta,
    delta,
    pheromone,
    context,
    *,
    a: float,
    rng: random.Random,
) -> list[Gene]:
    """One discrete multi-leader GWO move for a complete task schedule."""
    if not current:
        return []

    source = [(int(g[0]), int(g[1]), index) for index, g in enumerate(current)]
    source_map = _provider_map(source)
    future_index = (
        build_provider_future_index(context, source_map)
        if _context_flag(context, "gwo_cache_guidance", False)
        else {}
    )
    task_ranks = _get_task_ranks(context)
    budget = _change_budget(source, alpha, beta, delta, a=float(a), rng=rng)
    selected = set(
        _select_tasks(source, alpha, beta, delta, budget, task_ranks, rng)
    )

    result = []
    for index, (task, provider, _) in enumerate(source):
        new_provider = int(provider)
        if int(task) in selected:
            new_provider = _choose_provider(
                int(task),
                int(provider),
                alpha,
                beta,
                delta,
                pheromone,
                context,
                source_map,
                future_index,
                a=float(a),
                rng=rng,
            )
        result.append((int(task), int(new_provider), index))
    return result


def generate_adaptive_children(
    population,
    alpha,
    beta,
    delta,
    pheromone,
    context,
    *,
    a: float,
    count: int,
    rng: random.Random,
) -> list:
    """Generate at most one ordinary child per requested slot.

    There is no separate exploitation/exploration/refinement population.  The
    same adaptive discrete move changes its behavior through ``a``.
    """
    if not population or count <= 0:
        return []

    sources = list(population)
    rng.shuffle(sources)
    children = []
    for index in range(int(count)):
        source = sources[index % len(sources)]
        child = discrete_adaptive_move(
            source,
            alpha,
            beta,
            delta,
            pheromone,
            context,
            a=float(a),
            rng=rng,
        )
        child = context.repair_solution(child)
        if child:
            children.append(child)
    return children


def generate_escape_children(
    source_pool,
    context,
    *,
    a: float,
    count: int,
    rng: random.Random,
) -> list:
    """Unguided feasible jump used only after sustained stagnation."""
    if not source_pool or count <= 0:
        return []

    children = []
    phase = _search_phase(a)
    for _ in range(int(count)):
        source = copy.deepcopy(rng.choice(source_pool))
        if not source:
            continue

        jump_strength = max(
            1,
            min(
                len(source),
                int(math.ceil(math.sqrt(len(source)) * (1.0 + phase))),
            ),
        )
        positions = rng.sample(range(len(source)), jump_strength)

        for position in positions:
            task, current_provider, _ = source[position]
            alternatives = [
                int(provider)
                for provider in _valid_providers(context, int(task))
                if int(provider) != int(current_provider)
            ]
            if alternatives:
                source[position] = (
                    int(task),
                    int(rng.choice(alternatives)),
                    int(position),
                )

        child = context.repair_solution(source)
        if child:
            children.append(child)
    return children
