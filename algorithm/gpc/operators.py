
from __future__ import annotations

import math


DEFAULT_GRAVITY = 9.8
DEFAULT_RAMP_ANGLE_DEGREES = 14.0
DEFAULT_FRICTION_MIN = 1.0
DEFAULT_FRICTION_MAX = 10.0
DEFAULT_SUBSTITUTION_PROBABILITY = 0.50


def _copy_gene(gene):
    return (int(gene[0]), int(gene[1]), int(gene[2]))


def _valid_providers(context, task):
    return list(dict.fromkeys(int(value) for value in context.valid_provider(int(task))))


def _rank_map(context):
    for key in ("task_rank", "global_ranks"):
        values = context.get(key)
        if isinstance(values, dict) and values:
            return {int(k): float(v) for k, v in values.items()}
    return {}


def physical_gpc_move(
    worker,
    pharaoh,
    context,
    rng,
    *,
    progress=0.0,
    gravity=DEFAULT_GRAVITY,
    ramp_angle_degrees=DEFAULT_RAMP_ANGLE_DEGREES,
    friction_min=DEFAULT_FRICTION_MIN,
    friction_max=DEFAULT_FRICTION_MAX,
    substitution_probability=DEFAULT_SUBSTITUTION_PROBABILITY,
):
    """Discrete GPC move retaining the published physical controls.

    The original GPC samples initial velocity and friction, computes the stone
    and worker travel distances on a ramp, and then applies component-wise
    substitution.  Provider identifiers are categorical rather than Euclidean,
    so the two normalized distances control *how many* task assignments move;
    substituted assignments follow the Pharaoh increasingly over time while
    early moves prefer a feasible alternative.  At least one component is
    substituted, matching the reference implementation's forced index.

    Returns ``(solution, diagnostics)`` so the benchmark can report the actual
    physical values used by the discrete adapter.
    """
    result = [_copy_gene(gene) for gene in worker]
    if not result:
        return result, {
            "initial_velocity": 0.0,
            "friction": 0.0,
            "stone_distance": 0.0,
            "worker_distance": 0.0,
            "substituted_tasks": 0,
        }

    gravity = max(1e-12, float(gravity))
    theta = math.radians(float(ramp_angle_degrees))
    sine = max(1e-12, math.sin(theta))
    cosine = max(0.0, math.cos(theta))
    friction_low = min(float(friction_min), float(friction_max))
    friction_high = max(float(friction_min), float(friction_max))
    friction = rng.uniform(friction_low, friction_high)
    velocity = rng.random()

    worker_distance = (velocity ** 2) / (2.0 * gravity * sine)
    stone_distance = (velocity ** 2) / (
        2.0 * gravity * (sine + friction * cosine)
    )
    worker_max = 1.0 / (2.0 * gravity * sine)
    stone_max = 1.0 / (
        2.0 * gravity * (sine + friction_low * cosine)
    )
    movement = 0.5 * (worker_distance / worker_max) + 0.5 * (
        stone_distance / max(1e-12, stone_max)
    )

    pss = max(0.0, min(1.0, float(substitution_probability)))
    # Directly applying pSS to every coordinate is unsuitable for this large
    # categorical schedule (hundreds of tasks): pSS=0.5 would replace hundreds
    # of assignments in one move. Preserve the published physical control but
    # map it to a dimension-normalized substitution count. The move remains
    # non-empty, grows with pSS and the physical travel distance, and is O(sqrt D)
    # rather than O(D) destructive.
    substitution_count = max(
        1,
        min(
            len(result),
            int(math.ceil(math.sqrt(len(result)) * pss * (0.50 + movement))),
        ),
    )
    pharaoh_map = {int(g[0]): int(g[1]) for g in pharaoh or []}
    disagreements = [
        index
        for index, gene in enumerate(result)
        if pharaoh_map.get(int(gene[0])) is not None
        and int(pharaoh_map[int(gene[0])]) != int(gene[1])
    ]
    rng.shuffle(disagreements)
    remaining = [index for index in range(len(result)) if index not in set(disagreements)]
    rng.shuffle(remaining)
    selected = (disagreements + remaining)[:substitution_count]

    progress = max(0.0, min(1.0, float(progress)))
    follow_probability = 0.25 + 0.60 * progress

    changed = 0
    for index in selected:
        task, current, position = result[index]
        domain = _valid_providers(context, int(task))
        if not domain:
            continue
        leader = pharaoh_map.get(int(task))
        alternatives = [p for p in domain if int(p) != int(current)]
        if not alternatives:
            continue
        if (
            leader is not None
            and int(leader) in alternatives
            and rng.random() < follow_probability
        ):
            provider = int(leader)
        else:
            exploratory = [p for p in alternatives if leader is None or int(p) != int(leader)]
            provider = int(rng.choice(exploratory or alternatives))
        result[index] = (int(task), provider, int(position))
        changed += 1

    return result, {
        "initial_velocity": float(velocity),
        "friction": float(friction),
        "stone_distance": float(stone_distance),
        "worker_distance": float(worker_distance),
        "substituted_tasks": int(changed),
    }


def rank_guided_discrete_mutation(
    solution,
    context,
    rng,
    mutation_probability=0.15,
    rank_guided=True,
):
    """
    Phase 3: Rank-guided adaptive discrete mutation (RADM).

    Purpose:
        Add exploration without changing GPC identity.

    Rules:
        - chromosome remains (task, provider, position)
        - only valid providers are selected
        - repair is handled by the core after this operator
        - higher priority tasks have higher mutation chance when ranks exist
    """
    result = [_copy_gene(g) for g in solution]

    if rng.random() > float(mutation_probability):
        return result

    rank_map = _rank_map(context) if bool(rank_guided) else {}

    def providers(task):
        return _valid_providers(context, int(task))

    candidates = []
    for index, gene in enumerate(result):
        task = int(gene[0])
        domain = [
            p for p in providers(task)
            if int(p) != int(gene[1])
        ]
        if domain:
            priority = rank_map.get(task, 0.0)
            candidates.append((index, domain, priority))

    if not candidates:
        return result

    # Preserve important assignments more often.  Lower-ranked tasks receive
    # the larger exploratory probability, but every task remains reachable.
    priorities = [float(item[2]) for item in candidates]
    lo = min(priorities) if priorities else 0.0
    hi = max(priorities) if priorities else 0.0
    weighted = []
    for item in candidates:
        normalized = 0.5 if hi <= lo else (float(item[2]) - lo) / (hi - lo)
        weighted.append((0.35 + 0.65 * (1.0 - normalized), item))

    # A quarter of a 600+ task chromosome is not a mutation; it is a near
    # restart. Keep the rank-guided move local and dimension-normalised.
    count = max(1, int(math.ceil(0.50 * math.sqrt(len(candidates)))))
    selected = []
    pool = list(weighted)
    while pool and len(selected) < count:
        total = sum(weight for weight, _ in pool)
        threshold = rng.random() * total
        running = 0.0
        chosen = len(pool) - 1
        for index, (weight, _item) in enumerate(pool):
            running += weight
            if threshold <= running:
                chosen = index
                break
        selected.append(pool.pop(chosen)[1])

    for index, domain, _ in selected:
        old = result[index]
        result[index] = (
            int(old[0]),
            int(rng.choice(domain)),
            int(old[2]),
        )

    return result


def adaptive_levy_escape(solution, context, rng, stagnation_count, probability=0.15):
    """
    Adaptive escape operator for GPC.
    Activated only during stagnation. It performs larger discrete jumps
    instead of only moving toward Pharaoh.
    """
    if stagnation_count < 5:
        return solution

    result = [_copy_gene(g) for g in solution]
    if not result or rng.random() > probability:
        return result

    # Preserve the long-jump role without replacing hundreds of categorical
    # assignments at once. Stagnation increases the radius up to 2*sqrt(D).
    scale = min(2.0, 1.0 + 0.15 * max(0, int(stagnation_count) - 5))
    jump = max(1, min(len(result), int(math.ceil(math.sqrt(len(result)) * scale))))
    positions = rng.sample(range(len(result)), jump)

    for idx in positions:
        task, provider, pos = result[idx]
        providers = _valid_providers(context, int(task))

        alternatives = [int(p) for p in providers if int(p) != int(provider)]
        if alternatives:
            result[idx] = (int(task), int(rng.choice(alternatives)), int(pos))

    return result


def cache_aware_mutation(solution, context, rng, probability=0.1):
    """
    Safe GPC enhancement:
    - keeps chromosome and evaluator unchanged
    - only modifies feasible provider choices
    - uses cache hints when available
    - falls back to normal random provider mutation
    """
    if rng.random() > float(probability):
        return [_copy_gene(g) for g in solution]

    result = [_copy_gene(g) for g in solution]
    cache_hint = context.get("cache_guidance", {}) or {}
    task_types = context.get("task_type_ids", {}) or {}
    ranks = _rank_map(context)
    # Build the service-reuse profile from the candidate schedule itself. This
    # is guidance only: the unchanged evaluator still decides whether the move
    # is useful after queueing, communication and cache-capacity effects.
    reuse = {}
    for task, provider, _pos in result:
        task_type = task_types.get(int(task), task_types.get(str(int(task))))
        if task_type is not None:
            key = (int(task_type), int(provider))
            reuse[key] = reuse.get(key, 0) + 1

    candidates = []
    for index, gene in enumerate(result):
        task, provider, _pos = gene
        domain = [
            int(value)
            for value in _valid_providers(context, int(task))
            if int(value) != int(provider)
        ]
        hinted = cache_hint.get(int(task), cache_hint.get(str(int(task))))
        if hinted is not None and int(hinted) in domain:
            domain = [int(hinted)] + [p for p in domain if p != int(hinted)]
        if not domain:
            continue
        task_type = task_types.get(int(task), task_types.get(str(int(task))))
        current_reuse = reuse.get((int(task_type), int(provider)), 0) if task_type is not None else 0
        scored = []
        for candidate_provider in domain:
            candidate_reuse = (
                reuse.get((int(task_type), int(candidate_provider)), 0)
                if task_type is not None
                else 0
            )
            hint_bonus = 1.0 if hinted is not None and int(candidate_provider) == int(hinted) else 0.0
            scored.append((float(candidate_reuse - current_reuse) + hint_bonus, candidate_provider))
        best_gain = max(score for score, _provider in scored)
        preferred = [provider for score, provider in scored if score == best_gain]
        # Lower-ranked tasks are safer cache-locality experiments; all tasks
        # remain reachable through the physical and Levy operators.
        importance = float(ranks.get(int(task), 0.0))
        candidates.append((index, preferred, best_gain, importance))

    if not candidates:
        return result

    positive = [item for item in candidates if item[2] > 0.0]
    pool = positive or candidates
    max_rank = max((item[3] for item in pool), default=0.0)
    weights = [1.0 + max(0.0, max_rank - item[3]) for item in pool]
    threshold = rng.random() * sum(weights)
    running = 0.0
    selected = pool[-1]
    for item, weight in zip(pool, weights):
        running += weight
        if threshold <= running:
            selected = item
            break
    index, providers, _gain, _importance = selected
    task, _provider, pos = result[index]
    result[index] = (int(task), int(rng.choice(providers)), int(pos))
    return result


def success_memory_mutation(
    solution, context, rng, memory, *, probability=0.20, exploration_floor=0.15
):
    """Change one feasible coordinate using bounded accepted-move memory."""
    result = [_copy_gene(gene) for gene in solution]
    if memory is None or not result or rng.random() > float(probability):
        return result
    ranks = _rank_map(context)
    rank_values = list(ranks.values())
    lo, hi = (min(rank_values), max(rank_values)) if rank_values else (0.0, 0.0)
    span = max(1e-12, hi - lo)
    choices = []
    for index, (task, provider, _position) in enumerate(result):
        alternatives = [int(p) for p in _valid_providers(context, task) if int(p) != provider]
        if not alternatives:
            continue
        importance = (float(ranks.get(task, lo)) - lo) / span if rank_values else 0.5
        scores = memory.provider_scores(task, alternatives)
        choices.append((index, alternatives, scores, 0.20 + max(scores.values(), default=0.0) + 0.35 * (1.0 - importance)))
    if not choices:
        return result
    threshold, cumulative = rng.random() * sum(x[3] for x in choices), 0.0
    selected = choices[-1]
    for item in choices:
        cumulative += item[3]
        if threshold <= cumulative:
            selected = item
            break
    index, alternatives, scores, _ = selected
    weights = {p: 0.05 + scores.get(p, 0.0) for p in alternatives}
    if rng.random() < max(0.0, min(1.0, float(exploration_floor))):
        provider = rng.choice(alternatives)
    else:
        threshold, cumulative, provider = rng.random() * sum(weights.values()), 0.0, alternatives[-1]
        for candidate, weight in weights.items():
            cumulative += weight
            if threshold <= cumulative:
                provider = candidate
                break
    task, _old, position = result[index]
    result[index] = (int(task), int(provider), int(position))
    return result
