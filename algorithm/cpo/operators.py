from __future__ import annotations

import math


DEFENSE_EXPLORATION = ("sight", "sound")
DEFENSE_EXPLOITATION = ("odor", "physical_attack")
DEFENSE_NAMES = DEFENSE_EXPLORATION + DEFENSE_EXPLOITATION


def solution_key(solution):
    return tuple(
        (int(gene[0]), int(gene[1]))
        for gene in solution or []
        if isinstance(gene, (tuple, list)) and len(gene) >= 2
    )


def provider_map(solution):
    return dict(solution_key(solution))


def valid_providers(context, task: int) -> list[int]:
    return list(
        dict.fromkeys(int(value) for value in context.valid_provider(int(task)))
    )


def _energy_opportunity(context, task):
    if not bool(context.get("cpo_model_guidance", True)):
        return 0.0
    table = context.get("task_energy_opportunity", {})
    return max(
        0.0,
        float(table.get(int(task), table.get(str(int(task)), 0.0)) or 0.0),
    )


def _deadline_urgency_weights(context, tasks):
    """Return normalized deadline pressure when the benchmark exposes it.

    Deadline guidance is intentionally a proposal bias only. It never changes
    the evaluator or fitness. Missing task-level deadlines keep the old
    structural-only behaviour.
    """
    if not bool(context.get("cpo_deadline_guidance", False)):
        return {int(task): 0.0 for task in tasks}

    deadlines = (
        context.get("task_deadline_s")
        or context.get("task_deadlines_s")
        or context.get("deadline_by_task")
        or {}
    )
    if not deadlines:
        return {int(task): 0.0 for task in tasks}

    values = {}
    for task in tasks:
        value = deadlines.get(int(task), deadlines.get(str(int(task))))
        if value is not None:
            values[int(task)] = 1.0 / max(float(value), 1e-9)

    if not values:
        return {int(task): 0.0 for task in tasks}

    low, high = min(values.values()), max(values.values())
    if high <= low:
        return {task: 1.0 for task in values}
    return {task: (value-low)/(high-low) for task, value in values.items()}


def _criticality_weights(context, tasks):
    """Return combined structural and deadline-aware criticality."""
    if not bool(context.get("cpo_criticality_guidance", True)):
        return {int(task): 1.0 for task in tasks}
    ranks = (
        context.get("task_structural_criticality")
        or context.get("task_rank", context.get("global_ranks", {}))
        or {}
    )
    values = {
        int(task): float(ranks.get(int(task), ranks.get(str(int(task)), 0.0)))
        for task in tasks
    }
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high <= low:
        # Equal structural ranks are neutral evidence, not an early exit:
        # deadline pressure must still distinguish applications.
        structural = {task: 1.0 for task in values}
    else:
        structural = {
            task: 0.25 + 0.75 * (value - low) / (high - low)
            for task, value in values.items()
        }

    deadline = _deadline_urgency_weights(context, tasks)
    if bool(context.get("cpo_deadline_guidance", False)):
        return {
            task: 0.7 * structural.get(task, 1.0) + 0.3 * deadline.get(task, 0.0)
            for task in values
        }
    return structural


def hamming_distance(left, right):
    a, b = provider_map(left), provider_map(right)
    tasks = set(a) | set(b)
    return sum(a.get(task) != b.get(task) for task in tasks)


def _weighted_sample_without_replacement(weighted_items, count, rng):
    pool = [(item, max(1e-12, float(weight))) for item, weight in weighted_items]
    result = []
    for _ in range(min(max(0, int(count)), len(pool))):
        threshold = rng.random() * sum(weight for _item, weight in pool)
        cumulative = 0.0
        chosen = len(pool) - 1
        for index, (_item, weight) in enumerate(pool):
            cumulative += weight
            if threshold <= cumulative:
                chosen = index
                break
        result.append(pool.pop(chosen)[0])
    return result


def _changed_task_budget(dimension, progress, *, exploration):
    """Dimension-safe categorical radius.

    ``sqrt(D)`` is a useful continuous heuristic, but for the project's
    600+ categorical coordinates it changed dozens of assignments at once.
    The logarithmic exploration radius and 1--4 coordinate exploitation
    radius preserve multi-scale search without destroying a strong greedy
    schedule in one proposal.
    """
    dimension = max(1, int(dimension))
    progress = max(0.0, min(1.0, float(progress)))
    if exploration:
        maximum = max(2, int(math.ceil(math.log2(dimension + 1))))
        radius = 2.0 + (maximum - 2.0) * (1.0 - progress)
    else:
        radius = 1.0 + 3.0 * (1.0 - progress)
    return max(1, min(dimension, int(math.ceil(radius))))


def _choose_memory_or_random(task, alternatives, memory, rng, exploration_floor):
    alternatives = list(dict.fromkeys(int(value) for value in alternatives))
    if not alternatives:
        return None
    if memory is None or rng.random() < float(exploration_floor):
        return int(rng.choice(alternatives))
    scores = memory.provider_scores(int(task), alternatives)
    weights = [(provider, 0.05 + float(scores.get(provider, 0.0))) for provider in alternatives]
    return int(_weighted_sample_without_replacement(weights, 1, rng)[0])


def _choose_model_guided_provider(
    task,
    alternatives,
    context,
    memory,
    rng,
    progress,
    *,
    exploration_floor,
):
    """Use a bounded paper-model prior only to propose an exploitation move."""
    alternatives = list(dict.fromkeys(int(value) for value in alternatives))
    if not alternatives:
        return None
    if not bool(context.get("cpo_model_guidance", True)):
        return _choose_memory_or_random(
            task, alternatives, memory, rng, exploration_floor
        )
    table = context.get("task_provider_model_prior", {})
    model_scores = table.get(int(task), table.get(str(int(task)), {}))
    model_scores = model_scores if isinstance(model_scores, dict) else {}
    available_scores = {
        provider: max(
            0.0,
            float(model_scores.get(provider, model_scores.get(str(provider), 0.0))),
        )
        for provider in alternatives
    }
    max_probability = float(context.get("cpo_model_guidance_max_probability", 0.35))
    progress = max(0.0, min(1.0, float(progress)))
    guidance_probability = max(0.0, min(0.50, max_probability)) * (
        0.55 + 0.45 * progress
    )
    if (
        any(value > 0.0 for value in available_scores.values())
        and rng.random() < guidance_probability
    ):
        weights = [
            (provider, 0.05 + available_scores[provider])
            for provider in alternatives
        ]
        return int(_weighted_sample_without_replacement(weights, 1, rng)[0])
    return _choose_memory_or_random(
        task,
        alternatives,
        memory,
        rng,
        exploration_floor,
    )


def _force_change(candidate, reference, context, rng, memory=None):
    if solution_key(candidate) != solution_key(reference):
        return context.repair_solution(candidate)
    child = [tuple(int(value) for value in gene[:3]) for gene in reference]
    mutable = []
    for index, (task, provider, _position) in enumerate(child):
        alternatives = [value for value in valid_providers(context, task) if value != provider]
        if alternatives:
            mutable.append((index, task, alternatives))
    if not mutable:
        return context.repair_solution(child)
    index, task, alternatives = rng.choice(mutable)
    provider = _choose_memory_or_random(task, alternatives, memory, rng, 0.35)
    child[index] = (int(task), int(provider), int(index))
    return context.repair_solution(child)


def _exploration_task_weights(current, best, peer_a, peer_b, context):
    """Prioritize disagreement while protecting settled critical tasks.

    Critical DAG coordinates are expensive to disturb indiscriminately: a
    poor provider change can delay every descendant.  During the two global
    CPO defenses we therefore explore low-criticality coordinates first, and
    let population disagreement override that protection.  The exploitative
    defenses retain their existing high-criticality focus.

    When criticality guidance is disabled the expression is exactly the
    former neutral weighting, preserving the DCPO-base ablation.
    """
    current_map = provider_map(current)
    maps = [provider_map(best), provider_map(peer_a), provider_map(peer_b)]
    rank = _criticality_weights(context, current_map)
    guided = bool(context.get("cpo_criticality_guidance", True))
    rows = []
    for task, provider in current_map.items():
        disagreement = sum(mapping.get(task, provider) != provider for mapping in maps)
        if guided:
            # rank is in [0.25, 1.0].  Reversing only this bounded term keeps
            # exploration broad without allowing criticality to overwhelm
            # the stronger population-disagreement evidence.
            structural_weight = 1.25 - rank.get(task, 1.0)
        else:
            structural_weight = 1.0
        rows.append(
            (
                int(task),
                1.0 + 1.15 * disagreement + 0.85 * structural_weight,
            )
        )
    return rows


def _elite_provider_profile(elite_maps, task, domain, limit=8):
    """Return rank-weighted provider votes from the best population rows."""
    domain = set(int(value) for value in domain)
    votes = {}
    for rank, mapping in enumerate(list(elite_maps or [])[: max(1, int(limit))]):
        provider = mapping.get(int(task))
        if provider in domain:
            votes[int(provider)] = votes.get(int(provider), 0.0) + 1.0 / (1.0 + rank)
    return votes


def _elite_disagreement(elite_maps, task, current_provider, domain, limit=8):
    profile = _elite_provider_profile(elite_maps, task, domain, limit=limit)
    if not profile:
        return 0.0
    total = sum(profile.values())
    return max(0.0, 1.0 - profile.get(int(current_provider), 0.0) / max(1e-12, total))


def sight_defense(current, best, peer, context, progress, rng, memory=None):
    """Long-range visual defense: peer-disagreement plus feasible random sighting."""
    child = [tuple(int(value) for value in gene[:3]) for gene in current]
    current_map, best_map, peer_map = provider_map(current), provider_map(best), provider_map(peer)
    budget = _changed_task_budget(len(child), progress, exploration=True)
    tasks = _weighted_sample_without_replacement(
        _exploration_task_weights(current, best, peer, peer, context), budget, rng
    )
    index_by_task = {int(gene[0]): index for index, gene in enumerate(child)}
    for task in tasks:
        current_provider = current_map[task]
        domain = valid_providers(context, task)
        suggested = [
            provider for provider in (peer_map.get(task), best_map.get(task))
            if provider in domain and provider != current_provider
        ]
        if suggested and rng.random() < 0.65:
            provider = int(rng.choice(suggested))
        else:
            provider = _choose_memory_or_random(
                task, [value for value in domain if value != current_provider], memory, rng, 0.60
            )
        if provider is not None:
            index = index_by_task[task]
            child[index] = (int(task), int(provider), int(index))
    return _force_change(child, current, context, rng, memory)


def sound_defense(current, best, peer_a, peer_b, context, progress, rng, memory=None):
    """Acoustic warning: multi-peer categorical recombination."""
    child = [tuple(int(value) for value in gene[:3]) for gene in current]
    current_map = provider_map(current)
    maps = [provider_map(peer_a), provider_map(peer_b), provider_map(best)]
    budget = _changed_task_budget(len(child), progress, exploration=True)
    tasks = _weighted_sample_without_replacement(
        _exploration_task_weights(current, best, peer_a, peer_b, context), budget, rng
    )
    index_by_task = {int(gene[0]): index for index, gene in enumerate(child)}
    for task in tasks:
        domain = valid_providers(context, task)
        candidates = [
            mapping.get(task) for mapping in maps
            if mapping.get(task) in domain and mapping.get(task) != current_map[task]
        ]
        if candidates and rng.random() < 0.80:
            counts = {provider: candidates.count(provider) for provider in set(candidates)}
            provider = _weighted_sample_without_replacement(counts.items(), 1, rng)[0]
        else:
            provider = _choose_memory_or_random(
                task, [value for value in domain if value != current_map[task]], memory, rng, 0.45
            )
        if provider is not None:
            index = index_by_task[task]
            child[index] = (int(task), int(provider), int(index))
    return _force_change(child, current, context, rng, memory)


def odor_defense(current, best, context, progress, rng, memory=None, elite_maps=None):
    """Olfactory defense with a bounded service-affinity neighborhood.

    The anchor follows the best scent or successful memory.  At most a few
    tasks of the same service type follow the anchor when the chosen provider
    is feasible for them.  This exposes cache-reuse combinations while the
    unchanged objective still decides whether queueing and radio costs make
    the block move worthwhile.
    """
    child = [tuple(int(value) for value in gene[:3]) for gene in current]
    current_map, best_map = provider_map(current), provider_map(best)
    budget = _changed_task_budget(len(child), progress, exploration=False)
    rank = _criticality_weights(context, current_map)
    anchors = _weighted_sample_without_replacement(
        [
            (
                task,
                rank.get(task, 1.0)
                * (2.0 if best_map.get(task) != provider else 0.35)
                * (1.0 + 0.35 * _energy_opportunity(context, task)),
            )
            for task, provider in current_map.items()
        ],
        1,
        rng,
    )
    if not anchors:
        return context.repair_solution(child)
    anchor = int(anchors[0])
    task_types = context.get("task_type_ids", {}) or {}
    anchor_type = task_types.get(anchor, task_types.get(str(anchor)))
    ordered_tasks = list(current_map)
    anchor_position = ordered_tasks.index(anchor)
    # Cache placement can only benefit tasks that have not yet been scheduled.
    # Coupling an anchor to an earlier same-service task spent search effort on
    # a reuse that the sequential evaluator can no longer realize.
    same_service = [
        int(task)
        for task in ordered_tasks[anchor_position + 1 :]
        if task != anchor
        and anchor_type is not None
        and task_types.get(int(task), task_types.get(str(int(task)))) == anchor_type
    ] if bool(context.get("cpo_cache_coupling", True)) else []
    # A useful cache-reuse proposal needs the placement anchor plus at least
    # one later request, even after exploitation contracts late in the run.
    extra_count = max(0, budget - 1)
    if same_service:
        extra_count = max(1, extra_count)
    extra = _weighted_sample_without_replacement(
        [(task, 0.40 + rank.get(task, 1.0)) for task in same_service],
        extra_count,
        rng,
    )
    tasks = [anchor] + extra
    index_by_task = {int(gene[0]): index for index, gene in enumerate(child)}
    anchor_target = best_map.get(anchor)
    # The cache block must follow the provider that the anchor actually
    # selects.  In the previous implementation, later same-service tasks kept
    # following ``best_map[anchor]`` even when cache affinity or success/model
    # memory moved the anchor somewhere else.  That split the intended block
    # across providers and could not realise the proposed forward reuse.
    selected_anchor_provider = None
    for task in tasks:
        current_provider = current_map[task]
        domain = valid_providers(context, task)
        elite_provider = best_map.get(task)
        shared_target = (
            selected_anchor_provider
            if task != anchor and selected_anchor_provider is not None
            else anchor_target
        )
        shared_provider = shared_target if shared_target in domain else None
        cache_affinity = context.get("task_provider_cache_affinity", {}) or {}
        task_affinity = cache_affinity.get(int(task), cache_affinity.get(str(int(task)), {}))
        affinity_target = None
        if bool(context.get("cpo_cache_coupling", True)) and isinstance(task_affinity, dict):
            feasible_affinity = {
                int(candidate): float(value)
                for candidate, value in task_affinity.items()
                if int(candidate) in domain and int(candidate) != int(current_provider)
            }
            if feasible_affinity:
                best_affinity = max(feasible_affinity.values())
                if best_affinity > 0.0:
                    affinity_target = int(rng.choice([
                        candidate for candidate, value in feasible_affinity.items()
                        if abs(value - best_affinity) <= 1e-12
                    ]))
        if (
            task != anchor
            and selected_anchor_provider is not None
            and selected_anchor_provider in domain
        ):
            # Keep the selected forward reuse block on the anchor's actual
            # provider. The unchanged evaluator accepts or rejects the move.
            provider = int(selected_anchor_provider)
        elif affinity_target is not None and rng.random() < 0.45:
            provider = affinity_target
        elif shared_provider is not None and shared_provider != current_provider and rng.random() < 0.55:
            provider = int(shared_provider)
        elif elite_provider in domain and elite_provider != current_provider and rng.random() < 0.80:
            provider = int(elite_provider)
        else:
            alternatives = [value for value in domain if value != current_provider]
            elite_votes = _elite_provider_profile(elite_maps, task, alternatives)
            if elite_votes and rng.random() < 0.70:
                provider = int(_weighted_sample_without_replacement(elite_votes.items(), 1, rng)[0])
            else:
                provider = _choose_model_guided_provider(
                    task,
                    alternatives,
                    context,
                    memory,
                    rng,
                    progress,
                    exploration_floor=0.20,
                )
        if provider is not None:
            index = index_by_task[task]
            child[index] = (int(task), int(provider), int(index))
            if task == anchor:
                selected_anchor_provider = int(provider)
        elif task == anchor:
            selected_anchor_provider = int(current_provider)
    return _force_change(child, current, context, rng, memory)


def physical_attack(current, best, context, progress, rng, memory=None, elite_maps=None):
    """Close physical attack around the archive with elite disagreement.

    A one-coordinate move is retained.  Critical tasks on which strong
    solutions disagree are sampled more often, and elite provider votes are
    preferred before success memory and uniform exploration.
    """
    base = best if rng.random() < 0.88 else current
    child = [tuple(int(value) for value in gene[:3]) for gene in base]
    rank = _criticality_weights(context, [int(gene[0]) for gene in child])
    mutable = []
    for index, (task, provider, _position) in enumerate(child):
        domain = valid_providers(context, task)
        alternatives = [value for value in domain if value != provider]
        if alternatives:
            disagreement = _elite_disagreement(
                elite_maps, task, provider, domain
            )
            weight = (
                0.25
                + rank.get(task, 1.0)
                + 2.25 * disagreement
                + 0.75 * _energy_opportunity(context, task)
            )
            mutable.append(((index, task, alternatives), weight))
    if not mutable:
        return context.repair_solution(child)
    index, task, alternatives = _weighted_sample_without_replacement(mutable, 1, rng)[0]
    elite_votes = _elite_provider_profile(elite_maps, task, alternatives)
    if elite_votes and rng.random() < 0.68:
        provider = int(_weighted_sample_without_replacement(elite_votes.items(), 1, rng)[0])
    else:
        provider = _choose_model_guided_provider(
            task,
            alternatives,
            context,
            memory,
            rng,
            progress,
            exploration_floor=0.10 + 0.12 * (1.0 - progress),
        )
    child[index] = (int(task), int(provider), int(index))
    return _force_change(child, current, context, rng, memory)


def choose_defense(progress, stagnation, memory, rng):
    """NFE-stage adaptive selection across all four CPO defenses.

    Exploitation always receives meaningful budget because initialization is
    already 75% paper-greedy.  Success credit adjusts, but cannot collapse,
    any defense probability.
    """
    progress = max(0.0, min(1.0, float(progress)))
    base = {
        "sight": 0.24 * (1.0 - progress) + 0.06,
        "sound": 0.19 * (1.0 - progress) + 0.06,
        "odor": 0.18 + 0.08 * progress,
        "physical_attack": 0.27 + 0.30 * progress,
    }
    escape = min(0.12, 0.025 * max(0, int(stagnation)))
    base["sight"] += escape
    base["physical_attack"] += 0.50 * escape
    learned = memory.strategy_weights(DEFENSE_NAMES) if memory is not None else {
        name: 0.25 for name in DEFENSE_NAMES
    }
    weights = {
        name: max(0.04, 0.78 * base[name] + 0.22 * learned.get(name, 0.25))
        for name in DEFENSE_NAMES
    }
    return _weighted_sample_without_replacement(weights.items(), 1, rng)[0]


def build_defense_schedule(count, progress, stagnation, memory, rng):
    """Create a shuffled, non-starving defense schedule for one generation."""
    count = max(0, int(count))
    if count == 0:
        return []
    schedule = [choose_defense(progress, stagnation, memory, rng) for _ in range(count)]
    minimum_ratios = {
        "sight": 0.10,
        "sound": 0.10,
        "odor": 0.16,
        "physical_attack": 0.28,
    }
    required = {
        name: min(count, int(math.floor(count * ratio)))
        for name, ratio in minimum_ratios.items()
    }
    for name in DEFENSE_NAMES:
        deficit = max(0, required[name] - schedule.count(name))
        for _ in range(deficit):
            donors = [
                candidate
                for candidate in DEFENSE_NAMES
                if schedule.count(candidate) > required[candidate]
            ]
            if not donors:
                break
            donor = max(donors, key=lambda candidate: schedule.count(candidate) - required[candidate])
            schedule[schedule.index(donor)] = name
    rng.shuffle(schedule)
    return schedule


def generate_candidate(strategy, current, best, population, context, progress, rng, memory=None):
    peers = [solution for solution in population if solution_key(solution) != solution_key(current)]
    peer_a = rng.choice(peers or population)
    peer_b = rng.choice(peers or population)
    if strategy == "sight":
        return sight_defense(current, best, peer_a, context, progress, rng, memory)
    if strategy == "sound":
        return sound_defense(current, best, peer_a, peer_b, context, progress, rng, memory)
    if strategy == "odor":
        elite_maps = [provider_map(solution) for solution in list(population or [])[:8]]
        return odor_defense(current, best, context, progress, rng, memory, elite_maps)
    if strategy == "physical_attack":
        elite_maps = [provider_map(solution) for solution in list(population or [])[:8]]
        return physical_attack(current, best, context, progress, rng, memory, elite_maps)
    raise ValueError(f"Unknown CPO defense strategy: {strategy}")
