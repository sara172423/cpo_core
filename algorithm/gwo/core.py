from __future__ import annotations

import copy
import random

from .initial_population import create_initial_population
from .memory import initialize_pheromone, update_pheromone
from .operators import (
    generate_alpha_neighborhood_children,
    generate_adaptive_children,
    generate_escape_children,
    hamming_distance,
)


STAGNATION_ESCAPE_AFTER = 5
ESCAPE_FRACTION = 0.15


def _set_context_flag(context, key, value):
    """Set an optimizer-local flag on mapping and object adapters safely."""
    try:
        context[key] = value
    except (AttributeError, TypeError):
        setattr(context, key, value)


def _solution_key(solution):
    return tuple(
        (int(gene[0]), int(gene[1]))
        for gene in solution
        if isinstance(gene, (tuple, list)) and len(gene) >= 2
    )


def _sort_unique(rows):
    result = []
    seen = set()
    for solution, score in sorted(rows, key=lambda row: float(row[1]), reverse=True):
        key = _solution_key(solution)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append((solution, float(score)))
    return result


def _mean_pairwise_hamming(rows) -> float:
    if len(rows) < 2:
        return 0.0
    dimension = max(1, len(rows[0][0]))
    total = 0.0
    pairs = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            total += hamming_distance(rows[i][0], rows[j][0]) / float(dimension)
            pairs += 1
    return total / float(max(1, pairs))


def _adaptive_a(iteration: int, iterations: int, rows, stagnation_count: int) -> float:
    """Nonlinear GWO decay with bounded diversity/stagnation correction."""
    if iterations <= 0:
        return 0.0
    progress = float(max(0, iteration - 1)) / float(max(1, iterations))
    base = 2.0 * (1.0 - progress)
    diversity = _mean_pairwise_hamming(rows)
    low_diversity = max(0.0, min(1.0, 0.35 - diversity))
    stagnation = min(1.0, float(max(0, stagnation_count)) / 10.0)
    return max(
        0.0,
        min(2.0, base + 0.35 * low_diversity + 0.25 * stagnation),
    )


def run_gwo_aco(
    context,
    population_size=20,
    iterations=50,
    initial_population=None,
    *,
    stagnation_escape_after=STAGNATION_ESCAPE_AFTER,
    escape_fraction=ESCAPE_FRACTION,
    pheromone_elite_ratio=0.20,
    pheromone_evaporation=0.10,
    use_pheromone=False,
    use_rank_guidance=False,
    use_cache_guidance=False,
    max_function_evaluations=None,
):
    """Run a categorical GWO baseline with optional legacy enhancements.

    Deliberately removed from the search loop:
      * historical leader archive,
      * quality-diversity survival scoring,
      * separate refinement population,
      * generation-level regional RSU cache memory.

    Each generation creates only ``N`` children.  Parent scores are reused, so
    at most ``N`` new objective evaluations are needed per generation before
    memoization/deduplication.  The benchmark evaluator remains the sole source
    of the final Q fitness.
    """
    context = copy.deepcopy(context)
    _set_context_flag(context, "gwo_rank_guidance", bool(use_rank_guidance))
    _set_context_flag(context, "gwo_cache_guidance", bool(use_cache_guidance))
    seed = context.get("seed")
    rng = random.Random(seed)

    size = max(3, int(population_size))
    iterations = max(0, int(iterations))
    evaluation_budget = (
        None
        if max_function_evaluations is None
        else max(1, int(max_function_evaluations))
    )
    if evaluation_budget is not None and evaluation_budget < size:
        raise ValueError(
            "max_function_evaluations must be at least population_size so the "
            "initial population can be evaluated"
        )
    population = initial_population or create_initial_population(context, size, rng=rng)
    population = [context.repair_solution(solution) for solution in population]
    population = [solution for solution in population if solution][:size]
    if len(population) < size:
        raise RuntimeError(f"Initial population is {len(population)}, expected {size}")

    # Joint benchmark adapters expose these as properties while their ``get``
    # method intentionally delegates to the immutable problem context.  Read
    # the explicit optimizer contract first so greedy-constructor evaluations
    # are neither recomputed nor omitted from the fair NFE budget.
    initial_memo = getattr(context, "initial_evaluation_memo", None)
    if initial_memo is None:
        initial_memo = context.get("initial_evaluation_memo", {})
    initial_memo = initial_memo or {}
    evaluation_cache = {
        key: float(score)
        for key, score in initial_memo.items()
    }
    initial_count = getattr(context, "initial_function_evaluations", None)
    if initial_count is None:
        initial_count = context.get("initial_function_evaluations", 0)
    function_evaluations_total = max(
        len(evaluation_cache),
        int(initial_count or 0),
    )

    def evaluate_batch(batch):
        nonlocal function_evaluations_total
        rows = []
        for raw in batch:
            solution = context.repair_solution(raw)
            if not solution:
                continue
            key = _solution_key(solution)
            if key not in evaluation_cache:
                if (
                    evaluation_budget is not None
                    and function_evaluations_total >= evaluation_budget
                ):
                    break
                evaluation_cache[key] = float(context.evaluate(solution))
                function_evaluations_total += 1
            rows.append((solution, float(evaluation_cache[key])))
        return _sort_unique(rows)

    evaluated = evaluate_batch(population)
    if len(evaluated) < size:
        raise RuntimeError(
            f"Initial population has only {len(evaluated)} unique evaluated wolves; expected {size}"
        )
    evaluated = evaluated[:size]

    task_order = [int(task) for task in context.task_order]
    pheromone: dict[tuple[int, int], float] = {}
    if use_pheromone:
        initialize_pheromone(
            pheromone,
            {task: context.valid_provider(task) for task in task_order},
        )
        update_pheromone(
            pheromone,
            evaluated,
            elite_ratio=float(pheromone_elite_ratio),
            evaporation=max(0.0, float(pheromone_evaporation) - 0.02),
        )

    best_solution, best_score = evaluated[0]
    history = []
    stagnation_count = 0

    def record(iteration: int):
        history.append(
            {
                "iteration": int(iteration),
                "best_total_efficiency": float(best_score),
                "population_total_efficiencies": [float(score) for _, score in evaluated],
                "population_mean_efficiency": float(
                    sum(float(score) for _, score in evaluated) / float(max(1, len(evaluated)))
                ),
                "population_diversity_mean": float(_mean_pairwise_hamming(evaluated)),
                "function_evaluations": int(function_evaluations_total),
            }
        )

    record(0)

    for iteration in range(1, iterations + 1):
        a = _adaptive_a(iteration, iterations, evaluated, stagnation_count)
        alpha, beta, delta = evaluated[0][0], evaluated[1][0], evaluated[2][0]
        current_population = [solution for solution, _ in evaluated]

        trigger_escape = stagnation_count >= max(1, int(stagnation_escape_after))
        escape_count = (
            max(1, int(round(size * max(0.0, min(0.50, float(escape_fraction))))))
            if trigger_escape
            else 0
        )
        local_count = max(1, int(round(size * 0.20)))
        ordinary_count = max(0, size - escape_count - local_count)

        children = generate_adaptive_children(
            current_population,
            alpha,
            beta,
            delta,
            pheromone,
            context,
            a=float(a),
            count=ordinary_count,
            rng=rng,
        )
        children.extend(
            generate_alpha_neighborhood_children(
                alpha,
                context,
                count=local_count,
                rng=rng,
                pheromone=pheromone,
                beta=beta,
                delta=delta,
                a=float(a),
            )
        )
        if escape_count:
            children.extend(
                generate_escape_children(
                    current_population,
                    context,
                    a=float(a),
                    count=escape_count,
                    rng=rng,
                )
            )

        # Defensive fill without creating a second operator population.
        fill_attempts = 0
        while len(children) < size and fill_attempts < size * 4:
            fill_attempts += 1
            source = rng.choice(current_population)
            children.extend(
                generate_adaptive_children(
                    [source],
                    alpha,
                    beta,
                    delta,
                    pheromone,
                    context,
                    a=float(a),
                    count=1,
                    rng=rng,
                )
            )
        if len(children) < size:
            raise RuntimeError(
                f"Could generate only {len(children)} children; expected {size}"
            )
        children = children[:size]

        child_rows = evaluate_batch(children)
        selection_pool = _sort_unique(evaluated + child_rows)
        if len(selection_pool) < size:
            raise RuntimeError(
                f"Selection pool has {len(selection_pool)} unique wolves; expected at least {size}"
            )
        evaluated = selection_pool[:size]

        previous_best = float(best_score)
        best_solution, best_score = evaluated[0]
        if float(best_score) > previous_best + 1e-4:
            stagnation_count = 0
        else:
            stagnation_count = 0 if trigger_escape else stagnation_count + 1

        if use_pheromone:
            update_pheromone(
                pheromone,
                evaluated,
                evaporation=float(pheromone_evaporation) + 0.04 * (1.0 - a / 2.0),
                elite_ratio=float(pheromone_elite_ratio),
                stagnation=stagnation_count,
            )
        record(iteration)
        if (
            evaluation_budget is not None
            and function_evaluations_total >= evaluation_budget
        ):
            break

    return context.repair_solution(best_solution), float(best_score), history
