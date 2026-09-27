from __future__ import annotations

import copy
import math
import random
from algorithm.optimizer_common import (
    EvaluationBudgetReached,
    ObjectiveEvaluationProtocol,
    assignment_key,
)

from .initial_population import create_initial_population, create_random_solution
from .memory import ServiceAffinityMemory
from .operators import (
    DEFAULT_FRICTION_MAX,
    DEFAULT_FRICTION_MIN,
    DEFAULT_GRAVITY,
    DEFAULT_RAMP_ANGLE_DEGREES,
    DEFAULT_SUBSTITUTION_PROBABILITY,
    adaptive_levy_escape,
    cache_aware_mutation,
    physical_gpc_move,
    rank_guided_discrete_mutation,
    success_memory_mutation,
)


STAGNATION_ESCAPE_AFTER = 5
RESTART_FRACTION = 0.15
ELITE_RATIO = 0.20


def _population_diversity_metrics(population):
    signatures = [assignment_key(solution) for solution in population]
    signatures = [signature for signature in signatures if signature]
    if not signatures:
        return {"population_unique_count": 0, "population_mean_hamming": 0.0}

    dimension = max(1, len(signatures[0]))
    distances = []
    for left in range(len(signatures)):
        for right in range(left + 1, len(signatures)):
            a = dict(signatures[left])
            b = dict(signatures[right])
            tasks = set(a) | set(b)
            distances.append(
                sum(a.get(task) != b.get(task) for task in tasks) / float(dimension)
            )
    return {
        "population_unique_count": int(len(set(signatures))),
        "population_mean_hamming": float(
            sum(distances) / len(distances) if distances else 0.0
        ),
    }


def _deduplicate_ranked(rows):
    result = []
    seen = set()
    for solution, score in sorted(rows, key=lambda item: float(item[1]), reverse=True):
        key = assignment_key(solution)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append((copy.deepcopy(solution), float(score)))
    return result


def _local_pharaoh_candidate(
    pharaoh, context, rng, memory=None, *, problem_guidance=False
):
    """One feasible single-task neighborhood move around the best worker."""
    result = [tuple(int(value) for value in gene[:3]) for gene in pharaoh]
    if problem_guidance:
        learned = success_memory_mutation(
            result, context, rng, memory,
            probability=1.0, exploration_floor=0.10,
        )
        if assignment_key(learned) != assignment_key(result):
            return context.repair_solution(learned)
        guided = cache_aware_mutation(result, context, rng, probability=1.0)
        if assignment_key(guided) != assignment_key(result):
            return context.repair_solution(guided)
    mutable = []
    for index, (task, provider, _position) in enumerate(result):
        alternatives = [
            int(value)
            for value in context.valid_provider(int(task))
            if int(value) != int(provider)
        ]
        if alternatives:
            mutable.append((index, alternatives))
    if not mutable:
        return result
    index, alternatives = rng.choice(mutable)
    task, _provider, position = result[index]
    result[index] = (int(task), int(rng.choice(alternatives)), int(position))
    return context.repair_solution(result)


def _restart_from_pharaoh(
    pharaoh,
    context,
    rng,
    stagnation_counter,
    memory=None,
    *,
    problem_guidance=False,
):
    """Generate a bounded long jump instead of a destructive full random nest."""
    candidate = adaptive_levy_escape(
        pharaoh,
        context,
        rng,
        max(STAGNATION_ESCAPE_AFTER, int(stagnation_counter)),
        probability=1.0,
    )
    candidate = rank_guided_discrete_mutation(
        candidate,
        context,
        rng,
        mutation_probability=1.0,
        rank_guided=bool(problem_guidance),
    )
    if problem_guidance:
        candidate = success_memory_mutation(
            candidate, context, rng, memory,
            probability=0.65, exploration_floor=0.35,
        )
    candidate = context.repair_solution(candidate)
    if assignment_key(candidate) == assignment_key(pharaoh):
        candidate = _local_pharaoh_candidate(
            pharaoh, context, rng, memory,
            problem_guidance=problem_guidance,
        )
    return candidate


def run_gpc(
    context,
    population_size=30,
    iterations=20,
    seed=None,
    initial_population=None,
    *,
    gravity=DEFAULT_GRAVITY,
    ramp_angle_degrees=DEFAULT_RAMP_ANGLE_DEGREES,
    friction_min=DEFAULT_FRICTION_MIN,
    friction_max=DEFAULT_FRICTION_MAX,
    substitution_probability=DEFAULT_SUBSTITUTION_PROBABILITY,
    service_memory_enabled=False,
    problem_guidance=False,
    service_memory_weight=0.65,
    service_memory_evaporation=0.08,
    max_function_evaluations=None,
):
    """Run adaptive discrete GPC on the task-provider assignment space.

    The reference controls ``G``, ``Theta``, ``MuMin``, ``MuMax`` and ``pSS``
    are retained. Physical travel is normalized into categorical substitution
    intensity; greedy acceptance, a Pharaoh archive, rank-aware mutation, and
    bounded restarts adapt it to the constrained discrete problem.
    """
    context = copy.deepcopy(context)
    if seed is None:
        seed = context.get("seed")
    if seed is not None:
        context["seed"] = int(seed)
    rng = random.Random(seed)

    if not context.task_order:
        raise RuntimeError("GPC requires a non-empty ranked task order")
    size = max(2, int(population_size))
    generations = max(0, int(iterations))
    raw_population = list(
        initial_population or create_initial_population(context, size, rng)
    )
    population = []
    seen = set()
    for raw in raw_population:
        solution = context.repair_solution(raw)
        key = assignment_key(solution)
        if key and key not in seen:
            seen.add(key)
            population.append(solution)
        if len(population) >= size:
            break
    attempts = 0
    while len(population) < size and attempts < size * 50:
        attempts += 1
        solution = create_random_solution(context, rng)
        key = assignment_key(solution)
        if key and key not in seen:
            seen.add(key)
            population.append(solution)
    if len(population) < size:
        raise RuntimeError(
            f"Initial GPC population has {len(population)} unique solutions; expected {size}"
        )

    evaluations = ObjectiveEvaluationProtocol(
        context,
        population_size=size,
        max_function_evaluations=max_function_evaluations,
    )

    def evaluate(solution):
        result = evaluations.evaluate(solution)
        return result.solution, result.score

    scored = _deduplicate_ranked(evaluate(solution) for solution in population)
    if len(scored) < size:
        raise RuntimeError("GPC initialization collapsed after repair")
    scored = scored[:size]
    global_pharaoh = copy.deepcopy(scored[0][0])
    global_score = float(scored[0][1])
    service_memory = ServiceAffinityMemory(
        context, weight=service_memory_weight, evaporation=service_memory_evaporation
    ) if bool(service_memory_enabled) else None

    history = []
    stagnation_counter = 0
    total_accepted_moves = 0
    total_abandoned_workers = 0
    total_generated_replacements = 0

    def record(
        iteration,
        movement_rows=None,
        mutation_probability=0.0,
        local_refinement_trials=0,
    ):
        movement_rows = movement_rows or []
        population_now = [solution for solution, _score in scored]
        history.append(
            {
                "iteration": int(iteration),
                "best_total_efficiency": float(global_score),
                "current_pharaoh_score": float(scored[0][1]),
                "global_pharaoh_score": float(global_score),
                "population_total_efficiencies": [
                    float(score) for _solution, score in scored
                ],
                "population_mean_efficiency": float(
                    sum(float(score) for _solution, score in scored) / len(scored)
                ),
                "function_evaluations": int(evaluations.count),
                "function_evaluations_total": int(evaluations.count),
                "accepted_moves": int(total_accepted_moves),
                "abandoned_workers": int(total_abandoned_workers),
                "generated_replacements": int(total_generated_replacements),
                "stagnation_count": int(stagnation_counter),
                "mutation_probability": float(mutation_probability),
                "mean_substituted_tasks": float(
                    sum(row["substituted_tasks"] for row in movement_rows)
                    / len(movement_rows)
                    if movement_rows
                    else 0.0
                ),
                "local_refinement_trials": int(local_refinement_trials),
                **(service_memory.profile() if service_memory is not None else {
                    "service_memory_successful_updates": 0,
                    "service_memory_changed_coordinates": 0,
                    "service_memory_task_entries": 0,
                    "service_memory_service_entries": 0,
                }),
                **_population_diversity_metrics(population_now),
            }
        )

    record(0)

    for generation in range(1, generations + 1):
        if service_memory is not None:
            service_memory.begin_generation()
        progress = float(generation) / float(max(1, generations))
        previous_global_score = float(global_score)
        diversity = _population_diversity_metrics(
            [solution for solution, _score in scored]
        )["population_mean_hamming"]
        mutation_probability = (
            0.22 * (1.0 - progress)
            + 0.04
            + min(0.18, 0.03 * stagnation_counter)
            + max(0.0, 0.08 - 0.20 * diversity)
        )
        mutation_probability = max(0.03, min(0.45, mutation_probability))
        abandonment_probability = min(0.40, 0.05 + 0.04 * stagnation_counter)

        candidates = [(copy.deepcopy(global_pharaoh), float(global_score))]
        budget_exhausted = False
        movement_rows = []
        for worker, worker_score in scored[1:size]:
            if budget_exhausted:
                candidates.append((copy.deepcopy(worker), float(worker_score)))
                continue
            moved, diagnostics = physical_gpc_move(
                worker,
                global_pharaoh,
                context,
                rng,
                progress=progress,
                gravity=gravity,
                ramp_angle_degrees=ramp_angle_degrees,
                friction_min=friction_min,
                friction_max=friction_max,
                substitution_probability=substitution_probability,
            )
            movement_rows.append(diagnostics)
            moved = rank_guided_discrete_mutation(
                moved,
                context,
                rng,
                mutation_probability,
                rank_guided=bool(problem_guidance),
            )
            if problem_guidance:
                moved = success_memory_mutation(
                    moved, context, rng, service_memory,
                    probability=min(0.45, 0.10 + 0.60 * mutation_probability),
                )
            moved = adaptive_levy_escape(
                moved,
                context,
                rng,
                stagnation_counter,
                probability=min(0.45, 0.12 + 0.04 * stagnation_counter),
            )
            if problem_guidance:
                moved = cache_aware_mutation(
                    moved,
                    context,
                    rng,
                    probability=0.5 * mutation_probability,
                )
            try:
                moved, moved_score = evaluate(moved)
            except EvaluationBudgetReached:
                budget_exhausted = True
                candidates.append((copy.deepcopy(worker), float(worker_score)))
                continue

            if moved_score >= float(worker_score):
                total_accepted_moves += 1
                if service_memory is not None and moved_score > float(worker_score):
                    service_memory.reward(moved, moved_score - float(worker_score), reference=worker)
                candidates.append((moved, moved_score))
            elif rng.random() < abandonment_probability:
                replacement = _restart_from_pharaoh(
                    global_pharaoh,
                    context,
                    rng,
                    stagnation_counter,
                    service_memory,
                    problem_guidance=problem_guidance,
                )
                try:
                    replacement, replacement_score = evaluate(replacement)
                except EvaluationBudgetReached:
                    budget_exhausted = True
                    candidates.append((copy.deepcopy(worker), float(worker_score)))
                else:
                    total_abandoned_workers += 1
                    total_generated_replacements += 1
                    if service_memory is not None and replacement_score > float(worker_score):
                        service_memory.reward(replacement, replacement_score - float(worker_score), reference=worker)
                    candidates.append((replacement, replacement_score))
            else:
                candidates.append((copy.deepcopy(worker), float(worker_score)))

        # A small memetic neighborhood around the Pharaoh improves discrete
        # exploitation without overwhelming the population search budget.
        local_refinement_trials = max(
            1,
            int(round(size * (0.20 + 0.10 * min(1.0, stagnation_counter / 5.0)))),
        )
        completed_local_trials = 0
        if not budget_exhausted:
            for _ in range(local_refinement_trials):
                neighbor = _local_pharaoh_candidate(
                    global_pharaoh,
                    context,
                    rng,
                    service_memory,
                    problem_guidance=problem_guidance,
                )
                try:
                    neighbor, neighbor_score = evaluate(neighbor)
                    candidates.append((neighbor, neighbor_score))
                except EvaluationBudgetReached:
                    budget_exhausted = True
                    break
                if service_memory is not None and neighbor_score > float(global_score):
                    service_memory.reward(neighbor, neighbor_score - float(global_score), reference=global_pharaoh)
                completed_local_trials += 1

        elite_count = max(1, min(size, int(math.ceil(size * ELITE_RATIO))))
        retained_parents = scored if budget_exhausted else scored[:elite_count]
        pool = _deduplicate_ranked(candidates + retained_parents)
        if (
            not budget_exhausted
            and stagnation_counter >= STAGNATION_ESCAPE_AFTER
        ):
            restart_count = max(1, int(math.ceil(size * RESTART_FRACTION)))
            pool = pool[: max(1, size - restart_count)]

        pool_keys = {assignment_key(solution) for solution, _score in pool}
        fill_attempts = 0
        while len(pool) < size and fill_attempts < size * 50:
            fill_attempts += 1
            replacement = _restart_from_pharaoh(
                global_pharaoh,
                context,
                rng,
                stagnation_counter,
                service_memory,
                problem_guidance=problem_guidance,
            )
            key = assignment_key(replacement)
            if not key or key in pool_keys:
                continue
            try:
                replacement, replacement_score = evaluate(replacement)
            except EvaluationBudgetReached:
                budget_exhausted = True
                break
            pool.append((replacement, replacement_score))
            pool_keys.add(key)
            total_generated_replacements += 1
        if len(pool) < size and budget_exhausted:
            pool = _deduplicate_ranked(pool + scored)
        if len(pool) < size:
            raise RuntimeError(
                f"GPC could retain only {len(pool)} unique workers; expected {size}"
            )

        scored = _deduplicate_ranked(pool)[:size]
        if float(scored[0][1]) > float(global_score):
            global_pharaoh = copy.deepcopy(scored[0][0])
            global_score = float(scored[0][1])

        if float(global_score) > previous_global_score + 1e-12:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        record(
            generation,
            movement_rows,
            mutation_probability,
            completed_local_trials,
        )
        if budget_exhausted:
            break

    return context.repair_solution(global_pharaoh), float(global_score), history
