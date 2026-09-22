from __future__ import annotations

import copy
import math
import random

from .initial_population import create_initial_population
from .memory import DefenseSuccessMemory
from .operators import (
    build_defense_schedule,
    choose_defense,
    generate_candidate,
    hamming_distance,
    solution_key,
)


CPR_CYCLES = 2
CPR_MINIMUM_RATIO = 0.80
STAGNATION_RESTART_AFTER = 3
RESTART_FRACTION = 0.15


def _sort_unique(rows):
    result, seen = [], set()
    for solution, score in sorted(rows, key=lambda row: float(row[1]), reverse=True):
        key = solution_key(solution)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append((copy.deepcopy(solution), float(score)))
    return result


def _mean_hamming(rows):
    if len(rows) < 2:
        return 0.0
    dimension = max(1, len(rows[0][0]))
    distances = [
        hamming_distance(rows[left][0], rows[right][0]) / float(dimension)
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
    ]
    return float(sum(distances) / len(distances)) if distances else 0.0


def _search_progress(function_evaluations, initial_evaluations, budget, iteration, iterations):
    """Measure optimizer stage by paid NFE when a fair budget is active."""
    if budget is not None and int(budget) > int(initial_evaluations):
        used = max(0, int(function_evaluations) - int(initial_evaluations))
        available = max(1, int(budget) - int(initial_evaluations))
        return max(0.0, min(1.0, float(used) / float(available)))
    return max(0.0, min(1.0, float(iteration) / float(max(1, iterations))))


def _cyclic_active_size_at_progress(progress, initial_size, minimum_ratio, cycles):
    initial_size = max(2, int(initial_size))
    minimum_size = max(
        2, min(initial_size, int(round(initial_size * float(minimum_ratio))))
    )
    if initial_size == minimum_size:
        return initial_size
    phase = (max(0.0, min(1.0, float(progress))) * max(1, int(cycles))) % 1.0
    return max(
        minimum_size,
        min(
            initial_size,
            int(math.ceil(initial_size - (initial_size - minimum_size) * phase)),
        ),
    )


def run_cpo(
    context,
    population_size=50,
    iterations=14,
    seed=None,
    initial_population=None,
    *,
    cpr_cycles=CPR_CYCLES,
    cpr_minimum_ratio=CPR_MINIMUM_RATIO,
    stagnation_restart_after=STAGNATION_RESTART_AFTER,
    restart_fraction=RESTART_FRACTION,
    memory_evaporation=0.10,
    provider_memory_weight=0.65,
    criticality_guidance=True,
    cache_coupling=True,
    success_memory=True,
    model_guidance=True,
    deadline_guidance=True,
    max_function_evaluations=None,
):
    """Run the discrete CPO family in the task-provider search space.

    The four defensive mechanisms and cyclic population reduction are retained
    from CPO.  Continuous displacements are replaced by domain-valid
    categorical neighborhoods; this is explicitly an adapted algorithm, not a
    source-exact continuous CPO implementation.  The four guidance switches
    define reproducible ablations of the proposed method; none changes the
    objective, feasibility rules, repair operator, or evaluation budget.
    """
    context = copy.deepcopy(context)
    context["cpo_criticality_guidance"] = bool(criticality_guidance)
    context["cpo_cache_coupling"] = bool(cache_coupling)
    context["cpo_model_guidance"] = bool(model_guidance)
    context["cpo_success_memory"] = bool(success_memory)
    context["cpo_deadline_guidance"] = bool(deadline_guidance)
    seed = context.get("seed") if seed is None else seed
    if seed is not None:
        context["seed"] = int(seed)
    rng = random.Random(seed)
    size = max(2, int(population_size))
    iterations = max(0, int(iterations))
    budget = None if max_function_evaluations is None else max(1, int(max_function_evaluations))
    if budget is not None and budget < size:
        raise ValueError(
            "max_function_evaluations must be at least population_size so the initial population can be evaluated"
        )

    raw_population = list(initial_population or create_initial_population(context, size, rng))
    population, seen = [], set()
    for raw in raw_population:
        solution = context.repair_solution(raw)
        key = solution_key(solution)
        if key and key not in seen:
            seen.add(key)
            population.append(solution)
        if len(population) >= size:
            break
    if len(population) < size:
        raise RuntimeError(f"Initial CPO population has {len(population)} unique solutions; expected {size}")

    initial_memo = context.get("initial_evaluation_memo", {}) or {}
    evaluation_cache = {key: float(value) for key, value in initial_memo.items()}
    function_evaluations = max(
        len(evaluation_cache), int(context.initial_function_evaluations or 0)
    )

    def evaluate(raw):
        nonlocal function_evaluations
        solution = context.repair_solution(raw)
        key = solution_key(solution)
        if not key:
            return None
        if key not in evaluation_cache:
            if budget is not None and function_evaluations >= budget:
                return None
            evaluation_cache[key] = float(context.evaluate_solution(solution))
            function_evaluations += 1
        return solution, float(evaluation_cache[key])

    evaluated = _sort_unique(row for row in (evaluate(solution) for solution in population) if row)
    if len(evaluated) < size:
        raise RuntimeError(f"CPO initialization produced {len(evaluated)} evaluated solutions; expected {size}")
    evaluated = evaluated[:size]
    initial_function_evaluations = int(function_evaluations)
    best_solution, best_score = copy.deepcopy(evaluated[0][0]), float(evaluated[0][1])
    memory = (
        DefenseSuccessMemory(
            context,
            evaporation=memory_evaporation,
            provider_weight=provider_memory_weight,
        )
        if bool(success_memory)
        else None
    )
    history = []
    stagnation = 0

    def record(
        iteration,
        active_size,
        generated=0,
        accepted=0,
        attempts=0,
        strategy_counts=None,
        progress=0.0,
        best_improved=False,
    ):
        profile = memory.profile() if memory is not None else {
            "cpo_strategy_credit": {},
            "cpo_accepted_by_strategy": {},
            "cpo_task_provider_memory_entries": 0,
            "cpo_service_provider_memory_entries": 0,
        }
        history.append(
            {
                "iteration": int(iteration),
                "best_total_efficiency": float(best_score),
                "population_total_efficiencies": [float(score) for _solution, score in evaluated],
                "population_mean_efficiency": float(
                    sum(score for _solution, score in evaluated) / float(max(1, len(evaluated)))
                ),
                "population_unique_count": int(len(evaluated)),
                "population_mean_hamming": float(_mean_hamming(evaluated)),
                "active_population_size": int(active_size),
                "generated_trials": int(generated),
                "unique_trial_count": int(generated),
                "duplicate_trial_count": int(max(0, attempts - generated)),
                "accepted_candidates": int(accepted),
                "candidate_attempts": int(attempts),
                "stagnation_generations": int(stagnation),
                "search_progress": float(progress),
                "best_improved": bool(best_improved),
                "cpo_variant": str(context.get("cpo_variant", "dcc_dcpo")),
                "criticality_guidance": bool(criticality_guidance),
                "cache_coupling": bool(cache_coupling),
                "success_memory": bool(success_memory),
                "model_guidance": bool(model_guidance),
                "defense_trials": dict(strategy_counts or {}),
                "function_evaluations": int(function_evaluations),
                **profile,
            }
        )

    record(0, size)
    for iteration in range(1, iterations + 1):
        if budget is not None and function_evaluations >= budget:
            break
        progress = _search_progress(
            function_evaluations,
            initial_function_evaluations,
            budget,
            iteration - 1,
            iterations,
        )
        active_size = _cyclic_active_size_at_progress(
            progress, size, cpr_minimum_ratio, cpr_cycles
        )
        if memory is not None:
            memory.begin_generation()
        current_rows = list(evaluated)
        current_population = [solution for solution, _score in current_rows]
        # CPR changes the number of porcupines that move, while a reservoir is
        # retained so a new cycle can restore population size without losing
        # already paid objective evaluations.
        active_indices = list(range(min(active_size, len(current_rows))))
        if len(current_rows) > active_size:
            tail = list(range(active_size, len(current_rows)))
            rng.shuffle(tail)
            active_indices[-max(1, active_size // 5):] = tail[:max(1, active_size // 5)]

        candidate_rows = []
        improvement_records = []
        accepted = 0
        attempts = 0
        generated = 0
        strategy_counts = {
            name: 0
            for name in (
                "sight",
                "sound",
                "odor",
                "physical_attack",
                "stagnation_escape",
            )
        }
        target_new_evaluations = min(
            active_size,
            (budget - function_evaluations) if budget is not None else active_size,
        )
        defense_schedule = build_defense_schedule(
            target_new_evaluations,
            progress,
            stagnation,
            memory,
            rng,
        )
        new_evaluations_before = function_evaluations
        max_attempts = max(active_size * 10, target_new_evaluations * 12)
        cursor = 0
        while (
            function_evaluations - new_evaluations_before < target_new_evaluations
            and attempts < max_attempts
        ):
            attempts += 1
            parent_index = active_indices[cursor % len(active_indices)]
            cursor += 1
            parent, parent_score = current_rows[parent_index]
            stagnation_escape = (
                stagnation >= max(1, int(stagnation_restart_after))
                and rng.random() < min(0.50, float(restart_fraction))
            )
            if stagnation_escape:
                strategy = "sight"
                strategy_counts["stagnation_escape"] += 1
            else:
                completed = function_evaluations - new_evaluations_before
                strategy = (
                    defense_schedule[completed]
                    if completed < len(defense_schedule)
                    else choose_defense(progress, stagnation, memory, rng)
                )
            strategy_counts[strategy] += 1
            candidate = generate_candidate(
                strategy,
                parent,
                best_solution,
                current_population,
                context,
                progress,
                rng,
                memory,
            )
            previous_nfe = function_evaluations
            row = evaluate(candidate)
            if row is None:
                break
            if function_evaluations == previous_nfe:
                continue
            generated += 1
            child, child_score = row
            if child_score > float(parent_score) + 1e-12:
                improvement_records.append(
                    (solution_key(child), strategy, child, parent, child_score - float(parent_score))
                )
            candidate_rows.append((child, child_score))

        previous_best = float(best_score)
        selection_pool = _sort_unique(current_rows + candidate_rows)
        if len(selection_pool) < size:
            raise RuntimeError("CPO selection lost feasible population diversity")
        evaluated = selection_pool[:size]
        surviving_keys = {solution_key(solution) for solution, _score in evaluated}
        for key, strategy, child, parent, gain in improvement_records:
            if key not in surviving_keys:
                continue
            accepted += 1
            if memory is not None:
                memory.reward(strategy, child, parent, gain)
        if float(evaluated[0][1]) > best_score + 1e-12:
            best_solution = copy.deepcopy(evaluated[0][0])
            best_score = float(evaluated[0][1])
        stagnation = 0 if best_score > previous_best + 1e-12 else stagnation + 1
        end_progress = _search_progress(
            function_evaluations,
            initial_function_evaluations,
            budget,
            iteration,
            iterations,
        )
        record(
            iteration,
            active_size,
            generated,
            accepted,
            attempts,
            strategy_counts,
            end_progress,
            best_score > previous_best + 1e-12,
        )

        if generated == 0:
            break

    return context.repair_solution(best_solution), float(best_score), history
