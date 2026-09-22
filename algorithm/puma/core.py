from __future__ import annotations

import copy
import random
from collections import deque

from .initial_population import create_initial_population
from .operators import (
    exploration_candidate,
    exploitation_candidate,
    hamming_distance,
    solution_key,
)


PF = (0.5, 0.5, 0.3)
MEGA_EXPLORATION = 0.99
MEGA_EXPLOITATION = 0.99
PCR_INITIAL = 0.20
Q = 0.67
BETA = 2.0
UNEXPERIENCED_ITERATIONS = 3


def _sort_unique(rows):
    result, seen = [], set()
    for solution, score in sorted(rows, key=lambda row: float(row[1]), reverse=True):
        key = solution_key(solution)
        if key and key not in seen:
            seen.add(key)
            result.append((copy.deepcopy(solution), float(score)))
    return result


def _mean_hamming(rows):
    if len(rows) < 2:
        return 0.0
    dimension = max(1, len(rows[0][0]))
    values = [
        hamming_distance(rows[left][0], rows[right][0]) / float(dimension)
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
    ]
    return float(sum(values) / len(values)) if values else 0.0


def run_puma(
    context,
    population_size=50,
    iterations=14,
    seed=None,
    initial_population=None,
    *,
    max_function_evaluations=None,
    pcr_initial=PCR_INITIAL,
    q=Q,
    beta=BETA,
    unexperienced_iterations=UNEXPERIENCED_ITERATIONS,
):
    """Run an NFE-aware categorical adaptation of Puma Optimizer.

    The source PO architecture is preserved: both phases are learned during an
    unexperienced period and a deterministic score then selects exploration or
    exploitation. Only continuous vector arithmetic is replaced by feasible
    categorical task-provider moves.
    """
    context = copy.deepcopy(context)
    seed = context.get("seed") if seed is None else seed
    if seed is not None:
        context["seed"] = int(seed)
    rng = random.Random(seed)
    size = max(2, int(population_size))
    iterations = max(0, int(iterations))
    budget = None if max_function_evaluations is None else max(1, int(max_function_evaluations))
    if budget is not None and budget < size:
        raise ValueError("max_function_evaluations must be at least population_size")

    raw_population = list(
        initial_population or create_initial_population(context, size, rng)
    )
    population = _sort_unique(
        (context.repair_solution(solution), 0.0) for solution in raw_population
    )
    if len(population) < size:
        raise RuntimeError(
            f"Initial Puma population has {len(population)} unique solutions; expected {size}"
        )
    population = [solution for solution, _score in population[:size]]

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

    evaluated = _sort_unique(
        row for row in (evaluate(solution) for solution in population) if row
    )
    if len(evaluated) < size:
        raise RuntimeError(
            f"Puma initialization produced {len(evaluated)} evaluated solutions; expected {size}"
        )
    evaluated = evaluated[:size]
    best_solution, best_score = copy.deepcopy(evaluated[0][0]), float(evaluated[0][1])
    initial_evaluations = int(function_evaluations)
    pcr = max(0.01, min(0.95, float(pcr_initial)))
    phase_state = {
        "exploration": {"gains": deque(maxlen=3), "efficiencies": deque(maxlen=3), "idle": 0},
        "exploitation": {"gains": deque(maxlen=3), "efficiencies": deque(maxlen=3), "idle": 0},
    }
    history = []

    def progress(iteration):
        if budget is not None and budget > initial_evaluations:
            return max(0.0, min(
                1.0,
                (function_evaluations - initial_evaluations)
                / float(budget - initial_evaluations),
            ))
        return max(0.0, min(1.0, iteration / float(max(1, iterations))))

    def phase_scores():
        gain_total = sum(sum(state["gains"]) for state in phase_state.values())
        efficiency_total = sum(sum(state["efficiencies"]) for state in phase_state.values())
        scores = {}
        for name, state in phase_state.items():
            gain_component = sum(state["gains"]) / gain_total if gain_total > 0.0 else 0.0
            efficiency_component = (
                sum(state["efficiencies"]) / efficiency_total
                if efficiency_total > 0.0 else 0.0
            )
            mega = MEGA_EXPLORATION if name == "exploration" else MEGA_EXPLOITATION
            idle_component = 1.0 - mega ** max(0, int(state["idle"]))
            scores[name] = (
                PF[0] * gain_component
                + PF[1] * efficiency_component
                + PF[2] * idle_component
            )
        return scores

    def record(iteration, phase, generated=0, accepted=0, scores=None):
        history.append({
            "iteration": int(iteration),
            "best_total_efficiency": float(best_score),
            "population_total_efficiencies": [float(score) for _solution, score in evaluated],
            "population_mean_efficiency": float(
                sum(score for _solution, score in evaluated) / float(max(1, len(evaluated)))
            ),
            "population_unique_count": int(len(evaluated)),
            "population_mean_hamming": float(_mean_hamming(evaluated)),
            "selected_phase": str(phase),
            "phase_scores": dict(scores or {}),
            "pcr": float(pcr),
            "generated_trials": int(generated),
            "accepted_candidates": int(accepted),
            "search_progress": float(progress(iteration)),
            "function_evaluations": int(function_evaluations),
        })

    def run_phase(name, source_rows, target_evaluations, iteration):
        nonlocal pcr
        rows = [(copy.deepcopy(solution), float(score)) for solution, score in source_rows]
        occupied = {solution_key(solution) for solution, _score in rows}
        before_nfe = int(function_evaluations)
        old_best = max(score for _solution, score in rows)
        accepted = 0
        attempts = 0
        cursor = 0
        max_attempts = max(50, int(target_evaluations) * 30)
        while function_evaluations - before_nfe < target_evaluations and attempts < max_attempts:
            if budget is not None and function_evaluations >= budget:
                break
            attempts += 1
            parent_index = cursor % len(rows)
            cursor += 1
            parent, parent_score = rows[parent_index]
            current_population = [solution for solution, _score in rows]
            if name == "exploration":
                candidate = exploration_candidate(parent, current_population, context, pcr, rng)
            else:
                candidate = exploitation_candidate(
                    parent,
                    best_solution,
                    current_population,
                    context,
                    progress(iteration),
                    q,
                    beta,
                    rng,
                )
            key = solution_key(candidate)
            if not key or key in occupied:
                continue
            result = evaluate(candidate)
            if result is None:
                break
            solution, score = result
            if score > parent_score + 1e-12:
                occupied.discard(solution_key(parent))
                occupied.add(key)
                rows[parent_index] = (copy.deepcopy(solution), float(score))
                accepted += 1
            elif name == "exploration":
                # Same adaptive crossover update used by the PO exploration
                # source, bounded for a categorical search space.
                pcr = min(0.95, pcr + 0.8 / float(max(1, size)))

        rows = _sort_unique(rows)
        gain = max(0.0, max(score for _solution, score in rows) - old_best)
        paid = max(0, int(function_evaluations) - before_nfe)
        phase_state[name]["gains"].append(float(gain))
        phase_state[name]["efficiencies"].append(float(gain) / float(max(1, paid)))
        phase_state[name]["idle"] = 0
        other = "exploitation" if name == "exploration" else "exploration"
        phase_state[other]["idle"] += 1
        return rows, paid, accepted, attempts

    record(0, "initialization")
    for iteration in range(1, iterations + 1):
        if budget is not None and function_evaluations >= budget:
            break
        generated = accepted = 0
        scores = phase_scores()

        if iteration <= max(0, int(unexperienced_iterations)):
            remaining = size * 2 if budget is None else max(0, budget - function_evaluations)
            exploration_target = min(size, (remaining + 1) // 2)
            exploitation_target = min(size, max(0, remaining - exploration_target))
            explored, paid, kept, _attempts = run_phase(
                "exploration", evaluated, exploration_target, iteration
            )
            generated += paid
            accepted += kept
            exploited = evaluated
            if exploitation_target > 0:
                exploited, paid, kept, _attempts = run_phase(
                    "exploitation", evaluated, exploitation_target, iteration
                )
                generated += paid
                accepted += kept
                phase_state["exploration"]["idle"] = 0
                phase_state["exploitation"]["idle"] = 0
            pooled = _sort_unique(list(evaluated) + list(explored) + list(exploited))
            evaluated = pooled[:size]
            selected_phase = "both"
        else:
            scores = phase_scores()
            if abs(scores["exploration"] - scores["exploitation"]) <= 1e-15:
                selected_phase = (
                    "exploration"
                    if phase_state["exploration"]["idle"] >= phase_state["exploitation"]["idle"]
                    else "exploitation"
                )
            else:
                selected_phase = max(scores, key=scores.get)
            remaining = size if budget is None else max(0, budget - function_evaluations)
            target = min(size, remaining)
            evaluated, paid, kept, _attempts = run_phase(
                selected_phase, evaluated, target, iteration
            )
            generated += paid
            accepted += kept

        evaluated = _sort_unique(evaluated)[:size]
        if evaluated and evaluated[0][1] > best_score + 1e-12:
            best_solution = copy.deepcopy(evaluated[0][0])
            best_score = float(evaluated[0][1])
        # Update even on equality so final returned coordinates are guaranteed
        # to be an actually evaluated incumbent.
        if evaluated and evaluated[0][1] >= best_score - 1e-12:
            best_solution = copy.deepcopy(evaluated[0][0])
            best_score = float(evaluated[0][1])
        record(iteration, selected_phase, generated, accepted, phase_scores())

    return best_solution, float(best_score), history
