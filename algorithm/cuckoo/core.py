

from __future__ import annotations

import copy
import random

from parameter.services import load_params_obj

from algorithm.greedy_nests import (
    _providers,
    _reset_assignment,
    procedure1_greedy_initialization,
    rate,
)
from algorithm.optimizer_common import (
    compute_task_ranks_and_order,
    evaluate_solution_quality,
    materialize_solution,
    raise_if_cancelled,
    refresh_shared_parameters,
)

from . import operators as operators_module
from .operators import procedure3_generate_new_solution
from .search import run_population_search


params = load_params_obj()


def _refresh_algorithm_params():
    """Load one consistent parameter snapshot for the whole Cuckoo run."""
    current_params, _params_lib = refresh_shared_parameters()

    global params
    params = current_params
    operators_module.params = current_params
    operators_module.levy_lambda = float(current_params.levy_lambda)


class _CuckooContext(dict):
    """Use lightweight copies for per-solution mutable scheduling state."""

    def __deepcopy__(self, memo):
        clone = type(self)(self)
        memo[id(self)] = clone
        clone["cache"] = {
            int(sp_id): set(items)
            for sp_id, items in self.get("cache", {}).items()
        }
        _reset_assignment(clone)
        return clone


def _nest_key(nest):
    return tuple(
        (int(task_id), int(provider_id), int(rank))
        for task_id, provider_id, rank in nest
    )


def sort_population(ctx, population, task_order, evaluation_cache=None):
    if evaluation_cache is None:
        evaluation_cache = {}

    evaluated = []
    for nest in population:
        raise_if_cancelled(ctx)
        key = _nest_key(nest)
        cached = evaluation_cache.get(key)
        if cached is None:
            quality, cache_state, _scheduled_ctx = evaluate_solution_quality(
                ctx,
                nest,
                task_order,
            )
            cached = (quality, cache_state)
            evaluation_cache[key] = cached
        evaluated.append((nest, cached[0], cached[1]))

    evaluated.sort(key=lambda item: item[1], reverse=True)
    return (
        [item[0] for item in evaluated],
        [item[1] for item in evaluated],
        [item[2] for item in evaluated],
    )


def dcsga_run(ctx):
    """Run the paper DCSGA without changing its numerical search path."""
    _refresh_algorithm_params()
    ctx = _CuckooContext(ctx)
    raise_if_cancelled(ctx)

    seed = ctx.get("seed")
    if seed is not None:
        random.seed(seed)

    tmax = int(ctx.get("tmax", 10))
    population_size = int(params.S)
    ctx["rates"] = {
        sp_id: rate(ctx, sp_id)
        for sp_id in _providers(ctx)
    }
    task_order = compute_task_ranks_and_order(ctx)
    initial_population = procedure1_greedy_initialization(
        S=population_size,
        task_order=task_order,
        ctx=ctx,
    )
    raise_if_cancelled(ctx)
    evaluation_cache = {}

    def evaluate_population(population):
        ranked, qualities, _caches = sort_population(
            ctx,
            list(population),
            task_order,
            evaluation_cache,
        )
        return list(zip(ranked, qualities))

    def generate(source_nest, best_nest):
        return procedure3_generate_new_solution(
            source_nest,
            best_nest,
            copy.deepcopy(ctx),
        )

    _population, best_nest, _evaluated = run_population_search(
        initial_population,
        population_size=population_size,
        tmax=tmax,
        initial_discard_probability=float(params.p_discard_init),
        rng=random,
        generate_new_solution=generate,
        evaluate_population=evaluate_population,
        cancel_check=lambda: raise_if_cancelled(ctx),
    )
    return materialize_solution(ctx, best_nest, task_order)


__all__ = ["dcsga_run", "sort_population"]
