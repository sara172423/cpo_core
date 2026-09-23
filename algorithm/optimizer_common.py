"""Shared optimizer contracts, task ranking, and solution evaluation.

This module contains only problem-level operations used by more than one
optimizer.  Algorithm-specific searches live in their own packages.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Tuple


def assignment_key(solution: Any) -> Tuple[Tuple[int, int], ...]:
    """Return the task-provider part of a repaired optimizer solution."""
    if isinstance(solution, Mapping):
        return tuple(
            (int(task), int(provider))
            for task, provider in solution.items()
        )
    return tuple(
        (int(gene[0]), int(gene[1]))
        for gene in solution or []
        if isinstance(gene, (tuple, list)) and len(gene) >= 2
    )


def provider_map(solution: Any) -> Dict[int, int]:
    """Convert a repaired solution to a task-to-provider mapping."""
    return dict(assignment_key(solution))


def hamming_distance(left: Any, right: Any) -> int:
    """Count provider assignments that differ between two solutions."""
    left_map = provider_map(left)
    right_map = provider_map(right)
    tasks = set(left_map) | set(right_map)
    return sum(
        left_map.get(task) != right_map.get(task)
        for task in tasks
    )


class AlgorithmCancelled(RuntimeError):
    pass


def raise_if_cancelled(ctx) -> None:
    cancel_event = ctx.get("cancel_event") if isinstance(ctx, dict) else None
    if cancel_event is not None and cancel_event.is_set():
        raise AlgorithmCancelled("Algorithm execution was cancelled")


def refresh_shared_parameters():
    """Load one parameter snapshot for shared model and greedy equations."""
    from parameter.services import load_params_for_lib, load_params_obj
    from algorithm import greedy_nests, low_complexity

    params = load_params_obj()
    params_lib = load_params_for_lib()
    greedy_nests.params = params
    greedy_nests.params_lib = params_lib
    low_complexity.params = params
    low_complexity.params_lib = params_lib
    return params, params_lib


def natural_topological_order(ctx) -> List[int]:
    task_ids = [
        int(task_id)
        for task_id in ctx.get(
            "task_ids",
            ctx.get("tasks", {}).get("all", []),
        )
    ]
    dependencies = {
        int(task_id): [
            int(predecessor)
            for predecessor in ctx.get("dependencies", {}).get(task_id, [])
        ]
        for task_id in task_ids
    }
    indegree = {task_id: 0 for task_id in task_ids}
    children: Dict[int, List[int]] = {
        task_id: [] for task_id in task_ids
    }

    for task_id, predecessors in dependencies.items():
        for predecessor in predecessors:
            if predecessor in indegree:
                indegree[task_id] += 1
                children.setdefault(predecessor, []).append(task_id)

    ready = sorted(
        task_id
        for task_id in task_ids
        if indegree.get(task_id, 0) == 0
    )
    order = []
    while ready:
        task_id = ready.pop(0)
        order.append(task_id)
        for child_id in sorted(children.get(task_id, [])):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                ready.append(child_id)
                ready.sort()

    if len(order) != len(task_ids):
        raise ValueError("Task graph is not a DAG")
    return order


def compute_local_ranks(ctx):
    """Compute the shared HEFT local rank from the common system model."""
    from algorithm.greedy_nests import _providers, link_rate, t_comp
    from monarch_pylib.model import task_ranking

    children = ctx["children"]
    sizes_bits = ctx.get(
        "task_output_size_bits",
        ctx.get("output_size", {}),
    )
    edge_data_bits = ctx.get("edge_data_bits", {})
    all_tasks = [int(task_id) for task_id in ctx["tasks"]["all"]]
    providers = [int(sp_id) for sp_id in _providers(ctx)]
    if not providers:
        raise ValueError("At least one service provider is required")

    average_compute_time = {
        task_id: sum(
            float(t_comp(ctx, sp_id, task_id))
            for sp_id in providers
        ) / len(providers)
        for task_id in all_tasks
    }

    average_link_rates = []
    for src_sp_id in providers:
        for dst_sp_id in providers:
            if src_sp_id == dst_sp_id:
                average_link_rates.append(None)
                continue
            if (
                ctx.get("sp_types", {}).get(src_sp_id) == "rsu"
                and ctx.get("sp_types", {}).get(dst_sp_id) == "rsu"
            ):
                average_link_rates.append(None)
                continue
            rate_value = float(link_rate(ctx, src_sp_id, dst_sp_id))
            if rate_value > 0.0:
                average_link_rates.append(rate_value)

    ranks = {}
    for task_id in reversed(natural_topological_order(ctx)):
        successors = [int(value) for value in children[task_id]]
        if not successors:
            ranks[task_id] = average_compute_time[task_id]
            continue

        comm_times = []
        successor_ranks = []
        for successor_id in successors:
            data_bits = edge_data_bits.get(task_id, {}).get(
                successor_id,
                sizes_bits.get(task_id, 0.0),
            )
            transfer_samples = [
                float(data_bits) / rate_value
                for rate_value in average_link_rates
                if rate_value is not None and rate_value > 0.0
            ]
            comm_times.append(
                sum(transfer_samples) / len(transfer_samples)
                if transfer_samples
                else 0.0
            )
            successor_ranks.append(ranks[successor_id])

        ranks[task_id] = task_ranking.heft_task_local_rank(
            task_time_s=average_compute_time[task_id],
            succ_comm_times_s=comm_times,
            succ_ranks_s=successor_ranks,
        )
    return ranks


def compute_global_ranks(ctx, local_ranks):
    from monarch_pylib.model import task_ranking

    return {
        task_id: task_ranking.heft_task_global_rank(
            local_rank_s=float(local_rank),
            max_deadline_s=float(ctx["deadline_max_s"]),
            app_deadline_s=float(ctx["deadline_s"]),
        )
        for task_id, local_rank in local_ranks.items()
    }


def get_task_order(global_ranks):
    return [
        task_id
        for task_id, _rank in sorted(
            global_ranks.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def compute_task_ranks_and_order(ctx):
    """Return the common ranked order, excluding the local entry task."""
    from algorithm.greedy_nests import _entry_task_id

    entry_task_id = _entry_task_id(ctx)
    if ctx.get("use_ranking", True) is False:
        order = natural_topological_order(ctx)
    else:
        local_ranks = compute_local_ranks(ctx)
        global_ranks = compute_global_ranks(ctx, local_ranks)
        order = get_task_order(global_ranks)
    return [
        int(task_id)
        for task_id in order
        if int(task_id) != entry_task_id
    ]


def evaluate_solution_quality(base_ctx, nest, task_order):
    """Evaluate any repaired task-provider solution with the common model."""
    from algorithm.greedy_nests import (
        _apply_assignment,
        _apply_entry_task,
        _empty_state,
        _entry_task_id,
        _providers,
        _reset_assignment,
        compute_Q1,
    )
    from algorithm.update_service_cache import update_cache

    raise_if_cancelled(base_ctx)
    work_ctx = copy.deepcopy(base_ctx)
    _reset_assignment(work_ctx)
    work_ctx["_schedule_state"] = _empty_state(work_ctx)

    entry_task_id = _entry_task_id(work_ctx)
    local_sp_id = work_ctx.get("local_sp_id")
    if local_sp_id is None:
        raise ValueError(
            "Local service provider is required for the entry task"
        )
    local_sp_id = int(local_sp_id)
    normalized_order = [
        int(task_id)
        for task_id in task_order
        if int(task_id) != entry_task_id
    ]
    task_provider = {
        int(task_id): int(sp_id)
        for task_id, sp_id, _rank in nest
    }
    if (
        entry_task_id in task_provider
        and task_provider[entry_task_id] != local_sp_id
    ):
        raise ValueError(
            "Entry task must be assigned to the local service provider"
        )

    expected_tasks = set(normalized_order)
    provided_tasks = set(task_provider) - {entry_task_id}
    missing_tasks = sorted(expected_tasks - provided_tasks)
    unexpected_tasks = sorted(provided_tasks - expected_tasks)
    if missing_tasks:
        raise ValueError(f"Nest is missing tasks: {missing_tasks}")
    if unexpected_tasks:
        raise ValueError(
            f"Nest contains unexpected tasks: {unexpected_tasks}"
        )

    rank_counter = {
        sp_id: 0 for sp_id in _providers(work_ctx)
    }
    rank_counter.setdefault(local_sp_id, 0)
    _apply_entry_task(
        work_ctx,
        work_ctx["_schedule_state"],
        rank_counter,
    )

    for index, task_id in enumerate(normalized_order):
        raise_if_cancelled(work_ctx)
        sp_id = task_provider[task_id]
        rank_counter[sp_id] += 1
        _apply_assignment(
            work_ctx,
            work_ctx["_schedule_state"],
            task_id,
            sp_id,
            rank_counter[sp_id],
        )
        if work_ctx.get("use_caching", True):
            update_cache(
                work_ctx,
                sp_id,
                task_id,
                remaining_task_ids=normalized_order[index + 1 :],
            )

    return compute_Q1(work_ctx), work_ctx["cache"], work_ctx


def materialize_solution(ctx, nest, task_order):
    quality, cache_state, scheduled_ctx = evaluate_solution_quality(
        ctx,
        nest,
        task_order,
    )
    from algorithm.greedy_nests import _entry_task_id

    entry_task_id = _entry_task_id(scheduled_ctx)
    ordered_task_ids = [entry_task_id] + [
        int(task_id)
        for task_id in task_order
        if int(task_id) != entry_task_id
    ]
    compact = scheduled_ctx.get("z", {}).get("compact", {})
    solution = []
    for task_id in ordered_task_ids:
        assignment = compact.get(task_id, {})
        provider_id = assignment.get("provider")
        rank = assignment.get("rank")
        if provider_id is None or rank is None:
            raise ValueError(
                f"Task {task_id} has no finalized assignment"
            )
        solution.append((int(task_id), int(provider_id), int(rank)))
    return solution, quality, cache_state


__all__ = [
    "AlgorithmCancelled",
    "assignment_key",
    "compute_global_ranks",
    "compute_local_ranks",
    "compute_task_ranks_and_order",
    "evaluate_solution_quality",
    "get_task_order",
    "hamming_distance",
    "materialize_solution",
    "natural_topological_order",
    "provider_map",
    "raise_if_cancelled",
    "refresh_shared_parameters",
]
