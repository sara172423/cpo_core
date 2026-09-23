from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from algorithm.optimizer_common import (
    compute_task_ranks_and_order,
    materialize_solution,
    refresh_shared_parameters,
)
from algorithm.greedy_nests import (
    _apply_assignment,
    _apply_entry_task,
    _empty_state,
    _providers,
    _reset_assignment,
    compute_Q,
    compute_Q1,
)
from algorithm.update_service_cache import update_cache


Nest = List[Tuple[int, int, int]]


@dataclass
class _DTOSCDPNode:
    """One compressed dynamic-programming state for a task stage.

    DTOSC processes the ranked tasks stage by stage.  For every possible
    provider of the current task, the best predecessor state is retained.
    The state includes the complete partial schedule and cache, so the
    next-stage transition is evaluated with the same timing, energy and
    cache equations used by the rest of the project.
    """

    ctx: dict
    rank_counter: Dict[int, int]
    nest: Nest
    cumulative_utility: float
    provider_path: Tuple[int, ...]


def _dtosc_providers(ctx) -> List[int]:
    """Return the execution domain of the 2022 VEC baseline.

    The DTOSC baseline predates the vehicular-fog extension of the main
    paper, therefore its non-entry tasks use the local vehicle or MEC/RSU
    servers, not collaborative vehicle providers.
    """

    local_sp_id = ctx.get("local_sp_id")
    if local_sp_id is None:
        raise ValueError("DTOSC requires a local service provider")

    local_sp_id = int(local_sp_id)
    providers = sorted(
        {
            int(sp_id)
            for sp_id in _providers(ctx)
            if int(sp_id) == local_sp_id
            or ctx.get("sp_types", {}).get(int(sp_id)) == "rsu"
        }
    )

    if not providers:
        raise ValueError("DTOSC has no feasible local or RSU providers")

    return providers


def _schedule_finish_and_energy(ctx: dict) -> Tuple[float, float]:
    state = ctx.get("_schedule_state", {})
    finish = max(
        (float(value) for value in state.get("task_finish", {}).values()),
        default=0.0,
    )
    energy = sum(
        float(value) for value in state.get("task_energy", {}).values()
    )
    return float(finish), float(energy)


def _node_numeric_key(node: _DTOSCDPNode, *, final: bool) -> Tuple[float, ...]:
    application_utility = float(compute_Q1(node.ctx))
    finish, energy = _schedule_finish_and_energy(node.ctx)

    if final:
        # The completed application objective is the authoritative DTOSC
        # objective.  The accumulated stage utility is only a deterministic
        # secondary criterion between equal final objectives.
        return (
            application_utility,
            float(node.cumulative_utility),
            -finish,
            -energy,
        )

    # During the recurrence, accumulated utility represents the Bellman
    # value of the partial path.  Partial application utility and resource
    # use provide stable tie-breaking without introducing randomness.
    return (
        float(node.cumulative_utility),
        application_utility,
        -finish,
        -energy,
    )


def _is_better_node(
    candidate: _DTOSCDPNode,
    incumbent: Optional[_DTOSCDPNode],
    *,
    final: bool,
) -> bool:
    if incumbent is None:
        return True

    candidate_key = _node_numeric_key(candidate, final=final)
    incumbent_key = _node_numeric_key(incumbent, final=final)
    if candidate_key != incumbent_key:
        return candidate_key > incumbent_key

    # Prefer the lexicographically smaller provider path when all numerical
    # criteria are equal, making repeated seeded runs exactly reproducible.
    return candidate.provider_path < incumbent.provider_path


def _initial_dp_node(ctx: dict, providers: List[int]) -> _DTOSCDPNode:
    work_ctx = copy.deepcopy(ctx)
    _reset_assignment(work_ctx)
    work_ctx["_schedule_state"] = _empty_state(work_ctx)
    rank_counter: Dict[int, int] = {int(sp_id): 0 for sp_id in providers}

    local_sp_id = int(work_ctx["local_sp_id"])
    rank_counter.setdefault(local_sp_id, 0)
    _apply_entry_task(work_ctx, work_ctx["_schedule_state"], rank_counter)

    return _DTOSCDPNode(
        ctx=work_ctx,
        rank_counter=rank_counter,
        nest=[],
        cumulative_utility=0.0,
        provider_path=(),
    )


def _transition_node(
    node: _DTOSCDPNode,
    *,
    task_id: int,
    provider_id: int,
    remaining_task_ids: List[int],
) -> _DTOSCDPNode:
    task_id = int(task_id)
    provider_id = int(provider_id)

    # q_i,x is evaluated from the predecessor DP state before the current
    # assignment is committed, exactly as in the scheduling evaluator.
    stage_utility = float(compute_Q(node.ctx, provider_id, task_id))

    child_ctx = copy.deepcopy(node.ctx)
    child_state = child_ctx["_schedule_state"]
    child_ranks = dict(node.rank_counter)
    child_ranks[provider_id] = child_ranks.get(provider_id, 0) + 1

    _apply_assignment(
        child_ctx,
        child_state,
        task_id,
        provider_id,
        child_ranks[provider_id],
    )

    if child_ctx.get("use_caching", True):
        update_cache(
            child_ctx,
            provider_id,
            task_id,
            remaining_task_ids=remaining_task_ids,
        )

    return _DTOSCDPNode(
        ctx=child_ctx,
        rank_counter=child_ranks,
        nest=node.nest
        + [(task_id, provider_id, child_ranks[provider_id])],
        cumulative_utility=float(node.cumulative_utility + stage_utility),
        provider_path=node.provider_path + (provider_id,),
    )


def _build_dtosc_solution(ctx, task_order: List[int]) -> Nest:
    """Solve one application with stage-wise dynamic programming.

    Recurrence
    ----------
    At task stage ``i``, every retained predecessor state is expanded to
    every feasible local/MEC provider.  For each provider ``x`` of task
    ``i`` only the best path ending at ``x`` is retained.  Consequently the
    frontier contains at most ``|X_n|`` states and the recurrence requires
    ``O(I_n |X_n|^2)`` transitions, instead of the former one-step greedy
    choice.
    """

    providers = _dtosc_providers(ctx)
    normalized_order = [int(task_id) for task_id in task_order]
    frontier: List[_DTOSCDPNode] = [_initial_dp_node(ctx, providers)]

    for index, task_id in enumerate(normalized_order):
        remaining = normalized_order[index + 1 :]
        best_by_provider: Dict[int, _DTOSCDPNode] = {}

        for node in frontier:
            for provider_id in providers:
                candidate = _transition_node(
                    node,
                    task_id=task_id,
                    provider_id=provider_id,
                    remaining_task_ids=remaining,
                )
                incumbent = best_by_provider.get(provider_id)
                if _is_better_node(candidate, incumbent, final=False):
                    best_by_provider[provider_id] = candidate

        if not best_by_provider:
            raise RuntimeError(
                f"DTOSC dynamic-programming frontier became empty at task {task_id}"
            )

        frontier = [
            best_by_provider[provider_id]
            for provider_id in sorted(best_by_provider)
        ]

    best_node: Optional[_DTOSCDPNode] = None
    for candidate in frontier:
        if _is_better_node(candidate, best_node, final=True):
            best_node = candidate

    if best_node is None:
        raise RuntimeError("DTOSC dynamic programming produced no solution")

    return list(best_node.nest)


def dtosc_run(ctx):
    """Run the complete local/MEC DTOSC dynamic-programming baseline."""

    refresh_shared_parameters()
    seed = ctx.get("seed")
    if seed is not None:
        # The DP is deterministic; seeding is retained for interface
        # consistency and deterministic lower-level channel realizations.
        random.seed(seed)

    work_ctx = copy.deepcopy(ctx)
    work_ctx["scheme"] = "dtosc"
    work_ctx["use_ranking"] = True
    work_ctx["use_caching"] = True
    work_ctx["v2i_only"] = False
    work_ctx["provider_scope"] = "local_and_rsu"
    work_ctx["baseline_algorithm"] = "dtosc"
    work_ctx["dtosc_solver"] = "semi-distributed-stage-dynamic-programming"

    task_order = compute_task_ranks_and_order(work_ctx)
    nest = _build_dtosc_solution(work_ctx, task_order)
    return materialize_solution(work_ctx, nest, task_order)
