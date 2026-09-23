from __future__ import annotations

import random
from typing import Dict, List, Tuple

from parameter.services import load_params_obj
from algorithm.greedy_nests import _providers
from .search import mutate_provider_map

params = load_params_obj()
levy_lambda = params.levy_lambda


def nest_to_dict(nest: List[Tuple[int, int, int]]) -> Dict:
    return {i: {"provider": sp, "rank": r} for i, sp, r in nest}


def dict_to_nest(X: Dict, task_list: List[int]) -> List[Tuple[int, int, int]]:
    return [(i, X[i]["provider"], X[i]["rank"]) for i in task_list]


def _mode(ctx, sp_id: int) -> str:
    return ctx.get(
        "sp_modes", {},
    ).get(sp_id, "v2i" if ctx.get("sp_types", {}).get(sp_id) == "rsu" else "v2v")


def rebuild_solution_state(ctx, X: Dict, task_list: List[int], sp_list: List[int]) -> None:
    z_binary = ctx["z"]["binary"]
    z_compact = ctx["z"]["compact"]
    z_binary.clear()
    counter = {sp: 0 for sp in sp_list}

    for task_id in task_list:
        sp_id = X[task_id]["provider"]

        if sp_id not in counter:
            sp_id = random.choice(sp_list)
            X[task_id]["provider"] = sp_id

        counter[sp_id] += 1
        rank = counter[sp_id]
        X[task_id]["rank"] = rank
        z_compact[task_id]["provider"] = sp_id
        z_compact[task_id]["rank"] = rank
        z_binary.setdefault(sp_id, {})
        z_binary[sp_id].setdefault(rank, {})
        z_binary[sp_id][rank][task_id] = 1


def procedure3_generate_new_solution(
    Xs_nest: List[Tuple[int, int, int]],
    Xb_nest: List[Tuple[int, int, int]] | None,
    ctx,
):
    """Runtime adapter for the shared canonical Procedure 3 kernel."""
    sp_list = _providers(ctx)
    task_list = [int(task_id) for task_id, _, _ in Xs_nest]
    Xs = nest_to_dict(Xs_nest)
    Xb = nest_to_dict(Xb_nest) if Xb_nest else None

    provider_map = mutate_provider_map(
        task_list,
        {task_id: int(Xs[task_id]["provider"]) for task_id in task_list},
        (
            {task_id: int(Xb[task_id]["provider"]) for task_id in task_list}
            if Xb is not None
            else None
        ),
        levy_lambda=float(levy_lambda),
        rng=random,
        domain_for_task=lambda _task_id: sp_list,
        mode_for_task_provider=lambda _task_id, sp_id: _mode(ctx, int(sp_id)),
    )

    Xnew = {
        task_id: {
            "provider": int(provider_map[task_id]),
            "rank": int(Xs[task_id]["rank"]),
        }
        for task_id in task_list
    }
    rebuild_solution_state(ctx, Xnew, task_list, sp_list)
    return dict_to_nest(Xnew, task_list)
