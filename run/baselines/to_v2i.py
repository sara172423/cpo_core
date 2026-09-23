from __future__ import annotations

import copy
from typing import Any, Dict, Optional


class TO_V2I:
    name = "TO_V2I"
    key = "to_v2i"

    def run(
        self,
        base_ctx: Dict[str, Any],
        seed: Optional[int] = None,
    ):
        from algorithm.cuckoo.core import dcsga_run

        if not isinstance(base_ctx, dict):
            raise TypeError("base_ctx must be a dictionary")

        ctx = copy.deepcopy(base_ctx)
        ctx["seed"] = seed
        ctx["scheme"] = self.key
        ctx["use_ranking"] = True
        ctx["use_caching"] = True
        ctx["v2i_only"] = True
        return dcsga_run(ctx)

    def run_joint(
        self,
        joint_ctx: Dict[str, Any],
        *,
        seed: int,
        tmax: int,
        population_size: int | None = None,
        max_function_evaluations: int | None = None,
    ):
        from run.benchmark.search import run_joint_dcsga

        return run_joint_dcsga(
            joint_ctx,
            algorithm=self.key,
            seed=seed,
            tmax=tmax,
            population_size=population_size,
            max_function_evaluations=max_function_evaluations,
        )
