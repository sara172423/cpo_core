
from __future__ import annotations

from typing import Any, Dict, Optional

from parameter.services import load_params_obj
from .context import StandaloneOptimizerContext



class GWO_ACO:
    name = "D-GWO"
    key = "gwo_aco"
    article_exact = False
    implementation = "categorical-alpha-beta-delta-gwo-reference-baseline"
    reference_doi = "10.1016/j.advengsoft.2013.12.007"

    def run(
        self,
        base_ctx: Dict[str, Any],
        seed: Optional[int] = None,
    ):
        from algorithm.gwo.core import run_gwo_aco

        ctx = StandaloneOptimizerContext(base_ctx, seed=seed)
        ctx["scheme"] = self.key
        ctx["use_ranking"] = True
        ctx["use_caching"] = True
        ctx["v2i_only"] = False

        params = load_params_obj()
        tmax = max(1, int(ctx.get("tmax", 10)))
        return run_gwo_aco(
            ctx,
            population_size=int(params.S),
            iterations=tmax - 1,
            use_pheromone=False,
            use_rank_guidance=False,
            use_cache_guidance=False,
        )

    def run_joint(
        self,
        joint_ctx: Dict[str, Any],
        *,
        seed: int,
        tmax: int,
        population_size: int | None = None,
        max_function_evaluations: int | None = None,
    ):
        from run.benchmark.search import run_joint_gwo_aco
        return run_joint_gwo_aco(
            joint_ctx,
            algorithm=self.key,
            seed=seed,
            tmax=tmax,
            population_size=population_size,
            max_function_evaluations=max_function_evaluations,
        )


class GWO(GWO_ACO):
    """Scientific key for the isolated categorical GWO comparator."""

    key = "gwo"
