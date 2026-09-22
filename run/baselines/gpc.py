from __future__ import annotations

from typing import Any, Dict, Optional
from parameter.services import load_params_obj
from .context import StandaloneOptimizerContext


class GPC:
    name = "D-GPC"
    key = "gpc"
    article_exact = False
    implementation = "reference-physics-categorical-gpc"
    reference_doi = "10.1007/s12065-020-00451-3"

    def run(self, base_ctx: Dict[str, Any], seed: Optional[int] = None):
        from algorithm.gpc.core import run_gpc
        ctx = StandaloneOptimizerContext(base_ctx, seed=seed)
        params = load_params_obj()
        return run_gpc(
            ctx,
            population_size=int(params.S),
            iterations=max(0, int(ctx.get("tmax", 10)) - 1),
            seed=seed,
            service_memory_enabled=False,
            problem_guidance=False,
        )

    def run_joint(self, joint_ctx: Dict[str, Any], *, seed: int, tmax: int, population_size: int | None = None, max_function_evaluations: int | None = None):
        from run.benchmark.search import run_joint_gpc
        return run_joint_gpc(joint_ctx, algorithm=self.key, seed=seed, tmax=tmax, population_size=population_size, max_function_evaluations=max_function_evaluations)
