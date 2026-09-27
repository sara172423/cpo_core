from __future__ import annotations

from typing import Any, Dict


class GPC:
    name = "D-GPC"
    key = "gpc"
    article_exact = False
    implementation = "reference-physics-categorical-gpc"
    reference_doi = "10.1007/s12065-020-00451-3"

    def run_joint(self, joint_ctx: Dict[str, Any], *, seed: int, tmax: int, population_size: int | None = None, max_function_evaluations: int | None = None):
        from run.benchmark.search import run_joint_gpc
        return run_joint_gpc(joint_ctx, algorithm=self.key, seed=seed, tmax=tmax, population_size=population_size, max_function_evaluations=max_function_evaluations)
