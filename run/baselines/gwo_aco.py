
from __future__ import annotations

from typing import Any, Dict



class GWO_ACO:
    name = "D-GWO"
    key = "gwo_aco"
    article_exact = False
    implementation = "categorical-alpha-beta-delta-gwo-reference-baseline"
    reference_doi = "10.1016/j.advengsoft.2013.12.007"

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
