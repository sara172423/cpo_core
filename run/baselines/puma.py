from __future__ import annotations

from typing import Any, Dict


class PUMA:
    name = "D-PO"
    key = "puma"
    article_exact = False
    algorithm_complete = True
    implementation = "nfe-aware-categorical-puma-optimizer"
    reference_doi = "10.1007/s10586-023-04221-5"

    def run_joint(
        self,
        joint_ctx: Dict[str, Any],
        *,
        seed: int,
        tmax: int,
        population_size: int | None = None,
        max_function_evaluations: int | None = None,
    ):
        from run.benchmark.search import run_joint_puma

        return run_joint_puma(
            joint_ctx,
            algorithm=self.key,
            seed=seed,
            tmax=tmax,
            population_size=population_size,
            max_function_evaluations=max_function_evaluations,
        )
