from __future__ import annotations

from typing import Any, Dict


class CPO:
    name = "DCC-DCPO"
    key = "cpo"
    article_exact = False
    algorithm_complete = True
    implementation = "deadline-dependency-cache-coupled-discrete-cpo-v7"
    reference_doi = "10.1016/j.knosys.2023.111257"

    def run_joint(
        self,
        joint_ctx: Dict[str, Any],
        *,
        seed: int,
        tmax: int,
        population_size: int | None = None,
        max_function_evaluations: int | None = None,
    ):
        from run.benchmark.search import run_joint_cpo

        return run_joint_cpo(
            joint_ctx,
            algorithm=self.key,
            seed=seed,
            tmax=tmax,
            population_size=population_size,
            max_function_evaluations=max_function_evaluations,
        )


class CPOAblation(CPO):
    """Registered CPO ablation; intended for the ablation table, not baselines."""

    def __init__(self, key, name, implementation):
        self.key = str(key)
        self.name = str(name)
        self.implementation = str(implementation)
