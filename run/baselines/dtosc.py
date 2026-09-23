from __future__ import annotations

import copy
from typing import Any, Dict, Optional


class DTOSC:
    name = "DTOSC"
    key = "dtosc"
    # The implementation is complete and uses the published algorithm class
    # (semi-distributed dynamic programming).  The 2022 full pseudocode is not
    # included in this project, so source-exact line-by-line verification is
    # deliberately reported separately rather than asserted without evidence.
    algorithm_complete = True
    dynamic_programming_complete = True
    source_exact_verified = False
    article_exact = False
    implementation = "semi-distributed-stage-dynamic-programming"
    reference_alignment = "published-description-and-project-equations"
    reference_doi = "10.1109/TVT.2022.3196544"

    def run(
        self,
        base_ctx: Dict[str, Any],
        seed: Optional[int] = None,
    ):
        from algorithm.dtosc.core import dtosc_run

        if not isinstance(base_ctx, dict):
            raise TypeError("base_ctx must be a dictionary")

        ctx = copy.deepcopy(base_ctx)
        ctx["seed"] = seed
        ctx["scheme"] = self.key
        ctx["use_ranking"] = True
        ctx["use_caching"] = True
        ctx["v2i_only"] = False
        ctx["provider_scope"] = "local_and_rsu"
        ctx["baseline_algorithm"] = self.key
        return dtosc_run(ctx)

    def run_joint(
        self,
        joint_ctx: Dict[str, Any],
        *,
        seed: int,
        tmax: int,
        population_size: int | None = None,
        max_function_evaluations: int | None = None,
    ):
        # Population controls are intentionally unused: DTOSC is a
        # deterministic DP baseline, not a population metaheuristic.
        from run.benchmark.search import run_joint_dtosc

        return run_joint_dtosc(
            joint_ctx,
            seed=seed,
        )
