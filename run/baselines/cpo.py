from __future__ import annotations

from typing import Any, Dict, Optional

from parameter.services import load_params_obj
from .context import StandaloneOptimizerContext


class CPO:
    name = "DCC-DCPO"
    key = "cpo"
    article_exact = False
    algorithm_complete = True
    implementation = "deadline-dependency-cache-coupled-discrete-cpo-v7"
    reference_doi = "10.1016/j.knosys.2023.111257"

    def run(self, base_ctx: Dict[str, Any], seed: Optional[int] = None):
        from algorithm.cpo.core import run_cpo

        ctx = StandaloneOptimizerContext(base_ctx, seed=seed)
        ctx["scheme"] = self.key
        ctx["use_ranking"] = True
        ctx["use_caching"] = True
        ctx["v2i_only"] = False
        params = load_params_obj()
        return run_cpo(
            ctx,
            population_size=int(params.S),
            iterations=max(0, int(ctx.get("tmax", 10)) - 1),
            seed=seed,
            criticality_guidance=True,
            cache_coupling=True,
            success_memory=True,
            model_guidance=True,
            deadline_guidance=True,
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

    def run(self, base_ctx: Dict[str, Any], seed: Optional[int] = None):
        from algorithm.cpo.core import run_cpo

        flags = {
            "dcpo_base": dict(
                criticality_guidance=False,
                cache_coupling=False,
                success_memory=False,
                model_guidance=False,
                deadline_guidance=False,
            ),
            "dcpo_criticality": dict(
                criticality_guidance=True,
                cache_coupling=False,
                success_memory=False,
                model_guidance=False,
                deadline_guidance=False,
            ),
            "dcpo_cache": dict(
                criticality_guidance=False,
                cache_coupling=True,
                success_memory=False,
                model_guidance=False,
                deadline_guidance=False,
            ),
        }[self.key]
        ctx = StandaloneOptimizerContext(base_ctx, seed=seed)
        ctx["scheme"] = self.key
        ctx["use_ranking"] = True
        ctx["use_caching"] = True
        ctx["v2i_only"] = False
        ctx["cpo_variant"] = self.key
        params = load_params_obj()
        return run_cpo(
            ctx,
            population_size=int(params.S),
            iterations=max(0, int(ctx.get("tmax", 10)) - 1),
            seed=seed,
            **flags,
        )
