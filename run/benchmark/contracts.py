"""Small immutable contracts shared by the paper benchmark modules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class JointApplicationResult:
    application_id: int
    vehicle_id: int | None
    deadline_s: float
    alpha_n: float
    beta_n: float
    delay_s: float
    energy_j: float
    efficiency: float
    completed: bool
    task_count: int
    optimized_task_count: int
    scheduled_task_count: int
    entry_task_id: int
    entry_provider_id: int
    providers_used: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointAlgorithmResult:
    algorithm: str
    seed: int
    population_size: int
    tmax: int
    total_efficiency: float
    applications: List[JointApplicationResult]
    metrics: Dict[str, float]
    iteration_history: List[Dict[str, float]] = field(default_factory=list)
    scientific_status: str = (
        "article-aligned-runtime-final-with-declared-limitations"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "population_size": self.population_size,
            "tmax": self.tmax,
            "total_efficiency": self.total_efficiency,
            "applications": [item.to_dict() for item in self.applications],
            "metrics": dict(self.metrics),
            "iteration_history": list(self.iteration_history),
            "scientific_status": self.scientific_status,
        }


@dataclass(frozen=True)
class JointScheme:
    name: str
    use_ranking: bool
    use_caching: bool
    v2i_only: bool
    provider_scope: str = "all"

    def context_flags(self) -> Dict[str, Any]:
        return {
            "use_ranking": self.use_ranking,
            "use_caching": self.use_caching,
            "v2i_only": self.v2i_only,
            "provider_scope": self.provider_scope,
        }


SCHEMES: Dict[str, JointScheme] = {
    "dcsga": JointScheme("dcsga", True, True, False, "all"),
    "dtosc": JointScheme("dtosc", True, True, False, "local_and_rsu"),
    "to_v2i": JointScheme("to_v2i", True, True, True, "rsu_only"),
    "to_wo_c": JointScheme("to_wo_c", True, False, False, "all"),
    "to_wo_r": JointScheme("to_wo_r", False, True, False, "all"),
    "gwo_aco": JointScheme("gwo_aco", True, True, False, "all"),
    "gwo": JointScheme("gwo", True, True, False, "all"),
    "gpc": JointScheme("gpc", True, True, False, "all"),
    "cpo": JointScheme("cpo", True, True, False, "all"),
    "dcpo_base": JointScheme("dcpo_base", True, True, False, "all"),
    "dcpo_criticality": JointScheme(
        "dcpo_criticality", True, True, False, "all"
    ),
    "dcpo_cache": JointScheme("dcpo_cache", True, True, False, "all"),
    "puma": JointScheme("puma", True, True, False, "all"),
}
SUPPORTED_JOINT_ALGORITHMS = tuple(SCHEMES)


def get_joint_scheme(name: str) -> JointScheme:
    key = str(name).strip().lower()
    if key not in SCHEMES:
        raise ValueError(f"Unsupported joint algorithm: {name}")
    return SCHEMES[key]


PAPER_REPRODUCTION = "paper_reproduction"
FAIR_OPTIMIZER_COMPARISON = "fair_optimizer_comparison"
EXPERIMENT_MODES = (PAPER_REPRODUCTION, FAIR_OPTIMIZER_COMPARISON)
POPULATION_ALGORITHMS = frozenset(
    {
        "dcsga", "to_v2i", "to_wo_c", "to_wo_r", "gpc", "gwo",
        "gwo_aco", "cpo", "dcpo_base", "dcpo_criticality",
        "dcpo_cache", "puma",
    }
)
ADDED_OPTIMIZERS = frozenset(
    {
        "gpc", "gwo", "gwo_aco", "cpo", "dcpo_base",
        "dcpo_criticality", "dcpo_cache", "puma",
    }
)


def resolve_experiment_mode(
    requested_mode: str | None,
    *,
    figure: str,
    selected_algorithms: Iterable[str],
    paper_algorithms: Iterable[str],
    max_function_evaluations: int | None,
) -> str:
    """Resolve paper-reproduction versus exact-NFE comparison semantics."""
    selected = tuple(
        dict.fromkeys(str(name).strip().lower() for name in selected_algorithms)
    )
    paper = tuple(
        dict.fromkeys(str(name).strip().lower() for name in paper_algorithms)
    )
    mode = (
        FAIR_OPTIMIZER_COMPARISON
        if requested_mode is None and set(selected) & ADDED_OPTIMIZERS
        else PAPER_REPRODUCTION if requested_mode is None
        else str(requested_mode).strip().lower()
    )
    if mode not in EXPERIMENT_MODES:
        raise ValueError(
            "experiment_mode must be 'paper_reproduction' or "
            "'fair_optimizer_comparison'"
        )
    if mode == PAPER_REPRODUCTION:
        unsupported = sorted(set(selected) - set(paper))
        if unsupported:
            raise ValueError(
                "paper_reproduction accepts only algorithms printed in the "
                f"selected source figure; unsupported: {unsupported}. Use "
                "experiment_mode='fair_optimizer_comparison' for added optimizers."
            )
        if max_function_evaluations is not None:
            raise ValueError(
                "paper_reproduction uses the paper-native generation limit. "
                "Remove max_function_evaluations, or switch to "
                "fair_optimizer_comparison."
            )
        return mode
    if figure == "all":
        raise ValueError(
            "fair_optimizer_comparison must be run one figure at a time because "
            "Figures 6-10 have different scenarios and source baselines."
        )
    if not (set(selected) & ADDED_OPTIMIZERS):
        raise ValueError(
            "fair_optimizer_comparison must include at least one added optimizer."
        )
    if set(selected) & POPULATION_ALGORITHMS and max_function_evaluations is None:
        raise ValueError(
            "fair_optimizer_comparison requires max_function_evaluations so "
            "population optimizers use the same objective-call budget."
        )
    return mode


__all__ = [
    "ADDED_OPTIMIZERS",
    "EXPERIMENT_MODES",
    "FAIR_OPTIMIZER_COMPARISON",
    "JointAlgorithmResult",
    "JointApplicationResult",
    "JointScheme",
    "PAPER_REPRODUCTION",
    "POPULATION_ALGORITHMS",
    "SCHEMES",
    "SUPPORTED_JOINT_ALGORITHMS",
    "get_joint_scheme",
    "resolve_experiment_mode",
]
