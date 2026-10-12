

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable


class PaperModelError(ValueError):
    """Raised when inputs violate the mathematical domain of the paper model."""


class ProviderKind(str, Enum):
    VEHICLE = "vehicle"
    RSU = "rsu"


@dataclass(frozen=True)
class ApplicationWeights:
    alpha: float
    beta: float


@dataclass(frozen=True)
class LocalReference:
    t_local_s: float
    t_ref_s: float
    e_local_j: float




def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise PaperModelError(f"{name} must be a finite value greater than zero")
    return value






def article_weights(deadline_s: float) -> ApplicationWeights:
    deadline_s = _positive_finite(deadline_s, "deadline_s")

    alpha = 0.01 / deadline_s + 0.6
    beta = 1.0 - alpha
    if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
        raise PaperModelError("The article weights must be between zero and one")

    return ApplicationWeights(
        alpha=float(alpha),
        beta=float(beta),
    )

@dataclass(frozen=True)
class CPOObjectiveWeights:
    efficiency: float = 0.70
    deadline: float = 0.15
    cache: float = 0.10
    load_balance: float = 0.05


def cpo_weights(deadline_s: float) -> ApplicationWeights:
   
    deadline_s = _positive_finite(deadline_s, "deadline_s")

    alpha = 0.01 / deadline_s + 0.6
    beta = 1.0 - alpha

    if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
        raise PaperModelError("The CPO weights must be between zero and one")

    return ApplicationWeights(
        alpha=float(alpha),
        beta=float(beta),
    )


def cpo_framework_score(
    efficiency_score: float,
    *,
    deadline_score: float = 0.0,
    cache_score: float = 0.0,
    load_balance_score: float = 0.0,
    weights: CPOObjectiveWeights = CPOObjectiveWeights(),
) -> float:
    """Optional CPO framework score.

    This is not used by the shared fair optimizer comparison. It is intended
    only for CPO self-comparison/ablation experiments.
    """
    return float(
        weights.efficiency * efficiency_score
        + weights.deadline * deadline_score
        + weights.cache * cache_score
        + weights.load_balance * load_balance_score
    )

def local_reference(
    cpu_cycles: Iterable[float],
    *,
    vehicle_fmax_hz: float,
    deadline_s: float,
    kappa: float,
) -> LocalReference:
    """Compute equations (23)-(25) for the all-local reference execution."""
    cycles = [float(value) for value in cpu_cycles]
    if not cycles:
        raise PaperModelError("cpu_cycles must contain at least one task")
    if any((not isfinite(value) or value < 0.0) for value in cycles):
        raise PaperModelError("cpu_cycles values must be finite and non-negative")

    fmax = _positive_finite(vehicle_fmax_hz, "vehicle_fmax_hz")
    deadline = _positive_finite(deadline_s, "deadline_s")
    kappa = _positive_finite(kappa, "kappa")

    total_cycles = sum(cycles)
    t_local = total_cycles / fmax
    t_ref = min(t_local, deadline)
    e_local = sum(kappa * (fmax**2) * value for value in cycles)

    if t_ref <= 0.0 or e_local <= 0.0:
        raise PaperModelError(
            "The reference time and energy must be positive; verify task workloads"
        )

    return LocalReference(
        t_local_s=float(t_local),
        t_ref_s=float(t_ref),
        e_local_j=float(e_local),
    )


def offloading_efficiency(
    *,
    t_off_s: float,
    e_off_j: float,
    reference: LocalReference,
    weights: ApplicationWeights,
) -> float:
    """Compute equation (22) without clipping the objective value."""
    t_off = float(t_off_s)
    e_off = float(e_off_j)
    if not isfinite(t_off) or t_off < 0.0:
        raise PaperModelError("t_off_s must be finite and non-negative")
    if not isfinite(e_off) or e_off < 0.0:
        raise PaperModelError("e_off_j must be finite and non-negative")

    return float(
        weights.alpha * (reference.t_ref_s - t_off) / reference.t_ref_s
        + weights.beta * (reference.e_local_j - e_off) / reference.e_local_j
    )


def allocated_cpu_frequency_hz(
    *,
    provider_kind: ProviderKind | str,
    provider_fmax_hz: float,
    reference: LocalReference,
    weights: ApplicationWeights,
    kappa: float,
) -> float:
    """Compute equation (30); MEC servers use their maximum CPU frequency."""
    try:
        kind = ProviderKind(provider_kind)
    except ValueError as exc:
        raise PaperModelError(f"Unknown provider kind: {provider_kind!r}") from exc

    fmax = _positive_finite(provider_fmax_hz, "provider_fmax_hz")
    kappa = _positive_finite(kappa, "kappa")

    if kind is ProviderKind.RSU:
        return float(fmax)

    if weights.beta <= 0.0:
        return float(fmax)

    numerator = weights.alpha * reference.e_local_j
    denominator = 2.0 * weights.beta * kappa * reference.t_ref_s
    f_star = (numerator / denominator) ** (1.0 / 3.0)
    return float(min(f_star, fmax))








