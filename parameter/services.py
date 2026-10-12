from dataclasses import dataclass
from typing import Any, Dict, Tuple
from parameter.models import Parameter


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Cannot convert to float: {x}")


def _to_int(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        raise ValueError(f"Cannot convert to int: {x}")




def _to_int_tuple(x: Any, *, expected_len: int | None = None) -> Tuple[int, ...]:
    if isinstance(x, str):
        values = [part.strip() for part in x.split(",") if part.strip()]
    elif isinstance(x, (list, tuple)):
        values = list(x)
    else:
        raise ValueError(f"Cannot convert to integer tuple: {x}")
    result = tuple(_to_int(value) for value in values)
    if expected_len is not None and len(result) != int(expected_len):
        raise ValueError(
            f"Expected {expected_len} integer values, got {len(result)}"
        )
    return result


def _require(raw: Dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise KeyError(f"Missing parameter key: {key}")
    return raw[key]


def dbm_to_watt(dbm: float) -> float:
    return 10 ** ((float(dbm) - 30.0) / 10.0)


def mhz_to_hz(mhz: float) -> float:
    return float(mhz) * 1e6


def ghz_to_hz(ghz: float) -> float:
    return float(ghz) * 1e9


@dataclass(frozen=True)
class SimParams:
    B: float
    delta2: float
    Y_v2i: float
    Y_v2v: float
    sigma_v2i: float
    sigma_v2v: float
    h_rsu: float
    h_vehicle: float
    G_rsu: float
    G_vehicle: float
    levy_lambda: float
    p_discard_init: float
    S: float
    k: float
    alpha_n: float
    cell_radius_rsu: float
    rec_noi_rsu: float
    rec_noi_vehicle: float
    pmax_vehicle: float
    pmax_rsu: float
    fmax_vehicle: float
    fmax_rsu: float
    simulate_time: int
    taking_task_time: int
    vehicle_speed_min_kmh: float
    vehicle_speed_max_kmh: float
    application_rate_per_second: float
    simulation_seed: int
    deadline_sampling_weights: Tuple[int, ...]
    source_program_fraction: float

    @property
    def beta_n(self) -> float:
        return 1.0 - self.alpha_n


@dataclass(frozen=True)
class SimParamsLib:
    B_hz: float
    delta2_w: float
    pmax_vehicle_w: float
    pmax_rsu_w: float
    fmax_vehicle_hz: float
    fmax_rsu_hz: float


def load_params_obj() -> SimParams:
    raw = {p.key: p.value for p in Parameter.objects.all()}
    deadline_sampling_weights = _to_int_tuple(
        raw.get("deadline_sampling_weights", [5, 10, 15, 20, 20, 15, 10, 5]),
        expected_len=8,
    )
    if any(value < 0 for value in deadline_sampling_weights) or sum(deadline_sampling_weights) <= 0:
        raise ValueError(
            "deadline_sampling_weights must contain eight non-negative integers with a positive sum"
        )
    source_program_fraction = _to_float(raw.get("source_program_fraction", 0.10))
    if source_program_fraction <= 0.0:
        raise ValueError("source_program_fraction must be greater than zero")
    return SimParams(
        B=_to_float(_require(raw, "B")),
        delta2=_to_float(_require(raw, "delta2")),
        Y_v2i=_to_float(_require(raw, "Y_v2i")),
        Y_v2v=_to_float(_require(raw, "Y_v2v")),
        sigma_v2i=_to_float(_require(raw, "sigma_v2i")),
        sigma_v2v=_to_float(_require(raw, "sigma_v2v")),
        h_rsu=_to_float(_require(raw, "h_rsu")),
        h_vehicle=_to_float(_require(raw, "h_vehicle")),
        G_rsu=_to_float(_require(raw, "G_rsu")),
        G_vehicle=_to_float(_require(raw, "G_vehicle")),
        levy_lambda=_to_float(_require(raw, "levy_lambda")),
        p_discard_init=_to_float(_require(raw, "p_discard_init")),
        S=_to_float(_require(raw, "S")),
        k=_to_float(_require(raw, "k")),
        alpha_n=_to_float(_require(raw, "alpha_n")),
        cell_radius_rsu=_to_float(_require(raw, "cell_radius_rsu")),
        rec_noi_rsu=_to_float(_require(raw, "rec_noi_rsu")),
        rec_noi_vehicle=_to_float(_require(raw, "rec_noi_vehicle")),
        pmax_vehicle=_to_float(_require(raw, "pmax_vehicle")),
        pmax_rsu=_to_float(_require(raw, "pmax_rsu")),
        fmax_vehicle=_to_float(_require(raw, "fmax_vehicle")),
        fmax_rsu=_to_float(_require(raw, "fmax_rsu")),
        simulate_time=_to_int(_require(raw, "simulate_time")),
        taking_task_time=_to_int(_require(raw, "taking_task_time")),
        vehicle_speed_min_kmh=_to_float(raw.get("vehicle_speed_min_kmh", 60.0)),
        vehicle_speed_max_kmh=_to_float(raw.get("vehicle_speed_max_kmh", 80.0)),
        application_rate_per_second=_to_float(raw.get("application_rate_per_second", 10.0)),
        simulation_seed=_to_int(raw.get("simulation_seed", 1)),
        deadline_sampling_weights=deadline_sampling_weights,
        source_program_fraction=source_program_fraction,
    )


def load_params_for_lib() -> SimParamsLib:
    p = load_params_obj()
    return SimParamsLib(
        B_hz=mhz_to_hz(p.B),
        delta2_w=dbm_to_watt(p.delta2),
        pmax_vehicle_w=dbm_to_watt(p.pmax_vehicle),
        pmax_rsu_w=dbm_to_watt(p.pmax_rsu),
        fmax_vehicle_hz=ghz_to_hz(p.fmax_vehicle),
        fmax_rsu_hz=ghz_to_hz(p.fmax_rsu),
    )