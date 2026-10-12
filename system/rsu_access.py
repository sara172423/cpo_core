from __future__ import annotations

from math import log10
from random import Random
from typing import Any, Iterable, Tuple

from monarch_pylib.model import transmission


def _v2i_pathloss_db(distance_m: float) -> float:
    """Table-III V2I path-loss model used by the project/paper reconstruction."""
    distance_km = max(float(distance_m), 1.0) / 1000.0
    return float(128.1 + 37.6 * log10(distance_km))


def _association_seed(base_seed: int, vehicle_id: int, rsu_id: int) -> int:
    """Stable CSI seed for one vehicle/RSU association snapshot.

    Association must be reproducible for a fixed experiment seed.  The seed is
    intentionally independent of optimizer/application IDs because access-RSU
    selection happens before a particular application is optimized.
    """
    return (
        int(base_seed) * 1_000_003
        + int(vehicle_id) * 97_409
        + int(rsu_id) * 9_973
        + 104_729
    ) & 0xFFFFFFFF


def v2i_channel_gain_for_access(
    *,
    position_xyz: Tuple[float, float, float],
    rsu: Any,
    params: Any,
    base_seed: int,
    vehicle_id: int,
) -> Tuple[float, float]:
    """Return instantaneous V2I channel gain and 3-D distance for association.

    The gain follows the same physical ingredients used elsewhere in the
    project: Table-III V2I path loss, log-normal shadowing, Rayleigh fast
    fading, and vehicle/RSU antenna gains.
    """
    vx, vy, vh = (float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2]))
    distance_m = float(
        transmission.distance_3d(
            vx,
            vy,
            vh,
            float(rsu.x_coord),
            float(rsu.y_coord),
            float(params.h_rsu),
        )
    )

    rng = Random(_association_seed(base_seed, vehicle_id, int(rsu.id)))
    shadowing_db = rng.gauss(0.0, float(params.sigma_v2i))
    shadowing_linear = 10.0 ** (-shadowing_db / 10.0)
    rayleigh_power = max(rng.expovariate(1.0), 1e-12)
    pathloss_db = _v2i_pathloss_db(distance_m)
    mean_path_gain = 10.0 ** (
        (
            float(params.G_vehicle)
            + float(params.G_rsu)
            - pathloss_db
        )
        / 10.0
    )
    gain = float(max(mean_path_gain * shadowing_linear * rayleigh_power, 1e-30))
    return gain, distance_m


def select_access_rsu_by_csi(
    *,
    rsus: Iterable[Any],
    position_xyz: Tuple[float, float, float],
    params: Any,
    base_seed: int,
    vehicle_id: int,
):
    """Choose the covering RSU with the largest instantaneous V2I channel gain.

    Returns ``(rsu, gain, distance_m)``.  If no RSU covers the vehicle, all
    three values are ``None``.  This replaces the older nearest-RSU shortcut
    and matches the article's CSI-based access rule more closely.
    """
    best_rsu = None
    best_gain = None
    best_distance = None
    radius = max(0.0, float(params.cell_radius_rsu))

    for rsu in rsus:
        if getattr(rsu, "x_coord", None) is None or getattr(rsu, "y_coord", None) is None:
            continue
        gain, distance_m = v2i_channel_gain_for_access(
            position_xyz=position_xyz,
            rsu=rsu,
            params=params,
            base_seed=base_seed,
            vehicle_id=vehicle_id,
        )
        if distance_m > radius:
            continue
        if (
            best_rsu is None
            or best_gain is None
            or gain > best_gain
            or (gain == best_gain and int(rsu.id) < int(best_rsu.id))
        ):
            best_rsu = rsu
            best_gain = float(gain)
            best_distance = float(distance_m)

    return best_rsu, best_gain, best_distance
