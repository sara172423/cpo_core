from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from uuid import uuid4

from django.conf import settings

from dag.models import ApplicationType, Task, TaskDependency
from monarch_pylib.model import transmission
from object.models import RSU, ServiceProvider, Vehicle
from parameter.services import load_params_obj
from resource.models import Resource

from run.benchmark.protocol import (
    FAIR_OPTIMIZER_COMPARISON,
    PAPER_REPRODUCTION,
    POPULATION_ALGORITHMS,
    resolve_experiment_mode,
)

from .context import BenchmarkSnapshot, build_synthetic_joint_context
from .runner import run_joint_context_benchmark


ARTICLE_DOI = "10.1109/TVT.2025.3540639"
DEADLINES_MS = (30, 40, 50, 60, 70, 80, 90, 100)
VEHICLE_COUNTS = (52, 56, 60, 64, 68, 72, 76)
PAPER_MAX_MISSION_VEHICLES = max(VEHICLE_COUNTS)
TABLE_III_MIN_SPEED_KMH = 60.0
VEHICLE_SPEEDS_KMH = (75, 80, 85, 90, 95, 100, 105)
MEC_CAPACITIES_GHZ = (30, 40, 50, 60, 70, 80)
ALGORITHM_LABELS = {
    "dcsga": "DCSGA",
    "dtosc": "DTOSC",
    "to_v2i": "TO-V2I",
    "to_wo_c": "TO-w.o.-C",
    "to_wo_r": "TO-w.o.-R",
    "gwo_aco": "D-GWO",
    "gwo": "D-GWO",
    "gpc": "D-GPC",
    "cpo": "DCC-DCPO",
    "dcpo_base": "DCPO-base",
    "dcpo_criticality": "DCPO+C",
    "dcpo_cache": "DCPO+K",
    "puma": "D-PO",
}
PAPER_FIGURE_ALGORITHMS = {
    "figure_6": ("dcsga",),
    "figure_7": ("dcsga", "dtosc", "to_v2i", "to_wo_c", "to_wo_r"),
    "figure_8": ("dcsga", "to_wo_c", "to_wo_r"),
    "figure_9": ("dcsga", "dtosc"),
    "figure_10": ("dcsga", "dtosc"),
}

FIGURE_ALGORITHMS = dict(PAPER_FIGURE_ALGORITHMS)



def _close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    limit = max(1e-30, abs(float(expected)) * tolerance)
    return abs(float(actual) - float(expected)) <= limit


def _validate_paper_parameters() -> None:
    params = load_params_obj()
    expected = {
        "B": 20.0,
        "delta2": -114.0,
        "Y_v2i": 3.76,
        "Y_v2v": 1.8,
        "sigma_v2i": 8.0,
        "sigma_v2v": 3.0,
        "h_rsu": 5.0,
        "h_vehicle": 1.5,
        "G_rsu": 8.0,
        "G_vehicle": 3.0,
        "levy_lambda": 1.5,
        "p_discard_init": 0.2,
        "S": 50.0,
        "k": 1e-25,
        "cell_radius_rsu": 250.0,
        "rec_noi_rsu": 5.0,
        "rec_noi_vehicle": 9.0,
        "pmax_vehicle": 23.0,
        "pmax_rsu": 30.0,
        "fmax_vehicle": 3.0,
        "fmax_rsu": 50.0,
        "vehicle_speed_min_kmh": 60.0,
        "vehicle_speed_max_kmh": 80.0,
        "application_rate_per_second": 10.0,
    }
    mismatches = []
    for name, expected_value in expected.items():
        actual_value = float(getattr(params, name))
        if not _close(actual_value, expected_value):
            mismatches.append(
                f"{name}={actual_value} (paper value: {expected_value})"
            )
    if mismatches:
        raise ValueError(
            "Paper experiment parameters do not match Table III: "
            + "; ".join(mismatches)
        )


def _validate_paper_data(
    vehicles: Sequence[Vehicle],
    rsus: Sequence[RSU],
    application_types: Dict[int, ApplicationType],
) -> None:
    expected_cache_bytes = 625000000
    invalid_vehicles = [
        int(vehicle.id)
        for vehicle in vehicles
        if int(vehicle.cpu_capacity) != 3000000000
        or int(vehicle.cache_capacity) != expected_cache_bytes
    ]
    if invalid_vehicles:
        raise ValueError(
            f"Vehicles do not match Table III CPU/cache values: {invalid_vehicles}"
        )
    invalid_rsus = [
        int(rsu.id)
        for rsu in rsus
        if int(rsu.cpu_capacity) != 50000000000
        or int(rsu.cache_capacity) != expected_cache_bytes
    ]
    if invalid_rsus:
        raise ValueError(
            f"RSUs do not match Table III CPU/cache values: {invalid_rsus}"
        )
    task_rows = list(
        Task.objects.filter(
            application_type_id_id__in=[
                application_type.id
                for application_type in application_types.values()
            ]
        )
        .select_related("task_type_id")
        .order_by("application_type_id_id", "id")
    )
    tasks_by_type: Dict[int, List[Task]] = {}
    for task in task_rows:
        tasks_by_type.setdefault(int(task.application_type_id_id), []).append(task)
    for deadline_ms, application_type in application_types.items():
        tasks = tasks_by_type.get(int(application_type.id), [])
        if len(tasks) != 10:
            raise ValueError(
                f"Application type {application_type.id} for {deadline_ms} ms must have 10 tasks"
            )
        for task in tasks:
            cycles = int(task.workload_cycles)
            if cycles < 10000000 or cycles > 30000000:
                raise ValueError(
                    f"Task {task.id} workload is outside the Table III range"
                )
            task_type = task.task_type_id
            if task_type is None:
                raise ValueError(f"Task {task.id} has no task type")
            snapshot = task.initial_snapshot or {}
            task_type_snapshot = task_type.initial_snapshot or {}
            output_bits = snapshot.get(
                "output_size_bits",
                task_type_snapshot.get("communication_data_bits"),
            )
            service_bits = task_type_snapshot.get(
                "service_environment_size_bits",
                int(task_type.size) * 8,
            )
            if output_bits is None or not 100000 <= int(output_bits) <= 300000:
                raise ValueError(
                    f"Task {task.id} communication data is outside the Table III range"
                )
            if not 500000000 <= int(service_bits) <= 1000000000:
                raise ValueError(
                    f"Task type {task_type.id} service size is outside the Table III range"
                )

def _paper_reconstruction_audit(
    application_types: Dict[int, ApplicationType],
) -> Dict[str, Any]:
    """Describe published-vs-reconstructed benchmark inputs without mutating data."""
    type_ids = sorted({int(item.id) for item in application_types.values()})
    tasks = list(
        Task.objects.filter(application_type_id_id__in=type_ids)
        .select_related("task_type_id")
        .order_by("application_type_id_id", "id")
    )
    task_ids = [int(task.id) for task in tasks]
    dependencies = list(
        TaskDependency.objects.filter(
            parent_task_id_id__in=task_ids,
            child_task_id_id__in=task_ids,
        )
        .select_related("parent_task_id", "child_task_id")
        .order_by("parent_task_id__application_type_id_id", "id")
    )

    edge_counts: Dict[int, int] = {type_id: 0 for type_id in type_ids}
    explicit_edge_data = 0
    topology_signatures: Dict[int, List[str]] = {type_id: [] for type_id in type_ids}
    for dependency in dependencies:
        parent = dependency.parent_task_id
        child = dependency.child_task_id
        if parent is None or child is None:
            continue
        type_id = int(parent.application_type_id_id)
        edge_counts[type_id] = edge_counts.get(type_id, 0) + 1
        snapshot = dependency.initial_snapshot or {}
        if snapshot.get("communication_data_bits") is not None:
            explicit_edge_data += 1
        topology_signatures.setdefault(type_id, []).append(
            f"{str(parent.index or parent.id)}->{str(child.index or child.id)}"
        )

    task_sequences: Dict[int, List[int]] = {type_id: [] for type_id in type_ids}
    declared_compile_workloads = 0
    for task in tasks:
        type_id = int(task.application_type_id_id)
        task_sequences.setdefault(type_id, []).append(int(task.workload_cycles))
        task_type = task.task_type_id
        snapshot = (task_type.initial_snapshot or {}) if task_type is not None else {}
        if any(
            key in snapshot
            for key in (
                "compile_workload_cycles",
                "compile_cycles",
                "w_k_cycles",
                "W_k",
                "Wk",
            )
        ):
            declared_compile_workloads += 1

    unique_topologies = {
        tuple(sorted(values))
        for values in topology_signatures.values()
    }
    unique_workload_sequences = {
        tuple(values)
        for values in task_sequences.values()
    }
    total_edges = len(dependencies)

    return {
        "classification": {
            "ARTICLE_EXACT": [
                "Table-III parameter ranges and constants validated by the benchmark",
                "ten tasks per application type",
                "deadline set 30..100 ms",
            ],
            "REFERENCE_OR_ARTICLE_COMPATIBLE_RECONSTRUCTION": [
                "DAG realizations",
                "per-task workload realization inside the published range",
                "per-edge communication-data realization inside the published range",
            ],
            "DECLARED_ASSUMPTION": [
                "total road-vehicle population for Figures 6/7/8/10",
                "source-program size model when not numerically specified by the 2025 article",
                "service compile workload W_k when not numerically specified",
                "dynamic-arrival process omitted because its distribution/horizon is unpublished",
            ],
        },
        "dag": {
            "application_type_count": len(type_ids),
            "edge_counts_by_application_type": {
                str(key): int(value) for key, value in edge_counts.items()
            },
            "unique_topology_count": len(unique_topologies),
            "source_exact": False,
        },
        "edge_communication_data": {
            "dependency_count": int(total_edges),
            "explicit_edge_snapshot_count": int(explicit_edge_data),
            "parent_output_fallback_count": int(total_edges - explicit_edge_data),
            "source_exact": bool(total_edges > 0 and explicit_edge_data == total_edges),
        },
        "task_workloads": {
            "unique_workload_sequence_count": len(unique_workload_sequences),
            "article_range_validated": True,
            "source_exact_realization": False,
        },
        "compile_workload": {
            "tasks_with_task_type_compile_workload_snapshot": int(
                declared_compile_workloads
            ),
            "article_numeric_value_published": False,
        },
        "source_program_size": {
            "service_program_transfer_energy_in_2025_article": True,
            "article_equation": 27,
            "article_numeric_size_published": False,
            "current_size_model": "dtosc-2022-ratio-0.1-context-derived",
            "classification": "DECLARED_ASSUMPTION_FOR_SIZE_ONLY",
        },
    }


def _normalize_figure(value: str) -> str:
    compact = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "6": "figure_6",
        "7": "figure_7",
        "8": "figure_8",
        "9": "figure_9",
        "10": "figure_10",
        "fig6": "figure_6",
        "fig7": "figure_7",
        "fig8": "figure_8",
        "fig9": "figure_9",
        "fig10": "figure_10",
        "*": "all",
        "all_figures": "all",
    }
    compact = aliases.get(compact, compact)
    if compact == "all":
        return compact
    if compact not in FIGURE_ALGORITHMS:
        raise ValueError(f"Unsupported paper figure: {value}")
    return compact


def _stats(values: Sequence[float]) -> Dict[str, float]:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "ci95": 0.0,
        }

    def percentile(fraction: float) -> float:
        if len(numbers) == 1:
            return float(numbers[0])
        position = max(0.0, min(1.0, float(fraction))) * (len(numbers) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(numbers[lower])
        weight = position - lower
        return float(numbers[lower] * (1.0 - weight) + numbers[upper] * weight)

    standard_deviation = float(stdev(numbers)) if len(numbers) > 1 else 0.0
    q1 = percentile(0.25)
    q3 = percentile(0.75)
    return {
        "mean": float(mean(numbers)),
        "std": standard_deviation,
        "min": float(min(numbers)),
        "max": float(max(numbers)),
        "median": float(median(numbers)),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "ci95": float(1.96 * standard_deviation / math.sqrt(len(numbers))),
    }


def _paired_algorithm_statistics(
    rows: Sequence[Dict[str, Any]],
    metric: str = "total_efficiency",
) -> List[Dict[str, Any]]:
    """All-pairs paired-seed inference with optional SciPy Wilcoxon p-values."""
    by_algorithm: Dict[str, Dict[Tuple[str, int], float]] = {}
    for row in rows:
        if row.get(metric) is None:
            continue
        algorithm = str(row.get("algorithm", ""))
        pair_key = (str(row.get("scenario_id", "")), int(row.get("seed", 0)))
        by_algorithm.setdefault(algorithm, {})[pair_key] = float(row[metric])

    comparisons = []
    algorithms = sorted(by_algorithm)
    for left_index, left in enumerate(algorithms):
        for right in algorithms[left_index + 1 :]:
            keys = sorted(set(by_algorithm[left]) & set(by_algorithm[right]))
            differences = [by_algorithm[left][key] - by_algorithm[right][key] for key in keys]
            nonzero = [value for value in differences if abs(value) > 1e-12]
            ranks = []
            ordered = sorted(enumerate(nonzero), key=lambda item: abs(item[1]))
            position = 0
            while position < len(ordered):
                end = position + 1
                while end < len(ordered) and abs(abs(ordered[end][1]) - abs(ordered[position][1])) <= 1e-12:
                    end += 1
                average_rank = 0.5 * ((position + 1) + end)
                for source_index, value in ordered[position:end]:
                    ranks.append((source_index, value, average_rank))
                position = end
            positive = sum(rank for _index, value, rank in ranks if value > 0.0)
            negative = sum(rank for _index, value, rank in ranks if value < 0.0)
            denominator = positive + negative
            effect = 0.0 if denominator <= 0.0 else (positive - negative) / denominator
            p_value = None
            if nonzero:
                try:
                    from importlib import import_module

                    wilcoxon = import_module("scipy.stats").wilcoxon
                    p_value = float(
                        wilcoxon(
                            [by_algorithm[left][key] for key in keys],
                            [by_algorithm[right][key] for key in keys],
                            alternative="two-sided",
                            zero_method="wilcox",
                            method="auto",
                        ).pvalue
                    )
                except Exception:
                    p_value = None
            comparisons.append({
                "algorithm_a": left,
                "algorithm_b": right,
                "metric": metric,
                "paired_sample_count": len(keys),
                "mean_difference_a_minus_b": float(mean(differences)) if differences else 0.0,
                "median_difference_a_minus_b": float(median(differences)) if differences else 0.0,
                "wins_a": sum(value > 1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "wins_b": sum(value < -1e-12 for value in differences),
                "rank_biserial_a_minus_b": float(effect),
                "wilcoxon_two_sided_p": p_value,
                "holm_adjusted_p": None,
            })

    available = sorted(
        ((index, row["wilcoxon_two_sided_p"]) for index, row in enumerate(comparisons)
         if row["wilcoxon_two_sided_p"] is not None),
        key=lambda item: item[1],
    )
    running = 0.0
    count = len(available)
    for rank, (index, p_value) in enumerate(available):
        adjusted = min(1.0, float(p_value) * float(count - rank))
        running = max(running, adjusted)
        comparisons[index]["holm_adjusted_p"] = float(running)
    return comparisons


def _load_paper_environment() -> Tuple[List[Vehicle], List[RSU], Dict[int, ApplicationType]]:
    # Database ``is_mission`` is not used to choose the paper mission set.
    # A vehicle is eligible when it has a vehicle service-provider row; the
    # actual mission subset is selected per figure (52..76), and all remaining
    # selected road vehicles stay available as V2V cooperative providers.
    eligible_vehicle_ids = list(
        ServiceProvider.objects.filter(
            type="vehicle",
            vehicle_id_id__isnull=False,
        )
        .order_by("vehicle_id_id")
        .values_list("vehicle_id_id", flat=True)
        .distinct()
    )
    vehicles = list(
        Vehicle.objects.filter(id__in=eligible_vehicle_ids).order_by("id")
    )
    if len(vehicles) < PAPER_MAX_MISSION_VEHICLES:
        raise ValueError(
            "The paper experiments require at least "
            f"{PAPER_MAX_MISSION_VEHICLES} eligible vehicle service providers; "
            f"found {len(vehicles)}"
        )
    rsus = list(RSU.objects.order_by("id")[:5])
    if len(rsus) != 5:
        raise ValueError(f"The paper experiments require 5 RSUs; found {len(rsus)}")
    local_provider_count = ServiceProvider.objects.filter(
        type="vehicle",
        vehicle_id_id__in=[vehicle.id for vehicle in vehicles],
    ).values("vehicle_id_id").distinct().count()
    if local_provider_count != len(vehicles):
        raise ValueError("Every paper vehicle must have one local service provider")
    vehicle_provider_ids = list(
        ServiceProvider.objects.filter(
            type="vehicle",
            vehicle_id_id__in=[vehicle.id for vehicle in vehicles],
        )
        .order_by("id")
        .values_list("id", flat=True)
    )
    if len(vehicle_provider_ids) != len(vehicles):
        raise ValueError(
            "Every paper vehicle must have exactly one local service provider"
        )
    vehicle_resources = Resource.objects.filter(
        sp_id_id__in=vehicle_provider_ids
    )
    if (
        vehicle_resources.count() != len(vehicle_provider_ids)
        or vehicle_resources.values("sp_id_id").distinct().count()
        != len(vehicle_provider_ids)
    ):
        raise ValueError(
            "Every eligible vehicle service provider must have exactly one resource row"
        )
    rsu_provider_count = ServiceProvider.objects.filter(
        type="rsu",
        rsu_id_id__in=[rsu.id for rsu in rsus],
    ).values("rsu_id_id").distinct().count()
    if rsu_provider_count != len(rsus):
        raise ValueError("Every paper RSU must have one service provider")
    application_types = list(ApplicationType.objects.order_by("deadline", "id"))
    by_deadline: Dict[int, ApplicationType] = {}
    for application_type in application_types:
        deadline_ms = int(application_type.deadline)
        if deadline_ms in DEADLINES_MS and deadline_ms not in by_deadline:
            by_deadline[deadline_ms] = application_type
    missing = [deadline for deadline in DEADLINES_MS if deadline not in by_deadline]
    if missing:
        raise ValueError(f"Application types are missing paper deadlines: {missing}")
    _validate_paper_parameters()
    _validate_paper_data(vehicles, rsus, by_deadline)
    return vehicles, rsus, by_deadline


def _road_geometry(vehicles: Sequence[Vehicle]) -> Tuple[float, List[float]]:
    road_length = 1000.0
    lane_values: List[float] = []
    for vehicle in vehicles:
        snapshot = vehicle.initial_snapshot or {}
        if snapshot.get("road_length_m") is not None:
            road_length = float(snapshot["road_length_m"])
        lane_x = snapshot.get("lane_x_coord")
        if lane_x is not None:
            lane_values.append(float(lane_x))
    lanes = sorted(set(lane_values))
    if not _close(road_length, 1000.0):
        raise ValueError(
            f"Road length is {road_length} m; Table III requires 1000 m"
        )
    if len(lanes) != 4:
        raise ValueError(
            f"The paper experiment requires 4 lanes; found {len(lanes)}"
        )
    lane_gaps = [lanes[index + 1] - lanes[index] for index in range(3)]
    if any(not _close(gap, 4.0) for gap in lane_gaps):
        raise ValueError(
            f"Lane coordinates do not represent the Table III width of 4 m: {lanes}"
        )
    return road_length, lanes


def _nearest_rsu_id(
    position: Tuple[float, float],
    rsus: Sequence[RSU],
    cell_radius_m: float,
    h_vehicle: float,
    h_rsu: float,
) -> int:
    nearest_id = None
    nearest_distance = None
    for rsu in rsus:
        distance = float(
            transmission.distance_3d(
                float(position[0]),
                float(position[1]),
                float(h_vehicle),
                float(rsu.x_coord),
                float(rsu.y_coord),
                float(h_rsu),
            )
        )
        if distance > float(cell_radius_m):
            continue
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_id = int(rsu.id)
    if nearest_id is None:
        raise ValueError(f"Vehicle position {position} is outside all RSU cells")
    return nearest_id


def _vehicle_count_for_speed(
    speed_kmh: float,
    road_length_m: float,
    available_vehicle_count: int | None = None,
) -> int:
    """Return the 3GPP speed-density road population.

    The article uses a mean inter-vehicle distance of 2.5 seconds times
    vehicle speed.  The physical count is therefore derived from speed and
    road geometry, not hard-coded to the database size.  When a finite
    database pool is supplied, the result is capped by that available pool.
    """
    spacing_m = 2.5 * (float(speed_kmh) / 3.6)
    expected = max(1, int(round((4.0 * float(road_length_m)) / spacing_m)))
    if available_vehicle_count is None:
        return expected
    return min(expected, max(1, int(available_vehicle_count)))


def _fixed_road_vehicle_count(vehicles: Sequence[Vehicle]) -> int:
    """Choose a stable paper-compatible road pool for Figures 6, 7, 8, 10.

    The paper publishes mission counts but not a separate total road count.
    We use every eligible database vehicle up to the maximum physically
    consistent Table-III density (four 1-km lanes at 60 km/h).  Thus changing
    the database size requires no code edit, while an unrealistically large
    database cannot make the road denser than the paper model permits.
    """
    road_length_m, _ = _road_geometry(vehicles)
    physical_cap = _vehicle_count_for_speed(
        TABLE_III_MIN_SPEED_KMH,
        road_length_m,
    )
    selected_count = min(len(vehicles), physical_cap)
    if selected_count < PAPER_MAX_MISSION_VEHICLES:
        raise ValueError(
            "The paper mission sweep requires at least "
            f"{PAPER_MAX_MISSION_VEHICLES} road vehicles; "
            f"only {selected_count} are available after applying the "
            "Table-III density model"
        )
    return int(selected_count)


def _road_state(
    vehicles: Sequence[Vehicle],
    rsus: Sequence[RSU],
    seed: int,
    road_vehicle_count: int,
    speed_kmh: float | None,
) -> Tuple[BenchmarkSnapshot, List[int], Dict[int, float], Dict[str, Any]]:
    if road_vehicle_count < 1 or road_vehicle_count > len(vehicles):
        raise ValueError("Invalid road vehicle count")
    params = load_params_obj()
    road_length_m, lane_x_values = _road_geometry(vehicles)
    rng = random.Random(int(seed) * 1000003 + 7919)
    ordered = list(vehicles)
    rng.shuffle(ordered)
    selected = ordered[:road_vehicle_count]
    positions: Dict[int, Tuple[float, float]] = {}
    speeds: Dict[int, float] = {}
    base_count, remainder = divmod(road_vehicle_count, len(lane_x_values))
    lane_counts = [
        base_count + (1 if index < remainder else 0)
        for index in range(len(lane_x_values))
    ]
    selected_index = 0
    for lane_x, lane_count in zip(lane_x_values, lane_counts):
        lane_positions = sorted(
            rng.uniform(0.0, road_length_m)
            for _ in range(lane_count)
        )
        for y_coord in lane_positions:
            vehicle = selected[selected_index]
            selected_index += 1
            positions[int(vehicle.id)] = (float(lane_x), float(y_coord))
            speeds[int(vehicle.id)] = (
                float(speed_kmh)
                if speed_kmh is not None
                else float(rng.uniform(60.0, 80.0))
            )
    vehicle_rsu_ids = {
        vehicle_id: _nearest_rsu_id(
            position,
            rsus,
            float(params.cell_radius_rsu),
            float(params.h_vehicle),
            float(params.h_rsu),
        )
        for vehicle_id, position in positions.items()
    }
    rsu_vehicle_counts: Dict[int, int] = {}
    for rsu_id in vehicle_rsu_ids.values():
        rsu_vehicle_counts[rsu_id] = rsu_vehicle_counts.get(rsu_id, 0) + 1
    snapshot = BenchmarkSnapshot(
        at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        time_step_s=0.0,
        positions=positions,
        vehicle_rsu_ids=vehicle_rsu_ids,
        rsu_vehicle_counts=rsu_vehicle_counts,
    )
    speed_values = list(speeds.values())
    selected_ids = [int(vehicle.id) for vehicle in selected]
    metadata = {
        "road_vehicle_count": int(road_vehicle_count),
        "road_length_m": float(road_length_m),
        "number_of_lanes": 4,
        "mean_vehicle_speed_kmh": float(mean(speed_values)),
        "vehicle_speed_min_kmh": float(min(speed_values)),
        "vehicle_speed_max_kmh": float(max(speed_values)),
        "spatial_process": "spatial_poisson_conditioned_on_lane_vehicle_counts",
        "lane_vehicle_counts": lane_counts,
        "road_snapshot_signature": _snapshot_signature(
            snapshot,
            speeds,
            selected_ids,
        ),
    }
    return snapshot, selected_ids, speeds, metadata



def _snapshot_signature(
    snapshot: BenchmarkSnapshot,
    speeds: Dict[int, float],
    vehicle_ids: Sequence[int],
) -> str:
    """Stable signature for proving that a diagnostic keeps the base road state frozen."""
    rows = []
    for vehicle_id in sorted(int(value) for value in vehicle_ids):
        position = snapshot.positions.get(vehicle_id)
        rsu_id = snapshot.vehicle_rsu_ids.get(vehicle_id)
        if position is None or rsu_id is None:
            raise ValueError(
                f"Vehicle {vehicle_id} is missing from the benchmark snapshot"
            )
        rows.append({
            "vehicle_id": int(vehicle_id),
            "x": round(float(position[0]), 12),
            "y": round(float(position[1]), 12),
            "speed_kmh": round(float(speeds[vehicle_id]), 12),
            "rsu_id": int(rsu_id),
        })
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extend_road_state_preserving_baseline(
    baseline: Tuple[
        BenchmarkSnapshot,
        List[int],
        Dict[int, float],
        Dict[str, Any],
    ],
    all_vehicles: Sequence[Vehicle],
    rsus: Sequence[RSU],
    seed: int,
    target_road_vehicle_count: int,
    speed_kmh: float | None,
) -> Tuple[BenchmarkSnapshot, List[int], Dict[int, float], Dict[str, Any]]:
    """Add cooperative-only vehicles without moving/reseeding the baseline road state.

    This is diagnostic-only.  The positions, speeds and access-RSU association of
    every baseline vehicle remain byte-for-byte deterministic with the ordinary
    paper reconstruction.  Additional vehicles are drawn from the remaining DB
    pool with an independent RNG stream and fill only the extra lane slots.
    """
    base_snapshot, base_ids, base_speeds, base_metadata = baseline
    base_ids = [int(value) for value in base_ids]
    base_count = len(base_ids)
    target_count = int(target_road_vehicle_count)
    if target_count < base_count:
        raise ValueError(
            "Controlled road extension cannot be smaller than the frozen baseline"
        )
    if target_count == base_count:
        metadata = dict(base_metadata)
        metadata.update({
            "baseline_road_vehicle_count": int(base_count),
            "additional_cooperative_vehicle_count": 0,
            "mission_snapshot_frozen": True,
            "baseline_snapshot_signature": _snapshot_signature(
                base_snapshot,
                base_speeds,
                base_ids,
            ),
            "spatial_process": "frozen-baseline-road-state",
        })
        return base_snapshot, list(base_ids), dict(base_speeds), metadata

    params = load_params_obj()
    road_length_m, lane_x_values = _road_geometry(all_vehicles)
    baseline_set = set(base_ids)
    remaining = [
        vehicle
        for vehicle in all_vehicles
        if int(vehicle.id) not in baseline_set
    ]
    extra_count = target_count - base_count
    if extra_count > len(remaining):
        raise ValueError(
            f"Controlled road extension needs {extra_count} extra vehicles; "
            f"only {len(remaining)} are available"
        )

    extension_rng = random.Random(int(seed) * 1000003 + 32452843)
    extension_rng.shuffle(remaining)
    extras = remaining[:extra_count]

    positions = {
        int(vehicle_id): (float(position[0]), float(position[1]))
        for vehicle_id, position in base_snapshot.positions.items()
    }
    speeds = {int(key): float(value) for key, value in base_speeds.items()}

    # Preserve every baseline point.  Allocate only the additional slots needed
    # to reach the balanced target lane counts.
    target_base, target_remainder = divmod(target_count, len(lane_x_values))
    target_lane_counts = [
        target_base + (1 if index < target_remainder else 0)
        for index in range(len(lane_x_values))
    ]
    current_lane_counts = [
        sum(
            1
            for vehicle_id in base_ids
            if abs(float(positions[vehicle_id][0]) - float(lane_x)) <= 1e-9
        )
        for lane_x in lane_x_values
    ]
    lane_extra_counts = [
        int(target - current)
        for target, current in zip(target_lane_counts, current_lane_counts)
    ]
    if any(value < 0 for value in lane_extra_counts):
        raise ValueError(
            "Target diagnostic lane density is smaller than the frozen baseline"
        )
    if sum(lane_extra_counts) != extra_count:
        raise ValueError("Controlled road extension lane accounting is inconsistent")

    extra_index = 0
    for lane_x, lane_extra_count in zip(lane_x_values, lane_extra_counts):
        for _ in range(lane_extra_count):
            vehicle = extras[extra_index]
            extra_index += 1
            vehicle_id = int(vehicle.id)
            positions[vehicle_id] = (
                float(lane_x),
                float(extension_rng.uniform(0.0, road_length_m)),
            )
            speeds[vehicle_id] = (
                float(speed_kmh)
                if speed_kmh is not None
                else float(extension_rng.uniform(60.0, 80.0))
            )

    vehicle_rsu_ids = {
        vehicle_id: _nearest_rsu_id(
            position,
            rsus,
            float(params.cell_radius_rsu),
            float(params.h_vehicle),
            float(params.h_rsu),
        )
        for vehicle_id, position in positions.items()
    }
    rsu_vehicle_counts: Dict[int, int] = {}
    for rsu_id in vehicle_rsu_ids.values():
        rsu_vehicle_counts[rsu_id] = rsu_vehicle_counts.get(rsu_id, 0) + 1

    snapshot = BenchmarkSnapshot(
        at=base_snapshot.at,
        time_step_s=float(base_snapshot.time_step_s),
        positions=positions,
        vehicle_rsu_ids=vehicle_rsu_ids,
        rsu_vehicle_counts=rsu_vehicle_counts,
    )
    road_vehicle_ids = list(base_ids) + [int(vehicle.id) for vehicle in extras]
    speed_values = [float(speeds[vehicle_id]) for vehicle_id in road_vehicle_ids]
    metadata = {
        "road_vehicle_count": int(target_count),
        "road_length_m": float(road_length_m),
        "number_of_lanes": int(len(lane_x_values)),
        "mean_vehicle_speed_kmh": float(mean(speed_values)),
        "vehicle_speed_min_kmh": float(min(speed_values)),
        "vehicle_speed_max_kmh": float(max(speed_values)),
        "spatial_process": "frozen-baseline-plus-seeded-cooperative-extension",
        "lane_vehicle_counts": target_lane_counts,
        "baseline_road_vehicle_count": int(base_count),
        "additional_cooperative_vehicle_count": int(extra_count),
        "mission_snapshot_frozen": True,
        "baseline_snapshot_signature": _snapshot_signature(
            base_snapshot,
            base_speeds,
            base_ids,
        ),
        "extended_baseline_snapshot_signature": _snapshot_signature(
            snapshot,
            speeds,
            base_ids,
        ),
        "baseline_snapshot_signature_match": (
            _snapshot_signature(base_snapshot, base_speeds, base_ids)
            == _snapshot_signature(snapshot, speeds, base_ids)
        ),
        "additional_cooperative_vehicle_ids": [
            int(vehicle.id) for vehicle in extras
        ],
        "road_snapshot_signature": _snapshot_signature(
            snapshot,
            speeds,
            road_vehicle_ids,
        ),
    }
    return snapshot, road_vehicle_ids, speeds, metadata


def _assignments(
    road_vehicle_ids: Sequence[int],
    mission_vehicle_count: int,
    app_types: Dict[int, ApplicationType],
    seed: int,
    *,
    assignment_pool_count: int | None = None,
) -> List[Dict[str, int]]:
    """Build deterministic and balanced deadline/DAG assignments.

    Deadline is a scenario attribute, while ``application_type_id`` selects
    only the DAG/task template.  The two are deliberately decoupled so a
    particular deadline is not permanently tied to one DAG topology.

    Figure 7 still uses one master assignment pool and nested prefixes.  Over
    each complete block of 64 assignments, every one of the eight deadlines
    is paired exactly once with every one of the eight DAG templates.
    """
    if mission_vehicle_count > len(road_vehicle_ids):
        raise ValueError("Mission vehicle count exceeds road vehicle count")

    pool_count = (
        int(mission_vehicle_count)
        if assignment_pool_count is None
        else int(assignment_pool_count)
    )
    if pool_count < mission_vehicle_count:
        raise ValueError(
            "assignment_pool_count cannot be smaller than mission_vehicle_count"
        )
    if pool_count > len(road_vehicle_ids):
        raise ValueError("assignment_pool_count exceeds road vehicle count")

    dag_type_ids = list(dict.fromkeys(
        int(application_type.id)
        for application_type in app_types.values()
    ))
    if len(dag_type_ids) != len(DEADLINES_MS):
        raise ValueError(
            "Balanced paper scenarios require one distinct DAG template for "
            "each configured paper deadline"
        )

    mission_rng = random.Random(int(seed) * 1000033 + 104729)
    pair_rng = random.Random(int(seed) * 1000033 + 130363)

    mission_order = list(road_vehicle_ids)
    mission_rng.shuffle(mission_order)
    mission_order = mission_order[:pool_count]

    # Non-uniform deadline distribution:
    # strict and loose deadlines are less frequent, while normal QoS
    # requirements are more frequent. The weights follow a Gaussian-like
    # symmetric distribution:
    # 30/100 -> 5%, 40/90 -> 10%, 50/80 -> 15%, 60/70 -> 20%.
    deadline_weights = {
        30: 5,
        40: 10,
        50: 15,
        60: 20,
        70: 20,
        80: 15,
        90: 10,
        100: 5,
    }

    weighted_deadlines: List[int] = []
    for deadline in DEADLINES_MS:
        weighted_deadlines.extend([int(deadline)] * deadline_weights[int(deadline)])

    deadline_order = list(weighted_deadlines)
    dag_order = list(dag_type_ids)

    pair_rng.shuffle(deadline_order)
    pair_rng.shuffle(dag_order)

    assignment_pairs: List[Tuple[int, int]] = []
    while len(assignment_pairs) < pool_count:
        for index, deadline in enumerate(deadline_order):
            assignment_pairs.append(
                (
                    int(deadline),
                    int(dag_order[index % len(dag_order)]),
                )
            )
            if len(assignment_pairs) >= pool_count:
                break
        pair_rng.shuffle(deadline_order)
        pair_rng.shuffle(dag_order)

    rows: List[Dict[str, int]] = []
    for index in range(mission_vehicle_count):
        deadline_ms, dag_type_id = assignment_pairs[index]
        rows.append(
            {
                "application_id": -(index + 1),
                "vehicle_id": int(mission_order[index]),
                "application_type_id": int(dag_type_id),
                "deadline_ms": int(deadline_ms),
            }
        )
    return rows


def _build_scenario(
    figure: str,
    seed: int,
    mission_vehicle_count: int,
    road_vehicle_count: int,
    speed_kmh: float | None,
    mec_capacity_ghz: float,
    sweep_parameter: str,
    sweep_value: float,
    environment: Tuple[
        List[Vehicle],
        List[RSU],
        Dict[int, ApplicationType],
    ],
    *,
    assignment_pool_count: int | None = None,
    mission_assignment_vehicle_ids: Sequence[int] | None = None,
    road_state_override: Tuple[
        BenchmarkSnapshot,
        List[int],
        Dict[int, float],
        Dict[str, Any],
    ] | None = None,
) -> Dict[str, Any]:
    vehicles, rsus, app_types = environment
    if road_state_override is None:
        snapshot, road_vehicle_ids, speeds, road_metadata = _road_state(
            vehicles,
            rsus,
            seed,
            road_vehicle_count,
            speed_kmh,
        )
    else:
        snapshot, road_vehicle_ids, speeds, road_metadata = road_state_override
        if len(road_vehicle_ids) != int(road_vehicle_count):
            raise ValueError(
                "road_state_override vehicle count does not match road_vehicle_count"
            )
    assignment_vehicle_ids = (
        list(road_vehicle_ids)
        if mission_assignment_vehicle_ids is None
        else [int(value) for value in mission_assignment_vehicle_ids]
    )
    missing_assignment_vehicles = sorted(
        set(assignment_vehicle_ids) - set(road_vehicle_ids)
    )
    if missing_assignment_vehicles:
        raise ValueError(
            "Mission assignment pool contains vehicles outside the road snapshot: "
            f"{missing_assignment_vehicles}"
        )
    assignments = _assignments(
        assignment_vehicle_ids,
        mission_vehicle_count,
        app_types,
        seed,
        assignment_pool_count=assignment_pool_count,
    )
    metadata = {
        "experiment_name": figure,
        "scenario_id": f"{figure}_seed_{seed}_{sweep_parameter}_{sweep_value}",
        "scenario_seed": int(seed),
        "sweep_parameter": sweep_parameter,
        "sweep_value": float(sweep_value),
        "mission_vehicle_count": int(mission_vehicle_count),
        "assignment_pool_count": int(
            assignment_pool_count
            if assignment_pool_count is not None
            else mission_vehicle_count
        ),
        "nested_mission_prefix": bool(assignment_pool_count is not None),
        "mean_mec_capacity_ghz": float(mec_capacity_ghz),
        "deadlines_s": [float(value) / 1000.0 for value in DEADLINES_MS],
        "deadline_dag_assignment_model": "balanced-independent-seeded-pairing",
        "deadline_dag_coupled": False,
        "article_speed_density_model": figure == "figure_9",
        "mission_vehicle_count_article_exact": figure != "figure_9",
        "total_road_vehicle_count_article_exact": False,
        "road_vehicle_count_model": (
            "diagnostic-frozen-baseline-plus-cooperative-extension"
            if bool(road_metadata.get("mission_snapshot_frozen"))
            and int(road_metadata.get("additional_cooperative_vehicle_count", 0)) > 0
            else (
                "speed-derived-density-capped-by-database-pool"
                if figure == "figure_9"
                else (
                    "mission-count-sweep-with-fixed-road-pool-reconstruction"
                    if figure == "figure_7"
                    else "article-stated-vehicle-count-used-as-complete-scenario"
                )
            )
        ),
        "vehicle_role_model": (
            "one active application per selected mission vehicle; all same-RSU "
            "peer vehicles may provide V2V service, including other mission vehicles"
        ),
        "mec_access_path_model": (
            "vehicle-to-single-access-rsu-then-zero-added-delay-broadband-to-selected-mec"
        ),
    }
    metadata.update(road_metadata)
    if speed_kmh is not None:
        metadata["average_intervehicle_distance_m"] = float(
            2.5 * (float(speed_kmh) / 3.6)
        )
    metadata["vehicle_speeds_kmh"] = {
        str(vehicle_id): float(value)
        for vehicle_id, value in speeds.items()
    }
    return build_synthetic_joint_context(
        assignments,
        snapshot,
        mec_capacity_ghz=mec_capacity_ghz,
        scenario_metadata=metadata,
    )


def _run_scenario(
    joint_ctx: Dict[str, Any],
    algorithms: Iterable[str],
    seed: int,
    tmax: int,
    population_size: int | None,
    max_function_evaluations: int | None,
) -> Dict[str, Any]:
    return run_joint_context_benchmark(
        joint_ctx,
        algorithms=algorithms,
        seeds=[seed],
        tmax=tmax,
        population_size=population_size,
        max_function_evaluations=max_function_evaluations,
        export_artifacts=False,
    )


def _standard_rows(
    figure: str,
    result: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    scenario = result["scenario"]
    run_rows: List[Dict[str, Any]] = []
    application_rows: List[Dict[str, Any]] = []
    for run in result["runs"]:
        row = {
            "figure": figure,
            "scenario_id": scenario["scenario_id"],
            "seed": int(run["seed"]),
            "sweep_parameter": scenario["sweep_parameter"],
            "sweep_value": float(scenario["sweep_value"]),
            "mission_vehicle_count": int(scenario["mission_vehicle_count"]),
            "road_vehicle_count": int(scenario["road_vehicle_count"]),
            "mean_vehicle_speed_kmh": float(scenario["mean_vehicle_speed_kmh"]),
            "mean_mec_capacity_ghz": float(scenario["mean_mec_capacity_ghz"]),
            "algorithm": run["algorithm"],
            "avg_delay": float(run["metrics"]["avg_delay"]),
            "avg_efficiency": float(run["metrics"]["avg_efficiency"]),
            "total_efficiency": float(run["metrics"]["total_efficiency"]),
            "completion_rate": float(run["metrics"]["completion_rate"]),
            # Objective evaluation budget used by the run. This is recorded for
            # figures 7-10 as a fairness diagnostic, not as a plotting axis.
            "function_evaluations": int(
                run.get("final_function_evaluations", 0)
            ),
            "max_function_evaluations": (
                None
                if run.get("evaluation_budget") is None
                else int(run.get("evaluation_budget"))
            ),
            "evaluation_budget_exhausted": bool(
                run.get("evaluation_budget_exhausted", False)
            ),
            # Runtime diagnostics are required by Figure 8 and are also useful
            # for the NFE/fairness audit of Figures 7-10. Keep safe defaults so
            # legacy/non-population baselines cannot trigger a KeyError.
            "runtime_seconds": float(run.get("runtime_seconds", 0.0) or 0.0),
            "executed_iterations": int(run.get("executed_iterations", 0) or 0),
        }
        run_rows.append(row)
        for application in run["applications"]:
            application_rows.append(
                {
                    **{key: row[key] for key in (
                        "figure",
                        "scenario_id",
                        "seed",
                        "sweep_parameter",
                        "sweep_value",
                        "algorithm",
                    )},
                    "application_id": int(application["application_id"]),
                    "vehicle_id": int(application["vehicle_id"]),
                    "deadline_s": float(application["deadline_s"]),
                    "alpha_n": float(application["alpha_n"]),
                    "beta_n": float(application["beta_n"]),
                    "delay_s": float(application["delay_s"]),
                    "energy_j": float(application["energy_j"]),
                    "efficiency": float(application["efficiency"]),
                    "completed": bool(application["completed"]),
                    "task_count": int(application["task_count"]),
                    "optimized_task_count": int(application["optimized_task_count"]),
                    "scheduled_task_count": int(application["scheduled_task_count"]),
                    "entry_task_id": int(application["entry_task_id"]),
                    "entry_provider_id": int(application["entry_provider_id"]),
                    "providers_used": json.dumps(
                        application.get("providers_used", []),
                        ensure_ascii=False,
                    ),
                }
            )
    return run_rows, application_rows


def _figure_8_rows(
    result: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_rows, application_rows = _standard_rows("figure_8", result)
    run_lookup = {
        (str(row["algorithm"]), int(row["seed"])): row
        for row in run_rows
    }
    grouped: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = {}
    for row in application_rows:
        key = (
            str(row["algorithm"]),
            int(row["seed"]),
            float(row["deadline_s"]),
        )
        grouped.setdefault(key, []).append(row)
    deadline_rows: List[Dict[str, Any]] = []
    for (algorithm, seed, deadline_s), rows in sorted(grouped.items()):
        run_row = run_lookup[(algorithm, seed)]
        deadline_rows.append(
            {
                "figure": "figure_8",
                "scenario_id": run_row["scenario_id"],
                "seed": seed,
                "algorithm": algorithm,
                "deadline_s": deadline_s,
                "deadline_ms": deadline_s * 1000.0,
                "avg_delay": float(mean(row["delay_s"] for row in rows)),
                "completion_rate": float(
                    mean(1.0 if row["completed"] else 0.0 for row in rows)
                ),
                "application_count": len(rows),
                "runtime_seconds": run_row["runtime_seconds"],
                "executed_iterations": run_row["executed_iterations"],
                "function_evaluations": run_row["function_evaluations"],
                "max_function_evaluations": run_row["max_function_evaluations"],
                "evaluation_budget_exhausted": run_row[
                    "evaluation_budget_exhausted"
                ],
            }
        )
    return deadline_rows, application_rows


def _summary_rows(
    rows: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    metrics: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        grouped.setdefault(key, []).append(row)
    result: List[Dict[str, Any]] = []
    for key, selected in sorted(grouped.items(), key=lambda item: item[0]):
        summary = {name: value for name, value in zip(keys, key)}
        for metric in metrics:
            stats = _stats([float(row[metric]) for row in selected])
            summary[metric] = stats["mean"]
            summary[f"{metric}_std"] = stats["std"]
            summary[f"{metric}_min"] = stats["min"]
            summary[f"{metric}_max"] = stats["max"]
            summary[f"{metric}_median"] = stats["median"]
            summary[f"{metric}_q1"] = stats["q1"]
            summary[f"{metric}_q3"] = stats["q3"]
            summary[f"{metric}_iqr"] = stats["iqr"]
            summary[f"{metric}_ci95"] = stats["ci95"]
        summary["sample_count"] = len(selected)
        result.append(summary)
    return result


def _figure_6_summary_rows(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _summary_rows(
        rows,
        ("figure", "algorithm", "iteration", "point_type"),
        ("total_efficiency",),
    )


def _figure_6_nfe_summary_rows(
    rows: Sequence[Dict[str, Any]],
    grid_points: int = 25,
) -> List[Dict[str, Any]]:
    """Create an anytime comparison on a common objective-evaluation axis.

    Generation numbers are algorithm-specific because one generation can use
    a different number of objective evaluations.  We therefore forward-fill
    each seed's incumbent best on a shared NFE grid that ends at the smallest
    completed budget among all algorithm/seed runs.
    """
    grouped: Dict[Tuple[str, int], List[Tuple[int, float]]] = {}
    for row in rows:
        if str(row.get("point_type", "")).lower() != "best":
            continue
        key = (str(row["algorithm"]), int(row["seed"]))
        grouped.setdefault(key, []).append(
            (int(row.get("function_evaluations", 0)), float(row["total_efficiency"]))
        )
    if not grouped:
        return []
    for values in grouped.values():
        values.sort(key=lambda item: item[0])

    common_start = max(values[0][0] for values in grouped.values())
    common_end = min(values[-1][0] for values in grouped.values())
    if common_end < common_start:
        return []
    count = max(2, int(grid_points))
    if common_end == common_start:
        grid = [common_end]
    else:
        grid = sorted({
            int(round(common_start + (common_end - common_start) * index / (count - 1)))
            for index in range(count)
        })

    sampled: Dict[Tuple[str, int], List[float]] = {}
    for (algorithm, seed), values in grouped.items():
        for nfe in grid:
            incumbent = values[0][1]
            for observed_nfe, score in values:
                if observed_nfe > nfe:
                    break
                incumbent = max(float(incumbent), float(score))
            sampled.setdefault((algorithm, nfe), []).append(float(incumbent))

    result = []
    for (algorithm, nfe), scores in sorted(sampled.items()):
        stats = _stats(scores)
        ci95 = (
            1.96 * stats["std"] / math.sqrt(len(scores))
            if len(scores) > 1
            else 0.0
        )
        result.append({
            "figure": "figure_6",
            "algorithm": algorithm,
            "function_evaluations": int(nfe),
            "best_total_efficiency": stats["mean"],
            "best_total_efficiency_std": stats["std"],
            "best_total_efficiency_ci95": float(ci95),
            "best_total_efficiency_min": stats["min"],
            "best_total_efficiency_max": stats["max"],
            "sample_count": len(scores),
            "common_budget_start": int(common_start),
            "common_budget_end": int(common_end),
        })
    return result


def _execution_audit_rows(
    figure: str,
    result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    scenario = result["scenario"]
    rows: List[Dict[str, Any]] = []
    for run in result["runs"]:
        applications = list(run.get("applications", []))
        schedule = list(run.get("schedule", []))
        mode_counts = {"local": 0, "v2v": 0, "v2i": 0}
        transfer_count = 0
        cache_update_count = 0
        cache_hit_count = 0
        cache_miss_count = 0
        cache_insertion_count = 0
        cache_eviction_count = 0
        for task in schedule:
            mode = str(task.get("provider_mode", ""))
            if mode in mode_counts:
                mode_counts[mode] += 1
            transfer_count += len(task.get("transfers", []))
            if bool(task.get("cache_update_required", False)):
                cache_update_count += 1
            if task.get("cache_hit") is True:
                cache_hit_count += 1
            elif task.get("cache_miss") is True:
                cache_miss_count += 1
            cache_insertion_count += int(task.get("cache_inserted_count", 0) or 0)
            cache_eviction_count += int(task.get("cache_evicted_count", 0) or 0)
        rows.append(
            {
                "figure": figure,
                "scenario_id": scenario["scenario_id"],
                "seed": int(run["seed"]),
                "algorithm": str(run["algorithm"]),
                "application_count": len(applications),
                "completed_application_count": sum(
                    1 for application in applications if application["completed"]
                ),
                "total_energy_j": float(
                    sum(float(application["energy_j"]) for application in applications)
                ),
                "avg_delay_s": float(run["metrics"]["avg_delay"]),
                "avg_efficiency": float(run["metrics"]["avg_efficiency"]),
                "total_efficiency": float(run["metrics"]["total_efficiency"]),
                "completion_rate": float(run["metrics"]["completion_rate"]),
                "runtime_seconds": float(run.get("runtime_seconds", 0.0)),
                "executed_iterations": int(run.get("executed_iterations", 0)),
                "function_evaluations": int(
                    run.get("final_function_evaluations", 0)
                ),
                "max_function_evaluations": run.get("evaluation_budget"),
                "evaluation_budget_exhausted": bool(
                    run.get("evaluation_budget_exhausted", False)
                ),
                "scheduled_task_count": len(schedule),
                "local_task_count": mode_counts["local"],
                "v2v_task_count": mode_counts["v2v"],
                "v2i_task_count": mode_counts["v2i"],
                "transfer_count": transfer_count,
                "cache_update_count": cache_update_count,
                "cache_hit_count": cache_hit_count,
                "cache_miss_count": cache_miss_count,
                "cache_insertion_count": cache_insertion_count,
                "cache_eviction_count": cache_eviction_count,
            }
        )
    return rows


def _iteration_diagnostics(item: Dict[str, Any]) -> Dict[str, Any]:
    aliases = {
        "accepted_candidates": ("accepted_candidates", "accepted_moves"),
        "accepted_guided_trials": ("accepted_guided_trials",),
        "accepted_cache_trials": ("accepted_cache_trials",),
        "generated_trials": ("generated_trials", "generated_replacements"),
        "unique_trial_count": ("unique_trial_count",),
        "duplicate_trial_count": ("duplicate_trial_count",),
        "mean_trial_hamming": ("mean_trial_hamming", "mean_substituted_tasks"),
        "population_unique_count": ("population_unique_count",),
        "population_mean_hamming": (
            "population_mean_hamming",
            "population_diversity_mean",
        ),
        "active_population_size": ("active_population_size",),
        "search_progress": ("search_progress",),
        "best_improved": ("best_improved",),
        "stagnation_generations": (
            "stagnation_generations",
            "stagnation_count",
        ),
        "initial_greedy_target": ("initial_greedy_target",),
        "initial_genetic_target": ("initial_genetic_target",),
        "initial_random_target": ("initial_random_target",),
        "initial_greedy_selected": ("initial_greedy_selected",),
        "initial_genetic_selected": ("initial_genetic_selected",),
        "initial_random_selected": ("initial_random_selected",),
        "ga_generations": ("ga_generations",),
        "genetic_candidate_pool_size": ("genetic_candidate_pool_size",),
        "random_candidate_pool_size": ("random_candidate_pool_size",),
    }
    result = {}
    for output_key, input_keys in aliases.items():
        result[output_key] = next(
            (item[key] for key in input_keys if item.get(key) is not None),
            None,
        )
    defenses = item.get("defense_trials", {}) or {}
    for name in (
        "sight",
        "sound",
        "odor",
        "physical_attack",
        "stagnation_escape",
    ):
        result[f"cpo_{name}_trials"] = int(defenses.get(name, 0) or 0)
    return result


def _figure_6_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in result["runs"]:
        for item in run.get("iteration_history", []):
            iteration = int(float(item["iteration"]))
            population = item.get("population_total_efficiencies", [])
            for index, value in enumerate(population):
                rows.append(
                    {
                        "figure": "figure_6",
                        "algorithm": str(run["algorithm"]),
                        "seed": int(run["seed"]),
                        "iteration": iteration,
                        "function_evaluations": int(
                            item.get("function_evaluations", 0)
                        ),
                        "point_type": "population",
                        "population_index": index,
                        "total_efficiency": float(value),
                        "runtime_seconds": run.get("runtime_seconds"),
                        **_iteration_diagnostics(item),
                    }
                )
            rows.append(
                {
                    "figure": "figure_6",
                    "algorithm": str(run["algorithm"]),
                    "seed": int(run["seed"]),
                    "iteration": iteration,
                    "function_evaluations": int(
                        item.get("function_evaluations", 0)
                    ),
                    "point_type": "best",
                    "population_index": None,
                    "total_efficiency": float(item["best_total_efficiency"]),
                    "runtime_seconds": run.get("runtime_seconds"),
                    **_iteration_diagnostics(item),
                }
            )
    return rows


def _convergence_diagnostics(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("point_type") != "best":
            continue
        key = (str(row["algorithm"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)

    diagnostics = []
    for (algorithm, seed), selected in sorted(grouped.items()):
        selected.sort(key=lambda row: int(row["iteration"]))
        initial = selected[0]
        final = selected[-1]
        def accumulated_or_sum(key: str) -> int:
            values = [
                int(row.get(key) or 0) for row in selected[1:]
            ]
            if not values:
                return 0
            if all(left <= right for left, right in zip(values, values[1:])):
                return values[-1]
            return sum(values)

        accepted = accumulated_or_sum("accepted_candidates")
        generated = accumulated_or_sum("generated_trials")
        unique_trials = sum(
            int(row.get("unique_trial_count") or 0)
            for row in selected[1:]
        )
        initial_best = float(initial["total_efficiency"])
        final_best = float(final["total_efficiency"])
        absolute_gain = final_best - initial_best
        improved_iterations = sum(
            1
            for previous, current in zip(selected, selected[1:])
            if float(current["total_efficiency"])
            > float(previous["total_efficiency"])
        )
        diagnostics.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "initial_best": initial_best,
                "final_best": final_best,
                "absolute_gain": absolute_gain,
                "relative_gain_percent": (
                    100.0 * absolute_gain / abs(initial_best)
                    if initial_best != 0.0
                    else None
                ),
                "improved_iterations": improved_iterations,
                "accepted_candidates": (
                    accepted if generated else None
                ),
                "generated_trials": generated if generated else None,
                "unique_trial_count": unique_trials if generated else None,
                "acceptance_rate": (
                    float(accepted / generated) if generated else None
                ),
                "function_evaluations": int(
                    final.get("function_evaluations", 0)
                ),
                "runtime_seconds": final.get("runtime_seconds"),
                "final_population_unique_count": final.get(
                    "population_unique_count"
                ),
                "final_population_mean_hamming": final.get(
                    "population_mean_hamming"
                ),
                "initial_greedy_selected": initial.get(
                    "initial_greedy_selected"
                ),
                "initial_genetic_selected": initial.get(
                    "initial_genetic_selected"
                ),
                "initial_random_selected": initial.get(
                    "initial_random_selected"
                ),
                "ga_generations": initial.get("ga_generations"),
                "genetic_candidate_pool_size": initial.get(
                    "genetic_candidate_pool_size"
                ),
                "random_candidate_pool_size": initial.get(
                    "random_candidate_pool_size"
                ),
                "stagnated": absolute_gain <= 0.0,
            }
        )
    return diagnostics


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative_path(path: Path) -> str:
    root = Path(settings.BASE_DIR).resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _algorithms_from_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
    algorithms: List[str] = []
    for row in rows:
        algorithm = str(row.get("algorithm", "")).strip()
        if algorithm and algorithm not in algorithms:
            algorithms.append(algorithm)
    return algorithms


def _plot_figure_6(rows: Sequence[Dict[str, Any]],output_base: Path,) -> None:
    """Plot all population members as points using the paper's axis scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algorithms = _algorithms_from_rows(rows)
    if not algorithms:
        raise ValueError("Figure 6 has no algorithm data")

    population_rows = [
        row
        for row in rows
        if str(row.get("point_type", "")).strip().lower() == "population"
    ]
    if not population_rows:
        raise ValueError("Figure 6 has no population data")

    styles = {
        "dcsga": {"color": "#0072B2", "marker": "o"},
        "gwo_aco": {"color": "#7B2CBF", "marker": "s"},
        "gwo": {"color": "#7B2CBF", "marker": "s"},
        "gpc": {"color": "#2A9D8F", "marker": "^"},
        "cpo": {"color": "#D81B60", "marker": "P"},
        "dcpo_base": {"color": "#6C757D", "marker": "x"},
        "dcpo_criticality": {"color": "#F4A261", "marker": "v"},
        "dcpo_cache": {"color": "#00A896", "marker": ">"},
        "puma": {"color": "#E76F51", "marker": "D"},
    }

    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    paper_mode = algorithms == ["dcsga"]

    if paper_mode:
        selected_rows = [
            row
            for row in population_rows
            if str(row["algorithm"]) == "dcsga"
        ]
        selected_rows.sort(
            key=lambda row: (
                int(row["iteration"]),
                int(row["seed"]),
                int(row.get("population_index") or 0),
            )
        )
        axis.scatter(
            [int(row["iteration"]) for row in selected_rows],
            [float(row["total_efficiency"]) for row in selected_rows],
            s=13,
            color="black",
            alpha=0.80,
            marker="o",
            linewidths=0,
        )
    else:
        algorithm_count = len(algorithms)
        offset_step = 0.14
        center = (algorithm_count - 1) / 2.0

        for algorithm_index, algorithm in enumerate(algorithms):
            selected_rows = [
                row
                for row in population_rows
                if str(row["algorithm"]) == algorithm
            ]
            if not selected_rows:
                continue

            selected_rows.sort(
                key=lambda row: (
                    int(row["iteration"]),
                    int(row["seed"]),
                    int(row.get("population_index") or 0),
                )
            )

            style = styles.get(
                algorithm,
                {"color": None, "marker": "o"},
            )
            x_offset = (algorithm_index - center) * offset_step

            axis.scatter(
                [
                    float(row["iteration"]) + x_offset
                    for row in selected_rows
                ],
                [
                    float(row["total_efficiency"])
                    for row in selected_rows
                ],
                s=16,
                color=style["color"],
                marker=style["marker"],
                alpha=0.72,
                linewidths=0,
                label=ALGORITHM_LABELS.get(algorithm, algorithm),
            )

    iterations = sorted({
        int(row["iteration"])
        for row in population_rows
    })
    max_iteration = max(iterations)
    article_x_max = max(
        5,
        ((max_iteration + 4) // 5) * 5,
    )

    # Keep the paper's five-iteration x spacing.  Preserve the article-like
    # 18..25 y scale whenever it contains the generated population, but expand
    # it when necessary so no valid population member is silently clipped.
    axis.set_xlim(-0.5, article_x_max + 0.5)
    axis.set_xticks(list(range(0, article_x_max + 1, 5)))

    population_values = [
        float(row["total_efficiency"])
        for row in population_rows
    ]
    data_min = min(population_values)
    data_max = max(population_values)
    y_lower = (
        18.0
        if data_min >= 18.0
        else float(int(data_min // 1) - 1)
    )
    y_upper = (
        25.0
        if data_max <= 25.0
        else float(int(-(-data_max // 1)) + 1)
    )
    if y_upper <= y_lower:
        y_upper = y_lower + 1.0
    y_span = y_upper - y_lower
    y_step = 1 if y_span <= 16 else 5 if y_span <= 80 else 10
    first_tick = int(-(-y_lower // y_step)) * y_step
    last_tick = int(y_upper // y_step) * y_step
    axis.set_ylim(y_lower, y_upper)
    axis.set_yticks(list(range(first_tick, last_tick + 1, y_step)))

    axis.set_xlabel("Number of iterations")
    axis.set_ylabel("Total offloading efficiency")
    axis.grid(alpha=0.25, linestyle=":")

    if not paper_mode:
        axis.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=300)
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)


def _plot_figure_6_nfe(
    rows: Sequence[Dict[str, Any]],
    output_base: Path,
) -> None:
    """Plot incumbent quality against the fair cross-algorithm NFE axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("Figure 6 has no common NFE interval")
    styles = {
        "dcsga": {"color": "#0072B2", "marker": "o"},
        "gwo_aco": {"color": "#7B2CBF", "marker": "s"},
        "gwo": {"color": "#7B2CBF", "marker": "s"},
        "gpc": {"color": "#2A9D8F", "marker": "^"},
        "cpo": {"color": "#D81B60", "marker": "P"},
        "dcpo_base": {"color": "#6C757D", "marker": "x"},
        "dcpo_criticality": {"color": "#F4A261", "marker": "v"},
        "dcpo_cache": {"color": "#00A896", "marker": ">"},
        "puma": {"color": "#E76F51", "marker": "D"},
    }
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    for algorithm in _algorithms_from_rows(rows):
        selected = sorted(
            (row for row in rows if row["algorithm"] == algorithm),
            key=lambda row: int(row["function_evaluations"]),
        )
        if not selected:
            continue
        x = [int(row["function_evaluations"]) for row in selected]
        y = [float(row["best_total_efficiency"]) for row in selected]
        ci = [float(row["best_total_efficiency_ci95"]) for row in selected]
        style = styles.get(algorithm, {"color": None, "marker": "o"})
        axis.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            markevery=max(1, len(x) // 8),
            linewidth=2.2,
            markersize=4.5,
            label=ALGORITHM_LABELS.get(algorithm, algorithm),
        )
        if any(value > 0.0 for value in ci):
            axis.fill_between(
                x,
                [value - width for value, width in zip(y, ci)],
                [value + width for value, width in zip(y, ci)],
                color=style["color"],
                alpha=0.14,
                linewidth=0,
            )
    axis.set_xlabel("Number of objective function evaluations (NFE)")
    axis.set_ylabel("Best-so-far total offloading efficiency")
    axis.grid(alpha=0.25, linestyle=":")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=300)
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)


def _figure_6_final_seed_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract one final incumbent per paired algorithm/seed run."""
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("point_type", "")).strip().lower() != "population":
            continue
        key = (str(row.get("algorithm", "")), int(row.get("seed", 0)))
        grouped.setdefault(key, []).append(dict(row))
    result = []
    for (algorithm, seed), selected in sorted(grouped.items()):
        final_iteration = max(int(row.get("iteration", 0)) for row in selected)
        final_rows = [
            row for row in selected
            if int(row.get("iteration", 0)) == final_iteration
        ]
        if not final_rows:
            continue
        result.append({
            "algorithm": algorithm,
            "seed": int(seed),
            "final_iteration": int(final_iteration),
            "function_evaluations": max(
                int(row.get("function_evaluations", 0) or 0)
                for row in final_rows
            ),
            "final_best_total_efficiency": max(
                float(row.get("total_efficiency", 0.0)) for row in final_rows
            ),
        })
    return result


def _plot_figure_6_final_distribution(
    final_rows: Sequence[Dict[str, Any]], output_base: Path
) -> None:
    """Plot the seed distribution; meaningful for the final multi-seed run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algorithms = _algorithms_from_rows(final_rows)
    values = [
        [
            float(row["final_best_total_efficiency"])
            for row in final_rows
            if row["algorithm"] == algorithm
        ]
        for algorithm in algorithms
    ]
    if not algorithms or any(not group for group in values):
        return
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    tick_labels = [ALGORITHM_LABELS.get(name, name) for name in algorithms]
    boxplot_options = {
        "showmeans": True,
        "meanline": True,
    }
    # Matplotlib 3.9 renamed ``labels`` to ``tick_labels`` and 3.10 removed
    # the old keyword.  Keep the benchmark export compatible with both the
    # older server environment and current laptop installations.
    try:
        axis.boxplot(values, tick_labels=tick_labels, **boxplot_options)
    except TypeError as exc:
        if "tick_labels" not in str(exc):
            raise
        axis.boxplot(values, labels=tick_labels, **boxplot_options)
    for x_position, group in enumerate(values, start=1):
        offsets = [
            0.0 if len(group) == 1 else -0.12 + 0.24 * index / (len(group) - 1)
            for index in range(len(group))
        ]
        axis.scatter(
            [x_position + offset for offset in offsets],
            group,
            s=16,
            color="black",
            alpha=0.55,
            linewidths=0,
        )
    axis.set_xlabel("Algorithm")
    axis.set_ylabel("Final total offloading efficiency")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

def _line_panel(axis, rows: Sequence[Dict[str, Any]], metric: str, x_key: str, x_label: str, y_label: str,
) -> None:
    """Draw one Figure 7/8 panel from aggregated summary rows.

    Standard deviations remain available in CSV/JSON outputs, but uncertainty
    bands are intentionally not drawn so the exported figures match the
    article layout and completion-rate plots never extend below zero.
    """
    algorithms = _algorithms_from_rows(rows)
    if not algorithms:
        raise ValueError("The plot has no algorithm data")

    styles = {
        "dcsga": {
            "color": "#0072B2",
            "marker": "o",
            "linestyle": "-",
        },
        "dtosc": {
            "color": "#009E73",
            "marker": "^",
            "linestyle": "-.",
        },
        "to_v2i": {
            "color": "#CC79A7",
            "marker": "D",
            "linestyle": ":",
        },
        "to_wo_c": {
            "color": "#56B4E9",
            "marker": "v",
            "linestyle": (0, (5, 2)),
        },
        "to_wo_r": {
            "color": "#E69F00",
            "marker": "P",
            "linestyle": (0, (1, 1)),
        },
        "gwo_aco": {
            "color": "#7B2CBF",
            "marker": "s",
            "linestyle": "--",
        },
        "gwo": {
            "color": "#7B2CBF",
            "marker": "s",
            "linestyle": "--",
        },
        "gpc": {
            "color": "#2A9D8F",
            "marker": "^",
            "linestyle": "--",
        },
        "puma": {
            "color": "#E76F51",
            "marker": "D",
            "linestyle": "--",
        },
        "cpo": {
            "color": "#D81B60",
            "marker": "X",
            "linestyle": "--",
        },
        "dcpo_base": {"color": "#6C757D", "marker": "x", "linestyle": ":"},
        "dcpo_criticality": {"color": "#F4A261", "marker": "v", "linestyle": "-."},
        "dcpo_cache": {"color": "#00A896", "marker": ">", "linestyle": "-."},
    }

    all_x_values = set()

    for algorithm in algorithms:
        algorithm_rows = sorted(
            (
                row
                for row in rows
                if row.get("algorithm") == algorithm
                and row.get(x_key) is not None
                and row.get(metric) is not None
            ),
            key=lambda row: float(row[x_key]),
        )
        if not algorithm_rows:
            continue

        style = styles.get(
            algorithm,
            {
                "color": None,
                "marker": "o",
                "linestyle": "-",
            },
        )
        x_values = [float(row[x_key]) for row in algorithm_rows]
        y_values = [float(row[metric]) for row in algorithm_rows]
        all_x_values.update(x_values)

        axis.plot(
            x_values,
            y_values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=5.5,
            label=ALGORITHM_LABELS.get(algorithm, algorithm),
        )

    if not all_x_values:
        raise ValueError(f"The plot has no valid values for {metric}")

    axis.set_xticks(sorted(all_x_values))
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25, linestyle=":")
    axis.legend(frameon=False)

def _plot_figure_7(rows: Sequence[Dict[str, Any]], output_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    specs = (
        (
            "avg_delay",
            "Average delay (s)",
            (0.02, 0.07),
            [0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        ),
        (
            "avg_efficiency",
            "Average offloading efficiency",
            (0.0, 0.6),
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ),
        (
            "total_efficiency",
            "Total offloading efficiency",
            (0.0, 40.0),
            [0.0, 10.0, 20.0, 30.0, 40.0],
        ),
        (
            "completion_rate",
            "Rate of completion",
            (0.6, 1.0),
            [0.6, 0.7, 0.8, 0.9, 1.0],
        ),
    )
    for axis, (metric, y_label, y_limits, y_ticks) in zip(axes.flat, specs):
        _line_panel(
            axis,
            list(rows),
            metric,
            "mission_vehicle_count",
            "Number of vehicles",
            y_label,
        )
        axis.set_ylim(*y_limits)
        axis.set_yticks(y_ticks)

    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=300)
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)



def _plot_figure_8(rows: Sequence[Dict[str, Any]],output_base: Path,) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    _line_panel(
        axes[0],
        list(rows),
        "avg_delay",
        "deadline_s",
        "Deadline of application (s)",
        "Average delay (s)",
    )
    _line_panel(
        axes[1],
        list(rows),
        "completion_rate",
        "deadline_s",
        "Deadline of application (s)",
        "Rate of completion",
    )

    deadline_ticks = [
        0.03, 0.04, 0.05, 0.06,
        0.07, 0.08, 0.09, 0.10,
    ]
    for axis in axes:
        axis.set_xticks(deadline_ticks)

    # Figure 8(a): 0.02 to 0.08, one-hundredth spacing.
    axes[0].set_ylim(0.02, 0.08)
    axes[0].set_yticks([
        0.02, 0.03, 0.04, 0.05,
        0.06, 0.07, 0.08,
    ])

    # Figure 8(b): 0 to 1, 0.2 spacing.
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_yticks([
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
    ])

    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=300)
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)

def _plot_figure_9(rows: Sequence[Dict[str, Any]],output_base: Path,) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speeds = sorted({
        int(round(float(row["speed_kmh"])))
        for row in rows
        if row.get("speed_kmh") is not None
    })
    algorithms = _algorithms_from_rows(rows)
    width = 0.8 / max(1, len(algorithms))
    x_values = list(range(len(speeds)))

    figure, axis = plt.subplots()
    colors = {
        "dcsga": "#0072B2",
        "dtosc": "#009E73",
        "gwo_aco": "#7B2CBF",
        "gwo": "#7B2CBF",
        "gpc": "#2A9D8F",
        "cpo": "#D81B60",
        "dcpo_base": "#6C757D",
        "dcpo_criticality": "#F4A261",
        "dcpo_cache": "#00A896",
        "puma": "#E76F51",
    }

    for algorithm_index, algorithm in enumerate(algorithms):
        values_by_speed = {
            int(round(float(row["speed_kmh"]))): float(
                row["total_efficiency"]
            )
            for row in rows
            if row["algorithm"] == algorithm
        }
        offset = (
            algorithm_index - (len(algorithms) - 1) / 2.0
        ) * width

        axis.bar(
            [value + offset for value in x_values],
            [
                values_by_speed.get(speed, 0.0)
                for speed in speeds
            ],
            width=width,
            color=colors.get(algorithm),
            label=ALGORITHM_LABELS.get(algorithm, algorithm),
        )

    # Keep the paper's 20-25 range when the generated values fit it.
    # Otherwise expand the axis so no bar is clipped. This changes only
    # the visualization; benchmark values and algorithm behavior are untouched.
    axis.set_xticks(
        x_values,
        [str(speed) for speed in speeds],
    )
    plotted_values = [
        float(row["total_efficiency"])
        for row in rows
        if row.get("total_efficiency") is not None
    ]
    if plotted_values and not (
        min(plotted_values) >= 20.0
        and max(plotted_values) <= 25.0
    ):
        data_min = min(plotted_values)
        data_max = max(plotted_values)
        data_span = max(0.0, data_max - data_min)
        padding = max(0.5, data_span * 0.10)
        y_lower = math.floor(data_min - padding)
        y_upper = math.ceil(data_max + padding)
        if y_upper <= y_lower:
            y_upper = y_lower + 1
        axis.set_ylim(float(y_lower), float(y_upper))
        axis.set_yticks([
            float(value)
            for value in range(y_lower, y_upper + 1)
        ])
    else:
        axis.set_ylim(20.0, 25.0)
        axis.set_yticks([
            20.0, 21.0, 22.0,
            23.0, 24.0, 25.0,
        ])

    axis.set_xlabel("Speed of vehicles (km/h)")
    axis.set_ylabel("Total offloading efficiency")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=300)
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)

def _plot_figure_10(
    rows: Sequence[Dict[str, Any]],
    output_base: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, efficiency_axis = plt.subplots(figsize=(8.0, 5.2))
    completion_axis = efficiency_axis.twinx()

    styles = {
        "dcsga": {"color": "#0072B2", "marker": "o"},
        "dtosc": {"color": "#009E73", "marker": "^"},
        "gwo_aco": {"color": "#7B2CBF", "marker": "s"},
        "gwo": {"color": "#7B2CBF", "marker": "s"},
        "gpc": {"color": "#2A9D8F", "marker": "^"},
        "cpo": {"color": "#D81B60", "marker": "X"},
        "dcpo_base": {"color": "#6C757D", "marker": "x"},
        "dcpo_criticality": {"color": "#F4A261", "marker": "v"},
        "dcpo_cache": {"color": "#00A896", "marker": ">"},
        "puma": {"color": "#E76F51", "marker": "D"},
    }

    for algorithm in _algorithms_from_rows(rows):
        selected = sorted(
            (
                row
                for row in rows
                if row["algorithm"] == algorithm
            ),
            key=lambda row: float(row["mec_capacity_ghz"]),
        )
        if not selected:
            continue

        style = styles.get(
            algorithm,
            {"color": None, "marker": "o"},
        )
        x_values = [
            float(row["mec_capacity_ghz"])
            for row in selected
        ]
        label = ALGORITHM_LABELS.get(algorithm, algorithm)

        efficiency_axis.plot(
            x_values,
            [
                float(row["avg_efficiency"])
                for row in selected
            ],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2.2,
            markersize=5.5,
            label=f"{label} efficiency",
        )
        completion_axis.plot(
            x_values,
            [
                float(row["completion_rate"])
                for row in selected
            ],
            color=style["color"],
            marker=style["marker"],
            linestyle="--",
            linewidth=1.8,
            markersize=5.0,
            label=f"{label} completion",
        )

    efficiency_axis.set_xlim(28.0, 82.0)
    efficiency_axis.set_xticks([30, 40, 50, 60, 70, 80])

    # Preserve the paper-style limits when the data fit them. Otherwise,
    # expand each Y-axis independently so every efficiency and completion
    # point remains visible. This is a plotting-only adjustment.
    efficiency_values = [
        float(row["avg_efficiency"])
        for row in rows
        if row.get("avg_efficiency") is not None
    ]
    if efficiency_values and not (
        min(efficiency_values) >= 0.20
        and max(efficiency_values) <= 0.50
    ):
        efficiency_min = min(efficiency_values)
        efficiency_max = max(efficiency_values)
        efficiency_span = max(0.0, efficiency_max - efficiency_min)
        efficiency_padding = max(0.02, efficiency_span * 0.10)
        efficiency_lower = max(
            0.0,
            math.floor(
                (efficiency_min - efficiency_padding) / 0.05
            ) * 0.05,
        )
        efficiency_upper = math.ceil(
            (efficiency_max + efficiency_padding) / 0.05
        ) * 0.05
        if efficiency_upper <= efficiency_lower:
            efficiency_upper = efficiency_lower + 0.05
        efficiency_axis.set_ylim(
            efficiency_lower, efficiency_upper
        )
        tick_count = int(round(
            (efficiency_upper - efficiency_lower) / 0.05
        ))
        efficiency_axis.set_yticks([
            round(efficiency_lower + index * 0.05, 10)
            for index in range(tick_count + 1)
        ])
    else:
        efficiency_axis.set_ylim(0.20, 0.50)
        efficiency_axis.set_yticks([
            0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50,
        ])

    completion_values = [
        float(row["completion_rate"])
        for row in rows
        if row.get("completion_rate") is not None
    ]
    if completion_values and not (
        min(completion_values) >= 0.988
        and max(completion_values) <= 1.000
    ):
        completion_min = min(completion_values)
        completion_max = max(completion_values)
        completion_span = max(0.0, completion_max - completion_min)
        completion_padding = max(0.005, completion_span * 0.10)
        completion_lower = max(
            0.0,
            math.floor(
                (completion_min - completion_padding) / 0.02
            ) * 0.02,
        )
        completion_upper = min(
            1.0,
            math.ceil(
                (completion_max + completion_padding) / 0.02
            ) * 0.02,
        )
        if completion_upper <= completion_lower:
            completion_lower = max(0.0, completion_upper - 0.02)
        completion_axis.set_ylim(
            completion_lower, completion_upper
        )
        tick_count = int(round(
            (completion_upper - completion_lower) / 0.02
        ))
        completion_axis.set_yticks([
            round(completion_lower + index * 0.02, 10)
            for index in range(tick_count + 1)
        ])
    else:
        completion_axis.set_ylim(0.988, 1.000)
        completion_axis.set_yticks([
            0.988, 0.990, 0.992, 0.994,
            0.996, 0.998, 1.000,
        ])

    efficiency_axis.set_xlabel(
        "Computing capacity of MEC servers (GHz)"
    )
    efficiency_axis.set_ylabel(
        "Average offloading efficiency"
    )
    completion_axis.set_ylabel(
        "Rate of completion"
    )
    efficiency_axis.grid(alpha=0.3, linestyle=":")

    handles_a, labels_a = (
        efficiency_axis.get_legend_handles_labels()
    )
    handles_b, labels_b = (
        completion_axis.get_legend_handles_labels()
    )
    figure.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
    )

    figure.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(figure)
def _export_experiment(
    figure: str,
    raw_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    application_rows: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    run_id = (
        f"{figure}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid4().hex[:8]}"
    )
    experiment_folder = (
        "algorithm_comparison"
        if metadata.get("comparison_mode", False)
        else "paper_figures"
    )
    output_dir = (
        Path(settings.BASE_DIR)
        / "benchmark_results"
        / experiment_folder
        / figure
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary_results.csv"
    metadata_path = output_dir / "metadata.json"
    nfe_summary_path = output_dir / "nfe_summary_results.csv"

    # خروجی های حجیم و مخصوص دیباگ (raw/application/audit/distribution)
    # در نسخه گزارش نگه داشته نمی شوند تا پوشه نتایج تمیز بماند.
    _write_csv(summary_path, summary_rows)
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, default=str)
    figure_base = output_dir / figure
    nfe_figure_base = output_dir / "figure_6_nfe"
    nfe_summary_rows: List[Dict[str, Any]] = []
    if figure == "figure_6":
        _plot_figure_6(raw_rows, figure_base)
        nfe_summary_rows = _figure_6_nfe_summary_rows(raw_rows)
        _write_csv(nfe_summary_path, nfe_summary_rows)
        _plot_figure_6_nfe(nfe_summary_rows, nfe_figure_base)
    elif figure == "figure_7":
        _plot_figure_7(summary_rows, figure_base)
    elif figure == "figure_8":
        _plot_figure_8(summary_rows, figure_base)
    elif figure == "figure_9":
        _plot_figure_9(summary_rows, figure_base)
    elif figure == "figure_10":
        _plot_figure_10(summary_rows, figure_base)
    return {
        "run_id": run_id,
        "output_directory": _relative_path(output_dir),
        "summary_csv": _relative_path(summary_path),
        "metadata_json": _relative_path(metadata_path),
        "nfe_summary_csv": (
            _relative_path(nfe_summary_path) if figure == "figure_6" else None
        ),
        "figure_png": _relative_path(figure_base.with_suffix(".png")),
        "figure_pdf": _relative_path(figure_base.with_suffix(".pdf")),
        "nfe_figure_png": (
            _relative_path(nfe_figure_base.with_suffix(".png"))
            if figure == "figure_6"
            else None
        ),
        "nfe_figure_pdf": (
            _relative_path(nfe_figure_base.with_suffix(".pdf"))
            if figure == "figure_6"
            else None
        ),
    }


def run_paper_experiment(
    figure: str,
    repetitions: int | None = None,
    seed_start: int = 1,
    tmax: int = 15,
    population_size: int | None = None,
    max_function_evaluations: int | None = None,
    experiment_mode: str | None = None,
    algorithms: Iterable[str] | None = None,
    diagnostic_vehicle_count: int | None = None,
    diagnostic_road_vehicle_count: int | None = None,
    diagnostic_sweep_values: Iterable[float] | None = None,
    export_artifacts: bool = True,
) -> Dict[str, Any]:
    figure = _normalize_figure(figure)
    if figure == "all" and (
        diagnostic_vehicle_count is not None
        or diagnostic_road_vehicle_count is not None
        or diagnostic_sweep_values is not None
    ):
        raise ValueError(
            "Diagnostic vehicle overrides must target one figure, not figure='all'"
        )
    if figure == "all":
        resolved_all_mode = str(
            experiment_mode or PAPER_REPRODUCTION
        ).strip().lower()
        if resolved_all_mode != PAPER_REPRODUCTION:
            raise ValueError(
                "figure='all' is reserved for paper_reproduction. Run fair "
                "optimizer comparisons one figure at a time."
            )
        figures = list(FIGURE_ALGORITHMS)
        return {
            "figure": "all",
            "requested_figures": figures,
            "seed_start": int(seed_start),
            "tmax": int(tmax),
            "population_size": (
                None if population_size is None else int(population_size)
            ),
            "max_function_evaluations": (
                None
                if max_function_evaluations is None
                else int(max_function_evaluations)
            ),
            "experiment_mode": resolved_all_mode,
            "results": {
                item: run_paper_experiment(
                    figure=item,
                    repetitions=repetitions,
                    seed_start=seed_start,
                    tmax=tmax,
                    population_size=population_size,
                    max_function_evaluations=max_function_evaluations,
                    experiment_mode=resolved_all_mode,
                    algorithms=algorithms,
                    diagnostic_vehicle_count=None,
                    diagnostic_road_vehicle_count=None,
                    diagnostic_sweep_values=None,
                    export_artifacts=export_artifacts,
                )
                for item in figures
            },
        }
    if repetitions is None:
        repetitions = 100 if figure == "figure_8" else 1
    repetitions = int(repetitions)
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    tmax = int(tmax)
    if tmax < 1:
        raise ValueError("tmax must be positive")
    if population_size is not None and int(population_size) < 2:
        raise ValueError("population_size must be at least 2")
    actual_population_size = int(
        population_size if population_size is not None else load_params_obj().S
    )
    if (
        max_function_evaluations is not None
        and int(max_function_evaluations) < actual_population_size
    ):
        raise ValueError(
            "max_function_evaluations must be at least population_size"
        )
    if diagnostic_vehicle_count is not None:
        diagnostic_vehicle_count = int(diagnostic_vehicle_count)
        if figure not in {"figure_6", "figure_7"}:
            raise ValueError(
                "diagnostic_vehicle_count is supported only for figure_6 or figure_7"
            )
        if diagnostic_vehicle_count < 2 or diagnostic_vehicle_count > 76:
            raise ValueError(
                "diagnostic_vehicle_count must be between 2 and 76"
            )
    if diagnostic_road_vehicle_count is not None:
        diagnostic_road_vehicle_count = int(diagnostic_road_vehicle_count)
        if figure == "figure_9":
            raise ValueError(
                "diagnostic_road_vehicle_count is not supported for figure_9; "
                "Figure 9 derives road density from speed"
            )
        if diagnostic_road_vehicle_count < 2:
            raise ValueError("diagnostic_road_vehicle_count must be at least 2")

    sweep_values = None
    if diagnostic_sweep_values is not None:
        if figure not in {"figure_9", "figure_10"}:
            raise ValueError(
                "diagnostic_sweep_values is supported only for figure_9 or figure_10"
            )
        sweep_values = tuple(
            dict.fromkeys(float(value) for value in diagnostic_sweep_values)
        )
        if not sweep_values:
            raise ValueError("diagnostic_sweep_values cannot be empty")
        allowed = (
            {float(value) for value in VEHICLE_SPEEDS_KMH}
            if figure == "figure_9"
            else {float(value) for value in MEC_CAPACITIES_GHZ}
        )
        unsupported = sorted(set(sweep_values) - allowed)
        if unsupported:
            raise ValueError(
                f"Unsupported diagnostic sweep values for {figure}: {unsupported}; "
                f"allowed: {sorted(allowed)}"
            )

    seeds = [int(seed_start) + index for index in range(repetitions)]
    default_algorithms = FIGURE_ALGORITHMS[figure]
    paper_algorithms = PAPER_FIGURE_ALGORITHMS[figure]
    if algorithms is None:
        selected_algorithms = tuple(default_algorithms)
    else:
        selected_algorithms = tuple(
            dict.fromkeys(str(name).strip().lower() for name in algorithms)
        )
    if not selected_algorithms:
        raise ValueError("At least one algorithm is required")
    unknown_algorithms = [
        name for name in selected_algorithms if name not in ALGORITHM_LABELS
    ]
    if unknown_algorithms:
        raise ValueError(f"Unsupported comparison algorithms: {unknown_algorithms}")
    if (
        population_size is not None
        and ({"gwo", "gwo_aco"} & set(selected_algorithms))
        and int(population_size) < 3
    ):
        raise ValueError(
            "gwo requires population_size >= 3 for alpha, beta, and delta leaders"
        )
    if figure == "figure_6" and "dtosc" in selected_algorithms:
        raise ValueError("DTOSC has no population convergence history for Figure 6")
    algorithms = selected_algorithms
    experiment_mode = resolve_experiment_mode(
        experiment_mode,
        figure=figure,
        selected_algorithms=algorithms,
        paper_algorithms=paper_algorithms,
        max_function_evaluations=max_function_evaluations,
    )
    comparison_mode = experiment_mode == FAIR_OPTIMIZER_COMPARISON
    environment = _load_paper_environment()
    available_vehicle_count = len(environment[0])
    fixed_road_vehicle_count = _fixed_road_vehicle_count(environment[0])
    canonical_road_vehicle_count = {
        "figure_6": 76,
        "figure_7": 76,
        "figure_8": 68,
        "figure_10": 68,
    }.get(figure)
    diagnostic_road_changed = bool(
        diagnostic_road_vehicle_count is not None
        and canonical_road_vehicle_count is not None
        and int(diagnostic_road_vehicle_count) != int(canonical_road_vehicle_count)
    )
    if (
        diagnostic_road_vehicle_count is not None
        and diagnostic_road_vehicle_count > fixed_road_vehicle_count
    ):
        raise ValueError(
            "diagnostic_road_vehicle_count exceeds the available paper-compatible "
            f"road pool ({fixed_road_vehicle_count})"
        )
    raw_rows: List[Dict[str, Any]] = []
    application_rows: List[Dict[str, Any]] = []
    execution_audit: List[Dict[str, Any]] = []
    scenario_records: List[Dict[str, Any]] = []

    if figure == "figure_6":
        figure_6_vehicle_count = int(diagnostic_vehicle_count or 76)
        figure_6_road_vehicle_count = int(
            diagnostic_road_vehicle_count or figure_6_vehicle_count
        )
        if figure_6_road_vehicle_count < figure_6_vehicle_count:
            raise ValueError(
                "Figure 6 diagnostic road vehicle count cannot be smaller than "
                "the mission vehicle count"
            )

        # The canonical Figure 6 baseline is always generated from the same
        # first-N vehicle pool.  A diagnostic road extension must add peers
        # without relocating/reseeding those mission vehicles.
        figure_6_vehicles = list(environment[0][:figure_6_vehicle_count])
        if len(figure_6_vehicles) != figure_6_vehicle_count:
            raise ValueError(
                f"Figure 6 requires {figure_6_vehicle_count} vehicles; "
                f"found {len(figure_6_vehicles)}"
            )
        figure_6_baseline_environment = (
            figure_6_vehicles,
            environment[1],
            environment[2],
        )

        for seed in seeds:
            road_state_override = None
            mission_assignment_vehicle_ids = None
            scenario_environment = figure_6_baseline_environment

            if diagnostic_road_vehicle_count is not None:
                baseline_state = _road_state(
                    figure_6_vehicles,
                    environment[1],
                    seed,
                    figure_6_vehicle_count,
                    None,
                )
                road_state_override = _extend_road_state_preserving_baseline(
                    baseline_state,
                    environment[0],
                    environment[1],
                    seed,
                    figure_6_road_vehicle_count,
                    None,
                )
                mission_assignment_vehicle_ids = list(baseline_state[1])
                scenario_environment = environment

            joint_ctx = _build_scenario(
                figure,
                seed,
                mission_vehicle_count=figure_6_vehicle_count,
                road_vehicle_count=figure_6_road_vehicle_count,
                speed_kmh=None,
                mec_capacity_ghz=50.0,
                sweep_parameter="iteration",
                sweep_value=float(tmax - 1),
                environment=scenario_environment,
                mission_assignment_vehicle_ids=mission_assignment_vehicle_ids,
                road_state_override=road_state_override,
            )
            result = _run_scenario(
                joint_ctx,
                algorithms,
                seed,
                tmax,
                population_size,
                max_function_evaluations,
            )
            raw_rows.extend(_figure_6_rows(result))
            _, apps = _standard_rows(figure, result)
            application_rows.extend(apps)
            execution_audit.extend(_execution_audit_rows(figure, result))
            scenario_records.append(dict(result["scenario"]))
        summary_rows = _figure_6_summary_rows(raw_rows)
    elif figure == "figure_7":
        figure_7_vehicle_counts = (
            (int(diagnostic_vehicle_count),)
            if diagnostic_vehicle_count is not None
            else VEHICLE_COUNTS
        )

        # Fig. 7 varies the number of *mission* vehicles. The paper does not
        # publish a separate larger total-road population for this figure.
        # Use the minimal 76-vehicle road scenario implied by the sweep maximum
        # and let _road_state() choose that 76-vehicle subset deterministically
        # from the full eligible database pool for each seed. Therefore:
        #   - the same seed uses exactly the same 76 road vehicles at 52..76;
        #   - a different seed can select a different 76-of-N road subset;
        #   - extra database vehicles are not forced into every Fig. 7 run.
        figure_7_road_vehicle_count = int(
            diagnostic_road_vehicle_count or max(VEHICLE_COUNTS)
        )
        figure_7_assignment_pool_count = (
            max(figure_7_vehicle_counts)
            if diagnostic_vehicle_count is not None
            else max(VEHICLE_COUNTS)
        )
        if figure_7_road_vehicle_count < figure_7_assignment_pool_count:
            raise ValueError(
                "Figure 7 road vehicle count cannot be smaller than the "
                "mission assignment pool"
            )

        figure_7_diagnostic_states: Dict[
            int,
            Tuple[
                Tuple[BenchmarkSnapshot, List[int], Dict[int, float], Dict[str, Any]],
                List[int],
            ],
        ] = {}
        if diagnostic_road_vehicle_count is not None:
            for seed in seeds:
                baseline_state = _road_state(
                    environment[0],
                    environment[1],
                    seed,
                    max(VEHICLE_COUNTS),
                    None,
                )
                extended_state = _extend_road_state_preserving_baseline(
                    baseline_state,
                    environment[0],
                    environment[1],
                    seed,
                    figure_7_road_vehicle_count,
                    None,
                )
                figure_7_diagnostic_states[int(seed)] = (
                    extended_state,
                    list(baseline_state[1]),
                )

        for vehicle_count in figure_7_vehicle_counts:
            for seed in seeds:
                road_state_override = None
                mission_assignment_vehicle_ids = None
                if diagnostic_road_vehicle_count is not None:
                    road_state_override, mission_assignment_vehicle_ids = (
                        figure_7_diagnostic_states[int(seed)]
                    )

                joint_ctx = _build_scenario(
                    figure,
                    seed,
                    mission_vehicle_count=vehicle_count,
                    road_vehicle_count=figure_7_road_vehicle_count,
                    speed_kmh=None,
                    mec_capacity_ghz=50.0,
                    sweep_parameter="mission_vehicle_count",
                    sweep_value=float(vehicle_count),
                    environment=environment,
                    assignment_pool_count=figure_7_assignment_pool_count,
                    mission_assignment_vehicle_ids=mission_assignment_vehicle_ids,
                    road_state_override=road_state_override,
                )
                result = _run_scenario(
                    joint_ctx,
                    algorithms,
                    seed,
                    tmax,
                    population_size,
                    max_function_evaluations,
                )
                run_rows, apps = _standard_rows(figure, result)
                raw_rows.extend(run_rows)
                application_rows.extend(apps)
                execution_audit.extend(
                    _execution_audit_rows(figure, result)
                )
                scenario_records.append(dict(result["scenario"]))
        summary_rows = _summary_rows(
            raw_rows,
            ("figure", "mission_vehicle_count", "algorithm"),
            ("avg_delay", "avg_efficiency", "total_efficiency", "completion_rate"),
        )
    elif figure == "figure_8":
        # Fig. 8 explicitly states that the number of vehicles is 68.
        # No separate larger road population is published for this figure.
        # Use exactly the same canonical 68-vehicle scenario regardless of
        # whether the database later contains 76, 96, or more vehicles.
        figure_8_vehicle_count = 68
        figure_8_road_vehicle_count = int(
            diagnostic_road_vehicle_count or figure_8_vehicle_count
        )
        if figure_8_road_vehicle_count < figure_8_vehicle_count:
            raise ValueError(
                "Figure 8 road vehicle count cannot be smaller than 68 mission vehicles"
            )
        figure_8_vehicles = list(environment[0][:figure_8_vehicle_count])
        if len(figure_8_vehicles) != figure_8_vehicle_count:
            raise ValueError(
                f"Figure 8 requires {figure_8_vehicle_count} vehicles; "
                f"found {len(figure_8_vehicles)}"
            )
        figure_8_baseline_environment = (
            figure_8_vehicles,
            environment[1],
            environment[2],
        )

        for seed in seeds:
            road_state_override = None
            mission_assignment_vehicle_ids = None
            scenario_environment = figure_8_baseline_environment
            if diagnostic_road_vehicle_count is not None:
                baseline_state = _road_state(
                    figure_8_vehicles,
                    environment[1],
                    seed,
                    figure_8_vehicle_count,
                    None,
                )
                road_state_override = _extend_road_state_preserving_baseline(
                    baseline_state,
                    environment[0],
                    environment[1],
                    seed,
                    figure_8_road_vehicle_count,
                    None,
                )
                mission_assignment_vehicle_ids = list(baseline_state[1])
                scenario_environment = environment

            joint_ctx = _build_scenario(
                figure,
                seed,
                mission_vehicle_count=figure_8_vehicle_count,
                road_vehicle_count=figure_8_road_vehicle_count,
                speed_kmh=None,
                mec_capacity_ghz=50.0,
                sweep_parameter="deadline_s",
                sweep_value=0.0,
                environment=scenario_environment,
                mission_assignment_vehicle_ids=mission_assignment_vehicle_ids,
                road_state_override=road_state_override,
            )
            result = _run_scenario(
                joint_ctx,
                algorithms,
                seed,
                tmax,
                population_size,
                max_function_evaluations,
            )
            deadline_rows, apps = _figure_8_rows(result)
            raw_rows.extend(deadline_rows)
            application_rows.extend(apps)
            execution_audit.extend(
                _execution_audit_rows(figure, result)
            )
            scenario_records.append(dict(result["scenario"]))
        summary_rows = _summary_rows(
            raw_rows,
            ("figure", "deadline_s", "deadline_ms", "algorithm"),
            ("avg_delay", "completion_rate"),
        )
    elif figure == "figure_9":
        road_length_m, _ = _road_geometry(environment[0])

        # The paper explicitly makes road density a function of speed in
        # Fig. 9.  Compute the required counts from that rule first, then use
        # only the canonical prefix needed by the densest point.  Consequently
        # extra database rows above that requirement cannot perturb any Fig. 9
        # realization.  If the database is smaller, the existing transparent
        # finite-pool cap still applies instead of inventing vehicle records.
        figure_9_required_counts = {
            float(speed): _vehicle_count_for_speed(
                float(speed),
                road_length_m,
                available_vehicle_count=None,
            )
            for speed in VEHICLE_SPEEDS_KMH
        }
        figure_9_pool_count = min(
            max(figure_9_required_counts.values()),
            available_vehicle_count,
        )
        figure_9_environment = (
            list(environment[0][:figure_9_pool_count]),
            environment[1],
            environment[2],
        )

        for speed_kmh in (sweep_values or VEHICLE_SPEEDS_KMH):
            vehicle_count = min(
                int(figure_9_required_counts[float(speed_kmh)]),
                figure_9_pool_count,
            )
            for seed in seeds:
                joint_ctx = _build_scenario(
                    figure,
                    seed,
                    mission_vehicle_count=vehicle_count,
                    road_vehicle_count=vehicle_count,
                    speed_kmh=float(speed_kmh),
                    mec_capacity_ghz=50.0,
                    sweep_parameter="speed_kmh",
                    sweep_value=float(speed_kmh),
                    environment=figure_9_environment,
                )
                result = _run_scenario(
                    joint_ctx,
                    algorithms,
                    seed,
                    tmax,
                    population_size,
                    max_function_evaluations,
                )
                run_rows, apps = _standard_rows(figure, result)
                for row in run_rows:
                    row["speed_kmh"] = float(speed_kmh)
                raw_rows.extend(run_rows)
                application_rows.extend(apps)
                execution_audit.extend(
                    _execution_audit_rows(figure, result)
                )
                scenario_records.append(dict(result["scenario"]))
        summary_rows = _summary_rows(
            raw_rows,
            ("figure", "speed_kmh", "algorithm"),
            ("total_efficiency",),
        )
    else:
        # Fig. 10 states that the number of vehicles is 68.  Hold that
        # complete vehicle scenario fixed while only MEC capacity changes.
        figure_10_vehicle_count = 68
        figure_10_road_vehicle_count = int(
            diagnostic_road_vehicle_count or figure_10_vehicle_count
        )
        if figure_10_road_vehicle_count < figure_10_vehicle_count:
            raise ValueError(
                "Figure 10 road vehicle count cannot be smaller than 68 mission vehicles"
            )
        figure_10_vehicles = list(environment[0][:figure_10_vehicle_count])
        if len(figure_10_vehicles) != figure_10_vehicle_count:
            raise ValueError(
                f"Figure 10 requires {figure_10_vehicle_count} vehicles; "
                f"found {len(figure_10_vehicles)}"
            )
        figure_10_baseline_environment = (
            figure_10_vehicles,
            environment[1],
            environment[2],
        )

        figure_10_diagnostic_states: Dict[
            int,
            Tuple[
                Tuple[BenchmarkSnapshot, List[int], Dict[int, float], Dict[str, Any]],
                List[int],
            ],
        ] = {}
        if diagnostic_road_vehicle_count is not None:
            for seed in seeds:
                baseline_state = _road_state(
                    figure_10_vehicles,
                    environment[1],
                    seed,
                    figure_10_vehicle_count,
                    None,
                )
                extended_state = _extend_road_state_preserving_baseline(
                    baseline_state,
                    environment[0],
                    environment[1],
                    seed,
                    figure_10_road_vehicle_count,
                    None,
                )
                figure_10_diagnostic_states[int(seed)] = (
                    extended_state,
                    list(baseline_state[1]),
                )

        for capacity_ghz in (sweep_values or MEC_CAPACITIES_GHZ):
            for seed in seeds:
                road_state_override = None
                mission_assignment_vehicle_ids = None
                scenario_environment = figure_10_baseline_environment
                if diagnostic_road_vehicle_count is not None:
                    road_state_override, mission_assignment_vehicle_ids = (
                        figure_10_diagnostic_states[int(seed)]
                    )
                    scenario_environment = environment

                joint_ctx = _build_scenario(
                    figure,
                    seed,
                    mission_vehicle_count=figure_10_vehicle_count,
                    road_vehicle_count=figure_10_road_vehicle_count,
                    speed_kmh=None,
                    mec_capacity_ghz=float(capacity_ghz),
                    sweep_parameter="mec_capacity_ghz",
                    sweep_value=float(capacity_ghz),
                    environment=scenario_environment,
                    mission_assignment_vehicle_ids=mission_assignment_vehicle_ids,
                    road_state_override=road_state_override,
                )
                result = _run_scenario(
                    joint_ctx,
                    algorithms,
                    seed,
                    tmax,
                    population_size,
                    max_function_evaluations,
                )
                run_rows, apps = _standard_rows(figure, result)
                for row in run_rows:
                    row["mec_capacity_ghz"] = float(capacity_ghz)
                raw_rows.extend(run_rows)
                application_rows.extend(apps)
                execution_audit.extend(
                    _execution_audit_rows(figure, result)
                )
                scenario_records.append(dict(result["scenario"]))
        summary_rows = _summary_rows(
            raw_rows,
            ("figure", "mec_capacity_ghz", "algorithm"),
            ("avg_efficiency", "completion_rate"),
        )

    budget_incomplete_rows = [
        row
        for row in execution_audit
        if str(row.get("algorithm", "")).lower() in POPULATION_ALGORITHMS
        if row.get("max_function_evaluations") is not None
        and not bool(row.get("evaluation_budget_exhausted", False))
    ]
    if experiment_mode == FAIR_OPTIMIZER_COMPARISON and budget_incomplete_rows:
        details = ", ".join(
            f"{row.get('algorithm')}@{row.get('scenario_id')}:"
            f"{row.get('function_evaluations')}/{row.get('max_function_evaluations')}"
            for row in budget_incomplete_rows[:8]
        )
        raise ValueError(
            "The common NFE budget was not consumed by every population "
            f"optimizer ({details}). Increase tmax; do not publish this run."
        )

    metadata = {
        "figure": figure,
        "article_doi": ARTICLE_DOI,
        "repetitions": repetitions,
        "seeds": seeds,
        "tmax": int(tmax),
        "population_size": int(
            population_size if population_size is not None else load_params_obj().S
        ),
        "population_size_matches_paper": int(
            population_size if population_size is not None else load_params_obj().S
        ) == 50,
        "max_function_evaluations": (
            None
            if max_function_evaluations is None
            else int(max_function_evaluations)
        ),
        "experiment_mode": experiment_mode,
        "stopping_rule": (
            "paper-native-generation-limit"
            if experiment_mode == PAPER_REPRODUCTION
            else "exact-common-objective-evaluation-budget"
        ),
        "paper_optimizer_settings_match": bool(
            experiment_mode == PAPER_REPRODUCTION
            and int(tmax) == 15
            and int(actual_population_size) == 50
        ),
        "comparison_budget_mode": (
            "equal-objective-evaluation-budget"
            if max_function_evaluations is not None
            else "generation-limited"
        ),
        "evaluation_budget_fully_consumed": not bool(budget_incomplete_rows),
        "budget_incomplete_runs": [
            {
                "scenario_id": row.get("scenario_id"),
                "seed": row.get("seed"),
                "algorithm": row.get("algorithm"),
                "function_evaluations": row.get("function_evaluations"),
                "max_function_evaluations": row.get("max_function_evaluations"),
            }
            for row in budget_incomplete_rows
        ],
        "dcsga_rank_seed_aligned": True,
        "dcsga_rank_recomputed_per_run": True,
        "proposed_method_guidance": {
            "algorithm": "cpo",
            "name": "DCC-DCPO",
            "fitness_policy": "paper evaluator shared unchanged with every comparator",
            "structural_criticality": "0.65 normalized seeded HEFT rank + 0.35 normalized transitive DAG reach; protected during exploration and targeted during exploitation",
            "cache_coupling": "capacity-feasible causal forward same-service reuse used only in odor proposals; block members follow the anchor provider actually selected",
            "success_memory": "accepted objective-improving moves only",
            "static_model_prior": "bounded proposal prior; never a fitness bonus",
            "prefetching": False,
            "final_fitness_bonus": False,
        },
        "baseline_isolation": {
            "gpc": "no CPO criticality, cache affinity, success memory, or model prior",
            "gwo": "D-GWO without pheromone, rank, or predictive-cache guidance",
            "gwo_aco": "deprecated compatibility alias for the same isolated D-GWO execution",
            "puma": "no CPO criticality, cache affinity, success memory, or model prior",
            "common_components": "scenario, feasible domains, task order, repair, exact evaluator, seed, population size, and NFE budget",
        },
        "algorithms": list(algorithms),
        "comparison_mode": comparison_mode,
        "statistical_protocol": {
            "independent_seed_count": int(repetitions),
            "minimum_recommended_independent_seeds": 30,
            "paper_ready": bool(repetitions >= 30),
            "paired_seed_design": True,
            "primary_report": "median and IQR plus mean and 95% confidence interval",
            "pairwise_test": "two-sided Wilcoxon signed-rank with Holm correction (p-values require SciPy; effect sizes are always computed)",
            "effect_size": "paired rank-biserial correlation",
            "pairing_unit": "same scenario_id and seed",
            "results_key": "pairwise_statistics",
        },
        "pairwise_statistics": _paired_algorithm_statistics(
            execution_audit,
            metric="total_efficiency",
        ),
        "cross_algorithm_convergence_axis": (
            "objective_function_evaluations"
            if figure == "figure_6" and comparison_mode
            else "paper_native"
        ),
        "generation_axis_warning": (
            "Generation counts are algorithm-specific and must not be used as "
            "the primary cross-algorithm compute budget. Use the exported NFE "
            "curve and, for final comparisons, set max_function_evaluations."
            if figure == "figure_6" and comparison_mode
            else None
        ),
        "diagnostic_mode": bool(
            diagnostic_road_changed
            or sweep_values is not None
            or (
                diagnostic_vehicle_count is not None
                and (
                    figure == "figure_7"
                    or int(diagnostic_vehicle_count) != 76
                )
            )
        ),
        "diagnostic_vehicle_count": (
            None
            if diagnostic_vehicle_count is None
            else int(diagnostic_vehicle_count)
        ),
        "diagnostic_road_vehicle_count": (
            None
            if diagnostic_road_vehicle_count is None
            else int(diagnostic_road_vehicle_count)
        ),
        "diagnostic_sweep_values": (
            None if sweep_values is None else list(sweep_values)
        ),
        "default_paper_algorithms": list(paper_algorithms),
        "default_run_algorithms": list(default_algorithms),
        "arrival_model": {
            "application_rate_per_second": float(
                load_params_obj().application_rate_per_second
            ),
            "dynamic_arrivals_applied": False,
            "figure_7_scheduling_epoch": (
                "one concurrent application per mission vehicle"
            ),
            "reason": (
                "The article reports 10 applications per second but does not "
                "specify the arrival distribution, observation horizon, or "
                "whether the rate is per vehicle or system-wide. Figure 7 is "
                "therefore evaluated as the joint scheduling epoch defined by "
                "Algorithm 2 instead of imposing an unverified arrival model."
            ),
        },
        "vehicle_pool": {
            "database_eligible_vehicle_count": int(available_vehicle_count),
            "fixed_road_vehicle_count": int(fixed_road_vehicle_count),
            "paper_max_mission_vehicle_count": int(PAPER_MAX_MISSION_VEHICLES),
            "table_iii_max_density_vehicle_count": int(
                _vehicle_count_for_speed(
                    TABLE_III_MIN_SPEED_KMH,
                    _road_geometry(environment[0])[0],
                )
            ),
            "selection_rule": (
                "figure-specific vehicle policies; diagnostic road extensions "
                "freeze the baseline mission snapshot and add cooperative-only peers "
                "without mutating the database"
            ),
            "figure_7_road_vehicle_count": int(
                diagnostic_road_vehicle_count or max(VEHICLE_COUNTS)
            ),
            "figure_7_seeded_subset_selection": True,
            "figure_vehicle_policy": {
                "figure_6": (
                    "default: 76 mission/road vehicles; diagnostic override freezes "
                    "that 76-vehicle snapshot and adds cooperative-only peers"
                ),
                "figure_7": (
                    "default: 52..76 mission vehicles inside a seeded 76-vehicle "
                    "road pool; diagnostic override freezes that 76-road snapshot "
                    "and adds cooperative-only peers"
                ),
                "figure_8": (
                    "default: 68 mission/road vehicles; diagnostic override freezes "
                    "that 68-vehicle snapshot and adds cooperative-only peers"
                ),
                "figure_9": (
                    "speed-derived road population from the article density "
                    "rule, capped only by a finite database pool"
                ),
                "figure_10": (
                    "default: 68 mission/road vehicles; diagnostic override freezes "
                    "that 68-vehicle snapshot and adds cooperative-only peers"
                ),
            },
        },
        "article_parameters": {
            "vehicle_counts": list(VEHICLE_COUNTS),
            "deadlines_ms": list(DEADLINES_MS),
            "vehicle_speeds_kmh": list(VEHICLE_SPEEDS_KMH),
            "mec_capacities_ghz": list(MEC_CAPACITIES_GHZ),
            "figure_8_repetitions": 100,
            "figure_8_vehicle_count": 68,
            "figure_10_vehicle_count": 68,
            "population_size": 50,
            "speed_density_rule": "mean inter-vehicle distance equals 2.5 times mean speed in m/s",
        },
        "optimizer_parameters": {
            "dcsga": {
                "population_size": int(actual_population_size),
                "levy_lambda": float(load_params_obj().levy_lambda),
                "initial_discard_probability": float(
                    load_params_obj().p_discard_init
                ),
                "discard_schedule": "min(1, 2*p0/max(iteration,1))",
                "initialization": "paper greedy plus one second-best task mutation per nest",
            },
            "gpc": {
                "name": "D-GPC",
                "population_size": int(actual_population_size),
                "greedy_ratio": 0.70,
                "diverse_dlhs_ratio": 0.30,
                "sparse_near_greedy_fallback": True,
                "gravity": 9.8,
                "ramp_angle_degrees": 14.0,
                "friction_min": 1.0,
                "friction_max": 10.0,
                "substitution_probability": 0.50,
                "stagnation_escape_after": 5,
                "restart_fraction": 0.15,
                "elite_ratio": 0.20,
                "service_affinity_memory": False,
                "rank_guidance": False,
                "cache_guidance": False,
                "domain_guidance": "none; comparator retains physical GPC controls and feasible categorical projection only",
                "categorical_move": "dimension-normalised sparse physical GPC projection",
            },
            "gwo": {
                "name": "D-GWO",
                "reference_doi": "10.1016/j.advengsoft.2013.12.007",
                "population_size": int(actual_population_size),
                "greedy_ratio": 0.90,
                "diverse_dlhs_ratio": 0.10,
                "sparse_near_greedy_fallback": True,
                "a_schedule": "nonlinear 2-to-0 with bounded diversity/stagnation correction",
                "stagnation_escape_after": 5,
                "escape_fraction": 0.15,
                "pheromone_guidance": False,
                "rank_guidance": False,
                "cache_guidance": False,
                "alpha_neighborhood": "one-coordinate alpha/beta/delta-guided with a phase-dependent exploration floor",
                "categorical_move": "alpha-beta-delta encircling inside an O(sqrt(D)) trust region",
                "article_exact": False,
            },
            "puma": {
                "name": "D-PO",
                "reference_doi": "10.1007/s10586-023-04221-5",
                "population_size": int(actual_population_size),
                "greedy_ratio": 0.0,
                "categorical_coverage_ratio": 1.0,
                "unexperienced_iterations": 3,
                "phase_weights": [0.50, 0.50, 0.30],
                "mega_exploration": 0.99,
                "mega_exploitation": 0.99,
                "initial_crossover_probability": 0.20,
                "exploitation_q": 0.67,
                "exploitation_beta": 2.0,
                "phase_selection": "rolling improvement and improvement-per-NFE hyper-heuristic",
                "categorical_move": "domain-valid random-global/six-peer differential exploration and best/peer/consensus exploitation",
                "survival": "greedy phase survival and best-N pooled survival in the first three learning iterations",
                "article_exact": False,
            },
            "cpo": {
                "name": "DCC-DCPO",
                "revision": "deadline-dependency-cache-coupled-v7",
                "reference_doi": "10.1016/j.knosys.2023.111257",
                "population_size": int(actual_population_size),
                "greedy_ratio": 0.75,
                "diverse_dlhs_ratio": 0.25,
                "four_defenses": ["sight", "sound", "odor", "physical_attack"],
                "exploration_defenses": ["sight", "sound"],
                "exploitation_defenses": ["odor", "physical_attack"],
                "cpr_cycles": 2,
                "cpr_minimum_ratio": 0.80,
                "stagnation_restart_after": 3,
                "restart_fraction": 0.15,
                "success_memory_evaporation": 0.10,
                "provider_memory_weight": 0.65,
                "adaptation_clock": "consumed search NFE when max_function_evaluations is set; generation fraction otherwise",
                "minimum_defense_shares": {
                    "sight": 0.10,
                    "sound": 0.10,
                    "odor": 0.16,
                    "physical_attack": 0.28,
                },
                "elite_profile_size": 8,
                "physical_archive_probability": 0.88,
                "exploration_radius": "ceil(2 + (ceil(log2(D+1))-2)*(1-progress))",
                "exploitation_radius": "odor uses 1-to-4 coordinates but keeps the minimum meaningful two-task block when prospective same-service reuse exists; physical attack changes one coordinate",
                "criticality_guidance": "0.70 structural criticality plus 0.30 normalized owning-application deadline pressure; structural criticality is 0.65 normalized seeded HEFT rank plus 0.35 normalized transitive DAG reach; proposal selection only",
                "elite_disagreement_guidance": "rank-weighted top-8 provider disagreement guides physical attacks",
                "service_affinity_guidance": "capacity-feasible causal forward-only same-service reuse; odor retains an anchor-plus-one-future-request block at late NFE and selected later block members deterministically follow the feasible provider actually chosen by the anchor; exact common evaluation accepts or rejects the block",
                "static_model_guidance": "per-task provider prior derived from the common computation-time/energy equations and the article alpha/beta weights; used only for bounded exploitation proposals",
                "static_model_guidance_max_probability": 0.35,
                "static_model_guidance_scope": "odor and physical-attack proposals only; queueing, dependency transfers, cache state, acceptance and final selection remain governed by the exact common evaluator",
                "categorical_move": "domain-valid logarithmic global and 1-to-4-coordinate local neighborhoods; provider IDs are never treated as continuous coordinates",
                "survival": "elitist parent/archive pool with exact objective-evaluation accounting",
                "article_exact": False,
            },
            "cpo_ablations": {
                "dcpo_base": "four categorical defenses plus CPR; no structural/cache/model/memory guidance",
                "dcpo_criticality": "base plus structural DAG criticality only",
                "dcpo_cache": "base plus capacity-feasible cache coupling only",
                "comparison_contract": "same objective, scenario, repair, initialization family, seed, population and NFE budget as DCC-DCPO",
            },
        },
        "declared_limitations": [
            "DTOSC uses the legacy pre-repair stage-wise provider-path dynamic-programming reconstruction with the common benchmark evaluator. This compatibility baseline is intentionally retained for reproducibility/sensitivity and is not claimed to be source-exact DTOSC 2022.",
            "The channel and sender-side power models remain declared approximations.",
            "Figures 6-10 evaluate one joint scheduling epoch. The article reports 10 applications per second but does not define a reproducible arrival distribution, observation horizon, or whether that rate is per vehicle or system-wide; no unverified arrival process is imposed on the paper figures.",
            "By default, Figures 6, 8, and 10 use the article-stated vehicle count as the complete scenario count because no separate larger road population is published. Figure 7 defaults to 52..76 mission vehicles inside a seeded 76-vehicle road pool. The optional diagnostic_road_vehicle_count separates mission and total road populations only for sensitivity analysis and is never labeled article-exact. Figure 9 alone changes road density with speed according to the article's 2.5-second spacing rule.",
            "Mission and cooperative roles are not forced to be disjoint: a mission vehicle may also provide V2V service to a same-RSU peer, consistent with the paper's alliance domain V_n excluding n; the paper does not publish a separate cooperative-only vehicle count.",
            "Each mission vehicle uses exactly one wireless access RSU. A selected remote MEC is reached through that access RSU and the inter-RSU broadband path adds zero modeled delay/energy because the paper does not publish backhaul parameters.",
            "Figure 9 uses the article speed-density equation and caps only when the finite database pool is smaller than the derived road population.",
            "The service compile workload Wk is not reported in Table III and is set equal to the corresponding task workload as a declared deterministic assumption.",
            "The database contains multiple distinct ten-task DAG templates, but the exact DAG realizations and generator parameters used from reference [48] are not published; the stored DAGs are therefore reference-compatible reconstructions rather than source-exact graphs.",
        ],
        "reconstruction_audit": _paper_reconstruction_audit(environment[2]),
        "scenario_records": scenario_records,
        "execution_audit": execution_audit,
    }
    convergence_diagnostics = (
        _convergence_diagnostics(raw_rows)
        if figure == "figure_6"
        else []
    )
    artifacts = {}
    if export_artifacts:
        artifacts = _export_experiment(
            figure,
            raw_rows,
            summary_rows,
            application_rows,
            metadata,
        )
    return {
        **metadata,
        "convergence_diagnostics": convergence_diagnostics,
        "summary": summary_rows,
        "artifacts": artifacts,
    }
