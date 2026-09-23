from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

from django.core.cache import cache as django_cache
from django.db.models import Q

from monarch_pylib.model import transmission

from algorithm.optimizer_common import compute_local_ranks
from application.models import Application
from dag.models import ApplicationType, Task
from object.models import RSUVehicle, ServiceProvider, Vehicle
from parameter.services import load_params_obj
from system.build_context import MiniSystemContextBuilder
from .paper_model import allocated_cpu_frequency_hz, article_weights, local_reference


@dataclass(frozen=True)
class BenchmarkSnapshot:
    at: datetime
    time_step_s: float
    positions: Dict[int, Tuple[float, float]]
    vehicle_rsu_ids: Dict[int, int]
    rsu_vehicle_counts: Dict[int, int]


def _active_relation_map(snapshot_at: datetime) -> Dict[int, int]:
    relations = list(
        RSUVehicle.objects.filter(start_time__lte=snapshot_at)
        .filter(Q(end_time__isnull=True) | Q(end_time__gt=snapshot_at))
        .exclude(vehicle_id_id__isnull=True)
        .exclude(rsu_id_id__isnull=True)
        .order_by("vehicle_id_id", "-start_time", "-id")
        .values_list("vehicle_id_id", "rsu_id_id")
    )
    result: Dict[int, int] = {}
    for vehicle_id, rsu_id in relations:
        vehicle_id = int(vehicle_id)
        rsu_id = int(rsu_id)
        if vehicle_id in result:
            raise ValueError(
                f"Vehicle {vehicle_id} has more than one RSU relation at {snapshot_at.isoformat()}"
            )
        result[vehicle_id] = rsu_id
    return result


def _snapshot_time_step(snapshot_at: datetime) -> float:
    base_time = (
        RSUVehicle.objects.filter(start_time__lte=snapshot_at)
        .order_by("start_time", "id")
        .values_list("start_time", flat=True)
        .first()
    )
    if base_time is None:
        return 0.0
    return max(0.0, float((snapshot_at - base_time).total_seconds()))


def _vehicle_position(vehicle: Vehicle, time_step_s: float) -> Tuple[float, float]:
    path = vehicle.path or []
    if len(path) < 2:
        return float(vehicle.x_coord), float(vehicle.y_coord)
    initial_snapshot = vehicle.initial_snapshot or {}
    speed_kmh = initial_snapshot.get("speed_kmh", vehicle.speed)
    speed_mps = float(speed_kmh or 0.0) / 3.6
    path_length = float(transmission.path_length_2d(path))
    if speed_mps <= 0.0 or path_length <= 0.0:
        return float(path[0][0]), float(path[0][1])
    travel_time = path_length / speed_mps
    return transmission.current_position(
        travel_time,
        min(max(0.0, float(time_step_s)), travel_time),
        path,
    )


def _build_snapshot(
    applications: List[Application],
    snapshot_at: datetime,
    *,
    use_live_vehicle_positions: bool = False,
) -> BenchmarkSnapshot:
    vehicle_rsu_ids = _active_relation_map(snapshot_at)
    rsu_vehicle_counts: Dict[int, int] = {}
    for rsu_id in vehicle_rsu_ids.values():
        rsu_vehicle_counts[int(rsu_id)] = rsu_vehicle_counts.get(int(rsu_id), 0) + 1
    vehicle_ids = set(vehicle_rsu_ids)
    vehicle_ids.update(
        int(app.vehicle_id_id)
        for app in applications
        if app.vehicle_id_id is not None
    )
    time_step_s = _snapshot_time_step(snapshot_at)
    vehicles = Vehicle.objects.filter(id__in=sorted(vehicle_ids)).in_bulk()
    missing = sorted(vehicle_ids - set(int(vehicle_id) for vehicle_id in vehicles))
    if missing:
        raise ValueError(f"Vehicles not found for benchmark snapshot: {missing}")
    if use_live_vehicle_positions:
        # Runtime simulation already advances Vehicle.x_coord/y_coord with the
        # route-derived speed.  Reconstructing positions here from the paper
        # seed speed would make the radio snapshot disagree with the active
        # RSUVehicle relation.  Use the frozen live coordinates instead.
        positions = {
            int(vehicle_id): (float(vehicle.x_coord), float(vehicle.y_coord))
            for vehicle_id, vehicle in vehicles.items()
        }
    else:
        positions = {
            int(vehicle_id): _vehicle_position(vehicle, time_step_s)
            for vehicle_id, vehicle in vehicles.items()
        }
    return BenchmarkSnapshot(
        at=snapshot_at,
        time_step_s=time_step_s,
        positions=positions,
        vehicle_rsu_ids=vehicle_rsu_ids,
        rsu_vehicle_counts=rsu_vehicle_counts,
    )


def _available_provider_ids(app: Application, snapshot: BenchmarkSnapshot) -> List[int]:
    provider_ids = list(
        ServiceProvider.objects.filter(type="rsu", rsu_id__isnull=False)
        .order_by("id")
        .values_list("id", flat=True)
    )
    access_rsu_id = snapshot.vehicle_rsu_ids.get(int(app.vehicle_id_id))
    if access_rsu_id is not None:
        peer_vehicle_ids = [
            vehicle_id
            for vehicle_id, rsu_id in snapshot.vehicle_rsu_ids.items()
            if int(rsu_id) == int(access_rsu_id)
            and int(vehicle_id) != int(app.vehicle_id_id)
        ]
        provider_ids.extend(
            ServiceProvider.objects.filter(
                type="vehicle",
                vehicle_id_id__in=peer_vehicle_ids,
            )
            .order_by("id")
            .values_list("id", flat=True)
        )
    return list(dict.fromkeys(int(sp_id) for sp_id in provider_ids))


def _apply_snapshot_context(
    ctx: Dict[str, Any],
    app: Application,
    snapshot: BenchmarkSnapshot,
    *,
    route_remote_mec_via_access_rsu: bool = False,
) -> None:
    params = load_params_obj()
    vehicle_id = int(app.vehicle_id_id)
    if vehicle_id not in snapshot.positions:
        raise ValueError(f"Vehicle {vehicle_id} has no benchmark snapshot position")
    mission_x, mission_y = snapshot.positions[vehicle_id]
    vehicle_height = float(params.h_vehicle)
    rsu_height = float(params.h_rsu)
    compute_sp_positions = copy.deepcopy(ctx.get("sp_position", {}))
    sp_positions = copy.deepcopy(compute_sp_positions)
    radio_endpoint_provider_ids = {
        int(sp_id): int(sp_id)
        for sp_id in ctx.get("provider_ids", [])
    }
    access_rsu_id = snapshot.vehicle_rsu_ids.get(vehicle_id)
    access_rsu_provider_id = None
    access_rsu_position = None
    if route_remote_mec_via_access_rsu:
        if access_rsu_id is None:
            raise ValueError(
                f"Mission vehicle {vehicle_id} has no access RSU in the paper scenario"
            )
        for provider_id, provider_rsu_id in ctx.get("sp_rsu_ids", {}).items():
            if provider_rsu_id is not None and int(provider_rsu_id) == int(access_rsu_id):
                access_rsu_provider_id = int(provider_id)
                break
        if access_rsu_provider_id is None:
            raise ValueError(
                f"Access RSU {access_rsu_id} has no service provider in the paper scenario"
            )
        access_rsu_position = compute_sp_positions.get(access_rsu_provider_id)
        if access_rsu_position is None:
            raise ValueError(
                f"Access RSU provider {access_rsu_provider_id} has no position"
            )

    for sp_id in ctx.get("provider_ids", []):
        sp_id = int(sp_id)
        provider_type = ctx.get("sp_types", {}).get(sp_id)
        if provider_type == "vehicle":
            provider_vehicle_id = ctx.get("sp_vehicle_ids", {}).get(sp_id)
            if provider_vehicle_id is None:
                raise ValueError(f"Vehicle provider {sp_id} has no vehicle")
            provider_vehicle_id = int(provider_vehicle_id)
            if provider_vehicle_id not in snapshot.positions:
                raise ValueError(
                    f"Vehicle provider {sp_id} has no benchmark snapshot position"
                )
            sp_positions[sp_id] = snapshot.positions[provider_vehicle_id]
        elif provider_type == "rsu" and route_remote_mec_via_access_rsu:
            # The paper states that a vehicle accesses exactly one RSU.  A task
            # may still execute on another MEC server through broadband RSU
            # cooperation.  Therefore every MEC candidate uses the wireless
            # endpoint of the mission vehicle's access RSU; the MEC provider ID
            # remains unchanged for CPU, queue and cache accounting.
            sp_positions[sp_id] = access_rsu_position
            radio_endpoint_provider_ids[sp_id] = int(access_rsu_provider_id)

    distances: Dict[int, float] = {}
    connected_counts: Dict[int, float] = {}
    alliance_count = float(
        max(snapshot.rsu_vehicle_counts.get(int(access_rsu_id), 0), 1)
        if access_rsu_id is not None
        else 1
    )
    for sp_id in ctx.get("provider_ids", []):
        sp_id = int(sp_id)
        sp_x, sp_y = sp_positions[sp_id]
        provider_type = ctx.get("sp_types", {}).get(sp_id)
        provider_height = rsu_height if provider_type == "rsu" else vehicle_height
        distances[sp_id] = float(
            transmission.distance_3d(
                mission_x,
                mission_y,
                vehicle_height,
                float(sp_x),
                float(sp_y),
                provider_height,
            )
        )
        if provider_type == "rsu":
            if route_remote_mec_via_access_rsu:
                # Wireless bandwidth is shared at the access RSU, even when
                # the selected compute server is a remote MEC reached through
                # the broadband RSU backhaul.
                connected_counts[sp_id] = alliance_count
            else:
                rsu_id = ctx.get("sp_rsu_ids", {}).get(sp_id)
                connected_counts[sp_id] = float(
                    max(snapshot.rsu_vehicle_counts.get(int(rsu_id), 0), 1)
                    if rsu_id is not None
                    else 1
                )
        else:
            connected_counts[sp_id] = alliance_count
    ctx["sp_position"] = sp_positions
    ctx["sp_radio_endpoint_ids"] = radio_endpoint_provider_ids
    ctx["distance"] = distances
    ctx["v_m"] = connected_counts
    if route_remote_mec_via_access_rsu:
        ctx["sp_compute_position"] = compute_sp_positions
        ctx["access_rsu_id"] = int(access_rsu_id)
        ctx["access_rsu_provider_id"] = int(access_rsu_provider_id)
        ctx["paper_access_rsu_routing"] = True
        ctx["remote_mec_backhaul_model"] = "broadband-zero-added-delay"
        ctx["remote_mec_backhaul_delay_s"] = 0.0
    ctx["connected_vehicles_count"] = connected_counts
    ctx["cache"] = {
        int(sp_id): set()
        for sp_id in ctx.get("provider_ids", [])
    }
    ctx["cache_value_mu"] = {
        key: 0.0
        for key in ctx.get("cache_value_mu", {})
    }
    ctx["benchmark_snapshot_at"] = snapshot.at.isoformat()
    ctx["benchmark_snapshot_time_step_s"] = float(snapshot.time_step_s)
    ctx["benchmark_initial_cache"] = "empty"


def build_benchmark_context(
    application_id: int,
    snapshot: BenchmarkSnapshot | None = None,
    *,
    route_remote_mec_via_access_rsu: bool = False,
    use_live_vehicle_positions: bool = False,
) -> dict:
    app = (
        Application.objects.select_related("vehicle_id", "application_type_id")
        .get(id=int(application_id))
    )
    if app.vehicle_id_id is None:
        raise ValueError(f"Application {application_id} has no vehicle")
    if snapshot is None:
        snapshot = _build_snapshot(
            [app],
            app.start_at,
            use_live_vehicle_positions=use_live_vehicle_positions,
        )
    if app.start_at != snapshot.at:
        raise ValueError(
            f"Application {application_id} does not belong to benchmark snapshot "
            f"{snapshot.at.isoformat()}"
        )
    provider_ids = _available_provider_ids(app, snapshot)
    cache_key = f"vehicle_available_sps_{app.vehicle_id_id}"
    missing_value = object()
    previous_value = django_cache.get(cache_key, missing_value)
    django_cache.set(cache_key, provider_ids, timeout=600)
    try:
        ctx = MiniSystemContextBuilder(application_id=app.id).build_context()
    finally:
        if previous_value is missing_value:
            django_cache.delete(cache_key)
        else:
            django_cache.set(cache_key, previous_value, timeout=600)
    if not ctx.get("provider_ids"):
        raise ValueError(f"Application {application_id} has no available providers")
    _apply_snapshot_context(
        ctx,
        app,
        snapshot,
        route_remote_mec_via_access_rsu=route_remote_mec_via_access_rsu,
    )
    return ctx


def build_synthetic_benchmark_context(
    application_id: int,
    vehicle: Vehicle,
    application_type: ApplicationType,
    snapshot: BenchmarkSnapshot,
    *,
    deadline_ms: float | None = None,
) -> Dict[str, Any]:
    deadline_ms = float(
        application_type.deadline if deadline_ms is None else deadline_ms
    )
    deadline_s = deadline_ms / 1000.0

    alpha_n = 0.01 / deadline_s + 0.6
    beta_n = 1.0 - alpha_n
    if not 0.0 <= alpha_n <= 1.0 or not 0.0 <= beta_n <= 1.0:
        raise ValueError(
            f"Synthetic deadline {deadline_ms} ms produces invalid paper weights"
        )
    builder = MiniSystemContextBuilder(application_id=int(application_id))
    builder.ctx = {
        "application_id": int(application_id),
        "application": {
            "id": int(application_id),
            "application_type_id": int(application_type.id),
            "vehicle_id": int(vehicle.id),
        },
        "application_initial_snapshot": {
            "synthetic_paper_scenario": True,
            "scenario_deadline_ms": float(deadline_ms),
            "dag_template_application_type_id": int(application_type.id),
        },
        "application_type_initial_snapshot": application_type.initial_snapshot or {},
        "application_start_at": snapshot.at,
        "t_ddl_s": deadline_s,
        "deadline_s": deadline_s,
        "deadline_max_s": deadline_s,
        "alpha_n": float(alpha_n),
        "beta_n": float(beta_n),
        "vehicle_id": int(vehicle.id),
    }
    app_ref = SimpleNamespace(vehicle_id_id=int(vehicle.id))
    provider_ids = _available_provider_ids(app_ref, snapshot)
    cache_key = f"vehicle_available_sps_{vehicle.id}"
    missing_value = object()
    previous_value = django_cache.get(cache_key, missing_value)
    django_cache.set(cache_key, provider_ids, timeout=600)
    try:
        builder._build_task_info()
        builder._build_providers()
        builder._build_network_state()
        builder._build_cache_state()
        builder._build_assignment_state()
        builder._build_auxiliary_fields()
    finally:
        if previous_value is missing_value:
            django_cache.delete(cache_key)
        else:
            django_cache.set(cache_key, previous_value, timeout=600)
    ctx = builder.ctx
    if not ctx.get("provider_ids"):
        raise ValueError(
            f"Synthetic application {application_id} has no available providers"
        )
    _apply_snapshot_context(
        ctx,
        app_ref,
        snapshot,
        route_remote_mec_via_access_rsu=True,
    )
    return ctx


@dataclass(frozen=True)
class JointTaskRef:
    joint_task_id: int
    application_id: int
    task_id: int
    is_entry: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_task_id": self.joint_task_id,
            "application_id": self.application_id,
            "task_id": self.task_id,
            "is_entry": self.is_entry,
        }


def _unique_ints(values: Iterable[int]) -> List[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _task_index_map(application_type_ids: Iterable[int]) -> Dict[int, str]:
    return {
        int(task.id): str(task.index or "").strip()
        for task in Task.objects.filter(
            application_type_id_id__in=_unique_ints(application_type_ids)
        ).only("id", "index")
    }


def _looks_like_entry_index(value: str) -> bool:
    compact = value.strip().lower().replace("_", "").replace("-", "")
    return compact in {"1", "t1", "task1", "entry", "entrytask"}


def _entry_task_id(app_ctx: Dict[str, Any], task_indexes: Dict[int, str]) -> int:
    task_ids = [int(task_id) for task_id in app_ctx.get("task_ids", [])]
    if not task_ids:
        raise ValueError(
            f"Application {app_ctx.get('application_id')} has no tasks"
        )

    indexed_candidates = [
        task_id
        for task_id in task_ids
        if _looks_like_entry_index(task_indexes.get(task_id, ""))
    ]
    if len(indexed_candidates) == 1:
        return indexed_candidates[0]
    if len(indexed_candidates) > 1:
        raise ValueError(
            f"Application {app_ctx.get('application_id')} has more than one T1 task"
        )

    ready = [int(task_id) for task_id in app_ctx.get("tasks", {}).get("ready", [])]
    if len(ready) == 1:
        return ready[0]

    raise ValueError(
        "The paper requires exactly one entry task per application. "
        f"Application {app_ctx.get('application_id')} has ready tasks {ready}."
    )


def _natural_topological_order(app_ctx: Dict[str, Any]) -> List[int]:
    task_ids = [int(task_id) for task_id in app_ctx.get("task_ids", [])]
    dependencies = {
        task_id: [
            int(pred)
            for pred in app_ctx.get("dependencies", {}).get(task_id, [])
        ]
        for task_id in task_ids
    }

    indegree = {task_id: 0 for task_id in task_ids}
    children = {task_id: [] for task_id in task_ids}

    for task_id, predecessors in dependencies.items():
        for predecessor in predecessors:
            if predecessor in indegree:
                indegree[task_id] += 1
                children.setdefault(predecessor, []).append(task_id)

    ready = [task_id for task_id in task_ids if indegree[task_id] == 0]
    ready.sort()
    order: List[int] = []

    while ready:
        task_id = ready.pop(0)
        order.append(task_id)
        for child in sorted(children.get(task_id, [])):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()

    if len(order) != len(task_ids):
        raise ValueError(
            f"Application {app_ctx.get('application_id')} task graph is not a DAG"
        )

    return order


def _round_robin_orders(orders: Iterable[Iterable[int]]) -> List[int]:
    """Interleave per-application task orders without breaking either DAG.

    The paper defines TO-w.o.-R as selecting tasks from the top to the bottom
    *in turn* across application DAGs.  Concatenating complete DAGs makes the
    result depend on application-list position and can starve every later
    application.  A round-robin merge preserves each application's natural
    topological order while implementing the stated inter-application turn.
    """
    pending = [list(order) for order in orders]
    merged: List[int] = []
    while any(pending):
        for order in pending:
            if order:
                merged.append(int(order.pop(0)))
    return merged


def _apply_article_cpu_model(app_ctx: Dict[str, Any]) -> None:


    params = load_params_obj()
    kappa = float(params.k)
    deadline_s = float(app_ctx["deadline_s"])

    weights = article_weights(deadline_s)

    fmax_by_provider = {
        int(sp_id): float(value)
        for sp_id, value in app_ctx.get("sp_cpu_freq", {}).items()
    }

    local_sp_id = int(app_ctx["local_sp_id"])

    if local_sp_id not in fmax_by_provider:
        raise ValueError(
            f"Application {app_ctx.get('application_id')} local provider "
            f"{local_sp_id} has no CPU capacity"
        )

    reference = local_reference(
        app_ctx["cpu_cycles"].values(),
        vehicle_fmax_hz=fmax_by_provider[local_sp_id],
        deadline_s=deadline_s,
        kappa=kappa,
    )

    allocated_by_provider: Dict[int, float] = {}

    for sp_id, provider_fmax_hz in fmax_by_provider.items():
        provider_kind = str(
            app_ctx.get("sp_types", {}).get(sp_id, "")
        )

        allocated_by_provider[sp_id] = allocated_cpu_frequency_hz(
            provider_kind=provider_kind,
            provider_fmax_hz=provider_fmax_hz,
            reference=reference,
            weights=weights,
            kappa=kappa,
        )

    app_ctx["alpha_n"] = float(weights.alpha)
    app_ctx["beta_n"] = float(weights.beta)

    app_ctx["paper_weights"] = {
        "alpha_n": float(weights.alpha),
        "beta_n": float(weights.beta),
    }

    app_ctx["paper_reference"] = {
        "t_local_s": float(reference.t_local_s),
        "t_ref_s": float(reference.t_ref_s),
        "e_local_j": float(reference.e_local_j),
    }

    app_ctx["kappa"] = kappa

                         
    app_ctx["sp_cpu_fmax_hz"] = fmax_by_provider

                                     
    app_ctx["sp_cpu_allocated_hz"] = allocated_by_provider

                                               
    app_ctx["sp_cpu_freq"] = copy.deepcopy(
        allocated_by_provider
    )

    app_ctx["article_exact_cpu_allocation"] = True
def _merge_initial_cache(application_contexts: Dict[int, Dict[str, Any]]) -> Dict[int, set[int]]:
    merged: Dict[int, set[int]] = {}
    for app_ctx in application_contexts.values():
        for sp_id, cached_types in app_ctx.get("cache", {}).items():
            merged.setdefault(int(sp_id), set()).update(
                int(task_type_id) for task_type_id in cached_types
            )
    return merged


def _merge_provider_map(
    application_contexts: Dict[int, Dict[str, Any]],
    key: str,
) -> Dict[int, Any]:
    result: Dict[int, Any] = {}
    for app_ctx in application_contexts.values():
        for sp_id, value in app_ctx.get(key, {}).items():
            sp_id = int(sp_id)
            if sp_id in result and result[sp_id] != value:
                raise ValueError(
                    f"Provider {sp_id} has inconsistent {key} values across contexts"
                )
            result[sp_id] = copy.deepcopy(value)
    return result




def _validate_joint_order(
    application_contexts: Dict[int, Dict[str, Any]],
    task_refs: Dict[int, JointTaskRef],
    reverse_task_refs: Dict[Tuple[int, int], int],
    task_order: List[int],
) -> None:
    positions = {int(task_id): index for index, task_id in enumerate(task_order)}
    for app_id, app_ctx in application_contexts.items():
        entry_task_id = int(app_ctx["entry_task_id"])
        for child_id, predecessors in app_ctx.get("dependencies", {}).items():
            child_id = int(child_id)
            if child_id == entry_task_id:
                continue
            child_joint = reverse_task_refs[(int(app_id), child_id)]
            for predecessor_id in predecessors:
                predecessor_id = int(predecessor_id)
                if predecessor_id == entry_task_id:
                    continue
                predecessor_joint = reverse_task_refs[(int(app_id), predecessor_id)]
                if positions[predecessor_joint] >= positions[child_joint]:
                    raise ValueError(
                        "Task ranking produced a dependency-invalid joint order: "
                        f"application={app_id}, predecessor={predecessor_id}, child={child_id}"
                    )


def _apply_mec_capacity_override(
    application_contexts: Dict[int, Dict[str, Any]],
    mec_capacity_ghz: float | None,
) -> None:
    if mec_capacity_ghz is None:
        return
    capacity_hz = float(mec_capacity_ghz) * 1e9
    if capacity_hz <= 0.0:
        raise ValueError("MEC capacity must be positive")
    for app_ctx in application_contexts.values():
        for sp_id, provider_type in app_ctx.get("sp_types", {}).items():
            if provider_type == "rsu":
                app_ctx["sp_cpu_freq"][int(sp_id)] = capacity_hz


def _assemble_joint_context(
    application_ids: List[int],
    application_contexts: Dict[int, Dict[str, Any]],
    snapshot: BenchmarkSnapshot,
    scenario_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    max_deadline_s = max(
        float(app_ctx["deadline_s"])
        for app_ctx in application_contexts.values()
    )
    task_indexes = {
        int(task_id): str(index)
        for app_ctx in application_contexts.values()
        for task_id, index in app_ctx.get("task_indexes", {}).items()
    }
    task_refs: Dict[int, JointTaskRef] = {}
    reverse_task_refs: Dict[Tuple[int, int], int] = {}
    entry_task_ids: Dict[int, int] = {}
    optimized_task_ids: List[int] = []
    ranked_rows: List[Tuple[int, float]] = []
    unranked_orders: List[List[int]] = []
    task_domains: Dict[int, List[int]] = {}
    next_joint_task_id = 1

    for app_id in application_ids:
        app_ctx = application_contexts[app_id]
        entry_original_id = _entry_task_id(app_ctx, task_indexes)
        local_sp_id = app_ctx.get("local_sp_id")
        if local_sp_id is None:
            raise ValueError(f"Application {app_id} has no local service provider")
        provider_ids = _unique_ints(app_ctx.get("provider_ids", []))
        if int(local_sp_id) not in provider_ids:
            raise ValueError(
                f"Application {app_id} local provider {local_sp_id} is missing from its domain"
            )
        local_ranks = compute_local_ranks(app_ctx)
        urgency = max_deadline_s - float(app_ctx["deadline_s"])
        natural_order = _natural_topological_order(app_ctx)
        original_to_joint: Dict[int, int] = {}

        for original_task_id in app_ctx["task_ids"]:
            original_task_id = int(original_task_id)
            joint_task_id = next_joint_task_id
            next_joint_task_id += 1
            is_entry = original_task_id == entry_original_id
            task_refs[joint_task_id] = JointTaskRef(
                joint_task_id=joint_task_id,
                application_id=app_id,
                task_id=original_task_id,
                is_entry=is_entry,
            )
            reverse_task_refs[(app_id, original_task_id)] = joint_task_id
            original_to_joint[original_task_id] = joint_task_id
            task_domains[joint_task_id] = list(provider_ids)
            if is_entry:
                entry_task_ids[app_id] = joint_task_id
            else:
                optimized_task_ids.append(joint_task_id)
                ranked_rows.append(
                    (joint_task_id, float(local_ranks[original_task_id]) + urgency)
                )

        unranked_orders.append([
            original_to_joint[original_task_id]
            for original_task_id in natural_order
            if original_task_id != entry_original_id
        ])

        app_ctx["entry_task_id"] = int(entry_original_id)
        app_ctx["optimized_task_ids"] = [
            int(task_id)
            for task_id in app_ctx["task_ids"]
            if int(task_id) != entry_original_id
        ]

    unranked_task_ids = _round_robin_orders(unranked_orders)
    ranked_task_ids = [
        joint_task_id
        for joint_task_id, _ in sorted(
            ranked_rows,
            key=lambda row: (-row[1], row[0]),
        )
    ]
    _validate_joint_order(
        application_contexts,
        task_refs,
        reverse_task_refs,
        ranked_task_ids,
    )
    _validate_joint_order(
        application_contexts,
        task_refs,
        reverse_task_refs,
        unranked_task_ids,
    )
    provider_ids = sorted(
        {
            int(sp_id)
            for app_ctx in application_contexts.values()
            for sp_id in app_ctx.get("provider_ids", [])
        }
    )
    joint_cpu_cycles: Dict[int, float] = {}
    joint_task_type_ids: Dict[int, int | None] = {}
    joint_service_size_bits: Dict[int, int] = {}
    joint_source_program_size_bits: Dict[int, int] = {}
    joint_compile_workloads: Dict[int, float] = {}

    for joint_task_id, ref in task_refs.items():
        app_ctx = application_contexts[ref.application_id]
        task_type_id = app_ctx["task_type_ids"].get(ref.task_id)
        joint_cpu_cycles[joint_task_id] = float(app_ctx["cpu_cycles"][ref.task_id])
        joint_task_type_ids[joint_task_id] = (
            int(task_type_id) if task_type_id is not None else None
        )
        if task_type_id is not None:
            joint_service_size_bits[int(task_type_id)] = int(
                app_ctx.get("service_size_bits", {}).get(task_type_id, 0)
            )
            source_size_bits = int(
                app_ctx.get("source_program_size_bits", {}).get(task_type_id, 0)
            )
            if source_size_bits <= 0:
                raise ValueError(
                    "Application context is missing a valid source-program size "
                    f"for task type {task_type_id}"
                )
            joint_source_program_size_bits[int(task_type_id)] = source_size_bits
            joint_compile_workloads[int(task_type_id)] = float(
                app_ctx.get("compile_workloads", {}).get(task_type_id, 0.0)
            )

    return {
        "scientific_stage": "joint-stage-3a-cpu",
        "application_ids": list(application_ids),
        "applications": application_contexts,
        "task_refs": task_refs,
        "reverse_task_refs": reverse_task_refs,
        "entry_task_ids": entry_task_ids,
        "optimized_task_ids": optimized_task_ids,
        "ranked_task_ids": ranked_task_ids,
        "unranked_task_ids": unranked_task_ids,
        "unranked_task_order_model": (
            "round-robin natural-topological order across application DAGs"
        ),
        "task_domains": task_domains,
        "provider_ids": provider_ids,
        "initial_cache": _merge_initial_cache(application_contexts),
        "sp_cache_capacity": _merge_provider_map(
            application_contexts,
            "sp_cache_capacity",
        ),
        "service_size_bits": joint_service_size_bits,
        "source_program_size_bits": joint_source_program_size_bits,
        "source_program_size_model": application_contexts[application_ids[0]].get(
            "source_program_size_model"
        ),
        "source_program_size_ratio": application_contexts[application_ids[0]].get(
            "source_program_size_ratio"
        ),
        "source_program_size_reference_doi": application_contexts[
            application_ids[0]
        ].get("source_program_size_reference_doi"),
        "source_program_size_article_exact": bool(
            application_contexts[application_ids[0]].get(
                "source_program_size_article_exact", False
            )
        ),
        "compile_workloads": joint_compile_workloads,
        "cpu_cycles": joint_cpu_cycles,
        "task_type_ids": joint_task_type_ids,
        "deadline_max_s": max_deadline_s,
        "benchmark_snapshot_at": snapshot.at.isoformat(),
        "benchmark_snapshot_time_step_s": float(snapshot.time_step_s),
        "benchmark_initial_cache": "empty",
        "scenario_metadata": copy.deepcopy(scenario_metadata or {}),
        "article_alignment": {
            "application_specific_alpha_beta": True,
            "article_exact_cpu_allocation": True,
            "article_exact_channel_model": False,
            "article_exact_power_sender_model": False,
            "dtosc_provider_path_dynamic_programming": True,
            "dtosc_cache_knapsack_dynamic_programming": True,
            "dtosc_legacy_pre_repair_baseline": True,
            "dtosc_2022_policy_adapted_to_2025_model": False,
            "dtosc_source_exact_verified": False,
            "article_exact_dtosc": False,
        },
        "warnings": [
            (
                "Joint scheduling, shared SP queues, fixed local entry tasks, "
                "article alpha/beta, and Eq. (30) CPU allocation are enabled."
            ),
            (
                "The physical channel and sender-side transmission-power integration "
                "remain declared approximations. DTOSC uses the legacy pre-repair "
                "stage-wise provider-path dynamic-programming reconstruction with the "
                "common benchmark evaluator. It is retained as a reproducibility and "
                "sensitivity baseline and is not claimed to be source-exact DTOSC 2022."
            ),
        ],
    }


def build_joint_context(
    application_ids: Iterable[int],
    *,
    route_remote_mec_via_access_rsu: bool = False,
    use_live_vehicle_positions: bool = False,
) -> Dict[str, Any]:
    application_ids = _unique_ints(application_ids)
    if not application_ids:
        raise ValueError("At least one application_id is required")
    applications = list(
        Application.objects.filter(id__in=application_ids)
        .select_related("application_type_id", "vehicle_id")
    )
    by_id = {int(app.id): app for app in applications}
    missing = [app_id for app_id in application_ids if app_id not in by_id]
    if missing:
        raise ValueError(f"Applications not found: {missing}")
    vehicle_ids = [int(by_id[app_id].vehicle_id_id) for app_id in application_ids]
    if len(vehicle_ids) != len(set(vehicle_ids)):
        raise ValueError(
            "Joint paper scenario expects one active application per mission vehicle. "
            "Duplicate vehicle IDs were supplied."
        )
    start_times = {by_id[app_id].start_at for app_id in application_ids}
    if len(start_times) != 1:
        raise ValueError(
            "Joint benchmark applications must have one common start_at snapshot"
        )
    snapshot_at = next(iter(start_times))
    snapshot = _build_snapshot(
        [by_id[app_id] for app_id in application_ids],
        snapshot_at,
        use_live_vehicle_positions=use_live_vehicle_positions,
    )
    application_contexts: Dict[int, Dict[str, Any]] = {}
    for app_id in application_ids:
        app_ctx = copy.deepcopy(
            build_benchmark_context(
                app_id,
                snapshot=snapshot,
                route_remote_mec_via_access_rsu=route_remote_mec_via_access_rsu,
            )
        )
        app_ctx["application_id"] = int(app_id)
        app_ctx["application_start_at"] = snapshot_at.isoformat()
        _apply_article_cpu_model(app_ctx)
        application_contexts[app_id] = app_ctx
    return _assemble_joint_context(
        application_ids,
        application_contexts,
        snapshot,
    )


def build_synthetic_joint_context(
    assignments: Iterable[Dict[str, int]],
    snapshot: BenchmarkSnapshot,
    mec_capacity_ghz: float | None = None,
    scenario_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rows = [
        {
            "application_id": int(row["application_id"]),
            "vehicle_id": int(row["vehicle_id"]),
            "application_type_id": int(row["application_type_id"]),
            "deadline_ms": (
                None
                if row.get("deadline_ms") is None
                else float(row["deadline_ms"])
            ),
        }
        for row in assignments
    ]
    if not rows:
        raise ValueError("At least one synthetic application is required")
    application_ids = [row["application_id"] for row in rows]
    vehicle_ids = [row["vehicle_id"] for row in rows]
    if len(application_ids) != len(set(application_ids)):
        raise ValueError("Synthetic application IDs must be unique")
    if len(vehicle_ids) != len(set(vehicle_ids)):
        raise ValueError("Synthetic mission vehicle IDs must be unique")
    vehicles = Vehicle.objects.filter(id__in=vehicle_ids).in_bulk()
    app_type_ids = sorted({row["application_type_id"] for row in rows})
    application_types = ApplicationType.objects.filter(
        id__in=app_type_ids
    ).in_bulk()
    missing_vehicles = sorted(set(vehicle_ids) - set(int(key) for key in vehicles))
    missing_types = sorted(set(app_type_ids) - set(int(key) for key in application_types))
    if missing_vehicles:
        raise ValueError(f"Vehicles not found: {missing_vehicles}")
    if missing_types:
        raise ValueError(f"Application types not found: {missing_types}")
    application_contexts: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        app_id = row["application_id"]
        application_contexts[app_id] = build_synthetic_benchmark_context(
            app_id,
            vehicles[row["vehicle_id"]],
            application_types[row["application_type_id"]],
            snapshot,
            deadline_ms=row["deadline_ms"],
        )
    _apply_mec_capacity_override(application_contexts, mec_capacity_ghz)
    for app_ctx in application_contexts.values():
        _apply_article_cpu_model(app_ctx)
    return _assemble_joint_context(
        application_ids,
        application_contexts,
        snapshot,
        scenario_metadata=scenario_metadata,
    )

def joint_context_summary(joint_ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scientific_stage": joint_ctx["scientific_stage"],
        "benchmark_snapshot_at": joint_ctx.get("benchmark_snapshot_at"),
        "benchmark_snapshot_time_step_s": joint_ctx.get("benchmark_snapshot_time_step_s"),
        "benchmark_initial_cache": joint_ctx.get("benchmark_initial_cache"),
        "scenario_metadata": copy.deepcopy(
            joint_ctx.get("scenario_metadata", {})
        ),
        "application_ids": list(joint_ctx["application_ids"]),
        "application_count": len(joint_ctx["application_ids"]),
        "provider_ids": list(joint_ctx["provider_ids"]),
        "provider_count": len(joint_ctx["provider_ids"]),
        "task_count": len(joint_ctx["task_refs"]),
        "entry_task_count": len(joint_ctx["entry_task_ids"]),
        "optimized_task_count": len(joint_ctx["optimized_task_ids"]),
        "ranked_task_ids": list(joint_ctx["ranked_task_ids"]),
        "unranked_task_ids": list(joint_ctx["unranked_task_ids"]),
        "unranked_task_order_model": joint_ctx.get(
            "unranked_task_order_model"
        ),
        "entry_tasks": {
            str(app_id): joint_ctx["task_refs"][joint_task_id].to_dict()
            for app_id, joint_task_id in joint_ctx["entry_task_ids"].items()
        },
        "article_alignment": copy.deepcopy(
            joint_ctx.get("article_alignment", {})
        ),
        "warnings": list(joint_ctx.get("warnings", [])),
    }
