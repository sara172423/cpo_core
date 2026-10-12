from __future__ import annotations

import random
import threading
import traceback
from datetime import timedelta
from typing import Optional, Tuple, List

from django.db import close_old_connections, transaction
from django.db.models import Count
from django.utils import timezone
from django.core.cache import cache

from monarch_pylib.model import transmission

from parameter.services import load_params_obj
from system.rsu_access import select_access_rsu_by_csi
from object.models import RSU, RSUVehicle, Vehicle, ServiceProvider
from dag.models import ApplicationType
from application.models import Application


class SimulationConfig:
    def __init__(self, total_time: int, tick_seconds: int, cell_radius_rsu: float, base_time):
        self.total_time = int(total_time)
        self.tick_seconds = int(tick_seconds)
        self.cell_radius_rsu = float(cell_radius_rsu)
        self.base_time = base_time
        self.clock_tick_seconds = 1
        self.motion_tick_seconds = 1


def _sim_datetime(cfg: SimulationConfig, sim_seconds: float):
    base = cfg.base_time or timezone.now()
    return base + timedelta(seconds=max(0.0, float(sim_seconds)))


def _pick_application_type_id(
    vehicle_id: int,
    min_tasks: int = 3,
    rng: random.Random | None = None,
) -> int | None:
    ids = list(
        ApplicationType.objects
        .annotate(task_count=Count("task"))
        .filter(task_count__gte=int(min_tasks))
        .order_by("id")
        .values_list("id", flat=True)
    )
    if not ids:
        return None
    chooser = rng if rng is not None else random
    return int(chooser.choice(ids))


def _to_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _best_channel_rsu(
    rsus,
    pos_xyz: Tuple[float, float, float],
    *,
    vehicle_id: int,
    base_seed: int,
) -> Optional[RSU]:
    """Return the covering RSU with the best instantaneous V2I CSI.

    The previous runtime worker used the nearest RSU.  The paper states that
    access is selected from channel-state information, so runtime association
    now uses the same CSI-based baseline correction as the benchmark builder.
    """
    params = load_params_obj()
    rsu, _gain, _distance_m = select_access_rsu_by_csi(
        rsus=rsus,
        position_xyz=(
            float(pos_xyz[0]),
            float(pos_xyz[1]),
            float(pos_xyz[2]),
        ),
        params=params,
        base_seed=int(base_seed),
        vehicle_id=int(vehicle_id),
    )
    return rsu


def _close_rsu_vehicle_connections(queryset, end_time) -> int:
    """Close active RSU links through model.save() so connect_time is persisted."""
    closed = 0

    for link in queryset.select_for_update().order_by("id"):
        link.end_time = end_time
        link.is_current = False
        link.save(
            update_fields=[
                "end_time",
                "connect_time",
                "is_current",
            ]
        )
        closed += 1

    return closed


def _sync_rsu_vehicle(vehicle_id: int, new_rsu_id: Optional[int], now_dt):
    open_qs = RSUVehicle.objects.filter(
        vehicle_id_id=vehicle_id,
        end_time__isnull=True
    )

    if new_rsu_id is not None:
        _close_rsu_vehicle_connections(
            open_qs.exclude(rsu_id_id=new_rsu_id),
            now_dt,
        )
    else:
        _close_rsu_vehicle_connections(open_qs, now_dt)

    if new_rsu_id is None:
        return

    obj, created = RSUVehicle.objects.get_or_create(
        vehicle_id_id=vehicle_id,
        rsu_id_id=new_rsu_id,
        end_time__isnull=True,
        defaults={
            "start_time": now_dt,
            "is_current": True
        },
    )

    if not created and not obj.is_current:
        obj.is_current = True
        obj.save(update_fields=["is_current"])


def discover_available_sps(vehicle: Vehicle) -> List[int]:
    available = []

    rv = (
        RSUVehicle.objects
        .filter(
            vehicle_id=vehicle.id,
            is_current=True,
        )
        .select_related("rsu_id")
        .first()
    )

    if not rv:
        return available

    current_rsu = rv.rsu_id

    # M: تمام MEC Serverها مطابق X_n = M ∪ V_n
    rsu_provider_ids = (
        ServiceProvider.objects
        .filter(
            type="rsu",
            rsu_id__isnull=False,
        )
        .values_list("id", flat=True)
    )

    available.extend(
        int(sp_id)
        for sp_id in rsu_provider_ids
    )

    # V_n: خودروهای متصل به همان RSU، به‌جز خودروی مأموریت
    peer_vehicle_ids = (
        RSUVehicle.objects
        .filter(
            rsu_id=current_rsu,
            is_current=True,
        )
        .exclude(vehicle_id=vehicle.id)
        .values_list("vehicle_id_id", flat=True)
    )

    peer_provider_ids = (
        ServiceProvider.objects
        .filter(
            vehicle_id_id__in=peer_vehicle_ids,
            type="vehicle",
        )
        .values_list("id", flat=True)
    )

    available.extend(
        int(sp_id)
        for sp_id in peer_provider_ids
    )

    return list(dict.fromkeys(available))

class VehicleWorker(threading.Thread):
    def __init__(self, vehicle_id: int, cfg: SimulationConfig):
        super().__init__(daemon=True)
        self.vehicle_id = int(vehicle_id)
        self.cfg = cfg
        self._stop_flag = threading.Event()
        self.params = load_params_obj()
        self.h_vehicle = float(getattr(self.params, "h_vehicle", 0.0) or 0.0)
        self.h_rsu = float(getattr(self.params, "h_rsu", 0.0) or 0.0)

    def stop(self):
        self._stop_flag.set()

    def _speed_kmh(self, vehicle: Vehicle) -> float:
        """Use the route-derived speed stored on the Vehicle model."""
        speed_kmh = float(vehicle.speed or 0.0)
        if speed_kmh <= 0.0:
            raise ValueError(
                f"Vehicle {self.vehicle_id} has no valid route-derived speed"
            )
        return speed_kmh

    def run(self):
        """Run the motion loop with bounded recovery from transient DB errors."""
        retry_limit = max(1, int(getattr(self.cfg, "vehicle_worker_retry_limit", 3) or 3))
        failures = 0
        while not self._stop_flag.is_set():
            try:
                self._run_loop()
                return
            except Exception:
                failures += 1
                self.cfg.runtime_last_error = traceback.format_exc()
                traceback.print_exc()
                close_old_connections()
                if failures >= retry_limit:
                    self.cfg.runtime_failed_vehicle_id = int(self.vehicle_id)
                    return
                if self._stop_flag.wait(min(2.0, 0.25 * (2 ** (failures - 1)))):
                    return

    def _run_loop(self):
        close_old_connections()
        rsus = list(RSU.objects.all())
        total_time = float(self.cfg.total_time)
        motion_tick = max(
            0.1,
            float(
                getattr(
                    self.cfg,
                    "motion_tick_seconds",
                    1.0,
                )
                or 1.0
            ),
        )
        get_sim_time = getattr(
            self.cfg,
            "get_sim_time_s",
            None,
        )
        last_processed_time = None

        while not self._stop_flag.is_set():
            close_old_connections()

            if callable(get_sim_time):
                t = min(
                    total_time,
                    max(
                        0.0,
                        float(get_sim_time()),
                    ),
                )
            else:
                t = (
                    0.0
                    if last_processed_time is None
                    else min(
                        total_time,
                        float(last_processed_time) + motion_tick,
                    )
                )

            if (
                last_processed_time is not None
                and t <= float(last_processed_time)
            ):
                if self._stop_flag.wait(0.1):
                    break
                continue

            vehicle = Vehicle.objects.filter(
                id=self.vehicle_id
            ).first()

            if vehicle is None:
                break

            nearest_id = None
            available_sps = []
            route_active = False

            if vehicle.path and len(vehicle.path) >= 2:
                path_length = float(
                    transmission.path_length_2d(
                        vehicle.path
                    )
                )
                if path_length > 0.0:
                    speed_mps = self._speed_kmh(vehicle) / 3.6
                    travel_time = path_length / speed_mps
                else:
                    travel_time = 0.0

                route_active = (
                    travel_time > 0.0
                    and t <= travel_time
                )

                position_time = (
                    min(t, travel_time)
                    if travel_time > 0.0
                    else 0.0
                )

                x, y = transmission.current_position(
                    max(
                        travel_time,
                        motion_tick,
                    ),
                    position_time,
                    vehicle.path,
                )

                Vehicle.objects.filter(
                    id=self.vehicle_id
                ).update(
                    x_coord=x,
                    y_coord=y,
                )

                vehicle.x_coord = x
                vehicle.y_coord = y

                if route_active:
                    access_rsu = _best_channel_rsu(
                        rsus,
                        (
                            float(x),
                            float(y),
                            float(self.h_vehicle),
                        ),
                        vehicle_id=int(self.vehicle_id),
                        base_seed=int(self.cfg.simulation_seed),
                    )

                    nearest_id = (
                        access_rsu.id
                        if access_rsu
                        else None
                    )

            now_dt = _sim_datetime(
                self.cfg,
                t,
            )

            with transaction.atomic():
                _sync_rsu_vehicle(
                    self.vehicle_id,
                    nearest_id,
                    now_dt,
                )

            if route_active and nearest_id is not None:
                available_sps = discover_available_sps(
                    vehicle
                )

            cache.set(
                f"vehicle_available_sps_{self.vehicle_id}",
                available_sps,
                timeout=300,
            )

            last_processed_time = t

            if self._stop_flag.wait(0.1):
                break

        close_old_connections()

        final_t = (
            min(
                total_time,
                max(
                    0.0,
                    float(get_sim_time()),
                ),
            )
            if callable(get_sim_time)
            else min(
                total_time,
                float(last_processed_time or 0.0),
            )
        )

        end_dt = _sim_datetime(
            self.cfg,
            final_t,
        )

        with transaction.atomic():
            _close_rsu_vehicle_connections(
                RSUVehicle.objects.filter(
                    vehicle_id_id=self.vehicle_id,
                    end_time__isnull=True,
                ),
                end_dt,
            )


class ApplicationGeneratorWorker(threading.Thread):
    """Release one deterministic mission batch at fixed simulation intervals.

    The runtime simulation is intentionally separate from the paper benchmark:
    one fixed, seeded set of mission vehicles receives a new application at
    t=0, interval, 2*interval, ... while the remaining connected vehicles stay
    available as V2V helpers.
    """

    def __init__(self, cfg: SimulationConfig):
        super().__init__(daemon=True)
        self.cfg = cfg
        self._stop_flag = threading.Event()
        params = load_params_obj()
        self.min_tasks = int(getattr(params, "min_tasks_per_app_type", 3) or 3)
        self.seed = int(getattr(cfg, "simulation_seed", 1))
        self.interval_s = max(
            1,
            int(getattr(cfg, "runtime_batch_interval_s", 30) or 30),
        )
        self.batch_size = max(
            1,
            int(getattr(cfg, "runtime_batch_size", 20) or 20),
        )
        self._mission_vehicle_ids: List[int] = []
        self._released_batches = 0
        self._created_applications = 0
        self._skipped_applications = 0
        self._publish_runtime_state()

    def stop(self):
        self._stop_flag.set()

    def _publish_runtime_state(self) -> None:
        self.cfg.runtime_mission_vehicle_ids = list(self._mission_vehicle_ids)
        self.cfg.runtime_batches_released = int(self._released_batches)
        self.cfg.runtime_created_applications = int(self._created_applications)
        self.cfg.runtime_skipped_applications = int(self._skipped_applications)

        # Keep the database flag synchronized with the runtime mission set.
        # The scheduler itself uses runtime_mission_vehicle_ids, but storing the
        # same information in Vehicle makes database inspection/debugging clear.
        try:
            Vehicle.objects.update(is_mission=False)
            if self._mission_vehicle_ids:
                Vehicle.objects.filter(
                    id__in=[int(v) for v in self._mission_vehicle_ids]
                ).update(is_mission=True)
        except Exception:
            # Database synchronization must not stop the simulation workers.
            # Runtime mission selection remains the source of truth.
            pass

    def _connected_vehicle_ids(self) -> List[int]:
        return [
            int(vehicle_id)
            for vehicle_id in (
                RSUVehicle.objects.filter(
                    is_current=True,
                    end_time__isnull=True,
                    vehicle_id_id__isnull=False,
                )
                .order_by("vehicle_id_id")
                .values_list("vehicle_id_id", flat=True)
                .distinct()
            )
        ]

    def _select_mission_vehicle_ids(self, sim_second: int) -> List[int]:
        connected_ids = self._connected_vehicle_ids()
        connected_set = set(connected_ids)

        # Keep the original mission set whenever those vehicles remain connected.
        retained = [
            vehicle_id
            for vehicle_id in self._mission_vehicle_ids
            if vehicle_id in connected_set
        ]

        needed = min(self.batch_size, len(connected_ids)) - len(retained)
        if needed > 0:
            replacement_pool = [
                vehicle_id
                for vehicle_id in connected_ids
                if vehicle_id not in set(retained)
            ]
            rng = random.Random(
                self.seed * 1_000_003
                + int(sim_second) * 97_409
                + self._released_batches * 65_537
            )
            rng.shuffle(replacement_pool)
            retained.extend(replacement_pool[:needed])

        self._mission_vehicle_ids = retained[: self.batch_size]
        self._publish_runtime_state()
        return list(self._mission_vehicle_ids)

    def _create_batch(self, sim_second: int) -> int:
        selected_ids = self._select_mission_vehicle_ids(sim_second)
        requested = int(self.batch_size)

        if not selected_ids:
            self._released_batches += 1
            self._skipped_applications += requested
            self._publish_runtime_state()
            return 0

        start_at = _sim_datetime(self.cfg, float(sim_second))
        created = 0
        skipped = max(0, requested - len(selected_ids))
        rng = random.Random(
            self.seed * 1_000_003
            + int(sim_second) * 97_409
            + 17
        )

        with transaction.atomic():
            locked_ids = list(
                Vehicle.objects.select_for_update()
                .filter(id__in=selected_ids)
                .order_by("id")
                .values_list("id", flat=True)
            )

            for vehicle_id in locked_ids:
                # Normally the previous batch is already complete because the
                # logical clock pauses while DCSGA is running.  This guard keeps
                # a failed/stuck application from being duplicated silently.
                if Application.objects.filter(
                    vehicle_id_id=int(vehicle_id),
                    is_progress=True,
                ).exists():
                    skipped += 1
                    continue

                app_type_id = _pick_application_type_id(
                    int(vehicle_id),
                    min_tasks=self.min_tasks,
                    rng=rng,
                )
                if app_type_id is None:
                    skipped += 1
                    continue

                Application.objects.create(
                    vehicle_id_id=int(vehicle_id),
                    application_type_id_id=int(app_type_id),
                    start_at=start_at,
                    end_at=None,
                    is_progress=True,
                )
                created += 1

        self._released_batches += 1
        self._created_applications += created
        self._skipped_applications += skipped
        self.cfg.runtime_last_batch_time_s = int(sim_second)
        self.cfg.runtime_last_batch_requested = requested
        self.cfg.runtime_last_batch_created = created
        self.cfg.runtime_last_batch_skipped = skipped
        self._publish_runtime_state()
        return created

    def run(self):
        close_old_connections()
        total_time = int(self.cfg.total_time)
        next_batch_time = 0
        get_sim_time = getattr(self.cfg, "get_sim_time_s", None)
        scheduler_busy = getattr(self.cfg, "runtime_scheduler_busy", None)
        retry_limit = max(
            1,
            int(getattr(self.cfg, "runtime_batch_retry_limit", 5) or 5),
        )
        consecutive_failures = 0

        while not self._stop_flag.is_set():
            close_old_connections()

            if callable(get_sim_time):
                current_second = min(
                    total_time,
                    max(0, int(float(get_sim_time()))),
                )
            else:
                current_second = min(total_time, next_batch_time)

            while (
                next_batch_time < total_time
                and current_second >= next_batch_time
                and not self._stop_flag.is_set()
            ):
                # Freeze logical time at the release instant.  Vehicle workers
                # still run and can finish the topology snapshot for this tick.
                if scheduler_busy is not None:
                    scheduler_busy.set()

                if self._stop_flag.wait(0.2):
                    break

                try:
                    created = self._create_batch(next_batch_time)
                except Exception:
                    consecutive_failures += 1
                    self.cfg.runtime_last_error = traceback.format_exc()
                    self.cfg.runtime_batch_consecutive_failures = int(
                        consecutive_failures
                    )
                    traceback.print_exc()
                    if scheduler_busy is not None:
                        scheduler_busy.clear()
                    if consecutive_failures >= retry_limit:
                        self.cfg.runtime_generator_failed = True
                        close_old_connections()
                        return
                    if self._stop_flag.wait(0.5):
                        break
                    continue

                consecutive_failures = 0
                self.cfg.runtime_batch_consecutive_failures = 0

                if created <= 0 and scheduler_busy is not None:
                    scheduler_busy.clear()

                next_batch_time += self.interval_s

            if current_second >= total_time and next_batch_time >= total_time:
                break

            if self._stop_flag.wait(0.1):
                break

        close_old_connections()
