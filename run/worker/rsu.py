from __future__ import annotations

import threading
import time
import traceback
from datetime import timedelta
from typing import Dict, Any, List, Iterable

from django.db import close_old_connections, transaction, connection
from django.db.models import Max
from django.db.utils import OperationalError
from django.utils import timezone

from object.models import RSUVehicle, ServiceProvider
from execution.models import TaskExecution
from application.models import Application
from cache.models import cache as CacheModel
from resource.models import Resource
from state.models import State
from dag.models import TaskType
from parameter.services import load_params_obj

from run.benchmark.context import build_joint_context
from run.benchmark.search import run_joint_dcsga

from system.build_context import MiniSystemContextBuilder

from algorithm.optimizer_common import (
    AlgorithmCancelled,
    compute_task_ranks_and_order,
    evaluate_solution_quality,
)
from algorithm.greedy_nests import rate
from algorithm.low_complexity import channel_gain

_APP_LOCK_NAMESPACE = 742031
_RUNTIME_SCHEDULER_LOCK_NAMESPACE = 742032
_RUNTIME_SCHEDULER_LOCK_KEY = 1
_MAX_TRANSACTION_RETRIES = 3


def _try_lock_application(application_id: int) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_try_advisory_lock(%s, %s)",
            [_APP_LOCK_NAMESPACE, int(application_id)],
        )
        return bool(cursor.fetchone()[0])


def _unlock_application(application_id: int) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_unlock(%s, %s)",
            [_APP_LOCK_NAMESPACE, int(application_id)],
        )


def _try_lock_runtime_scheduler() -> bool:
    """Try to serialize runtime scheduling without blocking worker shutdown."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_try_advisory_lock(%s, %s)",
            [
                _RUNTIME_SCHEDULER_LOCK_NAMESPACE,
                _RUNTIME_SCHEDULER_LOCK_KEY,
            ],
        )
        return bool(cursor.fetchone()[0])


def _unlock_runtime_scheduler() -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_unlock(%s, %s)",
            [
                _RUNTIME_SCHEDULER_LOCK_NAMESPACE,
                _RUNTIME_SCHEDULER_LOCK_KEY,
            ],
        )


def _is_deadlock(error: BaseException) -> bool:
    current = error
    visited = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))

        if getattr(current, "sqlstate", None) == "40P01":
            return True

        if getattr(current, "pgcode", None) == "40P01":
            return True

        current = current.__cause__ or current.__context__

    return False


class RSUWorker(threading.Thread):

    def __init__(self, rsu_id, cfg):
        super().__init__(daemon=True)
        self.rsu_id = int(rsu_id)
        self.cfg = cfg
        self._stop_flag = threading.Event()

    def stop(self):
        self._stop_flag.set()

    def run(self):
        close_old_connections()

        coordinator_id = int(
            getattr(self.cfg, "runtime_coordinator_rsu_id", self.rsu_id)
            or self.rsu_id
        )

        # Keep the existing RSU worker registry intact, but let exactly one
        # worker coordinate each system-wide mission batch.  Vehicle workers
        # continue to maintain all RSU associations and V2V topology.
        if self.rsu_id != coordinator_id:
            while not self._stop_flag.is_set():
                if self._stop_flag.wait(0.2):
                    break
            close_old_connections()
            return

        while not self._stop_flag.is_set():
            close_old_connections()
            processed = self._process_next_batch()
            if not processed and self._stop_flag.wait(0.1):
                break

        close_old_connections()

    def _next_pending_batch(self) -> List[Application]:
        first = (
            Application.objects.filter(
                is_progress=True,
                start_at__isnull=False,
            )
            .order_by("start_at", "id")
            .first()
        )
        if first is None:
            return []

        return list(
            Application.objects.filter(
                is_progress=True,
                start_at=first.start_at,
            )
            .select_related("vehicle_id", "application_type_id")
            .order_by("id")
        )

    def _runtime_cache_map(self, provider_ids: Iterable[int]) -> Dict[int, set[int]]:
        normalized_ids = sorted({int(provider_id) for provider_id in provider_ids})
        result = {provider_id: set() for provider_id in normalized_ids}
        for provider_id, task_type_id in CacheModel.objects.filter(
            sp_id_id__in=normalized_ids
        ).values_list("sp_id_id", "task_type_id_id"):
            result[int(provider_id)].add(int(task_type_id))
        return result

    def _prepare_joint_runtime_context(
        self,
        applications: List[Application],
    ) -> Dict[str, Any]:
        application_ids = [int(app.id) for app in applications]
        joint_ctx = build_joint_context(
            application_ids,
            route_remote_mec_via_access_rsu=True,
            use_live_vehicle_positions=True,
        )
        mission_vehicle_ids = {
            int(app.vehicle_id_id)
            for app in applications
        }

        # Mission vehicles execute their own local entry task but are not used
        # as V2V helpers for the other mission applications in the same batch.
        # This leaves the non-mission vehicles available for cooperation.
        allowed_by_application: Dict[int, List[int]] = {}
        for app_id, app_ctx in joint_ctx["applications"].items():
            local_sp_id = int(app_ctx["local_sp_id"])
            allowed: List[int] = []

            for provider_id in app_ctx.get("provider_ids", []):
                provider_id = int(provider_id)
                provider_type = app_ctx.get("sp_types", {}).get(provider_id)
                provider_vehicle_id = app_ctx.get("sp_vehicle_ids", {}).get(provider_id)

                if (
                    provider_type == "vehicle"
                    and provider_vehicle_id is not None
                    and int(provider_vehicle_id) in mission_vehicle_ids
                    and provider_id != local_sp_id
                ):
                    continue

                allowed.append(provider_id)

            if local_sp_id not in allowed:
                allowed.append(local_sp_id)

            allowed_by_application[int(app_id)] = list(dict.fromkeys(allowed))
            app_ctx["provider_ids"] = list(allowed_by_application[int(app_id)])

        for joint_task_id, task_ref in joint_ctx["task_refs"].items():
            joint_ctx["task_domains"][int(joint_task_id)] = list(
                allowed_by_application[int(task_ref.application_id)]
            )

        provider_ids = sorted({
            int(provider_id)
            for provider_list in allowed_by_application.values()
            for provider_id in provider_list
        })
        joint_ctx["provider_ids"] = provider_ids
        joint_ctx["initial_cache"] = self._runtime_cache_map(provider_ids)

        batch_start = applications[0].start_at
        provider_initial_finish = {
            provider_id: 0.0
            for provider_id in provider_ids
        }

        if batch_start is not None and provider_ids:
            latest_rows = (
                TaskExecution.objects.filter(
                    sp_id_id__in=provider_ids,
                    end_time__isnull=False,
                    end_time__gt=batch_start,
                )
                .exclude(application_id_id__in=application_ids)
                .values("sp_id_id")
                .annotate(latest_end=Max("end_time"))
            )
            for row in latest_rows:
                latest_end = row["latest_end"]
                if latest_end is None:
                    continue
                provider_initial_finish[int(row["sp_id_id"])] = max(
                    0.0,
                    float((latest_end - batch_start).total_seconds()),
                )

        joint_ctx["initial_provider_finish"] = provider_initial_finish
        joint_ctx["scenario_metadata"] = {
            **dict(joint_ctx.get("scenario_metadata", {})),
            "runtime_mode": "fixed_seeded_mission_batch",
            "runtime_batch_size": len(applications),
            "runtime_mission_vehicle_ids": sorted(mission_vehicle_ids),
            "runtime_v2v_helper_vehicle_count": max(
                0,
                int(ServiceProvider.objects.filter(type="vehicle").count())
                - len(mission_vehicle_ids),
            ),
            "runtime_position_source": "live_vehicle_coordinates",
            "runtime_remote_mec_routing": "via_access_rsu_zero_backhaul_delay",
        }
        return joint_ctx

    def _process_next_batch(self) -> bool:
        applications = self._next_pending_batch()
        if not applications:
            return False

        scheduler_busy = getattr(self.cfg, "runtime_scheduler_busy", None)
        scheduler_locked = False
        completed = False

        try:
            scheduler_locked = _try_lock_runtime_scheduler()
            if not scheduler_locked:
                return False
            # Only the process that owns the advisory lock may freeze the
            # logical scheduler clock. A losing worker must not leave the
            # shared busy flag set forever.
            if scheduler_busy is not None:
                scheduler_busy.set()

            # Re-read after acquiring the process-wide scheduler lock.
            applications = self._next_pending_batch()
            if not applications:
                completed = True
                return False

            start_at = applications[0].start_at
            base_time = getattr(self.cfg, "base_time", None) or start_at or timezone.now()
            time_step = max(
                0,
                int(round((start_at - base_time).total_seconds()))
                if start_at is not None
                else 0,
            )

            joint_ctx = self._prepare_joint_runtime_context(applications)
            params = load_params_obj()
            seed = int(getattr(self.cfg, "simulation_seed", 1)) + int(time_step)

            _best_nest, evaluation, history = run_joint_dcsga(
                joint_ctx,
                algorithm="dcsga",
                seed=seed,
                tmax=int(getattr(self.cfg, "tmax", 10)),
                population_size=int(params.S),
            )

            self._persist_joint_batch(
                applications=applications,
                joint_ctx=joint_ctx,
                evaluation=evaluation,
                time_step=time_step,
            )

            self.cfg.runtime_batches_completed = int(
                getattr(self.cfg, "runtime_batches_completed", 0) or 0
            ) + 1
            self.cfg.runtime_last_error = None
            self.cfg.runtime_last_batch_metrics = {
                **dict(evaluation.metrics),
                "application_count": len(applications),
                "history_iterations": len(history),
                "time_step": int(time_step),
            }
            completed = True
            return True

        except AlgorithmCancelled:
            return False

        except Exception:
            self.cfg.runtime_last_error = traceback.format_exc()
            traceback.print_exc()
            return False

        finally:
            if scheduler_locked:
                _unlock_runtime_scheduler()

            # On failure, keep the logical clock frozen so the same batch is
            # retried instead of silently advancing and losing workload.
            if completed and scheduler_busy is not None:
                scheduler_busy.clear()

    def _persist_joint_batch(
        self,
        *,
        applications: List[Application],
        joint_ctx: Dict[str, Any],
        evaluation,
        time_step: int,
    ) -> None:
        application_ids = [int(app.id) for app in applications]
        schedule_rows = sorted(
            list(evaluation.schedule),
            key=lambda row: (
                int(row["application_id"]),
                float(row["start_s"]),
                int(row["task_id"]),
            ),
        )

        used_provider_ids = {
            int(row["provider_id"])
            for row in schedule_rows
        }

        with transaction.atomic():
            locked_apps = list(
                Application.objects.select_for_update(of=("self",))
                .filter(id__in=application_ids, is_progress=True)
                .select_related("vehicle_id")
                .order_by("id")
            )
            if {int(app.id) for app in locked_apps} != set(application_ids):
                raise ValueError(
                    "Runtime batch changed while it was being scheduled"
                )

            app_by_id = {int(app.id): app for app in locked_apps}
            final_cache = {
                int(provider_id): {
                    int(task_type_id)
                    for task_type_id in task_type_ids
                }
                for provider_id, task_type_ids in evaluation.cache_state.items()
            }
            current_cache = self._runtime_cache_map(final_cache.keys())
            changed_cache_provider_ids = {
                provider_id
                for provider_id, task_type_ids in final_cache.items()
                if task_type_ids != current_cache.get(provider_id, set())
            }

            lock_provider_ids = sorted(
                used_provider_ids | changed_cache_provider_ids
            )
            resources = self._lock_resources(lock_provider_ids)
            providers = ServiceProvider.objects.select_related(
                "vehicle_id",
                "rsu_id",
            ).in_bulk(lock_provider_ids)

            missing_providers = sorted(set(lock_provider_ids) - set(providers))
            if missing_providers:
                raise ValueError(
                    f"Missing service providers: {missing_providers}"
                )

            TaskExecution.objects.filter(
                application_id_id__in=application_ids
            ).delete()

            cpu_increments = {
                provider_id: 0
                for provider_id in used_provider_ids
            }
            max_finish_by_application = {
                app_id: 0.0
                for app_id in application_ids
            }

            for row in schedule_rows:
                app_id = int(row["application_id"])
                task_id = int(row["task_id"])
                provider_id = int(row["provider_id"])
                start_s = float(row["start_s"])
                finish_s = float(row["finish_s"])
                energy_j = float(row.get("energy_j", 0.0) or 0.0)

                if start_s < 0.0 or finish_s < start_s:
                    raise ValueError(
                        f"Invalid joint schedule for task {task_id}: "
                        f"start={start_s}, finish={finish_s}"
                    )

                app = app_by_id[app_id]
                app_ctx = joint_ctx["applications"][app_id]
                provider = providers[provider_id]
                local_sp_id = int(app_ctx["local_sp_id"])
                is_local = provider_id == local_sp_id
                sim_start = app.start_at or (
                    getattr(self.cfg, "base_time", None) or timezone.now()
                )

                gain_value = 0.0
                rate_value = 0.0
                distance_value = 0.0
                if not is_local:
                    gain_value = float(channel_gain(app_ctx, provider_id))
                    rate_value = float(rate(app_ctx, provider_id))
                    distance_value = float(
                        app_ctx.get("distance", {}).get(provider_id, 0.0)
                    )

                task_execution = TaskExecution.objects.create(
                    application_id=app,
                    sp_id=provider,
                    task_id_id=task_id,
                    start_time=sim_start + timedelta(seconds=start_s),
                    end_time=sim_start + timedelta(seconds=finish_s),
                    exec_time=finish_s - start_s,
                    energy=energy_j,
                )

                # State describes the physical wireless hop.  When a task is
                # computed on a remote MEC, TaskExecution.sp_id remains that
                # remote compute provider, while the radio hop terminates at
                # the mission vehicle's currently accessed RSU.
                access_rsu_id = app_ctx.get("access_rsu_id")
                state_rsu_id = None
                if provider.rsu_id_id:
                    state_rsu_id = int(
                        access_rsu_id
                        if access_rsu_id is not None
                        else provider.rsu_id_id
                    )

                State.objects.create(
                    time_step=int(time_step),
                    task_execution_id=task_execution,
                    from_vehicle_id=app.vehicle_id,
                    to_vehicle_id=(
                        provider.vehicle_id
                        if provider.vehicle_id_id
                        else None
                    ),
                    to_rsu_id_id=state_rsu_id,
                    gain=gain_value,
                    distance=distance_value,
                    rate=rate_value,
                    initial_snapshot={
                        "compute_provider_id": int(provider_id),
                        "compute_rsu_id": (
                            int(provider.rsu_id_id)
                            if provider.rsu_id_id
                            else None
                        ),
                        "wireless_access_rsu_id": state_rsu_id,
                        "remote_mec_via_access_rsu": bool(
                            provider.rsu_id_id
                            and state_rsu_id is not None
                            and int(provider.rsu_id_id) != int(state_rsu_id)
                        ),
                    },
                )

                cpu_increments[provider_id] += int(
                    app_ctx["cpu_cycles"].get(task_id, 0)
                )
                max_finish_by_application[app_id] = max(
                    max_finish_by_application[app_id],
                    finish_s,
                )

            for provider_id, increment in cpu_increments.items():
                resource = resources[provider_id]
                resource.cpu_used = int(resource.cpu_used or 0) + int(increment)
                resource.save(update_fields=["cpu_used"])

            self.apply_cache(
                final_cache,
                changed_cache_provider_ids,
                resources,
            )

            for app_id, app in app_by_id.items():
                self._finalize_application(
                    app,
                    int(time_step),
                    max_finish_by_application[app_id],
                )

    def _handle_vehicle(self, vehicle_id: int, t: int):
        try:
            app_ids = list(
                Application.objects.filter(
                    vehicle_id_id=vehicle_id,
                    is_progress=True,
                ).values_list("id", flat=True)
            )

            for app_id in app_ids:
                if self._stop_flag.is_set():
                    break

                if not _try_lock_application(app_id):
                    continue

                scheduler_locked = False

                try:
                    scheduler_locked = _try_lock_runtime_scheduler()

                    if not scheduler_locked:
                        continue

                    if self._stop_flag.is_set():
                        continue

                    app = Application.objects.filter(
                        id=app_id,
                        is_progress=True,
                    ).first()

                    if app is None:
                        continue

                    snapshot_time_step = self._current_sim_time_step(t)

                    ctx = MiniSystemContextBuilder(
                        application_id=app.id
                    ).build_context()
                    ctx["tmax"] = int(
                        getattr(self.cfg, "tmax", 10)
                    )
                    ctx["seed"] = int(
                        getattr(self.cfg, "simulation_seed", 1)
                    )
                    ctx["cancel_event"] = self._stop_flag
                    self._apply_runtime_queue_state(
                        ctx,
                        app,
                        snapshot_time_step,
                    )

                    from algorithm.cuckoo.core import dcsga_run

                    best_solution, quality, cache_state = dcsga_run(ctx)
                    task_order = compute_task_ranks_and_order(ctx)

                    # ``ctx`` contains ``cancel_event`` (a threading.Event).
                    # The final evaluator deep-copies its input, while Python
                    # thread locks are intentionally not deepcopy/pickle safe.
                    # Use a plain evaluation snapshot without the runtime-only
                    # cancellation handle.  Cancellation is checked immediately
                    # before and after this short deterministic materialization,
                    # so no result is persisted after a stop request.
                    if self._stop_flag.is_set():
                        raise AlgorithmCancelled(
                            "Algorithm execution was cancelled"
                        )

                    evaluation_ctx = dict(ctx)
                    evaluation_ctx.pop("cancel_event", None)

                    _, evaluated_cache, scheduled_ctx = evaluate_solution_quality(
                        evaluation_ctx,
                        best_solution,
                        task_order,
                    )

                    if self._stop_flag.is_set():
                        raise AlgorithmCancelled(
                            "Algorithm execution was cancelled"
                        )

                    final_cache_state = (
                        evaluated_cache
                        if evaluated_cache is not None
                        else cache_state
                    )

                    entry_task_id = int(scheduled_ctx["entry_task_id"])
                    assigned_provider_ids = sorted({
                        int(provider_id)
                        for _, provider_id, _ in best_solution
                    })
                    cache_provider_ids = sorted({
                        int(provider_id)
                        for task_id, provider_id, _ in best_solution
                        if int(task_id) != entry_task_id
                    })
                    lock_provider_ids = sorted(
                        set(assigned_provider_ids) | set(cache_provider_ids)
                    )

                    self._persist_application(
                        app_id=app_id,
                        best_solution=best_solution,
                        scheduled_ctx=scheduled_ctx,
                        final_cache_state=final_cache_state,
                        cache_provider_ids=cache_provider_ids,
                        lock_provider_ids=lock_provider_ids,
                        time_step=snapshot_time_step,
                    )

                except AlgorithmCancelled:
                    pass

                except Exception:
                    traceback.print_exc()

                finally:
                    try:
                        if scheduler_locked:
                            _unlock_runtime_scheduler()
                    finally:
                        _unlock_application(app_id)

        except Exception:
            traceback.print_exc()

    def _current_sim_time_step(self, fallback_time_step: int) -> int:
        get_sim_time = getattr(
            self.cfg,
            "get_sim_time_s",
            None,
        )
        total_time = max(
            0.0,
            float(getattr(self.cfg, "total_time", 120)),
        )

        if callable(get_sim_time):
            value = float(get_sim_time())
        else:
            value = float(fallback_time_step)

        return int(
            min(
                total_time,
                max(0.0, value),
            )
        )

    def _apply_runtime_queue_state(
        self,
        ctx: Dict[str, Any],
        app: Application,
        time_step: int,
    ) -> None:
        """Seed DCSGA with current simulation waiting and provider availability."""
        base_time = getattr(self.cfg, "base_time", None) or timezone.now()
        snapshot_time = base_time + timedelta(seconds=int(time_step))
        reference_time = app.start_at or snapshot_time
        scheduling_time = max(reference_time, snapshot_time)
        elapsed_wait = max(
            0.0,
            float((scheduling_time - reference_time).total_seconds()),
        )

        provider_ids = sorted({
            int(provider_id)
            for provider_id in ctx.get("provider_ids", [])
        })
        provider_initial_finish = {
            provider_id: elapsed_wait
            for provider_id in provider_ids
        }

        if provider_ids:
            latest_rows = (
                TaskExecution.objects
                .filter(
                    sp_id_id__in=provider_ids,
                    end_time__isnull=False,
                    end_time__gt=scheduling_time,
                )
                .exclude(application_id_id=app.id)
                .values("sp_id_id")
                .annotate(latest_end=Max("end_time"))
            )

            for row in latest_rows:
                provider_id = int(row["sp_id_id"])
                latest_end = row["latest_end"]

                if latest_end is None:
                    continue

                provider_initial_finish[provider_id] = max(
                    elapsed_wait,
                    float((latest_end - reference_time).total_seconds()),
                )

        ctx["runtime_snapshot_time_step"] = int(time_step)
        ctx["runtime_waiting_time_s"] = float(elapsed_wait)
        ctx["provider_initial_finish"] = provider_initial_finish

    def _persist_application(
        self,
        app_id: int,
        best_solution: List,
        scheduled_ctx: Dict,
        final_cache_state: Dict[int, Any],
        cache_provider_ids: List[int],
        lock_provider_ids: List[int],
        time_step: int,
    ) -> None:
        for attempt in range(_MAX_TRANSACTION_RETRIES):
            try:
                with transaction.atomic():
                    app = Application.objects.select_for_update().filter(
                        id=app_id,
                        is_progress=True,
                    ).first()

                    if app is None:
                        return
                    effective_time_step = max(0, int(time_step))
                    resources = self._lock_resources(lock_provider_ids)

                    TaskExecution.objects.filter(
                        application_id=app
                    ).delete()

                    max_finish_time = self.apply_offloading(
                        app,
                        best_solution,
                        scheduled_ctx,
                        effective_time_step,
                        resources,
                    )

                    self.apply_cache(
                        final_cache_state,
                        cache_provider_ids,
                        resources,
                    )

                    self._finalize_application(
                        app,
                        effective_time_step,
                        max_finish_time,
                    )

                return

            except OperationalError as error:
                if not _is_deadlock(error) or attempt + 1 >= _MAX_TRANSACTION_RETRIES:
                    raise

                time.sleep(0.05 * (attempt + 1))

    def _lock_resources(
        self,
        provider_ids: Iterable[int],
    ) -> Dict[int, Resource]:
        normalized_ids = sorted({int(provider_id) for provider_id in provider_ids})

        if not normalized_ids:
            return {}

        rows = list(
            Resource.objects.select_for_update()
            .filter(sp_id_id__in=normalized_ids)
            .order_by("sp_id_id", "id")
        )

        resources = {}
        duplicates = []

        for resource in rows:
            sp_id = int(resource.sp_id_id)

            if sp_id in resources:
                duplicates.append(sp_id)
                continue

            resources[sp_id] = resource

        if duplicates:
            raise ValueError(
                f"Duplicate resource rows for providers: {sorted(set(duplicates))}"
            )

        missing = sorted(set(normalized_ids) - set(resources))

        if missing:
            raise ValueError(
                f"Missing resource rows for providers: {missing}"
            )

        return resources

    def apply_offloading(
        self,
        app: Application,
        best_solution: List,
        ctx: Dict,
        time_step: int,
        resources: Dict[int, Resource],
    ) -> float:
        sim_now = app.start_at

        if sim_now is None:
            base_time = getattr(
                self.cfg,
                "base_time",
                timezone.now(),
            )
            sim_now = base_time + timedelta(seconds=time_step)

        schedule_state = ctx.get("_schedule_state", {})
        task_start = schedule_state.get("task_start", {})
        task_finish = schedule_state.get("task_finish", {})
        task_energy = schedule_state.get("task_energy", {})

        solution_task_ids = [
            int(task_id)
            for task_id, _, _ in best_solution
        ]

        missing_start = [
            task_id
            for task_id in solution_task_ids
            if task_id not in task_start
        ]
        missing_finish = [
            task_id
            for task_id in solution_task_ids
            if task_id not in task_finish
        ]

        if missing_start or missing_finish:
            raise ValueError(
                f"Incomplete schedule state: missing_start={missing_start}, "
                f"missing_finish={missing_finish}"
            )

        provider_ids = sorted({
            int(provider_id)
            for _, provider_id, _ in best_solution
        })
        providers = ServiceProvider.objects.select_related(
            "vehicle_id",
            "rsu_id",
        ).in_bulk(provider_ids)

        missing_providers = sorted(
            set(provider_ids) - set(providers)
        )
        if missing_providers:
            raise ValueError(
                f"Missing service providers: {missing_providers}"
            )

        missing_resources = sorted(
            set(provider_ids) - set(resources)
        )
        if missing_resources:
            raise ValueError(
                f"Missing locked resources: {missing_resources}"
            )

        overall_max_t = 0.0
        local_sp_id = ctx.get("local_sp_id")
        cpu_increments = {provider_id: 0 for provider_id in provider_ids}

        for task_id, provider_id, rank in best_solution:
            task_id = int(task_id)
            provider_id = int(provider_id)
            sp = providers[provider_id]

            start_time = float(task_start[task_id])
            finish_time = float(task_finish[task_id])
            energy_value = float(task_energy.get(task_id, 0.0))

            if start_time < 0.0 or finish_time < start_time:
                raise ValueError(
                    f"Invalid schedule for task {task_id}: "
                    f"start={start_time}, finish={finish_time}"
                )

            exec_time = finish_time - start_time
            overall_max_t = max(overall_max_t, finish_time)

            gain_value = 0.0
            rate_value = 0.0

            if local_sp_id is None or provider_id != int(local_sp_id):
                gain_value = float(
                    channel_gain(ctx, provider_id)
                )
                rate_value = float(
                    rate(ctx, provider_id)
                )

            distance_value = 0.0
            if local_sp_id is None or provider_id != int(local_sp_id):
                distance_value = float(
                    ctx.get("distance", {}).get(provider_id, 0.0)
                )

            te = TaskExecution.objects.create(
                application_id=app,
                sp_id=sp,
                task_id_id=task_id,
                start_time=sim_now + timedelta(seconds=start_time),
                end_time=sim_now + timedelta(seconds=finish_time),
                exec_time=exec_time,
                energy=energy_value,
            )

            State.objects.create(
                time_step=time_step,
                task_execution_id=te,
                from_vehicle_id=app.vehicle_id,
                to_vehicle_id=(
                    sp.vehicle_id
                    if sp.vehicle_id_id
                    else None
                ),
                to_rsu_id=(
                    sp.rsu_id
                    if sp.rsu_id_id
                    else None
                ),
                gain=gain_value,
                distance=distance_value,
                rate=rate_value,
            )

            cpu_increments[provider_id] += int(
                ctx["cpu_cycles"].get(task_id, 0)
            )

        for provider_id in provider_ids:
            resource = resources[provider_id]
            resource.cpu_used = int(resource.cpu_used or 0) + int(
                cpu_increments[provider_id]
            )
            resource.save(update_fields=["cpu_used"])

        return overall_max_t

    def apply_cache(
        self,
        cache_state: Dict[int, Any],
        provider_ids: Iterable[int],
        resources: Dict[int, Resource],
    ) -> None:
        normalized_provider_ids = sorted({
            int(provider_id)
            for provider_id in provider_ids
        })

        if not normalized_provider_ids:
            return

        normalized_cache = {
            sp_id: sorted({
                int(task_type_id)
                for task_type_id in (cache_state or {}).get(sp_id, set())
            })
            for sp_id in normalized_provider_ids
        }

        providers = ServiceProvider.objects.select_related(
            "vehicle_id",
            "rsu_id",
        ).in_bulk(normalized_provider_ids)

        missing_providers = sorted(
            set(normalized_provider_ids) - set(providers)
        )
        if missing_providers:
            raise ValueError(
                f"Missing cache service providers: {missing_providers}"
            )

        missing_resources = sorted(
            set(normalized_provider_ids) - set(resources)
        )
        if missing_resources:
            raise ValueError(
                f"Missing locked cache resources: {missing_resources}"
            )

        all_task_type_ids = sorted({
            task_type_id
            for task_types in normalized_cache.values()
            for task_type_id in task_types
        })
        task_type_sizes = dict(
            TaskType.objects.filter(
                id__in=all_task_type_ids
            ).values_list("id", "size")
        )

        missing_task_types = sorted(
            set(all_task_type_ids) - set(task_type_sizes)
        )
        if missing_task_types:
            raise ValueError(
                f"Missing cache task types: {missing_task_types}"
            )

        for sp_id in normalized_provider_ids:
            task_type_ids = normalized_cache[sp_id]
            cache_used = sum(
                int(task_type_sizes[task_type_id] or 0)
                for task_type_id in task_type_ids
            )
            resource = resources[sp_id]

            if cache_used > int(resource.cache_capacity or 0):
                raise ValueError(
                    f"Cache capacity exceeded for provider {sp_id}: "
                    f"used={cache_used}, capacity={resource.cache_capacity}"
                )

            CacheModel.objects.filter(sp_id_id=sp_id).delete()
            CacheModel.objects.bulk_create([
                CacheModel(
                    sp_id_id=sp_id,
                    task_type_id_id=task_type_id,
                )
                for task_type_id in task_type_ids
            ])

            resource.cache_used = int(cache_used)
            resource.save(update_fields=["cache_used"])

    def _finalize_application(
        self,
        app: Application,
        current_t: int,
        duration: float,
    ):
        start_dt = app.start_at

        if start_dt is None:
            base_time = getattr(
                self.cfg,
                "base_time",
                timezone.now(),
            )
            start_dt = base_time + timedelta(seconds=current_t)

        app.is_progress = False
        app.end_at = start_dt + timedelta(
            seconds=max(0.0, float(duration))
        )
        app.save(update_fields=["is_progress", "end_at"])
