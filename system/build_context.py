from collections import defaultdict
from math import isfinite
from django.core.cache import cache as django_cache
from application.models import Application
from dag.models import Task, TaskDependency, TaskType
from object.models import ServiceProvider, Vehicle, RSUVehicle
from cache.models import cache as CacheModel
from resource.models import Resource
from parameter.services import load_params_for_lib, load_params_obj
from monarch_pylib.model.transmission import distance_3d
from monarch_pylib.model import task_ranking
from algorithm.greedy_nests import t_comp, link_rate, _providers

SOURCE_PROGRAM_SIZE_REFERENCE_DOI = "10.1109/TVT.2022.3196544"


class MiniSystemContextBuilder:
    def __init__(self, application_id: int):
        self.application_id = int(application_id)
        self.ctx = {"application_id": self.application_id}

    def build_context(self) -> dict:
        self._build_application()
        self._build_task_info()
        self._build_providers()
        self._build_network_state()
        self._build_cache_state()
        self._build_assignment_state()
        self._build_task_ranking_context()
        self._build_auxiliary_fields()
        return self.ctx

    def _build_application(self):
        app = Application.objects.select_related(
            "application_type_id",
            "vehicle_id",
        ).get(id=self.application_id)

        app_type = app.application_type_id
        vehicle = app.vehicle_id

        if app_type is None:
            raise ValueError(f"Application {app.id} has no application type")
        if vehicle is None:
            raise ValueError(f"Application {app.id} has no vehicle")

        deadline_ms = float(app_type.deadline)
        deadline_s = deadline_ms / 1000.0

        alpha_n = 0.01 / deadline_s + 0.6
        beta_n = 1.0 - alpha_n

        if not 0.0 <= alpha_n <= 1.0 or not 0.0 <= beta_n <= 1.0:
            raise ValueError(
                f"Application {app.id} deadline produces invalid paper weights"
            )

        self.ctx["application"] = {
            "id": int(app.id),
            "application_type_id": int(app_type.id),
            "vehicle_id": int(vehicle.id),
        }
        self.ctx["application_initial_snapshot"] = app.initial_snapshot or {}
        self.ctx["application_type_initial_snapshot"] = app_type.initial_snapshot or {}
        self.ctx["application_start_at"] = app.start_at
        self.ctx["t_ddl_s"] = deadline_s
        self.ctx["deadline_s"] = deadline_s
        self.ctx["deadline_max_s"] = deadline_s
        self.ctx["alpha_n"] = float(alpha_n)
        self.ctx["beta_n"] = float(beta_n)
        self.ctx["vehicle_id"] = int(vehicle.id)

    def _build_task_info(self):
        app_type_id = self.ctx["application"]["application_type_id"]

        tasks = list(
            Task.objects.filter(application_type_id_id=app_type_id)
            .select_related("task_type_id")
            .order_by("id")
        )

        if not tasks:
            raise ValueError(f"Application type {app_type_id} has no tasks")

        task_ids = [int(task.id) for task in tasks]
        task_id_set = set(task_ids)
        cpu_cycles = {}
        task_type_ids = {}
        task_indexes = {}
        task_output_size_bits = {}
        task_type_size_bits = {}
        service_size_bits = {}
        service_size_bytes = {}
        source_program_size_bits = {}
        source_program_fraction = float(load_params_obj().source_program_fraction)

        for task in tasks:
            task_id = int(task.id)
            task_type = task.task_type_id

            if task_type is None:
                raise ValueError(f"Task {task_id} has no task type")

            cycles = float(task.workload_cycles or 0.0)
            if cycles <= 0.0:
                raise ValueError(f"Task {task_id} has an invalid workload")

            task_type_id = int(task_type.id)
            service_bytes = int(task_type.size or 0)
            if service_bytes <= 0:
                raise ValueError(
                    f"Task type {task_type_id} has an invalid service size"
                )

            task_snapshot = task.initial_snapshot or {}
            task_type_snapshot = task_type.initial_snapshot or {}
            output_bits = task_snapshot.get("output_size_bits")

            if output_bits is None:
                output_bits = task_type_snapshot.get("communication_data_bits")

            output_bits = int(float(output_bits or 0.0))
            if output_bits <= 0:
                raise ValueError(
                    f"Task {task_id} has no valid communication data size"
                )

            cpu_cycles[task_id] = cycles
            task_type_ids[task_id] = task_type_id
            task_indexes[task_id] = str(task.index or "")
            task_output_size_bits[task_id] = output_bits
            service_bits = service_bytes * 8
            task_type_size_bits[task_type_id] = service_bits
            service_size_bits[task_type_id] = service_bits
            service_size_bytes[task_type_id] = service_bytes
            source_program_size_bits[task_type_id] = max(
                1,
                int(round(service_bits * source_program_fraction)),
            )

        dependencies = defaultdict(list)
        children = defaultdict(list)
        task_dependencies = []
        edge_data_bits = defaultdict(dict)
        seen_edges = set()

        dep_qs = TaskDependency.objects.filter(
            parent_task_id_id__in=task_ids,
            child_task_id_id__in=task_ids,
        ).order_by("id")

        for dependency in dep_qs:
            parent_id = int(dependency.parent_task_id_id)
            child_id = int(dependency.child_task_id_id)

            if parent_id not in task_id_set or child_id not in task_id_set:
                raise ValueError(
                    f"Dependency {dependency.id} references an invalid task"
                )
            if parent_id == child_id:
                raise ValueError(
                    f"Dependency {dependency.id} contains a self-loop"
                )
            if (parent_id, child_id) in seen_edges:
                raise ValueError(
                    f"Duplicate dependency {parent_id}->{child_id}"
                )

            seen_edges.add((parent_id, child_id))
            dependency_snapshot = dependency.initial_snapshot or {}
            data_bits = dependency_snapshot.get("communication_data_bits")

            if data_bits is None:
                data_bits = task_output_size_bits[parent_id]

            data_bits = int(float(data_bits or 0.0))
            if data_bits <= 0:
                raise ValueError(
                    f"Dependency {parent_id}->{child_id} has an invalid data size"
                )

            dependencies[child_id].append(parent_id)
            children[parent_id].append(child_id)
            edge_data_bits[parent_id][child_id] = data_bits
            task_dependencies.append(
                {
                    "id": int(dependency.id),
                    "parent_task_id": parent_id,
                    "child_task_id": child_id,
                    "communication_data_bits": data_bits,
                }
            )

        for task_id in task_ids:
            dependencies[task_id] = sorted(dependencies.get(task_id, []))
            children[task_id] = sorted(children.get(task_id, []))
            edge_data_bits[task_id] = dict(edge_data_bits.get(task_id, {}))

        indegree = {
            task_id: len(dependencies[task_id])
            for task_id in task_ids
        }
        ready = sorted(
            task_id
            for task_id in task_ids
            if indegree[task_id] == 0
        )
        entry_tasks = list(ready)
        topological_order = []

        while ready:
            task_id = ready.pop(0)
            topological_order.append(task_id)

            for child_id in children[task_id]:
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    ready.append(child_id)
                    ready.sort()

        if len(topological_order) != len(task_ids):
            raise ValueError(
                f"Application type {app_type_id} task graph is not a DAG"
            )
        if len(entry_tasks) != 1:
            raise ValueError(
                f"Application type {app_type_id} must have exactly one entry task"
            )

        entry_task_id = int(entry_tasks[0])

        self.ctx["task_ids"] = task_ids
        self.ctx["cpu_cycles"] = cpu_cycles
        self.ctx["task_type_ids"] = task_type_ids
        self.ctx["task_indexes"] = task_indexes
        self.ctx["task_type_size_bits"] = task_type_size_bits
        self.ctx["task_output_size_bits"] = task_output_size_bits
        self.ctx["service_size_bits"] = service_size_bits
        self.ctx["service_size_bytes"] = service_size_bytes
        self.ctx["source_program_size_bits"] = source_program_size_bits
        self.ctx["source_program_size_model"] = (
            f"dtosc-2022-ratio-{source_program_fraction:.6g}-context-derived"
        )
        self.ctx["source_program_fraction"] = float(source_program_fraction)
        self.ctx["source_program_size_ratio"] = float(source_program_fraction)
        self.ctx["source_program_size_reference_doi"] = (
            SOURCE_PROGRAM_SIZE_REFERENCE_DOI
        )
        self.ctx["source_program_size_article_exact"] = False
        self.ctx["dependencies"] = dict(dependencies)
        self.ctx["children"] = dict(children)
        self.ctx["task_dependencies"] = task_dependencies
        self.ctx["edge_data_bits"] = dict(edge_data_bits)
        self.ctx["topological_order"] = topological_order
        self.ctx["entry_task_id"] = entry_task_id
        self.ctx["optimized_task_ids"] = [
            task_id
            for task_id in topological_order
            if task_id != entry_task_id
        ]
        self.ctx["tasks"] = {
            "all": topological_order,
            "ready": [entry_task_id],
        }

    def _build_providers(self):
        vehicle_id = self.ctx.get("vehicle_id")
        provider_ids = []

        cached_providers = django_cache.get(
            f"vehicle_available_sps_{vehicle_id}",
            [],
        )

        for item in cached_providers or []:
            sp_id = getattr(item, "id", item)
            if sp_id is not None:
                provider_ids.append(int(sp_id))

        local_sp = None
        if vehicle_id is not None:
            local_sp = ServiceProvider.objects.filter(
                vehicle_id_id=vehicle_id,
                type="vehicle",
            ).first()

        if local_sp is not None:
            provider_ids.append(int(local_sp.id))

        provider_ids = list(dict.fromkeys(provider_ids))

        sp_qs = list(
            ServiceProvider.objects.filter(id__in=provider_ids)
            .select_related("rsu_id", "vehicle_id")
            .order_by("id")
        )

        self.ctx["providers"] = {}
        self.ctx["provider_ids"] = []
        self.ctx["sp_cpu_freq"] = {}
        self.ctx["sp_cache_capacity"] = {}
        self.ctx["cache_capacity"] = {}
        self.ctx["sp_types"] = {}
        self.ctx["sp_modes"] = {}
        self.ctx["sp_position"] = {}
        self.ctx["sp_vehicle_ids"] = {}
        self.ctx["sp_rsu_ids"] = {}
        self.ctx["local_sp_id"] = int(local_sp.id) if local_sp else None

        for sp in sp_qs:
            sp_id = int(sp.id)
            res = Resource.objects.filter(sp_id=sp).first()

            if sp.type == "rsu":
                cpu_capacity = sp.rsu_id.cpu_capacity if sp.rsu_id else None
                cache_capacity = sp.rsu_id.cache_capacity if sp.rsu_id else None
                x = sp.rsu_id.x_coord if sp.rsu_id else 0.0
                y = sp.rsu_id.y_coord if sp.rsu_id else 0.0
                rsu_id = sp.rsu_id_id
                vehicle_ref_id = None
                mode = "v2i"
            else:
                cpu_capacity = sp.vehicle_id.cpu_capacity if sp.vehicle_id else None
                cache_capacity = sp.vehicle_id.cache_capacity if sp.vehicle_id else None
                x = sp.vehicle_id.x_coord if sp.vehicle_id else 0.0
                y = sp.vehicle_id.y_coord if sp.vehicle_id else 0.0
                rsu_id = None
                vehicle_ref_id = sp.vehicle_id_id
                mode = "local" if sp_id == self.ctx["local_sp_id"] else "v2v"

            if res is not None:
                cpu_capacity = res.cpu_capacity
                cache_capacity = res.cache_capacity

            cpu_capacity = float(cpu_capacity if cpu_capacity is not None else 0.0)
            cache_capacity = int(cache_capacity if cache_capacity is not None else 0)

            self.ctx["provider_ids"].append(sp_id)
            self.ctx["providers"][sp_id] = {
                "id": sp_id,
                "type": sp.type,
                "mode": mode,
                "vehicle_id": int(vehicle_ref_id) if vehicle_ref_id else None,
                "rsu_id": int(rsu_id) if rsu_id else None,
            }
            self.ctx["sp_cpu_freq"][sp_id] = cpu_capacity
            self.ctx["sp_cache_capacity"][sp_id] = cache_capacity
            self.ctx["cache_capacity"][sp_id] = cache_capacity
            self.ctx["sp_types"][sp_id] = sp.type
            self.ctx["sp_modes"][sp_id] = mode
            self.ctx["sp_position"][sp_id] = (float(x), float(y))
            self.ctx["sp_vehicle_ids"][sp_id] = int(vehicle_ref_id) if vehicle_ref_id else None
            self.ctx["sp_rsu_ids"][sp_id] = int(rsu_id) if rsu_id else None

    def _build_network_state(self):
        params = load_params_obj()
        params_lib = load_params_for_lib()

        vehicle_id = self.ctx.get("vehicle_id")
        app_vehicle = (
            Vehicle.objects.filter(id=vehicle_id).first()
            if vehicle_id is not None
            else None
        )

        if app_vehicle is not None:
            vx = float(app_vehicle.x_coord)
            vy = float(app_vehicle.y_coord)
            local_cpu = float(app_vehicle.cpu_capacity)
        else:
            vx = 0.0
            vy = 0.0
            local_cpu = float(params_lib.fmax_vehicle_hz)

        vh = float(params.h_vehicle)
        rh = float(params.h_rsu)

        distances = {}

        for sp_id in self.ctx["providers"].keys():
            sx, sy = self.ctx["sp_position"][sp_id]
            sh = rh if self.ctx["sp_types"][sp_id] == "rsu" else vh

            distances[sp_id] = float(
                distance_3d(
                    vx,
                    vy,
                    vh,
                    float(sx),
                    float(sy),
                    sh,
                )
            )

        connected_counts = {}

        # Every V2V link of application n reuses one V2I subchannel of
        # the RSU currently accessed by the mission vehicle. Therefore,
        # all vehicle SPs in the same alliance V_n use that RSU's V_m.
        access_rsu_id = (
            RSUVehicle.objects
            .filter(
                vehicle_id_id=vehicle_id,
                is_current=True,
            )
            .values_list("rsu_id_id", flat=True)
            .first()
        )
        alliance_vehicle_count = 1.0

        if access_rsu_id is not None:
            alliance_vehicle_count = float(
                max(
                    RSUVehicle.objects.filter(
                        rsu_id_id=access_rsu_id,
                        is_current=True,
                    ).count(),
                    1,
                )
            )

        for sp_id in self.ctx["providers"].keys():
            if self.ctx["sp_types"][sp_id] == "rsu":
                rsu_id = self.ctx["sp_rsu_ids"].get(sp_id)
                if rsu_id is not None:
                    count = RSUVehicle.objects.filter(
                        rsu_id_id=rsu_id,
                        is_current=True,
                    ).count()
                    connected_counts[sp_id] = float(max(count, 1))
                else:
                    connected_counts[sp_id] = 1.0
            else:
                connected_counts[sp_id] = alliance_vehicle_count

        self.ctx["distance"] = distances
        self.ctx["v_m"] = connected_counts
        self.ctx["connected_vehicles_count"] = connected_counts
        self.ctx["local_cpu_freq_hz"] = local_cpu

    def _build_cache_state(self):
        cache_state = {}

        for sp_id in self.ctx["providers"].keys():
            cached_types = CacheModel.objects.filter(
                sp_id_id=sp_id,
            ).values_list("task_type_id_id", flat=True)

            cache_state[sp_id] = set(int(x) for x in cached_types if x is not None)

        self.ctx["cache"] = cache_state

    def _build_assignment_state(self):
        self.ctx["z"] = {
            "binary": {},
            "compact": {
                tid: {
                    "provider": None,
                    "rank": None,
                }
                for tid in self.ctx["task_ids"]
            },
        }

    def _build_task_ranking_context(self):
        task_ids = [int(task_id) for task_id in self.ctx["tasks"]["all"]]
        providers = [int(sp_id) for sp_id in _providers(self.ctx)]

        if not providers:
            raise ValueError("Task ranking requires at least one provider")

        average_compute_time = {}

        for task_id in task_ids:
            values = [
                float(t_comp(self.ctx, sp_id, task_id))
                for sp_id in providers
            ]
            average_compute_time[task_id] = sum(values) / len(values)

        average_link_rates = []

        for src_sp in providers:
            for dst_sp in providers:
                if src_sp == dst_sp:
                    continue
                value = float(link_rate(self.ctx, src_sp, dst_sp))
                if value > 0:
                    average_link_rates.append(value)

        local_ranks = {}

        for task_id in reversed(self.ctx["topological_order"]):
            successors = [
                int(value)
                for value in self.ctx["children"].get(task_id, [])
            ]

            if not successors:
                local_ranks[task_id] = average_compute_time[task_id]
                continue

            comm_times = []
            successor_ranks = []

            for successor_id in successors:
                data_bits = (
                    self.ctx["edge_data_bits"]
                    .get(task_id, {})
                    .get(
                        successor_id,
                        self.ctx["task_output_size_bits"].get(task_id, 0),
                    )
                )

                transfer_times = [
                    float(data_bits) / rate_value
                    for rate_value in average_link_rates
                    if rate_value > 0
                ]

                comm_times.append(
                    sum(transfer_times) / len(transfer_times)
                    if transfer_times
                    else 0.0
                )
                successor_ranks.append(local_ranks[successor_id])

            local_ranks[task_id] = task_ranking.heft_task_local_rank(
                task_time_s=average_compute_time[task_id],
                succ_comm_times_s=comm_times,
                succ_ranks_s=successor_ranks,
            )

        global_ranks = {
            task_id: task_ranking.heft_task_global_rank(
                local_rank_s=float(rank),
                max_deadline_s=float(self.ctx["deadline_max_s"]),
                app_deadline_s=float(self.ctx["deadline_s"]),
            )
            for task_id, rank in local_ranks.items()
        }

        task_order = [
            int(task_id)
            for task_id, _ in sorted(
                global_ranks.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        entry_task_id = int(self.ctx["entry_task_id"])

        task_order = [
            task_id
            for task_id in task_order
            if task_id != entry_task_id
        ]

        self.ctx["local_ranks"] = {
            int(k): float(v)
            for k, v in local_ranks.items()
        }

        self.ctx["global_ranks"] = {
            int(k): float(v)
            for k, v in global_ranks.items()
        }

        self.ctx["task_order"] = list(task_order)
        self.ctx["ranked_task_ids"] = list(task_order)

    def _build_auxiliary_fields(self):
        self.ctx["idle_time"] = 0.0
        self.ctx["output_size"] = {
            tid: int(self.ctx["task_output_size_bits"].get(tid, 0))
            for tid in self.ctx["task_ids"]
        }

        task_type_ids = set(self.ctx["task_type_ids"].values()) - {None}
        task_types = {
            int(t.id): t
            for t in TaskType.objects.filter(id__in=task_type_ids)
        }

        compile_workloads = {}

        app_type_snapshot = self.ctx.get("application_type_initial_snapshot") or {}
        service_compile_map = app_type_snapshot.get("compile_workloads") or app_type_snapshot.get("service_compile_workloads") or {}

        for task_type_id in task_type_ids:
            task_type_id = int(task_type_id)
            task_type = task_types.get(task_type_id)
            value = None

            if task_type is not None and isinstance(task_type.initial_snapshot, dict):
                for key in (
                    "compile_workload_cycles",
                    "compile_cycles",
                    "w_k_cycles",
                    "W_k",
                    "Wk",
                ):
                    if key in task_type.initial_snapshot:
                        value = task_type.initial_snapshot[key]
                        break

            if value is None:
                value = service_compile_map.get(str(task_type_id), service_compile_map.get(task_type_id))

            compile_workloads[task_type_id] = float(value) if value is not None else 0.0

        self.ctx["compile_workloads"] = compile_workloads

        cpu_cycles_by_task = self.ctx["cpu_cycles"]
        task_type_by_task = self.ctx["task_type_ids"]

        cache_value_v_kj = {}
        cache_value_mu = {}
        cache_value_denom_cpu_cycles = {}
        cache_value_denom_v_kj = {}

        all_task_ids = list(self.ctx["task_ids"])
        all_cpu_cycles = [float(cpu_cycles_by_task[tid]) for tid in all_task_ids]

        cached_counts = {}
        for task_type_id in task_type_ids:
            task_type_id = int(task_type_id)
            cached_counts[task_type_id] = CacheModel.objects.filter(
                task_type_id_id=task_type_id
            ).count()

        for sp_id in self.ctx["providers"].keys():
            cache_value_denom_cpu_cycles[sp_id] = all_cpu_cycles

            for task_type_id in task_type_ids:
                task_type_id = int(task_type_id)
                flags = [
                    1.0 if task_type_by_task.get(tid) == task_type_id else 0.0
                    for tid in all_task_ids
                ]

                cache_value_v_kj[(sp_id, task_type_id)] = flags
                cache_value_mu[(sp_id, task_type_id)] = float(cached_counts.get(task_type_id, 0))
                cache_value_denom_v_kj[(sp_id, task_type_id)] = flags

        self.ctx["cache_value_v_kj"] = cache_value_v_kj
        self.ctx["cache_value_mu"] = cache_value_mu
        self.ctx["cache_value_denom_cpu_cycles"] = cache_value_denom_cpu_cycles
        self.ctx["cache_value_denom_v_kj"] = cache_value_denom_v_kj

        local_sp_id = self.ctx.get("local_sp_id")
        if local_sp_id is None or local_sp_id not in self.ctx["sp_cpu_freq"]:
            raise ValueError("Local service provider has no CPU capacity")

        fmax_by_provider = {
            int(sp_id): float(value)
            for sp_id, value in self.ctx["sp_cpu_freq"].items()
        }
        local_fmax = float(fmax_by_provider[int(local_sp_id)])
        if not isfinite(local_fmax) or local_fmax <= 0.0:
            raise ValueError("Local maximum CPU frequency must be positive")

        total_cycles = sum(float(value) for value in self.ctx["cpu_cycles"].values())
        kappa = float(load_params_obj().k)
        t_local = total_cycles / local_fmax
        e_local = kappa * (local_fmax ** 2) * total_cycles
        t_ref = min(t_local, float(self.ctx["deadline_s"]))

        alpha_n = float(self.ctx["alpha_n"])
        beta_n = float(self.ctx["beta_n"])

        if t_ref <= 0.0 or e_local <= 0.0 or kappa <= 0.0:
            raise ValueError("Invalid local reference for CPU allocation")

        if beta_n <= 0.0:
            f_star = local_fmax
        else:
            f_star = (
                alpha_n * e_local
                / (2.0 * beta_n * kappa * t_ref)
            ) ** (1.0 / 3.0)

        allocated_by_provider = {}
        for sp_id, fmax in fmax_by_provider.items():
            if not isfinite(fmax) or fmax <= 0.0:
                raise ValueError(f"Provider {sp_id} has an invalid CPU capacity")
            if self.ctx["sp_types"].get(sp_id) == "rsu":
                allocated_by_provider[sp_id] = float(fmax)
            else:
                allocated_by_provider[sp_id] = float(min(f_star, fmax))

        self.ctx["sp_cpu_fmax_hz"] = fmax_by_provider
        self.ctx["sp_cpu_allocated_hz"] = allocated_by_provider
        self.ctx["paper_reference"] = {
            "t_local_s": float(t_local),
            "t_ref_s": float(t_ref),
            "e_local_j": float(e_local),
        }
        self.ctx["paper_weights"] = {
            "alpha_n": alpha_n,
            "beta_n": beta_n,
        }
        self.ctx["article_exact_cpu_allocation"] = True