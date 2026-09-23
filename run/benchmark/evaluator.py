from __future__ import annotations

import copy
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from algorithm.optimizer_common import (
    compute_task_ranks_and_order,
    evaluate_solution_quality,
)
from algorithm.greedy_nests import (
    _apply_assignment,
    _candidate_eval,
    _empty_state,
    _reset_assignment,
    e_loc_j,
    t_ref_s,
)
from algorithm.update_service_cache import update_cache

from .models import ApplicationResult, JointApplicationResult
from .paper_model import (
    ApplicationWeights,
    LocalReference,
    offloading_efficiency,
)
from .schemes import get_joint_scheme

@dataclass(frozen=True)
class MetricResult:
    avg_delay: float
    avg_efficiency: float
    total_efficiency: float
    completion_rate: float


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    for key in ("application_id", "app_id", "application", "vehicle_id", "vehicle", "n"):
        if row.get(key) is not None:
            return key, row[key]
    return ("single_application",)


def _finish_time(row: Dict[str, Any]) -> Optional[float]:
    for key in (
        "t_off", "t_finish", "finish_time_s", "completion_time_s",
        "delay", "latency", "latency_s",
    ):
        value = _to_float(row.get(key))
        if value is not None:
            return value

    parts = [_to_float(row.get(key)) for key in ("exec_time_s", "transfer_time_s", "penalty_s")]
    if all(value is None for value in parts):
        return None
    return float(sum(value or 0.0 for value in parts))


def _efficiency(row: Dict[str, Any]) -> Optional[float]:
    for key in ("q", "Q", "efficiency", "offloading_efficiency", "q_n", "Qn"):
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _deadline(ctx: Dict[str, Any], row: Dict[str, Any]) -> Optional[float]:
    keys = ("deadline_s", "deadline", "T_ddl", "app_deadline_s")
    for source in (row, ctx, ctx.get("params") if isinstance(ctx.get("params"), dict) else {}):
        for key in keys:
            value = _to_float(source.get(key))
            if value is not None:
                return value
    return None


def _safe_decisions(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        decisions = value.get("decisions")
        if isinstance(decisions, list):
            return [item for item in decisions if isinstance(item, dict)]
        return [value]
    return []


def evaluate_metrics(ctx: Dict[str, Any], decisions: Any) -> MetricResult:
    rows = _safe_decisions(decisions)
    finish_by_group: Dict[Tuple[Any, ...], float] = {}
    efficiency_by_group: Dict[Tuple[Any, ...], float] = {}
    completion: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for row in rows:
        key = _group_key(row)
        finish = _finish_time(row)
        if finish is not None:
            finish_by_group[key] = max(finish_by_group.get(key, 0.0), finish)

        q_value = _efficiency(row)
        if q_value is not None:
            efficiency_by_group[key] = q_value

        item = completion.setdefault(
            key,
            {"finish": None, "deadline": _deadline(ctx, row), "explicit": None},
        )
        explicit = row.get("completed", row.get("ok"))
        if isinstance(explicit, bool):
            item["explicit"] = explicit
        if finish is not None:
            item["finish"] = finish if item["finish"] is None else max(item["finish"], finish)
        ddl = _deadline(ctx, row)
        if ddl is not None:
            item["deadline"] = ddl

    avg_delay = (
        sum(finish_by_group.values()) / len(finish_by_group)
        if finish_by_group else 0.0
    )
    avg_efficiency = (
        sum(efficiency_by_group.values()) / len(efficiency_by_group)
        if efficiency_by_group else 0.0
    )
    total_efficiency = sum(efficiency_by_group.values()) if efficiency_by_group else 0.0

    completed_count = 0
    for item in completion.values():
        if item["explicit"] is not None:
            completed_count += int(bool(item["explicit"]))
        elif (
            item["finish"] is not None
            and item["deadline"] is not None
            and item["finish"] <= item["deadline"]
        ):
            completed_count += 1
    completion_rate = completed_count / len(completion) if completion else 0.0

    return MetricResult(
        avg_delay=float(avg_delay),
        avg_efficiency=float(avg_efficiency),
        total_efficiency=float(total_efficiency),
        completion_rate=float(completion_rate),
    )


def configured_context(
    base_ctx: Dict[str, Any],
    algorithm: str,
    seed: int,
) -> Dict[str, Any]:
    scheme = get_joint_scheme(algorithm)
    ctx = copy.deepcopy(base_ctx)
    ctx.update(scheme.context_flags())
    ctx["seed"] = int(seed)
    ctx["scheme"] = scheme.name
    return ctx


def normalize_algorithm_output(
    base_ctx: Dict[str, Any],
    algorithm: str,
    seed: int,
    raw_output: Tuple[
        Iterable,
        float,
        Dict,
    ],
) -> ApplicationResult:
    if (
        not isinstance(raw_output, tuple)
        or len(raw_output) != 3
    ):
        raise ValueError(
            f"Invalid output from {algorithm}"
        )

    nest, raw_quality, _ = raw_output
    nest = list(nest)

    eval_ctx = configured_context(
        base_ctx,
        algorithm,
        seed,
    )

    task_order = compute_task_ranks_and_order(
        eval_ctx
    )

    quality, _, scheduled_ctx = (
        evaluate_solution_quality(
            eval_ctx,
            nest,
            task_order,
        )
    )

    state = scheduled_ctx.get(
        "_schedule_state",
        {},
    )

    task_finish = state.get(
        "task_finish",
        {},
    )

    task_energy = state.get(
        "task_energy",
        {},
    )

    task_provider = state.get(
        "task_provider",
        {},
    )

    delay_s = max(
        (
            float(value)
            for value in task_finish.values()
        ),
        default=0.0,
    )

    energy_j = sum(
        float(value)
        for value in task_energy.values()
    )

    deadline_s = float(
        scheduled_ctx.get(
            "deadline_s",
            0.0,
        )
        or 0.0
    )

    total_task_count = len(
        scheduled_ctx.get(
            "task_ids",
            task_order,
        )
    )

    scheduled_count = len(task_finish)

    completed = (
        scheduled_count == total_task_count
        and deadline_s > 0.0
        and delay_s <= deadline_s
    )

    application = scheduled_ctx.get(
        "application",
        {},
    )

    application_id = int(
        application.get(
            "id",
            scheduled_ctx.get(
                "application_id"
            ),
        )
    )

    vehicle_id = scheduled_ctx.get(
        "vehicle_id"
    )

    return ApplicationResult(
        application_id=application_id,
        vehicle_id=(
            int(vehicle_id)
            if vehicle_id is not None
            else None
        ),
        delay_s=float(delay_s),
        energy_j=float(energy_j),
        efficiency=float(
            quality
            if quality is not None
            else raw_quality
        ),
        deadline_s=deadline_s,
        completed=bool(completed),
        task_count=total_task_count,
        scheduled_task_count=scheduled_count,
        providers_used=sorted(
            set(
                int(value)
                for value
                in task_provider.values()
            )
        ),
    )


NestItem = Tuple[int, int, int]


@dataclass(frozen=True)
class JointEvaluation:
    total_efficiency: float
    applications: List[JointApplicationResult]
    metrics: Dict[str, float]
    cache_state: Dict[int, set[int]]
    schedule: List[Dict[str, Any]]


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / denominator

def _paper_reference_and_weights(
    app_ctx: Dict[str, Any],
) -> Tuple[LocalReference, ApplicationWeights]:

    reference_data = (
        app_ctx.get("paper_reference") or {}
    )

    weights_data = (
        app_ctx.get("paper_weights") or {}
    )

    reference = LocalReference(
        t_local_s=float(
            reference_data["t_local_s"]
        ),
        t_ref_s=float(
            reference_data["t_ref_s"]
        ),
        e_local_j=float(
            reference_data["e_local_j"]
        ),
    )

    weights = ApplicationWeights(
        alpha=float(
            weights_data["alpha_n"]
        ),
        beta=float(
            weights_data["beta_n"]
        ),
    )

    return reference, weights
def _application_efficiency(
    app_ctx: Dict[str, Any],
    state: Dict[str, Any],
) -> float:

    task_finish = state.get(
        "task_finish",
        {},
    )

    task_energy = state.get(
        "task_energy",
        {},
    )

    t_off_s = max(
        (
            float(value)
            for value in task_finish.values()
        ),
        default=0.0,
    )

    e_off_j = sum(
        float(value)
        for value in task_energy.values()
    )

    reference, weights = (
        _paper_reference_and_weights(app_ctx)
    )

    return offloading_efficiency(
        t_off_s=t_off_s,
        e_off_j=e_off_j,
        reference=reference,
        weights=weights,
    )
class JointScheduleState:
    def __init__(
        self,
        joint_ctx: Dict[str, Any],
        *,
        use_caching: bool,
        v2i_only: bool,
        record_schedule: bool = True,
    ):
        self.joint_ctx = joint_ctx
        self.use_caching = bool(use_caching)
        self.v2i_only = bool(v2i_only)
        self.record_schedule = bool(record_schedule)
        shared_cache_keys = (
            "_link_fading",
            "_channel_gain_cache",
            "_tx_power_cache",
            "_link_rate_cache",
            "_tx_time_energy_cache",
            "_service_program_energy_cache",
        )
        self.application_contexts: Dict[int, Dict[str, Any]] = {}
        for app_id, source_ctx in joint_ctx["applications"].items():
            app_ctx = dict(source_ctx)
            for cache_key in shared_cache_keys:
                app_ctx[cache_key] = source_ctx.setdefault(cache_key, {})
            self.application_contexts[int(app_id)] = app_ctx
        self.shared_cache: Dict[int, set[int]] = (
            copy.deepcopy(
                self.joint_ctx.get("initial_cache", {})
            )
            if self.use_caching
            else {
                int(sp_id): set()
                for sp_id in self.joint_ctx["provider_ids"]
            }
        )
        initial_provider_finish = self.joint_ctx.get(
            "initial_provider_finish",
            {},
        )
        self.shared_provider_finish: Dict[int, float] = {
            int(sp_id): max(
                0.0,
                float(initial_provider_finish.get(int(sp_id), 0.0)),
            )
            for sp_id in self.joint_ctx["provider_ids"]
        }
        initial_link_finish = self.joint_ctx.get(
            "initial_link_finish",
            {},
        )
        self.shared_link_finish: Dict[Tuple[int, int], float] = {
            (int(link_key[0]), int(link_key[1])): max(0.0, float(value))
            for link_key, value in initial_link_finish.items()
        }
        self.rank_counter: Dict[int, int] = {
            int(sp_id): 0 for sp_id in self.joint_ctx["provider_ids"]
        }
        self.app_states: Dict[int, Dict[str, Any]] = {}
        self.scheduled_joint_ids: set[int] = set()
        self.schedule_rows: List[Dict[str, Any]] = []

        for app_id, app_ctx in self.application_contexts.items():
            app_ctx["use_caching"] = self.use_caching
            app_ctx["v2i_only"] = self.v2i_only
            app_ctx["cache"] = self.shared_cache
            _reset_assignment(app_ctx)
            state = _empty_state(app_ctx)
            state["provider_finish"] = self.shared_provider_finish
            state["link_finish"] = self.shared_link_finish
            app_ctx["_schedule_state"] = state
            self.app_states[int(app_id)] = state

        self.joint_cache_ctx = {
            "use_caching": self.use_caching,
            "cache": self.shared_cache,
            "task_ids": list(self.joint_ctx["optimized_task_ids"]),
            "task_type_ids": dict(self.joint_ctx["task_type_ids"]),
            "cpu_cycles": dict(self.joint_ctx["cpu_cycles"]),
            "service_size_bits": dict(self.joint_ctx["service_size_bits"]),
            "compile_workloads": dict(self.joint_ctx["compile_workloads"]),
            "sp_cache_capacity": dict(self.joint_ctx["sp_cache_capacity"]),
        }

    def clone(self) -> "JointScheduleState":
        return copy.deepcopy(self)

    def clone_for_application(self, application_id: int) -> "JointScheduleState":
        """Clone a DP branch while copying only the mutable target app deeply.

        DTOSC expands several alternatives for one application at a time.
        Copying the complete multi-application object at every transition is
        prohibitively expensive for the 68/76-vehicle paper scenarios.  This
        method preserves branch isolation for the shared provider/link queues,
        cache and target application while safely sharing immutable history of
        applications that are not modified in the current DP stage.
        """

        application_id = int(application_id)
        if application_id not in self.application_contexts:
            raise KeyError(f"Unknown application_id: {application_id}")

        cloned = object.__new__(type(self))
        cloned.joint_ctx = self.joint_ctx
        cloned.use_caching = self.use_caching
        cloned.v2i_only = self.v2i_only
        cloned.record_schedule = self.record_schedule
        cloned.shared_cache = {
            int(sp_id): set(values)
            for sp_id, values in self.shared_cache.items()
        }
        cloned.shared_provider_finish = dict(self.shared_provider_finish)
        cloned.shared_link_finish = dict(self.shared_link_finish)
        cloned.rank_counter = dict(self.rank_counter)
        cloned.scheduled_joint_ids = set(self.scheduled_joint_ids)
        # Existing rows are never mutated after insertion; a shallow list copy
        # isolates append operations while avoiding quadratic deep copies.
        cloned.schedule_rows = list(self.schedule_rows)
        cloned.application_contexts = {}
        cloned.app_states = {}

        for app_id, source_ctx in self.application_contexts.items():
            app_id = int(app_id)
            new_ctx = dict(source_ctx)
            source_state = self.app_states[app_id]

            if app_id == application_id:
                new_state: Dict[str, Any] = {}
                for key, value in source_state.items():
                    if key in {"provider_finish", "link_finish"}:
                        continue
                    if key == "task_transfers":
                        new_state[key] = copy.deepcopy(value)
                    elif isinstance(value, dict):
                        new_state[key] = dict(value)
                    elif isinstance(value, set):
                        new_state[key] = set(value)
                    elif isinstance(value, list):
                        new_state[key] = list(value)
                    else:
                        new_state[key] = copy.deepcopy(value)

                # _apply_assignment mutates the binary and compact z maps.
                new_ctx["z"] = copy.deepcopy(source_ctx.get("z", {}))
            else:
                # Untouched application task maps are immutable in this branch.
                # The state shell is copied only to retarget the shared queues.
                new_state = dict(source_state)

            new_state["provider_finish"] = cloned.shared_provider_finish
            new_state["link_finish"] = cloned.shared_link_finish
            new_ctx["cache"] = cloned.shared_cache
            new_ctx["_schedule_state"] = new_state
            cloned.application_contexts[app_id] = new_ctx
            cloned.app_states[app_id] = new_state

        cloned.joint_cache_ctx = dict(self.joint_cache_ctx)
        cloned.joint_cache_ctx["cache"] = cloned.shared_cache
        return cloned

    def application_efficiency(self, application_id: int) -> float:
        application_id = int(application_id)
        return float(
            _application_efficiency(
                self.application_contexts[application_id],
                self.app_states[application_id],
            )
        )

    def application_finish_time(self, application_id: int) -> float:
        state = self.app_states[int(application_id)]
        return max(
            (float(value) for value in state.get("task_finish", {}).values()),
            default=0.0,
        )

    def application_energy(self, application_id: int) -> float:
        state = self.app_states[int(application_id)]
        return float(
            sum(float(value) for value in state.get("task_energy", {}).values())
        )

    def task_ref(self, joint_task_id: int):
        return self.joint_ctx["task_refs"][int(joint_task_id)]

    def allowed_providers(self, joint_task_id: int) -> List[int]:
        joint_task_id = int(joint_task_id)
        ref = self.task_ref(joint_task_id)
        app_ctx = self.application_contexts[ref.application_id]
        domain = [
            int(sp_id)
            for sp_id in self.joint_ctx["task_domains"][joint_task_id]
        ]

        if ref.is_entry:
            return [int(app_ctx["local_sp_id"])]

        if self.v2i_only:
            return [
                sp_id
                for sp_id in domain
                if app_ctx.get("sp_types", {}).get(sp_id) == "rsu"
            ]

        return domain

    def _record_assignment(
        self,
        joint_task_id: int,
        provider_id: int,
        *,
        is_entry: bool,
        cache_before: Iterable[int] | None = None,
        cache_after: Iterable[int] | None = None,
    ) -> None:
        if not self.record_schedule:
            return

        ref = self.task_ref(joint_task_id)
        app_ctx = self.application_contexts[ref.application_id]
        state = self.app_states[ref.application_id]
        original_task_id = ref.task_id
        before = sorted(int(value) for value in (cache_before or []))
        after = sorted(int(value) for value in (cache_after or []))
        task_type_id = app_ctx.get("task_type_ids", {}).get(original_task_id)
        provider_mode = app_ctx.get("sp_modes", {}).get(provider_id)
        cache_hit = (
            None
            if bool(is_entry) or provider_mode == "local" or task_type_id is None
            else int(task_type_id) in set(before)
        )

        self.schedule_rows.append(
            {
                "joint_task_id": int(joint_task_id),
                "application_id": int(ref.application_id),
                "task_id": int(original_task_id),
                "task_type_id": (
                    int(task_type_id)
                    if task_type_id is not None
                    else None
                ),
                "provider_id": int(provider_id),
                "provider_mode": provider_mode,
                "rank": int(self.rank_counter[provider_id]),
                "start_s": float(
                    state["task_start"][original_task_id]
                ),
                "finish_s": float(
                    state["task_finish"][original_task_id]
                ),
                "energy_j": float(
                    state["task_energy"][original_task_id]
                ),
                "transfers": copy.deepcopy(
                    state.get(
                        "task_transfers",
                        {},
                    ).get(
                        original_task_id,
                        [],
                    )
                ),
                "cpu_frequency_hz": float(
                    app_ctx.get(
                        "sp_cpu_allocated_hz",
                        app_ctx.get("sp_cpu_freq", {}),
                    )[int(provider_id)]
                ),
                "cpu_fmax_hz": float(
                    app_ctx.get(
                        "sp_cpu_fmax_hz",
                        app_ctx.get("sp_cpu_freq", {}),
                    )[int(provider_id)]
                ),
                "is_entry": bool(is_entry),
                "cache_before": before,
                "cache_after": after,
                "cache_hit": cache_hit,
                "cache_miss": (None if cache_hit is None else not cache_hit),
                "cache_inserted_count": len(set(after) - set(before)),
                "cache_evicted_count": len(set(before) - set(after)),
                "cache_update_required": bool(before != after),
            }
        )

    def assign_entry_tasks(self) -> None:
        for app_id in self.joint_ctx["application_ids"]:
            joint_task_id = int(self.joint_ctx["entry_task_ids"][app_id])
            ref = self.task_ref(joint_task_id)
            app_ctx = self.application_contexts[app_id]
            provider_id = int(app_ctx["local_sp_id"])
            state = self.app_states[app_id]
            cached = set(self.shared_cache.get(provider_id, set()))

            self.rank_counter[provider_id] += 1
            _apply_assignment(
                app_ctx,
                state,
                ref.task_id,
                provider_id,
                self.rank_counter[provider_id],
            )
            self.scheduled_joint_ids.add(joint_task_id)
            self._record_assignment(
                joint_task_id,
                provider_id,
                is_entry=True,
                cache_before=cached,
                cache_after=cached,
            )

    def assign_task(
        self,
        joint_task_id: int,
        provider_id: int,
        *,
        remaining_task_ids: Iterable[int] | None = None,
    ) -> None:
        joint_task_id = int(joint_task_id)
        provider_id = int(provider_id)
        ref = self.task_ref(joint_task_id)

        if ref.is_entry:
            raise ValueError("Entry tasks are fixed locally and cannot be placed in a nest")
        if joint_task_id in self.scheduled_joint_ids:
            raise ValueError(f"Joint task {joint_task_id} was scheduled more than once")

        allowed = self.allowed_providers(joint_task_id)
        if provider_id not in allowed:
            raise ValueError(
                f"Provider {provider_id} is outside the domain of joint task {joint_task_id}"
            )
        if not allowed:
            raise ValueError(f"Joint task {joint_task_id} has no feasible providers")

        app_ctx = self.application_contexts[ref.application_id]
        state = self.app_states[ref.application_id]
        cache_before = set(self.shared_cache.get(provider_id, set()))

        self.rank_counter[provider_id] += 1
        _apply_assignment(
            app_ctx,
            state,
            ref.task_id,
            provider_id,
            self.rank_counter[provider_id],
        )
        self.scheduled_joint_ids.add(joint_task_id)

        mode = app_ctx.get("sp_modes", {}).get(provider_id)
        if self.use_caching and mode != "local":
            update_cache(
                self.joint_cache_ctx,
                provider_id,
                joint_task_id,
                remaining_task_ids=remaining_task_ids,
            )

        cache_after = set(self.shared_cache.get(provider_id, set()))
        self._record_assignment(
            joint_task_id,
            provider_id,
            is_entry=False,
            cache_before=cache_before,
            cache_after=cache_after,
        )

    def candidate_provider_score(
        self,
        joint_task_id: int,
        provider_id: int,
    ) -> float:
        joint_task_id = int(joint_task_id)
        provider_id = int(provider_id)
        ref = self.task_ref(joint_task_id)
        if ref.is_entry:
            raise ValueError("Entry tasks are fixed locally and cannot be candidate tasks")
        if provider_id not in self.allowed_providers(joint_task_id):
            raise ValueError(
                f"Provider {provider_id} is outside the domain of joint task {joint_task_id}"
            )
        app_ctx = self.application_contexts[ref.application_id]
        state = self.app_states[ref.application_id]
        q_value, _, _, _, _, _ = _candidate_eval(
            app_ctx,
            state,
            provider_id,
            ref.task_id,
        )
        return float(q_value)

    def candidate_task_score(self, joint_task_id: int) -> float:
        ref = self.task_ref(joint_task_id)
        app_ctx = self.application_contexts[ref.application_id]
        state = self.app_states[ref.application_id]
        finish = float(state["task_finish"][ref.task_id])
        task_energy = float(state["task_energy"][ref.task_id])
        t_reference = float(t_ref_s(app_ctx))
        e_local = float(e_loc_j(app_ctx))

        return float(
            float(app_ctx["alpha_n"])
            * _safe_ratio(t_reference - finish, t_reference)
            + float(app_ctx["beta_n"])
            * _safe_ratio(e_local - task_energy, e_local)
        )

    def total_efficiency(self) -> float:
        """Return the exact joint objective without materializing report rows.

        The application order and floating-point operations intentionally match
        ``final_evaluation``.  This method is used only by the search fast path;
        the winning nest is still evaluated once with full details before it is
        returned to callers.
        """

        return float(
            sum(
                _application_efficiency(
                    self.application_contexts[app_id],
                    self.app_states[app_id],
                )
                for app_id in self.joint_ctx["application_ids"]
            )
        )

    def final_evaluation(self) -> JointEvaluation:
        application_results: List[JointApplicationResult] = []

        for app_id in self.joint_ctx["application_ids"]:
            app_ctx = self.application_contexts[app_id]
            state = self.app_states[app_id]
            task_finish = state.get("task_finish", {})
            task_energy = state.get("task_energy", {})
            task_provider = state.get("task_provider", {})

            delay_s = max(
                (float(value) for value in task_finish.values()),
                default=0.0,
            )
            energy_j = sum(float(value) for value in task_energy.values())
            efficiency = _application_efficiency(app_ctx, state)
            task_count = len(app_ctx["task_ids"])
            optimized_task_count = len(app_ctx["optimized_task_ids"])
            scheduled_task_count = len(task_finish)
            deadline_s = float(app_ctx["deadline_s"])
            entry_task_id = int(app_ctx["entry_task_id"])
            entry_provider_id = int(app_ctx["local_sp_id"])

            application_results.append(
                JointApplicationResult(
                    application_id=int(app_id),
                    vehicle_id=(
                        int(app_ctx["vehicle_id"])
                        if app_ctx.get("vehicle_id") is not None
                        else None
                    ),
                    deadline_s=deadline_s,
                    alpha_n=float(app_ctx["alpha_n"]),
                    beta_n=float(app_ctx["beta_n"]),
                    delay_s=delay_s,
                    energy_j=energy_j,
                    efficiency=efficiency,
                    completed=(
                        scheduled_task_count == task_count
                        and delay_s <= deadline_s
                    ),
                    task_count=task_count,
                    optimized_task_count=optimized_task_count,
                    scheduled_task_count=scheduled_task_count,
                    entry_task_id=entry_task_id,
                    entry_provider_id=entry_provider_id,
                    providers_used=sorted(
                        {int(provider_id) for provider_id in task_provider.values()}
                    ),
                )
            )

        total_efficiency = sum(
            result.efficiency for result in application_results
        )
        metrics = {
            "avg_delay": mean(
                result.delay_s for result in application_results
            ),
            "avg_efficiency": mean(
                result.efficiency for result in application_results
            ),
            "total_efficiency": float(total_efficiency),
            "completion_rate": mean(
                1.0 if result.completed else 0.0
                for result in application_results
            ),
        }

        return JointEvaluation(
            total_efficiency=float(total_efficiency),
            applications=application_results,
            metrics={key: float(value) for key, value in metrics.items()},
            cache_state=copy.deepcopy(self.shared_cache),
            schedule=copy.deepcopy(self.schedule_rows),
        )


def _nest_provider_map(nest: Sequence[NestItem]) -> Dict[int, int]:
    provider_map: Dict[int, int] = {}
    for joint_task_id, provider_id, _rank in nest:
        joint_task_id = int(joint_task_id)
        if joint_task_id in provider_map:
            raise ValueError(f"Joint task {joint_task_id} appears more than once in nest")
        provider_map[joint_task_id] = int(provider_id)
    return provider_map


def evaluate_joint_nest(
    joint_ctx: Dict[str, Any],
    nest: Sequence[NestItem],
    task_order: Sequence[int],
    *,
    use_caching: bool,
    v2i_only: bool,
) -> JointEvaluation:
    expected = [int(task_id) for task_id in joint_ctx["optimized_task_ids"]]
    task_order = [int(task_id) for task_id in task_order]
    provider_map = _nest_provider_map(nest)

    if set(provider_map) != set(expected):
        missing = sorted(set(expected) - set(provider_map))
        extra = sorted(set(provider_map) - set(expected))
        raise ValueError(f"Nest task mismatch; missing={missing}, extra={extra}")
    if set(task_order) != set(expected) or len(task_order) != len(expected):
        raise ValueError("Task order must contain every non-entry task exactly once")

    state = JointScheduleState(
        joint_ctx,
        use_caching=use_caching,
        v2i_only=v2i_only,
    )
    state.assign_entry_tasks()

    for index, joint_task_id in enumerate(task_order):
        state.assign_task(
            joint_task_id,
            provider_map[joint_task_id],
            remaining_task_ids=task_order[index + 1 :],
        )

    return state.final_evaluation()


def _evaluate_joint_nest_state(
    joint_ctx: Dict[str, Any],
    nest: Sequence[NestItem],
    task_order: Sequence[int],
    *,
    use_caching: bool,
    v2i_only: bool,
) -> JointScheduleState:
    """Schedule one intermediate nest once and return its mutable state.

    This is the shared kernel for objective-only and cache-state inspection.
    Keeping the validation and assignment loop in one place ensures every
    optimizer uses exactly the same paper scheduling path.
    """
    expected = [int(task_id) for task_id in joint_ctx["optimized_task_ids"]]
    normalized_order = [int(task_id) for task_id in task_order]
    provider_map = _nest_provider_map(nest)
    expected_set = set(expected)

    if set(provider_map) != expected_set:
        missing = sorted(expected_set - set(provider_map))
        extra = sorted(set(provider_map) - expected_set)
        raise ValueError(f"Nest task mismatch; missing={missing}, extra={extra}")
    if set(normalized_order) != expected_set or len(normalized_order) != len(expected):
        raise ValueError("Task order must contain every non-entry task exactly once")

    state = JointScheduleState(
        joint_ctx,
        use_caching=use_caching,
        v2i_only=v2i_only,
        record_schedule=False,
    )
    state.assign_entry_tasks()
    for index, joint_task_id in enumerate(normalized_order):
        state.assign_task(
            joint_task_id,
            provider_map[joint_task_id],
            remaining_task_ids=normalized_order[index + 1 :],
        )
    return state


def evaluate_joint_nest_total(
    joint_ctx: Dict[str, Any],
    nest: Sequence[NestItem],
    task_order: Sequence[int],
    *,
    use_caching: bool,
    v2i_only: bool,
) -> float:
    """Evaluate only the exact total objective for an intermediate nest.

    Scheduling, queues, transfers, cache updates, formulas, and task order are
    identical to :func:`evaluate_joint_nest`.  Reporting objects are omitted.
    """
    state = _evaluate_joint_nest_state(
        joint_ctx,
        nest,
        task_order,
        use_caching=use_caching,
        v2i_only=v2i_only,
    )
    return float(state.total_efficiency())
