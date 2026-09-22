from __future__ import annotations

from bisect import bisect_right
from collections import deque
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

# One compact problem-aware cache mechanism.
DEFAULT_LOOKAHEAD = 5
DEFAULT_RANK_AWARE = True
DEFAULT_DAG_AWARE = True
DEFAULT_PROVIDER_GUIDANCE_WEIGHT = 0.40

def build_joint_dag_distance_map(joint_ctx) -> Dict[Tuple[int, int], int]:
    """Precompute shortest descendant distance once for all joint DAG tasks."""
    reverse_refs = joint_ctx.get("reverse_task_refs", {})
    result: Dict[Tuple[int, int], int] = {}

    for app_id, app_ctx in joint_ctx.get("applications", {}).items():
        app_id = int(app_id)
        children = {
            int(task_id): [int(child_id) for child_id in values]
            for task_id, values in app_ctx.get("children", {}).items()
        }
        for source_task_id in [int(t) for t in app_ctx.get("task_ids", [])]:
            source_joint = reverse_refs.get((app_id, source_task_id))
            if source_joint is None:
                continue
            queue = deque((child, 1) for child in children.get(source_task_id, []))
            visited = set()
            while queue:
                target_task_id, distance = queue.popleft()
                target_task_id = int(target_task_id)
                if target_task_id in visited:
                    continue
                visited.add(target_task_id)
                target_joint = reverse_refs.get((app_id, target_task_id))
                if target_joint is not None:
                    result[(int(source_joint), int(target_joint))] = int(distance)
                for child in children.get(target_task_id, []):
                    if int(child) not in visited:
                        queue.append((int(child), int(distance) + 1))
    return result


def _matching_future_tasks(ctx, task_type_id: int, future_tasks: Iterable[int]) -> list[int]:
    task_types = ctx.get("task_type_ids", {}) or {}
    task_type_id = int(task_type_id)
    return [
        int(task_id)
        for task_id in future_tasks
        if task_types.get(int(task_id)) is not None
        and int(task_types[int(task_id)]) == task_type_id
    ]


def _rank_factor(ctx, matching_tasks: Iterable[int]) -> float:
    if not bool(ctx.get("predictive_cache_rank_aware", DEFAULT_RANK_AWARE)):
        return 1.0
    normalized = ctx.get("predictive_rank_weight", {}) or {}
    values = [
        max(0.0, min(1.0, float(normalized.get(int(task_id), 0.0))))
        for task_id in matching_tasks
    ]
    if not values:
        return 1.0
    return 0.5 + 0.5 * (sum(values) / len(values))


def _dag_factor(ctx, current_task_id: int, matching_tasks: Iterable[int]) -> float:
    if not bool(ctx.get("predictive_cache_dag_aware", DEFAULT_DAG_AWARE)):
        return 1.0
    distances = ctx.get("predictive_dag_distance", {}) or {}
    factors = []
    for task_id in matching_tasks:
        distance = distances.get((int(current_task_id), int(task_id)))
        factors.append(
            1.0 if distance is None or int(distance) <= 0
            else 1.0 + 1.0 / float(distance)
        )
    return sum(factors) / len(factors) if factors else 1.0


def build_provider_future_index(context: Any, provider_map: Dict[int, int]) -> Dict[str, Any]:
    """Index structural future assignments once per wolf move."""
    order = tuple(int(t) for t in (context.get("task_order", []) or []))
    positions = {task_id: index for index, task_id in enumerate(order)}
    provider_tasks: Dict[int, list[int]] = {}
    provider_positions: Dict[int, list[int]] = {}
    for task_id in order:
        provider_id = provider_map.get(int(task_id))
        if provider_id is None:
            continue
        provider_id = int(provider_id)
        provider_tasks.setdefault(provider_id, []).append(int(task_id))
        provider_positions.setdefault(provider_id, []).append(int(positions[task_id]))
    return {
        "positions": positions,
        "provider_tasks": {k: tuple(v) for k, v in provider_tasks.items()},
        "provider_positions": {k: tuple(v) for k, v in provider_positions.items()},
    }


def provider_reuse_guidance(
    context: Any,
    task_id: int,
    provider_id: int,
    provider_map: Dict[int, int],
    *,
    future_index: Optional[Dict[str, Any]] = None,
    lookahead: Optional[int] = None,
) -> float:
    """Search-only future same-provider reuse signal; never a fitness bonus."""
    task_id = int(task_id)
    provider_id = int(provider_id)
    task_types = context.get("task_type_ids", {}) or {}
    task_type_id = task_types.get(task_id)
    if task_type_id is None:
        return 0.0

    future_index = future_index or build_provider_future_index(context, provider_map)
    current_position = future_index.get("positions", {}).get(task_id)
    if current_position is None:
        return 0.0

    positions: Sequence[int] = future_index.get("provider_positions", {}).get(provider_id, ())
    tasks: Sequence[int] = future_index.get("provider_tasks", {}).get(provider_id, ())
    if not positions or not tasks:
        return 0.0

    limit = max(
        0,
        int(
            lookahead
            if lookahead is not None
            else context.get("predictive_cache_lookahead", DEFAULT_LOOKAHEAD)
        ),
    )
    if limit <= 0:
        return 0.0

    start = bisect_right(positions, int(current_position))
    future_tasks = [int(task) for task in tasks[start : start + limit]]
    matching = _matching_future_tasks(context, int(task_type_id), future_tasks)
    return float(
        sum(
            _rank_factor(context, [future_task])
            * _dag_factor(context, task_id, [future_task])
            for future_task in matching
        )
    )


def normalized_provider_reuse_guidance(
    context: Any,
    task_id: int,
    providers: Iterable[int],
    provider_map: Dict[int, int],
    *,
    future_index: Optional[Dict[str, Any]] = None,
) -> Dict[int, float]:
    provider_ids = list(dict.fromkeys(int(p) for p in providers))
    if not provider_ids:
        return {}
    future_index = future_index or build_provider_future_index(context, provider_map)
    raw = {
        provider_id: provider_reuse_guidance(
            context,
            int(task_id),
            provider_id,
            provider_map,
            future_index=future_index,
        )
        for provider_id in provider_ids
    }
    lo, hi = min(raw.values()), max(raw.values())
    if hi <= 0.0 or hi <= lo:
        return {provider_id: 0.0 for provider_id in provider_ids}
    return {
        provider_id: max(0.0, min(1.0, (value - lo) / (hi - lo)))
        for provider_id, value in raw.items()
    }
