from __future__ import annotations

import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from uuid import uuid4

from django.conf import settings

from parameter.services import load_params_obj

from run.baselines import (
    JOINT_BENCHMARK_ALGORITHMS,
    PAPER_ALGORITHM_NAMES,
)

from .context import joint_context_summary
from .contracts import JointAlgorithmResult


ALGORITHMS = JOINT_BENCHMARK_ALGORITHMS


def _apply_paper_source_program_model(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the source-program model used by the paper benchmark.

    The 2025 article explicitly includes the energy of transmitting service
    programs in Eq. (27).  What the article does *not* publish numerically is
    the source-program data size used by that energy term.  The project derives
    that size once in ``MiniSystemContextBuilder`` and the benchmark consumes
    the same stored value; it must not silently remove the Eq. (27) term or
    substitute the cache-environment size ``L_k``.
    """

    service_sizes = {
        int(task_type_id): int(size_bits)
        for task_type_id, size_bits in ctx.get("service_size_bits", {}).items()
    }
    explicit = ctx.get("source_program_size_bits")
    if explicit is None:
        raise ValueError(
            "Context is missing source_program_size_bits. Rebuild the context "
            "with MiniSystemContextBuilder so runtime and benchmark use the "
            "same source-program sizes."
        )

    source_sizes = {
        int(task_type_id): int(size_bits)
        for task_type_id, size_bits in explicit.items()
    }

    missing = sorted(set(service_sizes) - set(source_sizes))
    if missing:
        raise ValueError(
            "Context source-program sizes are missing task types: "
            f"{missing}"
        )

    invalid = sorted(
        task_type_id
        for task_type_id in service_sizes
        if source_sizes.get(task_type_id, 0) <= 0
    )
    if invalid:
        raise ValueError(
            "Context source-program sizes must be positive for task types: "
            f"{invalid}"
        )

    ctx["source_program_size_bits"] = source_sizes

    # Scientific provenance: Eq. (27) of the 2025 article contains a separate
    # service-program transmission-energy term.  Its mathematical presence is
    # article-aligned; only the numerical source-program size is reconstructed.
    # These flags are metadata only and do not alter runtime behaviour.
    ctx["service_program_transfer_energy_enabled"] = True
    ctx["service_program_transfer_energy_article_equation"] = 27
    ctx["service_program_transfer_energy_structure_article_exact"] = True
    return ctx


def _apply_joint_paper_source_program_model(
    joint_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate one shared source-size definition in joint/per-app contexts."""

    for app_ctx in joint_ctx.get("applications", {}).values():
        _apply_paper_source_program_model(app_ctx)

    _apply_paper_source_program_model(joint_ctx)
    return joint_ctx

def _unique_ints(values: Iterable[int]) -> List[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _unique_names(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(value).strip().lower() for value in values))


def _metric_stats(values: Sequence[float]) -> Dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(mean(numbers)),
        "std": float(stdev(numbers)) if len(numbers) > 1 else 0.0,
        "min": float(min(numbers)),
        "max": float(max(numbers)),
    }


def _summary_row(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for metric in (
        "avg_delay",
        "avg_efficiency",
        "total_efficiency",
        "completion_rate",
    ):
        stats = _metric_stats([run["metrics"][metric] for run in selected])
        result[metric] = stats["mean"]
        result[f"{metric}_std"] = stats["std"]
        result[f"{metric}_min"] = stats["min"]
        result[f"{metric}_max"] = stats["max"]
    result["seed_count"] = len(selected)
    return result


def _json_cache(cache_state: Dict[int, set[int]]) -> Dict[str, List[int]]:
    return {
        str(int(sp_id)): sorted(int(task_type_id) for task_type_id in values)
        for sp_id, values in cache_state.items()
    }


def _run_convergence(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {
            "initial_best": 0.0,
            "final_best": 0.0,
            "absolute_gain": 0.0,
            "relative_gain_percent": 0.0,
            "improved_iterations": 0,
            "stagnated": True,
            "function_evaluations": 0,
        }

    best_values = [
        float(item.get("best_total_efficiency", 0.0))
        for item in history
    ]
    initial_best = best_values[0]
    final_best = best_values[-1]
    absolute_gain = final_best - initial_best
    denominator = max(abs(initial_best), 1e-12)
    improved_iterations = sum(
        1
        for previous, current in zip(best_values, best_values[1:])
        if current > previous + 1e-12
    )

    return {
        "initial_best": float(initial_best),
        "final_best": float(final_best),
        "absolute_gain": float(absolute_gain),
        "relative_gain_percent": float(100.0 * absolute_gain / denominator),
        "improved_iterations": int(improved_iterations),
        "stagnated": bool(improved_iterations == 0),
        "function_evaluations": int(
            history[-1].get("function_evaluations", 0)
        ),
    }


def _joint_summary(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for algorithm in dict.fromkeys(run["algorithm"] for run in runs):
        selected = [run for run in runs if run["algorithm"] == algorithm]
        row = _summary_row(selected)

        runtime_stats = _metric_stats(
            [float(run.get("runtime_seconds", 0.0)) for run in selected]
        )
        nfe_stats = _metric_stats(
            [float(run.get("final_function_evaluations", 0)) for run in selected]
        )
        gain_stats = _metric_stats(
            [
                float(run.get("convergence", {}).get("absolute_gain", 0.0))
                for run in selected
            ]
        )

        row["runtime_seconds"] = runtime_stats["mean"]
        row["runtime_seconds_std"] = runtime_stats["std"]
        row["function_evaluations"] = nfe_stats["mean"]
        row["function_evaluations_std"] = nfe_stats["std"]
        row["convergence_gain"] = gain_stats["mean"]
        row["convergence_gain_std"] = gain_stats["std"]
        row["stagnated_seed_count"] = sum(
            1
            for run in selected
            if bool(run.get("convergence", {}).get("stagnated", False))
        )

        result[algorithm] = row
    return result


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _raw_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in result.get("runs", []):
        metrics = run.get("metrics", {})
        for application in run.get("applications", []):
            rows.append(
                {
                    "algorithm": run.get("algorithm"),
                    "seed": run.get("seed"),
                    "application_id": application.get("application_id"),
                    "vehicle_id": application.get("vehicle_id"),
                    "delay_s": application.get("delay_s"),
                    "energy_j": application.get("energy_j"),
                    "efficiency": application.get("efficiency"),
                    "deadline_s": application.get("deadline_s"),
                    "completed": application.get("completed"),
                    "task_count": application.get("task_count"),
                    "optimized_task_count": application.get("optimized_task_count"),
                    "scheduled_task_count": application.get("scheduled_task_count"),
                    "entry_task_id": application.get("entry_task_id"),
                    "entry_provider_id": application.get("entry_provider_id"),
                    "providers_used": json.dumps(
                        application.get("providers_used", []),
                        ensure_ascii=False,
                    ),
                    "run_avg_delay": metrics.get("avg_delay"),
                    "run_avg_efficiency": metrics.get("avg_efficiency"),
                    "run_total_efficiency": metrics.get("total_efficiency"),
                    "run_completion_rate": metrics.get("completion_rate"),
                    "population_size": run.get("population_size", 0),
                    "tmax": run.get("tmax", result.get("tmax")),
                    "article_exact": run.get("article_exact", False),
                    "implementation": run.get("implementation", "article-aligned"),
                    "reference_doi": run.get("reference_doi", ""),
                    "runtime_seconds": run.get("runtime_seconds", 0.0),
                    "function_evaluations": run.get(
                        "final_function_evaluations",
                        0,
                    ),
                    "algorithm_article_exact": run.get("algorithm_article_exact", False),
                    "system_article_exact": run.get("system_article_exact", False),
                    "mission_vehicle_count": result.get("scenario", {}).get("mission_vehicle_count"),
                    "mean_deadline_s": result.get("scenario", {}).get("mean_deadline_s"),
                    "mean_vehicle_speed_kmh": result.get("scenario", {}).get("mean_vehicle_speed_kmh"),
                    "mean_mec_capacity_ghz": result.get("scenario", {}).get("mean_mec_capacity_ghz"),
                    "application_rate_per_second": result.get("scenario", {}).get("application_rate_per_second"),
                }
            )
    return rows


def _summary_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for algorithm, values in result.get("summary", {}).items():
        row: Dict[str, Any] = {"algorithm": algorithm}
        row.update(values)
        rows.append(row)
    return rows


def _iteration_diagnostics(item: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "accepted_candidates",
        "accepted_guided_trials",
        "accepted_cache_trials",
        "generated_trials",
        "unique_trial_count",
        "duplicate_trial_count",
        "mean_trial_hamming",
        "population_unique_count",
        "population_mean_hamming",
        "best_improved",
        "stagnation_generations",
    )
    return {key: item.get(key) for key in keys}


def _convergence_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in result.get("runs", []):
        for item in run.get("iteration_history", []):
            iteration = item.get("iteration")
            rows.append(
                {
                    "algorithm": run.get("algorithm"),
                    "seed": run.get("seed"),
                    "iteration": iteration,
                    "function_evaluations": item.get(
                        "function_evaluations",
                        0,
                    ),
                    "point_type": "best",
                    "population_index": None,
                    "total_efficiency": item.get("best_total_efficiency"),
                    **_iteration_diagnostics(item),
                }
            )
            for index, value in enumerate(
                item.get("population_total_efficiencies", [])
            ):
                rows.append(
                    {
                        "algorithm": run.get("algorithm"),
                        "seed": run.get("seed"),
                        "iteration": iteration,
                        "function_evaluations": item.get(
                            "function_evaluations",
                            0,
                        ),
                        "point_type": "population",
                        "population_index": index,
                        "total_efficiency": value,
                        **_iteration_diagnostics(item),
                    }
                )
    return rows


def _relative_path(path: Path) -> str:
    root = Path(settings.BASE_DIR).resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _bar_plot(
    summary_rows: List[Dict[str, Any]],
    metric: str,
    title: str,
    y_label: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["algorithm"]) for row in summary_rows]
    values = [float(row.get(metric, 0.0)) for row in summary_rows]
    errors = [float(row.get(f"{metric}_std", 0.0)) for row in summary_rows]
    figure, axis = plt.subplots()
    axis.bar(labels, values, yerr=errors, capsize=4)
    axis.set_title(title)
    axis.set_xlabel("Algorithm")
    axis.set_ylabel(y_label)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _convergence_plot(rows: List[Dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_grouped: Dict[str, Dict[float, List[float]]] = {}
    population_grouped: Dict[str, List[Tuple[float, float]]] = {}
    for row in rows:
        algorithm = str(row["algorithm"])
        iteration = float(row["iteration"])
        value = float(row["total_efficiency"])
        if row.get("point_type") == "population":
            population_grouped.setdefault(algorithm, []).append((iteration, value))
        else:
            best_grouped.setdefault(algorithm, {}).setdefault(
                iteration, []
            ).append(value)

    figure, axis = plt.subplots()
    for algorithm, points in population_grouped.items():
        axis.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            s=8,
            alpha=0.35,
            label=f"{algorithm} population",
        )
    for algorithm, iteration_values in best_grouped.items():
        points = sorted(iteration_values.items())
        axis.plot(
            [item[0] for item in points],
            [mean(item[1]) for item in points],
            marker="o",
            label=f"{algorithm} best",
        )
    axis.set_title("Benchmark convergence")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Total offloading efficiency")
    axis.grid(alpha=0.3)
    if best_grouped or population_grouped:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)



def _scenario_metadata(joint_ctx: Dict[str, Any]) -> Dict[str, Any]:
    params = load_params_obj()
    application_contexts = joint_ctx.get("applications", {})
    deadlines = [
        float(app_ctx.get("deadline_s", 0.0))
        for app_ctx in application_contexts.values()
    ]
    rsu_frequencies_by_provider: Dict[int, float] = {}
    for app_ctx in application_contexts.values():
        sp_types = app_ctx.get("sp_types", {})
        for sp_id, value in app_ctx.get("sp_cpu_fmax_hz", {}).items():
            sp_id = int(sp_id)
            if sp_types.get(sp_id) != "rsu":
                continue
            frequency = float(value)
            previous = rsu_frequencies_by_provider.get(sp_id)
            if previous is not None and previous != frequency:
                raise ValueError(
                    f"Provider {sp_id} has inconsistent MEC capacity across contexts"
                )
            rsu_frequencies_by_provider[sp_id] = frequency
    rsu_frequencies = list(rsu_frequencies_by_provider.values())
    speed_min = float(getattr(params, "vehicle_speed_min_kmh", 60.0))
    speed_max = float(getattr(params, "vehicle_speed_max_kmh", 80.0))
    metadata = {
        "mission_vehicle_count": len(joint_ctx.get("application_ids", [])),
        "mean_deadline_s": float(mean(deadlines)) if deadlines else 0.0,
        "deadlines_s": sorted(set(deadlines)),
        "vehicle_speed_min_kmh": speed_min,
        "vehicle_speed_max_kmh": speed_max,
        "mean_vehicle_speed_kmh": (speed_min + speed_max) / 2.0,
        "application_rate_per_second": float(
            getattr(params, "application_rate_per_second", 10.0)
        ),
        "mean_mec_capacity_ghz": (
            float(mean(rsu_frequencies)) / 1e9
            if rsu_frequencies
            else 0.0
        ),
    }
    metadata.update(copy.deepcopy(joint_ctx.get("scenario_metadata", {})))
    return metadata

def _export_benchmark_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    root = Path(settings.BASE_DIR) / "benchmark_results"
    run_id = (
        f"joint_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid4().hex[:8]}"
    )
    output_dir = root / run_id
    warnings: List[str] = []

    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        raw_rows = _raw_rows(result)
        summary_rows = _summary_rows(result)
        convergence_rows = _convergence_rows(result)

        raw_path = output_dir / "raw_results.csv"
        summary_path = output_dir / "summary.csv"
        convergence_path = output_dir / "convergence.csv"
        json_path = output_dir / "benchmark_result.json"

        _write_csv(
            raw_path,
            raw_rows,
            [
                "algorithm",
                "seed",
                "application_id",
                "vehicle_id",
                "delay_s",
                "energy_j",
                "efficiency",
                "deadline_s",
                "completed",
                "task_count",
                "optimized_task_count",
                "scheduled_task_count",
                "entry_task_id",
                "entry_provider_id",
                "providers_used",
                "run_avg_delay",
                "run_avg_efficiency",
                "run_total_efficiency",
                "run_completion_rate",
                "population_size",
                "tmax",
                "article_exact",
                "implementation",
                "reference_doi",
                "algorithm_article_exact",
                "system_article_exact",
                "mission_vehicle_count",
                "mean_deadline_s",
                "mean_vehicle_speed_kmh",
                "mean_mec_capacity_ghz",
                "application_rate_per_second",
            ],
        )
        summary_fieldnames = ["algorithm"]
        if summary_rows:
            summary_fieldnames.extend(
                key for key in summary_rows[0] if key != "algorithm"
            )
        _write_csv(summary_path, summary_rows, summary_fieldnames)
        _write_csv(
            convergence_path,
            convergence_rows,
            [
                "algorithm",
                "seed",
                "iteration",
                "point_type",
                "population_index",
                "total_efficiency",
            ],
        )
        with json_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, default=str)

        figure_paths: List[Path] = []
        try:
            if summary_rows:
                plot_specs = (
                    (
                        "total_efficiency",
                        "Total efficiency by algorithm",
                        "Total efficiency",
                        "total_efficiency.png",
                    ),
                    (
                        "avg_delay",
                        "Average delay by algorithm",
                        "Average delay (s)",
                        "avg_delay.png",
                    ),
                    (
                        "completion_rate",
                        "Completion rate by algorithm",
                        "Completion rate",
                        "completion_rate.png",
                    ),
                )
                for metric, title, y_label, filename in plot_specs:
                    figure_path = output_dir / filename
                    _bar_plot(summary_rows, metric, title, y_label, figure_path)
                    figure_paths.append(figure_path)
            if convergence_rows:
                convergence_figure = output_dir / "convergence.png"
                _convergence_plot(convergence_rows, convergence_figure)
                figure_paths.append(convergence_figure)
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            warnings.append(f"Chart generation failed: {exc}")

        return {
            "created": True,
            "run_id": run_id,
            "output_directory": _relative_path(output_dir),
            "raw_csv": _relative_path(raw_path),
            "summary_csv": _relative_path(summary_path),
            "convergence_csv": _relative_path(convergence_path),
            "result_json": _relative_path(json_path),
            "figures": [_relative_path(path) for path in figure_paths],
            "sweep_csv": None,
            "sweep_figures": [],
            "warnings": warnings,
        }
    except OSError as exc:
        return {
            "created": False,
            "run_id": run_id,
            "output_directory": _relative_path(output_dir),
            "figures": [],
            "warnings": [f"Benchmark export failed: {exc}"],
        }


def run_joint_context_benchmark(
    joint_ctx: Dict[str, Any],
    *,
    algorithms: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    tmax: int = 10,
    population_size: int | None = None,
    max_function_evaluations: int | None = None,
    export_artifacts: bool = True,
) -> Dict[str, Any]:
    # Work on an isolated copy. Worker/simulation execution is separate from
    # this paper-benchmark runner and does not pass through this adaptation.
    joint_ctx = _apply_joint_paper_source_program_model(copy.deepcopy(joint_ctx))
    application_ids = _unique_ints(joint_ctx.get("application_ids", []))
    algorithm_names = _unique_names(algorithms or PAPER_ALGORITHM_NAMES)
    seed_values = _unique_ints(seeds or [1])
    if not application_ids:
        raise ValueError("Joint context has no applications")
    if not algorithm_names:
        raise ValueError("At least one algorithm is required")
    if not seed_values:
        raise ValueError("At least one seed is required")
    unknown = [name for name in algorithm_names if name not in ALGORITHMS]
    if unknown:
        raise ValueError(f"Unsupported joint algorithms: {unknown}")
    actual_population_size = int(
        population_size if population_size is not None else load_params_obj().S
    )
    if actual_population_size <= 0:
        raise ValueError("Population size must be positive")
    if (
        max_function_evaluations is not None
        and int(max_function_evaluations) < actual_population_size
    ):
        raise ValueError(
            "max_function_evaluations must be at least population_size"
        )
    runs: List[Dict[str, Any]] = []

    for seed in seed_values:
        for algorithm_name in algorithm_names:
            algorithm = ALGORITHMS[algorithm_name]
            started_at = perf_counter()
            best_nest, evaluation, history = algorithm.run_joint(
                copy.deepcopy(joint_ctx),
                seed=int(seed),
                tmax=max(1, int(tmax)),
                population_size=population_size,
                max_function_evaluations=(
                    int(max_function_evaluations)
                    if max_function_evaluations is not None
                    and algorithm_name != "dtosc"
                    else None
                ),
            )
            runtime_seconds = perf_counter() - started_at
            uses_population = algorithm_name != "dtosc"
            result = JointAlgorithmResult(
                algorithm=algorithm_name,
                seed=int(seed),
                population_size=actual_population_size if uses_population else 0,
                tmax=max(1, int(tmax)) if uses_population else 1,
                total_efficiency=float(evaluation.total_efficiency),
                applications=evaluation.applications,
                metrics=evaluation.metrics,
                iteration_history=history,
                scientific_status="article-aligned-with-declared-limitations",
            ).to_dict()
            result["runtime_seconds"] = float(runtime_seconds)
            result["convergence"] = _run_convergence(history)
            result["final_function_evaluations"] = int(
                result["convergence"]["function_evaluations"]
            )
            result["executed_iterations"] = max(0, len(history) - 1)
            result["evaluation_budget"] = (
                None
                if max_function_evaluations is None or not uses_population
                else int(max_function_evaluations)
            )
            result["evaluation_budget_exhausted"] = bool(
                uses_population
                and max_function_evaluations is not None
                and result["final_function_evaluations"]
                >= int(max_function_evaluations)
            )
            result["best_nest"] = [
                {
                    "joint_task_id": int(task_id),
                    "provider_id": int(provider_id),
                    "rank": int(rank),
                }
                for task_id, provider_id, rank in best_nest
            ]
            result["schedule"] = evaluation.schedule
            result["cache_state"] = _json_cache(evaluation.cache_state)
            result["algorithm_article_exact"] = bool(
                getattr(algorithm, "article_exact", False)
            )
            result["algorithm_complete"] = bool(
                getattr(algorithm, "algorithm_complete", True)
            )
            result["system_article_exact"] = False
            result["article_exact"] = False
            if algorithm_name == "dtosc":
                result["implementation"] = str(
                    getattr(algorithm, "implementation", "unknown")
                )
                result["reference_doi"] = str(
                    getattr(algorithm, "reference_doi", "")
                )
                result["dynamic_programming_complete"] = bool(
                    getattr(algorithm, "dynamic_programming_complete", False)
                )
                result["dynamic_programming_scope"] = str(
                    getattr(algorithm, "dynamic_programming_scope", "")
                )
                result["exit_policy"] = str(
                    getattr(algorithm, "exit_policy", "")
                )
                result["source_exact_verified"] = bool(
                    getattr(algorithm, "source_exact_verified", False)
                )
                result["reference_alignment"] = str(
                    getattr(algorithm, "reference_alignment", "")
                )
            elif algorithm_name == "dcsga":
                result["implementation"] = "article-aligned-dcsga"
                result["reference_doi"] = "10.1109/TVT.2025.3540639"
                result["rank_seed_aligned"] = True
                result["rank_seed"] = int(seed)
            else:
                result["implementation"] = str(
                    getattr(algorithm, "implementation", algorithm_name)
                )
                result["reference_doi"] = str(
                    getattr(algorithm, "reference_doi", "")
                )
            runs.append(result)

    scenario = _scenario_metadata(joint_ctx)
    budget_incomplete = [
        run["algorithm"]
        for run in runs
        if run.get("evaluation_budget") is not None
        and not bool(run.get("evaluation_budget_exhausted", False))
    ]
    speed_density_exact = bool(
        scenario.get("article_speed_density_model", False)
    )
    benchmark_result: Dict[str, Any] = {
        "scientific_stage": "article-aligned-with-declared-limitations",
        "article_alignment": {
            "joint_multi_application_fitness": True,
            "shared_provider_queues": True,
            "entry_tasks_fixed_local": True,
            "entry_tasks_excluded_from_nest": True,
            "application_specific_alpha_beta": True,
            "shared_cache_during_joint_schedule": True,
            "five_paper_schemes_exposed": True,
            "to_v2i_rsu_only": True,
            "to_wo_r_uses_dag_order_without_urgency": True,
            "to_wo_c_disables_cache_resources": True,
            "article_exact_channel_model": False,
            "article_exact_cpu_allocation": True,
            "article_exact_power_sender_model": False,
            "article_cache_update_persistence_timing": True,
            "source_program_separated_from_cache_environment": True,
            "service_program_transfer_energy_enabled": bool(
                joint_ctx.get("service_program_transfer_energy_enabled", True)
            ),
            "service_program_transfer_energy_article_equation": int(
                joint_ctx.get("service_program_transfer_energy_article_equation", 27)
            ),
            "service_program_transfer_energy_structure_article_exact": bool(
                joint_ctx.get(
                    "service_program_transfer_energy_structure_article_exact",
                    True,
                )
            ),
            "source_program_size_article_exact": False,
            "source_program_size_model": joint_ctx.get(
                "source_program_size_model"
            ),
            "source_program_size_ratio": joint_ctx.get(
                "source_program_size_ratio"
            ),
            "source_program_size_reference_doi": joint_ctx.get(
                "source_program_size_reference_doi"
            ),
            "article_vehicle_speed_parameterization": speed_density_exact,
            "article_application_arrival_rate": False,
            "dtosc_provider_path_dynamic_programming": True,
            "dtosc_cache_knapsack_dynamic_programming": True,
            "dtosc_legacy_pre_repair_baseline": True,
            "dtosc_2022_policy_adapted_to_2025_model": False,
            "dtosc_source_exact_verified": False,
            "article_exact_dtosc": False,
            "article_exact_system": False,
        },
        "warnings": [
            "DTOSC uses the legacy pre-repair stage-wise provider-path dynamic-programming reconstruction. It is retained for reproducibility/sensitivity and is not claimed to be source-exact DTOSC 2022.",
            "Equation (27) of the 2025 paper explicitly includes energy for transmitting service programs. The paper does not publish the numerical source-program data size, so the current 0.1 * L_k source-size value remains a declared DTOSC-2022-derived reconstruction. The benchmark keeps the Eq. (27) energy term enabled; cache capacity still uses L_k and compile/install work still uses W_k.",
            "The V2V channel and sender-side power model remain declared approximations.",
            "The 2025 paper reports 10 applications/s but does not specify how temporal arrivals are integrated into Figures 6-10; this static joint benchmark therefore does not invent an arrival process.",
        ] + (
            [
                "The requested objective-evaluation budget was not exhausted by: "
                + ", ".join(budget_incomplete)
                + ". Increase tmax; final algorithm comparisons are not budget-matched until every population algorithm reaches the same NFE cap."
            ]
            if budget_incomplete
            else []
        ),
        "context": joint_context_summary(joint_ctx),
        "scenario": scenario,
        "application_ids": application_ids,
        "algorithms": algorithm_names,
        "seeds": seed_values,
        "tmax": max(1, int(tmax)),
        "population_size": actual_population_size,
        "population_size_override": population_size,
        "max_function_evaluations": (
            None
            if max_function_evaluations is None
            else int(max_function_evaluations)
        ),
        "evaluation_budget_fully_consumed": not bool(budget_incomplete),
        "runs": runs,
        "summary": _joint_summary(runs),
    }
    if export_artifacts:
        benchmark_result["artifacts"] = _export_benchmark_result(
            benchmark_result,
        )
    return benchmark_result
