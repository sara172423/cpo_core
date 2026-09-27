from __future__ import annotations

import threading

from drf_spectacular.utils import extend_schema, inline_serializer
from rest_framework import serializers, status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from run.simulation import (
    run_simulation,
    stop_simulation as stop_simulation_fn,
    simulation_status as status_fn,
)
from run.simulation.service import reset_simulation as reset_fn
from run.benchmark.experiments import run_paper_experiment
from .serializer import (
    PaperExperimentRequestSerializer,
    StartSimulationSerializer,
)


@extend_schema(
    request=StartSimulationSerializer,
    responses=inline_serializer(
        name="StartSimulationResponse",
        fields={
            "status": serializers.CharField(),
            "running": serializers.BooleanField(required=False),
            "stopping": serializers.BooleanField(required=False),
            "alive_workers": serializers.ListField(
                child=serializers.CharField(),
                required=False,
            ),
        },
    ),
)
@api_view(["POST"])
def start_simulation(request):
    if status_fn().get("running"):
        return Response(
            {"status": "already running", **status_fn()},
            status=status.HTTP_200_OK,
        )

    ser = StartSimulationSerializer(data=request.data)
    ser.is_valid(raise_exception=True)

    tmax = ser.validated_data.get("tmax", 10)

    t = threading.Thread(
        target=run_simulation,
        kwargs={"tmax": tmax},
        daemon=True,
    )
    t.start()

    return Response(
        {"status": "started", **status_fn()},
        status=status.HTTP_200_OK,
    )


@extend_schema(
    responses=inline_serializer(
        name="StopSimulationResponse",
        fields={
            "status": serializers.CharField(),
            "running": serializers.BooleanField(required=False),
            "stopping": serializers.BooleanField(required=False),
            "alive_workers": serializers.ListField(
                child=serializers.CharField(),
                required=False,
            ),
        },
    )
)
@api_view(["POST"])
def stop_simulation(request):
    stop_result = stop_simulation_fn()
    current_status = status_fn()
    response_status = (
        "stopped"
        if stop_result.get("stopped", False)
        else "stopping"
    )

    return Response(
        {
            "status": response_status,
            "alive_workers": stop_result.get("alive_workers", []),
            **current_status,
        },
        status=status.HTTP_200_OK,
    )


@extend_schema(
    responses=inline_serializer(
        name="SimulationStatusResponse",
        fields={
            "running": serializers.BooleanField(),
            "stopping": serializers.BooleanField(required=False),
        },
    )
)
@api_view(["GET"])
def simulation_status(request):
    return Response(status_fn(), status=status.HTTP_200_OK)


@extend_schema(
    responses=inline_serializer(
        name="ResetSimulationResponse",
        fields={"status": serializers.CharField()},
    )
)
@api_view(["POST"])
def reset_simulation(request):
    stop_result = stop_simulation_fn(wait_timeout=10.0)

    if not stop_result.get("stopped", False):
        return Response(
            {
                "status": "stopping",
                "detail": "Simulation workers are still stopping. Retry reset shortly.",
                "alive_workers": stop_result.get("alive_workers", []),
            },
            status=status.HTTP_409_CONFLICT,
        )

    reset_fn()
    return Response({"status": "reset"}, status=status.HTTP_200_OK)
@extend_schema(
    request=PaperExperimentRequestSerializer,
    responses=serializers.DictField(),
)
@api_view(["POST"])
def paper_benchmark(request):
    if status_fn().get("running"):
        return Response(
            {
                "detail": (
                    "Stop the simulation before running the paper experiment."
                )
            },
            status=status.HTTP_409_CONFLICT,
        )
    ser = PaperExperimentRequestSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    try:
        result = run_paper_experiment(
            figure=ser.validated_data["figure"],
            repetitions=ser.validated_data.get("repetitions"),
            seed_start=ser.validated_data.get("seed_start", 1),
            tmax=ser.validated_data.get("tmax", 15),
            population_size=ser.validated_data.get("population_size"),
            max_function_evaluations=ser.validated_data.get(
                "max_function_evaluations"
            ),
            experiment_mode=ser.validated_data.get("experiment_mode"),
            algorithms=ser.validated_data.get("algorithms"),
            diagnostic_vehicle_count=ser.validated_data.get(
                "diagnostic_vehicle_count"
            ),
            diagnostic_road_vehicle_count=ser.validated_data.get(
                "diagnostic_road_vehicle_count"
            ),
            diagnostic_sweep_values=ser.validated_data.get(
                "diagnostic_sweep_values"
            ),
            export_artifacts=ser.validated_data.get("export_artifacts", True),
        )
    except (ValueError, KeyError, TypeError) as exc:
        return Response(
            {"detail": str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if ser.validated_data.get("summary_only", False):
        result = {
            key: result.get(key)
            for key in (
                "figure",
                "article_doi",
                "repetitions",
                "seeds",
                "tmax",
                "population_size",
                "max_function_evaluations",
                "experiment_mode",
                "stopping_rule",
                "algorithms",
                "comparison_mode",
                "diagnostic_mode",
                "diagnostic_vehicle_count",
                "diagnostic_road_vehicle_count",
                "diagnostic_sweep_values",
                "reconstruction_audit",
                "convergence_diagnostics",
                "statistical_protocol",
                "pairwise_statistics",
                "summary",
                "artifacts",
            )
            if key in result
        }
    return Response(result, status=status.HTTP_200_OK)
