from rest_framework import serializers


class StrictFieldsSerializer(serializers.Serializer):
    """Reject misspelled/unknown request fields instead of silently ignoring them."""

    def to_internal_value(self, data):
        if hasattr(data, "keys"):
            unknown = sorted(set(data.keys()) - set(self.fields.keys()))
            if unknown:
                raise serializers.ValidationError({
                    key: ["Unknown field."]
                    for key in unknown
                })
        return super().to_internal_value(data)


class StartSimulationSerializer(serializers.Serializer):
    tmax = serializers.IntegerField(
        required=False,
        min_value=1,
        max_value=1000,
        default=10,
    )


class SimulationConfigSerializer(serializers.Serializer):
    total_time = serializers.IntegerField()
    tick_seconds = serializers.IntegerField()
    cell_radius_rsu = serializers.FloatField()


class SimulationWorkersSerializer(serializers.Serializer):
    vehicle = serializers.IntegerField()
    rsu = serializers.IntegerField()
    task_generator = serializers.BooleanField()
    context = serializers.BooleanField()
    status = serializers.BooleanField()


class SimulationCountsSerializer(serializers.Serializer):
    vehicles = serializers.IntegerField()
    rsus = serializers.IntegerField()
    rsu_vehicle_total = serializers.IntegerField()
    rsu_vehicle_open = serializers.IntegerField()
    applications = serializers.IntegerField()
    applications_in_progress = serializers.IntegerField()
    tasks = serializers.IntegerField()
    taskexecutions = serializers.IntegerField()
    cache_items = serializers.IntegerField()


class SimulationContextSerializer(serializers.Serializer):
    ok = serializers.IntegerField()
    fail = serializers.IntegerField()
    sample = serializers.DictField(required=False, allow_null=True)
    vehicles = serializers.ListField(child=serializers.DictField(), required=False)
    sim_time_s = serializers.FloatField(required=False, allow_null=True)
    base_time = serializers.DateTimeField(required=False, allow_null=True)
    base_time_type = serializers.CharField(required=False, allow_null=True)
    last_error = serializers.CharField(required=False, allow_null=True)


class SimulationStatusSerializer(serializers.Serializer):
    running = serializers.BooleanField()
    stopping = serializers.BooleanField(required=False)
    workers = SimulationWorkersSerializer()
    counts = SimulationCountsSerializer()
    context = SimulationContextSerializer()
    cfg = SimulationConfigSerializer(required=False, allow_null=True)
    snapshot_ts = serializers.CharField(required=False, allow_null=True)
class BenchmarkRequestSerializer(StrictFieldsSerializer):
    application_ids = serializers.ListField(
        child=serializers.IntegerField(min_value=1),
        allow_empty=False,
    )

    algorithms = serializers.ListField(
        child=serializers.ChoiceField(
            choices=[
                "dcsga",
                "dtosc",
                "to_v2i",
                "to_wo_c",
                "to_wo_r",
                "gwo_aco",
                "gwo",
                "gpc",
                "cpo",
                "dcpo_base",
                "dcpo_criticality",
                "dcpo_cache",
                "puma",
            ]
        ),
        required=False,
        allow_empty=False,
    )
    seeds = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        allow_empty=False,
        default=[1],
    )

    tmax = serializers.IntegerField(
        required=False,
        min_value=1,
        max_value=1000,
        default=10,
    )

    population_size = serializers.IntegerField(
        required=False,
        min_value=2,
        max_value=500,
        allow_null=True,
    )

    max_function_evaluations = serializers.IntegerField(
        required=False,
        min_value=2,
        max_value=1000000,
        allow_null=True,
    )

    export_artifacts = serializers.BooleanField(
        required=False,
        default=True,
    )

    summary_only = serializers.BooleanField(
        required=False,
        default=False,
    )

    def validate(self, attrs):
        algorithms = attrs.get("algorithms") or []
        population_size = attrs.get("population_size")
        if (
            population_size is not None
            and ({"gwo", "gwo_aco"} & set(algorithms))
            and int(population_size) < 3
        ):
            raise serializers.ValidationError({
                "population_size": [
                    "gwo requires population_size >= 3 for alpha, beta, and delta leaders."
                ]
            })
        return attrs

class PaperExperimentRequestSerializer(StrictFieldsSerializer):
    figure = serializers.ChoiceField(
        choices=[
            "figure_6",
            "figure_7",
            "figure_8",
            "figure_9",
            "figure_10",
            "all",
            "*",
        ]
    )
    algorithms = serializers.ListField(
        child=serializers.ChoiceField(
            choices=[
                "dcsga",
                "dtosc",
                "to_v2i",
                "to_wo_c",
                "to_wo_r",
                "gwo_aco",
                "gwo",
                "gpc",
                "cpo",
                "dcpo_base",
                "dcpo_criticality",
                "dcpo_cache",
                "puma",
            ]
        ),
        required=False,
        allow_empty=False,
    )
    experiment_mode = serializers.ChoiceField(
        choices=["paper_reproduction", "fair_optimizer_comparison"],
        required=False,
        allow_null=True,
    )
    repetitions = serializers.IntegerField(
        required=False,
        min_value=1,
        max_value=100,
    )
    seed_start = serializers.IntegerField(
        required=False,
        default=1,
    )
    tmax = serializers.IntegerField(
        required=False,
        min_value=1,
        max_value=1000,
        default=15,
    )
    population_size = serializers.IntegerField(
        required=False,
        min_value=2,
        max_value=500,
        allow_null=True,
    )
    max_function_evaluations = serializers.IntegerField(
        required=False,
        min_value=2,
        max_value=1000000,
        allow_null=True,
    )
    diagnostic_vehicle_count = serializers.IntegerField(
        required=False,
        min_value=2,
        max_value=76,
        allow_null=True,
    )
    diagnostic_road_vehicle_count = serializers.IntegerField(
        required=False,
        min_value=2,
        max_value=500,
        allow_null=True,
    )
    diagnostic_sweep_values = serializers.ListField(
        child=serializers.FloatField(),
        required=False,
        allow_empty=False,
        allow_null=True,
    )
    export_artifacts = serializers.BooleanField(
        required=False,
        default=True,
    )
    summary_only = serializers.BooleanField(
        required=False,
        default=False,
    )

    def validate(self, attrs):
        figure = attrs.get("figure")
        algorithms = attrs.get("algorithms") or []
        population_size = attrs.get("population_size")
        diagnostic_vehicle_count = attrs.get("diagnostic_vehicle_count")
        diagnostic_road_vehicle_count = attrs.get("diagnostic_road_vehicle_count")
        diagnostic_sweep_values = attrs.get("diagnostic_sweep_values")
        experiment_mode = attrs.get("experiment_mode")
        max_function_evaluations = attrs.get("max_function_evaluations")

        if (
            population_size is not None
            and ({"gwo", "gwo_aco"} & set(algorithms))
            and int(population_size) < 3
        ):
            raise serializers.ValidationError({
                "population_size": [
                    "gwo requires population_size >= 3 for alpha, beta, and delta leaders."
                ]
            })

        if figure in {"all", "*"} and (
            diagnostic_vehicle_count is not None
            or diagnostic_road_vehicle_count is not None
            or diagnostic_sweep_values is not None
        ):
            raise serializers.ValidationError(
                "Diagnostic vehicle overrides must target one figure, not figure='all'."
            )

        if (
            diagnostic_vehicle_count is not None
            and figure not in {"figure_6", "figure_7"}
        ):
            raise serializers.ValidationError({
                "diagnostic_vehicle_count": [
                    "Supported only for figure_6 and figure_7."
                ]
            })

        if diagnostic_road_vehicle_count is not None and figure == "figure_9":
            raise serializers.ValidationError({
                "diagnostic_road_vehicle_count": [
                    "Figure 9 derives road population from the published speed-density rule."
                ]
            })

        if (
            diagnostic_sweep_values is not None
            and figure not in {"figure_9", "figure_10"}
        ):
            raise serializers.ValidationError({
                "diagnostic_sweep_values": [
                    "Supported only for figure_9 and figure_10 laptop smoke tests."
                ]
            })

        added_optimizers = {
            "gpc", "gwo", "gwo_aco", "cpo", "dcpo_base",
            "dcpo_criticality", "dcpo_cache", "puma",
        }
        if experiment_mode == "paper_reproduction" and set(algorithms) & added_optimizers:
            raise serializers.ValidationError({
                "experiment_mode": [
                    "Added optimizers and CPO ablations are not algorithms printed in the original figure. Use fair_optimizer_comparison."
                ]
            })
        if (
            (
                experiment_mode == "fair_optimizer_comparison"
                or set(algorithms) & added_optimizers
            )
            and max_function_evaluations is None
        ):
            raise serializers.ValidationError({
                "max_function_evaluations": [
                    "Required for a fair comparison of population optimizers."
                ]
            })
        if experiment_mode == "fair_optimizer_comparison" and not algorithms:
            raise serializers.ValidationError({
                "algorithms": [
                    "List the population optimizers explicitly in fair comparison mode."
                ]
            })
        if figure in {"all", "*"} and experiment_mode == "fair_optimizer_comparison":
            raise serializers.ValidationError({
                "figure": ["Run fair optimizer comparisons one figure at a time."]
            })

        return attrs
