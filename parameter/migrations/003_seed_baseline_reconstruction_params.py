from django.db import migrations


def seed_baseline_reconstruction_params(apps, schema_editor):
    Parameter = apps.get_model("parameter", "Parameter")
    defaults = [
        (
            "deadline_sampling_weights",
            [5, 10, 15, 20, 20, 15, 10, 5],
            "json",
            "-",
        ),
        ("source_program_fraction", 0.10, "float", "-"),
    ]
    for key, value, value_type, unit in defaults:
        Parameter.objects.update_or_create(
            key=key,
            defaults={
                "value": value,
                "value_type": value_type,
                "unit": unit,
            },
        )


def unseed_baseline_reconstruction_params(apps, schema_editor):
    Parameter = apps.get_model("parameter", "Parameter")
    Parameter.objects.filter(
        key__in=["deadline_sampling_weights", "source_program_fraction"]
    ).delete()


class Migration(migrations.Migration):
    dependencies = [
        ("parameter", "002_seed_default_params"),
    ]

    operations = [
        migrations.RunPython(
            seed_baseline_reconstruction_params,
            unseed_baseline_reconstruction_params,
        ),
    ]
