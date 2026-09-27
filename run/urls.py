from django.urls import path

from .views import (
    paper_benchmark,
    reset_simulation,
    simulation_status,
    start_simulation,
    stop_simulation,
)


urlpatterns = [
    path(
        "start-simulation/",
        start_simulation,
        name="start_simulation",
    ),
    path(
        "stop-simulation/",
        stop_simulation,
        name="stop_simulation",
    ),
    path(
        "status/",
        simulation_status,
        name="simulation_status",
    ),
    path(
        "reset/",
        reset_simulation,
        name="reset_simulation",
    ),
    path(
        "paper-benchmark/",
        paper_benchmark,
        name="paper_benchmark",
    ),
]
