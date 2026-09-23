"""DTOSC dynamic-programming baseline."""

__all__ = ["dtosc_run"]


def __getattr__(name):
    if name == "dtosc_run":
        from .core import dtosc_run

        return dtosc_run
    raise AttributeError(name)
