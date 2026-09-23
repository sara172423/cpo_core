"""Paper DCSGA/Cuckoo Search implementation.

Import the explicit submodule needed by the caller; package import itself has
no database or Django side effect.
"""

__all__ = ["dcsga_run"]


def __getattr__(name):
    if name == "dcsga_run":
        from .core import dcsga_run

        return dcsga_run
    raise AttributeError(name)
