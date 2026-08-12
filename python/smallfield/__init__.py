"""SmallField: GPU conformer generation with ETKDG and MMFF94."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.3.0"

_submodules = frozenset({"api", "log", "tools"})

__all__ = ["__version__", "api", "log", "tools"]


def __getattr__(name: str) -> Any:
    """Resolve a submodule on first attribute access.

    ``api`` binds the compiled CUDA extension at import time, so importing it eagerly here would
    make ``smallfield.log`` — which the parquet tooling and the CPU test job reach for — require a
    built extension. ``pyproject.toml`` puts ``python/`` on the path precisely so that a clone
    resolves without a build step, and only laziness keeps that true.
    """

    if name in _submodules:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
