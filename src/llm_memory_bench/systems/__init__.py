"""System registry: maps a name to its System subclass."""

from __future__ import annotations

from ..core.system import System
from .amem import AMem

REGISTRY: dict[str, type[System]] = {
    AMem.name: AMem,
}


def get_system(name: str) -> type[System]:
    if name not in REGISTRY:
        raise KeyError(f"unknown system {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]
