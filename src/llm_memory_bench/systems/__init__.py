"""System registry: maps a name to its System subclass."""

from __future__ import annotations

from ..core.system import System
from .amem import AMem
from .amem_lc import AMemLateChunk
from .hipporag import HippoRAG
from .hipporag2 import HippoRAG2

REGISTRY: dict[str, type[System]] = {
    AMem.name: AMem,
    AMemLateChunk.name: AMemLateChunk,
    HippoRAG.name: HippoRAG,
    HippoRAG2.name: HippoRAG2,
}


def get_system(name: str) -> type[System]:
    if name not in REGISTRY:
        raise KeyError(f"unknown system {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]
