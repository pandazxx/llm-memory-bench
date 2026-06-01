"""The System plugin interface.

A System provides the two core operations of the framework — Memory
construction and Memory retrieval — plus visualization and a freeze/re-render
path so visuals can be iterated on without re-calling the LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import Query, QueryResult, Statement


class System(ABC):
    """One concrete subclass per memory system (A-Mem, HippoRAG, ...)."""

    #: Registry key / results subdirectory name and CLI id.
    name: str = "system"
    #: Human-readable label shown in visualizations.
    label: str = "System"
    #: Default parameters; a param-set is overlaid on top in ``__init__``.
    DEFAULT_PARAMS: dict = {}

    def __init__(self, params: dict | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    @abstractmethod
    def construct_memory(self, statements: list[Statement]) -> None:
        """Build memory from the dataset's statements (order matters)."""

    @abstractmethod
    def retrieve(self, query: Query) -> QueryResult:
        """Answer one query, returning retrieved ids, answer, and a trail."""

    @abstractmethod
    def render_memory(self, out_dir: Path) -> None:
        """Write the memory-structure visualization to ``out_dir`` (index.html + assets)."""

    @abstractmethod
    def render_retrieval(
        self, results: list[QueryResult], out_dir: Path, scores: list[dict] | None = None
    ) -> None:
        """Write the retrieval-trail visualization to ``out_dir``.

        ``scores`` (per-query dicts from ``scoring.QueryScore``) drive the
        expected-vs-actual + precision/recall/F1 comparison table when present."""

    @abstractmethod
    def dump_memory_state(self, out_dir: Path) -> None:
        """Freeze the constructed memory to ``out_dir/state.json`` for re-rendering."""

    @classmethod
    @abstractmethod
    def render_from_state(cls, state_dir: Path, out_dir: Path) -> None:
        """Re-emit the memory visualization from a frozen state.json. No LLM calls."""

    @classmethod
    @abstractmethod
    def render_retrieval_from_state(
        cls,
        state_dir: Path,
        results: list[QueryResult],
        out_dir: Path,
        scores: list[dict] | None = None,
    ) -> None:
        """Re-emit the retrieval visualization from a frozen state.json + results. No LLM calls."""
