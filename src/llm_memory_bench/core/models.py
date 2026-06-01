"""Core data models shared across the framework.

These mirror the README's Dataset / Test-set / Query-result concepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Statement:
    """One Dataset entry. ``comment`` is human-only and ignored by systems."""

    id: str  # S00001
    statement: str
    timestamp: str | None = None
    comment: str | None = None

    def to_memory_item(self) -> dict[str, Any]:
        """Shape a system expects for ingestion."""
        return {"id": self.id, "content": self.statement, "timestamp": self.timestamp}


@dataclass
class Query:
    """One Test-set entry."""

    id: str  # Q00001
    query: str
    required_retrieval: list[str] = field(default_factory=list)
    expected_answer: str | None = None
    comment: str | None = None


@dataclass
class QueryResult:
    """Result of Memory retrieval for one query.

    ``trail`` is model-specific (seed nodes, links followed, PPR matrix, ...)
    and is used to render the retrieval visualization.
    """

    query_id: str
    retrieved_ids: list[str]
    answer: str
    trail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "retrieved_ids": list(self.retrieved_ids),
            "answer": self.answer,
            "trail": self.trail,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QueryResult:
        return cls(
            query_id=d["query_id"],
            retrieved_ids=list(d.get("retrieved_ids", [])),
            answer=d.get("answer", ""),
            trail=d.get("trail", {}),
        )
