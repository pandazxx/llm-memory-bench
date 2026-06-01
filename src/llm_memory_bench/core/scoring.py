"""Retrieval scoring: compare retrieved statement ids against the test-set's
``required_retrieval`` (the README's judgemental factor)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Query, QueryResult


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


@dataclass
class QueryScore:
    query_id: str
    precision: float
    recall: float
    f1: float
    required: list[str]
    retrieved: list[str]
    hits: list[str]
    missed: list[str]
    answer: str
    expected_answer: str | None
    answer_match: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Aggregate:
    macro_precision: float
    macro_recall: float
    macro_f1: float
    answer_accuracy: float
    n: int
    per_query: list[QueryScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "answer_accuracy": self.answer_accuracy,
            "n": self.n,
            "per_query": [q.to_dict() for q in self.per_query],
        }


def score_query(query: Query, result: QueryResult) -> QueryScore:
    required = list(query.required_retrieval)
    retrieved = list(result.retrieved_ids)
    req_set, ret_set = set(required), set(retrieved)
    hits = req_set & ret_set

    precision = len(hits) / len(ret_set) if ret_set else 0.0
    recall = len(hits) / len(req_set) if req_set else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    expected = query.expected_answer
    answer_match = bool(expected) and _norm(expected) in _norm(result.answer)

    return QueryScore(
        query_id=query.id,
        precision=precision,
        recall=recall,
        f1=f1,
        required=required,
        retrieved=retrieved,
        hits=sorted(hits),
        missed=sorted(req_set - ret_set),
        answer=result.answer,
        expected_answer=expected,
        answer_match=answer_match,
    )


def aggregate(scores: list[QueryScore]) -> Aggregate:
    n = len(scores)
    if n == 0:
        return Aggregate(0.0, 0.0, 0.0, 0.0, 0)
    return Aggregate(
        macro_precision=sum(s.precision for s in scores) / n,
        macro_recall=sum(s.recall for s in scores) / n,
        macro_f1=sum(s.f1 for s in scores) / n,
        answer_accuracy=sum(1 for s in scores if s.answer_match) / n,
        n=n,
        per_query=scores,
    )
