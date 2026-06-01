"""Load + validate a Test-set (list of queries) from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..core.models import Query


@dataclass
class TestSet:
    name: str
    dataset: str | None
    queries: list[Query]


def load_testset(path: Path | str) -> TestSet:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "tests" not in data:
        raise ValueError(f"{path}: test-set must be a mapping with a 'tests' list")

    name = data.get("name", path.stem)
    queries: list[Query] = []
    seen: set[str] = set()
    for i, raw in enumerate(data["tests"]):
        qid = raw.get("id")
        if not qid:
            raise ValueError(f"{path}: test #{i} is missing 'id'")
        if qid in seen:
            raise ValueError(f"{path}: duplicate query id {qid!r}")
        seen.add(qid)
        if "query" not in raw:
            raise ValueError(f"{path}: test {qid} is missing 'query' text")
        required = raw.get("required_retrieval", []) or []
        if not isinstance(required, list):
            raise ValueError(f"{path}: test {qid} 'required_retrieval' must be a list")
        queries.append(
            Query(
                id=qid,
                query=raw["query"],
                required_retrieval=list(required),
                expected_answer=raw.get("expected_answer"),
                comment=raw.get("comment"),
            )
        )
    return TestSet(name=name, dataset=data.get("dataset"), queries=queries)


def find_testset(name: str, root: Path | str = "data/testsets") -> Path:
    p = Path(root) / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"test-set {name!r} not found at {p}")
    return p
