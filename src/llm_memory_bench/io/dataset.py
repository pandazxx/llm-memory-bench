"""Load + validate a Dataset (list of statements) from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..core.models import Statement


@dataclass
class Dataset:
    name: str
    statements: list[Statement]


def load_dataset(path: Path | str) -> Dataset:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "statements" not in data:
        raise ValueError(f"{path}: dataset must be a mapping with a 'statements' list")

    name = data.get("name", path.stem)
    statements: list[Statement] = []
    seen: set[str] = set()
    for i, raw in enumerate(data["statements"]):
        sid = raw.get("id")
        if not sid:
            raise ValueError(f"{path}: statement #{i} is missing 'id'")
        if sid in seen:
            raise ValueError(f"{path}: duplicate statement id {sid!r}")
        seen.add(sid)
        if "statement" not in raw:
            raise ValueError(f"{path}: statement {sid} is missing 'statement' text")
        statements.append(
            Statement(
                id=sid,
                statement=raw["statement"],
                timestamp=raw.get("timestamp"),
                comment=raw.get("comment"),
            )
        )
    return Dataset(name=name, statements=statements)


def find_dataset(name: str, root: Path | str = "data/datasets") -> Path:
    p = Path(root) / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"dataset {name!r} not found at {p}")
    return p
