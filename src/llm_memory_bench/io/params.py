"""Load param-sets (system tuning parameters) from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ParamSet:
    name: str
    system: str | None
    params: dict = field(default_factory=dict)


def load_params(path: Path | str) -> ParamSet:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: param-set must be a mapping")
    return ParamSet(
        name=data.get("name", path.stem),
        system=data.get("system"),
        params=data.get("params", {}) or {},
    )


def find_params(name: str, root: Path | str = "data/params") -> Path:
    p = Path(root) / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"param-set {name!r} not found at {p}")
    return p
