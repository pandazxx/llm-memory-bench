"""Result directory layout, per README "Test result storage structure".

results/<dataset>_<testset>/<system>_<paramset>/
    memory/   index.html, state.json, assets
    result/   index.html, result.json
results/<dataset>_<testset>/index.html   (comparison)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunPaths:
    root: Path
    experiment_dir: Path
    run_dir: Path
    memory_dir: Path
    result_dir: Path

    @property
    def state_json(self) -> Path:
        return self.memory_dir / "state.json"

    @property
    def result_json(self) -> Path:
        return self.result_dir / "result.json"

    @property
    def comparison_index(self) -> Path:
        return self.experiment_dir / "index.html"


def run_paths(
    dataset: str,
    testset: str,
    system: str,
    paramset: str,
    root: Path | str = "results",
) -> RunPaths:
    root = Path(root)
    experiment_dir = root / f"{dataset}_{testset}"
    run_dir = experiment_dir / f"{system}_{paramset}"
    paths = RunPaths(
        root=root,
        experiment_dir=experiment_dir,
        run_dir=run_dir,
        memory_dir=run_dir / "memory",
        result_dir=run_dir / "result",
    )
    return paths


def ensure_dirs(paths: RunPaths) -> None:
    paths.memory_dir.mkdir(parents=True, exist_ok=True)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
