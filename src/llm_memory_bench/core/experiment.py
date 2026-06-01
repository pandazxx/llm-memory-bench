"""Experiment definition + runner.

An Experiment = System x param-set x dataset x test-set. The runner builds
memory, runs every query, scores retrieval, and writes the README's
``results/<dataset>_<testset>/<system>_<paramset>/`` tree.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..io.dataset import find_dataset, load_dataset
from ..io.params import find_params, load_params
from ..io.testset import find_testset, load_testset
from ..systems import get_system
from ..viz.common import esc, page
from . import scoring
from .models import QueryResult
from .storage import ensure_dirs, run_paths

log = logging.getLogger(__name__)


@dataclass
class Experiment:
    system: str
    params: str
    dataset: str
    testset: str


def run(exp: Experiment, results_root: Path | str = "results") -> Path:
    dataset = load_dataset(find_dataset(exp.dataset))
    testset = load_testset(find_testset(exp.testset))
    paramset = load_params(find_params(exp.params))

    system_cls = get_system(exp.system)
    system = system_cls(paramset.params)

    paths = run_paths(dataset.name, testset.name, system.name, paramset.name, results_root)
    ensure_dirs(paths)

    log.info("constructing memory: %d statements", len(dataset.statements))
    system.construct_memory(dataset.statements)
    system.dump_memory_state(paths.memory_dir)
    system.render_memory(paths.memory_dir)

    log.info("running %d queries", len(testset.queries))
    results = []
    scores = []
    for q in testset.queries:
        result = system.retrieve(q)
        results.append(result)
        scores.append(scoring.score_query(q, result))

    system.render_retrieval(results, paths.result_dir)
    agg = scoring.aggregate(scores)

    payload = {
        "experiment": exp.__dict__,
        "system_label": system.label,
        "params": system.params,
        "aggregate": {k: v for k, v in agg.to_dict().items() if k != "per_query"},
        "results": [r.to_dict() for r in results],
        "scores": [s.to_dict() for s in agg.per_query],
    }
    paths.result_json.write_text(json.dumps(payload, indent=2))

    _write_comparison_index(paths.experiment_dir)
    log.info(
        "done: F1=%.3f recall=%.3f answer_acc=%.3f -> %s",
        agg.macro_f1,
        agg.macro_recall,
        agg.answer_accuracy,
        paths.run_dir,
    )
    return paths.run_dir


def rerender(dataset: str, testset: str, results_root: Path | str = "results") -> Path:
    """Re-emit every run's memory + retrieval HTML and the comparison index from
    frozen state.json/result.json under ``results/<dataset>_<testset>/``. No LLM calls."""
    experiment_dir = Path(results_root) / f"{dataset}_{testset}"
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"no results to re-render at {experiment_dir}")

    n = 0
    for result_json in sorted(experiment_dir.glob("*/result/result.json")):
        run_dir = result_json.parent.parent
        data = json.loads(result_json.read_text())
        system_cls = get_system(data["experiment"]["system"])
        memory_dir = run_dir / "memory"
        result_dir = run_dir / "result"

        system_cls.render_from_state(memory_dir, memory_dir)
        results = [QueryResult.from_dict(d) for d in data.get("results", [])]
        system_cls.render_retrieval_from_state(memory_dir, results, result_dir)
        n += 1
        log.info("re-rendered %s", run_dir.name)

    _write_comparison_index(experiment_dir)
    log.info("re-rendered %d run(s) -> %s", n, experiment_dir)
    return experiment_dir


def _write_comparison_index(experiment_dir: Path) -> None:
    """Aggregate every <system>_<paramset>/result/result.json into a comparison table."""
    rows = []
    for result_json in sorted(experiment_dir.glob("*/result/result.json")):
        data = json.loads(result_json.read_text())
        agg = data.get("aggregate", {})
        run_name = result_json.parent.parent.name
        rows.append(
            "<tr>"
            f"<td><a href='{esc(run_name)}/result/index.html'>{esc(run_name)}</a></td>"
            f"<td><a href='{esc(run_name)}/memory/index.html'>memory</a></td>"
            f"<td>{agg.get('macro_precision', 0):.3f}</td>"
            f"<td>{agg.get('macro_recall', 0):.3f}</td>"
            f"<td>{agg.get('macro_f1', 0):.3f}</td>"
            f"<td>{agg.get('answer_accuracy', 0):.3f}</td>"
            f"<td>{agg.get('n', 0)}</td>"
            "</tr>"
        )
    body = (
        f"<h1>Comparison — {esc(experiment_dir.name)}</h1>"
        "<table><thead><tr>"
        "<th>System_paramset</th><th>Memory</th><th>Precision</th><th>Recall</th>"
        "<th>F1</th><th>Answer acc</th><th>N</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    (experiment_dir / "index.html").write_text(page(f"Comparison — {experiment_dir.name}", body))
