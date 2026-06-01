from pathlib import Path

import pytest

from llm_memory_bench.io.dataset import load_dataset
from llm_memory_bench.io.params import load_params
from llm_memory_bench.io.testset import load_testset


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_load_dataset_ok(tmp_path):
    p = _write(
        tmp_path,
        "d.yaml",
        "name: d\nstatements:\n"
        "  - id: S00001\n    statement: Sam lives in Oakland.\n    timestamp: t1\n",
    )
    ds = load_dataset(p)
    assert ds.name == "d"
    assert ds.statements[0].id == "S00001"
    assert ds.statements[0].statement == "Sam lives in Oakland."


def test_load_dataset_duplicate_id(tmp_path):
    p = _write(
        tmp_path,
        "d.yaml",
        "statements:\n  - id: S1\n    statement: a\n  - id: S1\n    statement: b\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(p)


def test_load_dataset_missing_text(tmp_path):
    p = _write(tmp_path, "d.yaml", "statements:\n  - id: S1\n")
    with pytest.raises(ValueError, match="missing 'statement'"):
        load_dataset(p)


def test_load_testset_ok(tmp_path):
    p = _write(
        tmp_path,
        "t.yaml",
        "name: t\ndataset: d\ntests:\n"
        "  - id: Q00001\n    query: Where?\n    required_retrieval: [S00001]\n"
        "    expected_answer: Oakland\n",
    )
    ts = load_testset(p)
    assert ts.dataset == "d"
    assert ts.queries[0].required_retrieval == ["S00001"]


def test_load_params_ok(tmp_path):
    p = _write(tmp_path, "p.yaml", "system: amem\nname: default\nparams:\n  retrieval_k: 7\n")
    ps = load_params(p)
    assert ps.params["retrieval_k"] == 7
