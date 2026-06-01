"""End-to-end runner test with a stub system (no NIM key / no LLM)."""

import json
from pathlib import Path

from llm_memory_bench.core import experiment as exp_mod
from llm_memory_bench.core.experiment import Experiment, run
from llm_memory_bench.core.models import Query, QueryResult, Statement
from llm_memory_bench.core.system import System


class StubSystem(System):
    name = "stub"
    label = "Stub"
    DEFAULT_PARAMS = {"k": 1}

    def construct_memory(self, statements: list[Statement]) -> None:
        self._ids = [s.id for s in statements]

    def retrieve(self, query: Query) -> QueryResult:
        # Echo back the required ids so scoring is deterministic.
        return QueryResult(
            query_id=query.id,
            retrieved_ids=list(query.required_retrieval),
            answer=query.expected_answer or "",
            trail={"seed_ids": list(query.required_retrieval), "links_followed": []},
        )

    def render_memory(self, out_dir: Path) -> None:
        (out_dir / "index.html").write_text("<html>mem</html>")

    def render_retrieval(self, results, out_dir: Path) -> None:
        (out_dir / "index.html").write_text("<html>res</html>")

    def dump_memory_state(self, out_dir: Path) -> None:
        (out_dir / "state.json").write_text(json.dumps({"memories": {}}))

    @classmethod
    def render_from_state(cls, state_dir: Path, out_dir: Path) -> None:
        (out_dir / "index.html").write_text("<html>rerender</html>")


def test_run_end_to_end(tmp_path, monkeypatch):
    (tmp_path / "ds.yaml").write_text(
        "name: ds\nstatements:\n  - id: S00001\n    statement: Sam lives in Oakland.\n"
    )
    (tmp_path / "ts.yaml").write_text(
        "name: ts\ndataset: ds\ntests:\n"
        "  - id: Q00001\n    query: Where?\n    required_retrieval: [S00001]\n"
        "    expected_answer: Oakland\n"
    )
    (tmp_path / "p.yaml").write_text("system: stub\nname: default\nparams:\n  k: 2\n")

    import llm_memory_bench.systems as systems_mod

    monkeypatch.setitem(systems_mod.REGISTRY, "stub", StubSystem)
    monkeypatch.setattr(exp_mod, "find_dataset", lambda n: tmp_path / "ds.yaml")
    monkeypatch.setattr(exp_mod, "find_testset", lambda n: tmp_path / "ts.yaml")
    monkeypatch.setattr(exp_mod, "find_params", lambda n: tmp_path / "p.yaml")

    results_root = tmp_path / "results"
    run_dir = run(Experiment("stub", "default", "ds", "ts"), results_root=results_root)

    result_json = json.loads((run_dir / "result" / "result.json").read_text())
    assert result_json["aggregate"]["macro_f1"] == 1.0
    assert result_json["aggregate"]["answer_accuracy"] == 1.0
    assert (run_dir / "memory" / "index.html").exists()
    assert (run_dir / "result" / "index.html").exists()
    assert (results_root / "ds_ts" / "index.html").exists()
