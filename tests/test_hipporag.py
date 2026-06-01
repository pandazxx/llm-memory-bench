"""Offline tests for the HippoRAG v1/v2 adapters (NIM calls are stubbed)."""

import numpy as np

from llm_memory_bench.backend import nim
from llm_memory_bench.core.models import Query, Statement
from llm_memory_bench.systems.hipporag import HippoRAG
from llm_memory_bench.systems.hipporag2 import HippoRAG2

STATEMENTS = [
    Statement(id="S00001", statement="Alice Chen mentors Bob Martinez."),
    Statement(id="S00002", statement="Bob Martinez leads the payments team."),
]

TRIPLES = {
    "Alice Chen mentors Bob Martinez.": [("Alice Chen", "mentors", "Bob Martinez")],
    "Bob Martinez leads the payments team.": [("Bob Martinez", "leads", "payments team")],
}


def _vec(text: str, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) or 1.0)


def _patch_nim(monkeypatch):
    monkeypatch.setattr(nim, "extract_triples", lambda passage: TRIPLES.get(passage, []))
    monkeypatch.setattr(
        nim, "embed_batch", lambda texts, input_type="passage": np.array([_vec(t) for t in texts])
    )
    monkeypatch.setattr(nim, "embed", lambda text, input_type="query": _vec(text))
    monkeypatch.setattr(nim, "chat_text", lambda prompt, system: "stubbed answer")
    monkeypatch.setattr(nim, "extract_query_entities", lambda q: ["Bob Martinez"])
    monkeypatch.setattr(nim, "filter_triples", lambda q, triples, top_k=4: list(triples)[:1])


def test_hipporag_v1_construct_and_retrieve(monkeypatch, tmp_path):
    _patch_nim(monkeypatch)
    sys = HippoRAG()
    sys.construct_memory(STATEMENTS)

    assert sys.index["entities"]  # entities extracted from triples
    assert sys._edges  # at least the relation edges

    result = sys.retrieve(Query(id="Q00001", query="Who does Bob report to?"))
    assert result.query_id == "Q00001"
    assert result.answer == "stubbed answer"
    assert result.trail["kind"] == "hipporag"
    assert result.trail["seed_entities"]  # NER matched a KG entity
    for rid in result.retrieved_ids:
        assert rid in {"S00001", "S00002"}

    sys.dump_memory_state(tmp_path)
    assert (tmp_path / "state.json").exists()
    sys.render_memory(tmp_path)
    sys.render_retrieval([result], tmp_path)
    assert (tmp_path / "index.html").exists()


def test_hipporag_v2_construct_and_retrieve(monkeypatch, tmp_path):
    _patch_nim(monkeypatch)
    sys = HippoRAG2()
    sys.construct_memory(STATEMENTS)

    assert sys.index["phrases"]
    assert sys.index["N"] == sys.index["N_phrase"] + sys.index["P"]

    result = sys.retrieve(Query(id="Q00002", query="What does Bob lead?"))
    assert result.trail["kind"] == "hipporag2"
    assert result.trail["filtered_triples"]  # recognition memory kept something
    for rid in result.retrieved_ids:
        assert rid in {"S00001", "S00002"}

    sys.dump_memory_state(tmp_path)
    HippoRAG2.render_from_state(tmp_path, tmp_path)
    assert (tmp_path / "index.html").exists()


def test_hipporag_v1_empty_dataset(monkeypatch):
    _patch_nim(monkeypatch)
    sys = HippoRAG()
    sys.construct_memory([])
    result = sys.retrieve(Query(id="Q0", query="anything?"))
    assert result.retrieved_ids == []
    assert result.answer == "unknown / not mentioned"
