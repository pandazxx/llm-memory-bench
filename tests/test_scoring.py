from llm_memory_bench.core import scoring
from llm_memory_bench.core.models import Query, QueryResult


def test_perfect_retrieval():
    q = Query(id="Q1", query="?", required_retrieval=["S1", "S2"], expected_answer="Oakland")
    r = QueryResult(query_id="Q1", retrieved_ids=["S1", "S2"], answer="It is Oakland.")
    s = scoring.score_query(q, r)
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.f1 == 1.0
    assert s.answer_match is True


def test_partial_retrieval():
    q = Query(id="Q1", query="?", required_retrieval=["S1", "S2"])
    r = QueryResult(query_id="Q1", retrieved_ids=["S1", "S3", "S4"], answer="x")
    s = scoring.score_query(q, r)
    assert s.precision == 1 / 3
    assert s.recall == 1 / 2
    assert s.hits == ["S1"]
    assert s.missed == ["S2"]


def test_empty_required_recall_is_one():
    q = Query(id="Q1", query="?", required_retrieval=[])
    r = QueryResult(query_id="Q1", retrieved_ids=[], answer="x")
    s = scoring.score_query(q, r)
    assert s.recall == 1.0


def test_aggregate():
    q1 = Query(id="Q1", query="?", required_retrieval=["S1"], expected_answer="a")
    q2 = Query(id="Q2", query="?", required_retrieval=["S2"], expected_answer="b")
    r1 = QueryResult(query_id="Q1", retrieved_ids=["S1"], answer="a")
    r2 = QueryResult(query_id="Q2", retrieved_ids=["S9"], answer="wrong")
    agg = scoring.aggregate([scoring.score_query(q1, r1), scoring.score_query(q2, r2)])
    assert agg.n == 2
    assert agg.macro_recall == 0.5
    assert agg.answer_accuracy == 0.5
