import json

from llm_memory_bench.convert import convert
from llm_memory_bench.io.dataset import load_dataset
from llm_memory_bench.io.testset import load_testset


def test_convert_remaps_ids(tmp_path):
    src = tmp_path / "dataset.json"
    src.write_text(
        json.dumps(
            {
                "memories": [
                    {"id": "m01", "content": "Sam lives in Oakland.", "timestamp": "t1"},
                    {"id": "m02", "content": "TechCorp is in SF.", "timestamp": "t2"},
                ],
                "questions": [
                    {
                        "id": "q01",
                        "category": "single_hop",
                        "question": "Where does Sam live?",
                        "expected_answer": "Oakland",
                        "requires_facts": ["m01"],
                    }
                ],
            }
        )
    )
    ds_path, ts_path = convert(
        src, name="t", datasets_dir=tmp_path / "ds", testsets_dir=tmp_path / "ts"
    )
    ds = load_dataset(ds_path)
    ts = load_testset(ts_path)

    assert [s.id for s in ds.statements] == ["S00001", "S00002"]
    assert ds.statements[0].statement == "Sam lives in Oakland."
    # m01 -> S00001 remap propagated into required_retrieval
    assert ts.queries[0].required_retrieval == ["S00001"]
    assert ts.queries[0].expected_answer == "Oakland"
