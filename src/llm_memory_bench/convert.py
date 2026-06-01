"""Convert the reference a-mem-reproduction ``dataset.json`` (combined
memories + questions) into our split YAML dataset + test-set.

Mapping:
  memories[].id/content/timestamp  -> dataset statements (m01 -> S00001)
  questions[].id/question          -> test-set queries  (q01 -> Q00001)
  questions[].requires_facts       -> required_retrieval (remapped ids)
  questions[].expected_answer      -> expected_answer
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def _sid(i: int) -> str:
    return f"S{i:05d}"


def _qid(i: int) -> str:
    return f"Q{i:05d}"


def convert(
    src: Path | str,
    name: str = "comparison",
    datasets_dir: Path | str = "data/datasets",
    testsets_dir: Path | str = "data/testsets",
) -> tuple[Path, Path]:
    data = json.loads(Path(src).read_text())
    memories = data["memories"]
    questions = data["questions"]

    # Remap original memory ids (e.g. "m01") to S-ids in encounter order.
    id_map = {m["id"]: _sid(i) for i, m in enumerate(memories, 1)}

    statements = [
        {
            "id": id_map[m["id"]],
            "statement": m["content"],
            "timestamp": m.get("timestamp"),
            "comment": f"orig {m['id']}",
        }
        for m in memories
    ]
    tests = [
        {
            "id": _qid(i),
            "query": q["question"],
            "required_retrieval": [id_map[f] for f in q.get("requires_facts", []) if f in id_map],
            "expected_answer": q.get("expected_answer"),
            "comment": f"orig {q['id']}" + (f" ({q['category']})" if q.get("category") else ""),
        }
        for i, q in enumerate(questions, 1)
    ]

    datasets_dir = Path(datasets_dir)
    testsets_dir = Path(testsets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    testsets_dir.mkdir(parents=True, exist_ok=True)

    ds_path = datasets_dir / f"{name}.yaml"
    ts_path = testsets_dir / f"{name}.yaml"
    ds_path.write_text(yaml.safe_dump({"name": name, "statements": statements}, sort_keys=False))
    ts_path.write_text(
        yaml.safe_dump({"name": name, "dataset": name, "tests": tests}, sort_keys=False)
    )
    return ds_path, ts_path
