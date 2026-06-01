"""Retrieval-trail visualization: per-query result table + interactive trail graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.models import QueryResult
from .common import esc, page
from .graph import TRACE_OPTIONS, graph_assets, graph_block

_QUERY_COLOR = "#ef4444"
_SEED_COLOR = "#60a5fa"
_LINK_COLOR = "#a78bfa"


def _result_table(results: list[QueryResult]) -> str:
    rows = []
    for r in results:
        rows.append(
            "<tr>"
            f"<td><code>{esc(r.query_id)}</code></td>"
            f"<td>{esc(r.answer)}</td>"
            f"<td>{', '.join(f'<code>{esc(i)}</code>' for i in r.retrieved_ids) or '—'}</td>"
            f"<td>{', '.join(f'<code>{esc(i)}</code>' for i in r.trail.get('seed_ids', [])) or '—'}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Query id</th><th>Answer</th><th>Retrieved</th><th>Seeds</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _content_of(state: dict[str, Any], mid: str) -> str:
    return state.get("memories", {}).get(mid, {}).get("content", "")


def _trail_graph(state: dict[str, Any], r: QueryResult) -> str:
    qid = f"_q_{r.query_id}"
    seeds = r.trail.get("seed_ids", [])
    links = r.trail.get("links_followed", [])

    nodes = [
        {
            "id": qid,
            "label": r.query_id,
            "color": _QUERY_COLOR,
            "shape": "diamond",
            "size": 22,
            "_detail": {
                "title": r.query_id,
                "subtitle": "query",
                "rows": [{"label": "Answer", "value": r.answer, "type": "text"}],
            },
        }
    ]
    edges = []
    added: set[str] = set()

    def _add_note(mid: str, subtitle: str, color: str) -> None:
        if mid in added:
            return
        added.add(mid)
        nodes.append(
            {
                "id": mid,
                "label": mid,
                "color": color,
                "shape": "dot",
                "size": 16,
                "_detail": {
                    "title": mid,
                    "subtitle": subtitle,
                    "rows": [
                        {"label": "Content", "value": _content_of(state, mid), "type": "text"}
                    ],
                },
            }
        )

    for s in seeds:
        _add_note(s, "seed (cosine match)", _SEED_COLOR)
        edges.append({"from": qid, "to": s, "label": "seed", "color": _SEED_COLOR, "width": 2})

    for src, dst in links:
        _add_note(src, "seed (cosine match)", _SEED_COLOR)
        _add_note(dst, "link-followed", _LINK_COLOR)
        edges.append(
            {"from": src, "to": dst, "label": "link", "color": _LINK_COLOR, "dashes": True}
        )

    for pid in r.retrieved_ids:
        _add_note(pid, "retrieved", _LINK_COLOR)

    return graph_block(nodes, edges, options=TRACE_OPTIONS, height=420)


def _trail(state: dict[str, Any], r: QueryResult) -> str:
    return f"<h3>{esc(r.query_id)}</h3><p class='muted'>answer: {esc(r.answer)}</p>" + _trail_graph(
        state, r
    )


def render(state: dict[str, Any], results: list[QueryResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    label = state.get("label", state.get("system", "System"))
    body = (
        f"<h1>Query results — {esc(label)}</h1>"
        "<h2>Results</h2>"
        + _result_table(results)
        + "<h2>Retrieval trails</h2>"
        + "".join(_trail(state, r) for r in results)
    )
    (out_dir / "index.html").write_text(page(f"Results — {label}", body, head=graph_assets()))
