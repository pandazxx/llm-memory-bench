"""Knowledge-graph visualization for the HippoRAG systems.

Both v1 and v2 build an entity/phrase graph over OpenIE triples, so they share
this renderer. The memory view shows passages, their extracted triples, and an
interactive entity graph; the retrieval view shows, per query, an interactive
graph of seeds → retrieved passages. Nodes and edges are clickable — details
appear in the side panel.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from ..core.models import QueryResult
from .common import esc, page, scored_result_table
from .graph import TRACE_OPTIONS, graph_assets, graph_block

_ENTITY_COLOR = "#60a5fa"
_PASSAGE_COLOR = "#34d399"
_SEED_COLOR = "#f59e0b"
_QUERY_COLOR = "#ef4444"


def _passage_table(items: list[dict]) -> str:
    rows = []
    for it in items:
        triples = it.get("triples", [])
        rows.append(
            "<tr>"
            f"<td><code>{esc(it.get('id', ''))}</code></td>"
            f"<td>{esc(it.get('content', ''))}</td>"
            f"<td>{len(triples)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Statement id</th><th>Content</th><th>Triples</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _triple_table(items: list[dict]) -> str:
    rows = []
    for it in items:
        pid = it.get("id", "")
        for t in it.get("triples", []):
            if len(t) < 3:
                continue
            rows.append(
                "<tr>"
                f"<td><code>{esc(pid)}</code></td>"
                f"<td>{esc(t[0])}</td><td>{esc(t[1])}</td><td>{esc(t[2])}</td>"
                "</tr>"
            )
    if not rows:
        rows.append("<tr><td colspan='4' class='muted'>no triples</td></tr>")
    return (
        "<table><thead><tr><th>From</th><th>Subject</th><th>Predicate</th><th>Object</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _entity_memberships(items: list[dict]) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """For each entity: the passages it appears in, and the triples it belongs to."""
    in_passages: dict[str, set[str]] = {}
    in_triples: dict[str, list[str]] = {}
    for it in items:
        pid = it.get("id", "")
        for t in it.get("triples", []):
            if len(t) < 3:
                continue
            text = f"({t[0]}, {t[1]}, {t[2]})"
            for ent in (t[0], t[2]):
                in_passages.setdefault(ent, set()).add(pid)
                in_triples.setdefault(ent, []).append(text)
    return in_passages, in_triples


def _memory_graph(items: list[dict], edges: list[dict]) -> str:
    in_passages, in_triples = _entity_memberships(items)
    entity_names = set(in_passages)
    for e in edges:
        if e.get("src"):
            entity_names.add(e["src"])
        if e.get("dst"):
            entity_names.add(e["dst"])

    nodes = []
    for ent in sorted(entity_names):
        passages = sorted(in_passages.get(ent, []))
        triples = in_triples.get(ent, [])
        nodes.append(
            {
                "id": ent,
                "label": textwrap.shorten(ent, 22),
                "color": _ENTITY_COLOR,
                "shape": "dot",
                "size": 14 + min(len(passages) * 3, 18),
                "_detail": {
                    "title": ent,
                    "subtitle": "entity",
                    "rows": [
                        {"label": "Appears in", "value": passages, "type": "chips"},
                        {"label": "Triples", "value": triples, "type": "list"},
                    ],
                },
            }
        )

    vis_edges = []
    for e in edges:
        src, dst = e.get("src"), e.get("dst")
        if not src or not dst:
            continue
        kind = e.get("kind", "relation")
        if kind == "synonymy":
            detail = {
                "title": "synonymy",
                "subtitle": "embedding similarity edge",
                "rows": [
                    {"label": "Between", "value": [src, dst], "type": "chips"},
                    {"label": "Cosine", "value": e.get("weight", ""), "type": "code"},
                ],
            }
            vis_edges.append(
                {"from": src, "to": dst, "color": "#c4b5fd", "dashes": True, "_detail": detail}
            )
        else:
            pred = e.get("predicate", "")
            detail = {
                "title": pred or "relation",
                "subtitle": "relation edge",
                "rows": [
                    {"label": "Subject", "value": src, "type": "code"},
                    {"label": "Predicate", "value": pred, "type": "text"},
                    {"label": "Object", "value": dst, "type": "code"},
                ],
            }
            vis_edges.append(
                {
                    "from": src,
                    "to": dst,
                    "label": textwrap.shorten(pred, 18),
                    "color": "#9ca3af",
                    "_detail": detail,
                }
            )
    return graph_block(nodes, vis_edges, height=600)


def render_memory(state: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    label = state.get("label", state.get("system", "System"))
    items = state.get("items", [])
    edges = state.get("edges", [])
    n_triples = sum(len(it.get("triples", [])) for it in items)
    body = (
        f"<h1>Memory structure — {esc(label)}</h1>"
        f"<p class='muted'>{len(items)} passages · {n_triples} triples · "
        f"{len(edges)} graph edges · params: <code>{esc(state.get('params', {}))}</code></p>"
        "<h2>Entity graph</h2>"
        + _memory_graph(items, edges)
        + "<h2>Passages</h2>"
        + _passage_table(items)
        + "<h2>Triples</h2>"
        + _triple_table(items)
    )
    (out_dir / "index.html").write_text(page(f"Memory — {label}", body, head=graph_assets()))


def _trail_graph(r: QueryResult, items_by_id: dict[str, dict]) -> str:
    t = r.trail
    qid = f"_q_{r.query_id}"
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

    seeds = t.get("seed_entities") or []
    filtered = t.get("filtered_triples") or []
    seed_labels: list[str] = []
    if seeds:
        for s in seeds:
            matched = s.get("matched", "")
            if not matched or matched in added:
                continue
            added.add(matched)
            seed_labels.append(matched)
            nodes.append(
                {
                    "id": matched,
                    "label": textwrap.shorten(matched, 22),
                    "color": _SEED_COLOR,
                    "shape": "dot",
                    "size": 16,
                    "_detail": {
                        "title": matched,
                        "subtitle": "seed entity",
                        "rows": [
                            {
                                "label": "From query entity",
                                "value": s.get("query_entity", ""),
                                "type": "code",
                            },
                            {
                                "label": "Similarity",
                                "value": s.get("similarity", ""),
                                "type": "code",
                            },
                        ],
                    },
                }
            )
            edges.append(
                {"from": qid, "to": matched, "label": "seed", "color": _SEED_COLOR, "width": 2}
            )
    elif filtered:
        for tr in filtered:
            if len(tr) < 3:
                continue
            for ent in (tr[0], tr[2]):
                if not ent or ent in added:
                    continue
                added.add(ent)
                seed_labels.append(ent)
                nodes.append(
                    {
                        "id": ent,
                        "label": textwrap.shorten(ent, 22),
                        "color": _SEED_COLOR,
                        "shape": "dot",
                        "size": 16,
                        "_detail": {
                            "title": ent,
                            "subtitle": "seed phrase (kept triple)",
                            "rows": [
                                {
                                    "label": "From triple",
                                    "value": f"({tr[0]}, {tr[1]}, {tr[2]})",
                                    "type": "code",
                                },
                            ],
                        },
                    }
                )
                edges.append(
                    {"from": qid, "to": ent, "label": "seed", "color": _SEED_COLOR, "width": 2}
                )

    for pid in r.retrieved_ids:
        if pid not in added:
            content = items_by_id.get(pid, {}).get("content", "")
            nodes.append(
                {
                    "id": pid,
                    "label": pid,
                    "color": _PASSAGE_COLOR,
                    "shape": "box",
                    "size": 16,
                    "_detail": {
                        "title": pid,
                        "subtitle": "retrieved passage",
                        "rows": [{"label": "Content", "value": content, "type": "text"}],
                    },
                }
            )
            added.add(pid)
        sources = seed_labels or [qid]
        for src in sources:
            edges.append(
                {"from": src, "to": pid, "label": "PPR", "color": _PASSAGE_COLOR, "dashes": True}
            )

    return graph_block(nodes, edges, options=TRACE_OPTIONS, height=420)


def _trail(r: QueryResult, items_by_id: dict[str, dict]) -> str:
    parts = [f"<h3>{esc(r.query_id)}</h3>", f"<p class='muted'>answer: {esc(r.answer)}</p>"]
    t = r.trail

    q_ents = t.get("query_entities")
    if q_ents:
        parts.append(
            "<p><b>Query entities:</b> "
            + ", ".join(f"<code>{esc(e)}</code>" for e in q_ents)
            + "</p>"
        )

    top_triples = t.get("top_triples")
    if top_triples:
        lis = "".join(
            f"<li><code>({esc(x[0])}, {esc(x[1])}, {esc(x[2])})</code></li>"
            for x in top_triples
            if len(x) >= 3
        )
        parts.append(f"<p><b>Top triples (by query cosine):</b></p><ul>{lis}</ul>")

    filtered = t.get("filtered_triples")
    if filtered is not None:
        if filtered:
            lis = "".join(
                f"<li><code>({esc(x[0])}, {esc(x[1])}, {esc(x[2])})</code></li>"
                for x in filtered
                if len(x) >= 3
            )
        else:
            lis = "<li class='muted'>none kept</li>"
        parts.append(f"<p><b>Recognition memory kept:</b></p><ul>{lis}</ul>")

    parts.append(_trail_graph(r, items_by_id))
    return "".join(parts)


def render_retrieval(
    state: dict[str, Any],
    results: list[QueryResult],
    out_dir: Path,
    scores: list[dict[str, Any]] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    label = state.get("label", state.get("system", "System"))
    items_by_id = {it.get("id"): it for it in state.get("items", [])}
    body = (
        f"<h1>Query results — {esc(label)}</h1>"
        "<h2>Results</h2>"
        + scored_result_table(results, scores)
        + "<h2>Retrieval trails</h2>"
        + "".join(_trail(r, items_by_id) for r in results)
    )
    (out_dir / "index.html").write_text(page(f"Results — {label}", body, head=graph_assets()))
