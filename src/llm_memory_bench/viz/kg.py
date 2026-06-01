"""Knowledge-graph visualization for the HippoRAG systems.

Both v1 and v2 build an entity/phrase graph over OpenIE triples, so they share
this renderer. The memory view shows passages, their extracted triples, and the
entity graph; the retrieval view shows, per query, the seeds (query entities for
v1, filtered triples for v2) and the passages PPR surfaced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.models import QueryResult
from .common import esc, mermaid_id, page


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


def _entity_graph(edges: list[dict]) -> str:
    lines = ["graph LR"]
    if not edges:
        lines.append("  empty[no edges]")
    style = {"synonymy": "-.->", "relation": "-->", "triple": "-->"}
    for e in edges:
        src, dst = e.get("src"), e.get("dst")
        if src is None or dst is None:
            continue
        arrow = style.get(e.get("kind", "triple"), "-->")
        label = e.get("predicate")
        if label and arrow == "-->":
            lines.append(
                f"  {mermaid_id(src)}[{esc(src)}] -->|{esc(label)}| {mermaid_id(dst)}[{esc(dst)}]"
            )
        else:
            lines.append(f"  {mermaid_id(src)}[{esc(src)}] {arrow} {mermaid_id(dst)}[{esc(dst)}]")
    return '<pre class="mermaid">\n' + "\n".join(lines) + "\n</pre>"


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
        "<h2>Passages</h2>"
        + _passage_table(items)
        + "<h2>Triples</h2>"
        + _triple_table(items)
        + "<h2>Entity graph</h2>"
        + _entity_graph(edges)
    )
    (out_dir / "index.html").write_text(page(f"Memory — {label}", body))


def _result_table(results: list[QueryResult]) -> str:
    rows = []
    for r in results:
        rows.append(
            "<tr>"
            f"<td><code>{esc(r.query_id)}</code></td>"
            f"<td>{esc(r.answer)}</td>"
            f"<td>{', '.join(f'<code>{esc(i)}</code>' for i in r.retrieved_ids) or '—'}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Query id</th><th>Answer</th><th>Retrieved</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _trail(r: QueryResult) -> str:
    t = r.trail
    parts = [f"<h3>{esc(r.query_id)}</h3>", f"<p class='muted'>answer: {esc(r.answer)}</p>"]

    q_ents = t.get("query_entities")
    if q_ents:
        parts.append(
            "<p><b>Query entities:</b> "
            + ", ".join(f"<code>{esc(e)}</code>" for e in q_ents)
            + "</p>"
        )

    seeds = t.get("seed_entities")
    if seeds:
        items = "".join(
            f"<li><code>{esc(s.get('query_entity', ''))}</code> → "
            f"<code>{esc(s.get('matched', ''))}</code> "
            f"(sim={esc(s.get('similarity', ''))})</li>"
            for s in seeds
        )
        parts.append(f"<p><b>Seed entities:</b></p><ul>{items}</ul>")

    top_triples = t.get("top_triples")
    if top_triples:
        items = "".join(
            f"<li><code>({esc(x[0])}, {esc(x[1])}, {esc(x[2])})</code></li>"
            for x in top_triples
            if len(x) >= 3
        )
        parts.append(f"<p><b>Top triples (by query cosine):</b></p><ul>{items}</ul>")

    filtered = t.get("filtered_triples")
    if filtered is not None:
        if filtered:
            items = "".join(
                f"<li><code>({esc(x[0])}, {esc(x[1])}, {esc(x[2])})</code></li>"
                for x in filtered
                if len(x) >= 3
            )
        else:
            items = "<li class='muted'>none kept</li>"
        parts.append(f"<p><b>Recognition memory kept:</b></p><ul>{items}</ul>")

    # Seed → retrieved-passage diagram.
    lines = ["graph LR", f'  Q_{mermaid_id(r.query_id)}(["{esc(r.query_id)}"])']
    seed_labels = []
    if seeds:
        seed_labels = [s.get("matched", "") for s in seeds]
    elif filtered:
        for x in filtered:
            if len(x) >= 3:
                seed_labels.extend([x[0], x[2]])
    for sl in dict.fromkeys(seed_labels):
        if sl:
            lines.append(f"  Q_{mermaid_id(r.query_id)} -->|seed| {mermaid_id(sl)}[{esc(sl)}]")
    for pid in r.retrieved_ids:
        if seed_labels:
            for sl in dict.fromkeys(seed_labels):
                if sl:
                    lines.append(f"  {mermaid_id(sl)} -.->|PPR| {mermaid_id(pid)}([{esc(pid)}])")
        else:
            lines.append(f"  Q_{mermaid_id(r.query_id)} -.->|PPR| {mermaid_id(pid)}([{esc(pid)}])")
    parts.append('<pre class="mermaid">\n' + "\n".join(lines) + "\n</pre>")
    return "".join(parts)


def render_retrieval(state: dict[str, Any], results: list[QueryResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    label = state.get("label", state.get("system", "System"))
    body = (
        f"<h1>Query results — {esc(label)}</h1>"
        "<h2>Results</h2>"
        + _result_table(results)
        + "<h2>Retrieval trails</h2>"
        + "".join(_trail(r) for r in results)
    )
    (out_dir / "index.html").write_text(page(f"Results — {label}", body))
