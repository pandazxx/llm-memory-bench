"""Retrieval-trail visualization: per-query result table + seed/link diagram."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.models import QueryResult
from .common import esc, mermaid_id, page


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


def _trail(r: QueryResult) -> str:
    seeds = r.trail.get("seed_ids", [])
    links = r.trail.get("links_followed", [])
    lines = ["graph LR", f'  Q_{esc(r.query_id)}(["{esc(r.query_id)}"])']
    for s in seeds:
        lines.append(f"  Q_{esc(r.query_id)} -->|seed| {mermaid_id(s)}[{esc(s)}]")
    for src, dst in links:
        lines.append(f"  {mermaid_id(src)} -->|link| {mermaid_id(dst)}[{esc(dst)}]")
    return (
        f"<h3>{esc(r.query_id)}</h3>"
        f"<p class='muted'>answer: {esc(r.answer)}</p>"
        '<pre class="mermaid">\n' + "\n".join(lines) + "\n</pre>"
    )


def render(state: dict[str, Any], results: list[QueryResult], out_dir: Path) -> None:
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
