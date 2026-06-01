"""Shared HTML scaffolding for visualizations."""

from __future__ import annotations

import html
from typing import Any

from ..core.models import QueryResult

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
h1 {{ font-size: 1.4rem; }}
h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; vertical-align: top; }}
th {{ background: #f4f4f4; }}
code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
.tag {{ display: inline-block; background: #eef; border-radius: 3px; padding: 0 6px; margin: 1px; }}
.hit {{ color: #137333; font-weight: 600; }}
.miss {{ color: #c5221f; font-weight: 600; }}
.muted {{ color: #777; }}
</style>
{head}
</head>
<body>
{body}
</body>
</html>
"""


def esc(s: object) -> str:
    return html.escape(str(s))


def tags_html(tags: list) -> str:
    return (
        "".join(f'<span class="tag">{esc(t)}</span>' for t in (tags or []))
        or '<span class="muted">—</span>'
    )


def page(title: str, body: str, head: str = "") -> str:
    return PAGE.format(title=esc(title), body=body, head=head)


def _ids_cell(ids: list[str], *, green: set[str] = frozenset(), red: set[str] = frozenset()) -> str:
    if not ids:
        return "<span class='muted'>—</span>"
    out = []
    for i in ids:
        cls = "hit" if i in green else ("miss" if i in red else "")
        out.append(f"<code class='{cls}'>{esc(i)}</code>" if cls else f"<code>{esc(i)}</code>")
    return " ".join(out)


def scored_result_table(
    results: list[QueryResult], scores: list[dict[str, Any]] | None = None
) -> str:
    """Per-query result table. With ``scores`` it compares retrieved vs required
    and answer vs expected, and shows precision/recall/F1; without, a plain table."""
    by_id = {s.get("query_id"): s for s in (scores or [])}
    if not by_id:
        rows = "".join(
            "<tr>"
            f"<td><code>{esc(r.query_id)}</code></td>"
            f"<td>{esc(r.answer)}</td>"
            f"<td>{_ids_cell(r.retrieved_ids)}</td>"
            "</tr>"
            for r in results
        )
        return (
            "<table><thead><tr><th>Query id</th><th>Answer</th><th>Retrieved</th>"
            "</tr></thead><tbody>" + rows + "</tbody></table>"
        )

    rows = []
    for r in results:
        s = by_id.get(r.query_id, {})
        required = list(s.get("required", []))
        retrieved = list(s.get("retrieved", r.retrieved_ids))
        req_set, ret_set = set(required), set(retrieved)
        hits = req_set & ret_set
        expected = s.get("expected_answer")
        expected_cell = esc(expected) if expected else "<span class='muted'>—</span>"
        match_cell = (
            "<span class='hit'>✓</span>" if s.get("answer_match") else "<span class='miss'>✗</span>"
        )
        rows.append(
            "<tr>"
            f"<td><code>{esc(r.query_id)}</code></td>"
            f"<td>{esc(s.get('answer', r.answer))}</td>"
            f"<td>{expected_cell}</td>"
            f"<td>{match_cell}</td>"
            f"<td>{_ids_cell(retrieved, green=hits)}</td>"
            f"<td>{_ids_cell(required, green=hits, red=req_set - ret_set)}</td>"
            f"<td>{s.get('precision', 0):.2f}</td>"
            f"<td>{s.get('recall', 0):.2f}</td>"
            f"<td>{s.get('f1', 0):.2f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Query id</th><th>Answer</th><th>Expected</th><th>Match</th>"
        "<th>Retrieved</th><th>Required</th><th>P</th><th>R</th><th>F1</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
