"""Memory-structure visualization: node table + link graph."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from .common import esc, mermaid_id, page, tags_html


def _node_table(memories: dict[str, dict]) -> str:
    rows = []
    for mid, n in memories.items():
        rows.append(
            "<tr>"
            f"<td><code>{esc(mid)}</code></td>"
            f"<td>{esc(n.get('content', ''))}</td>"
            f"<td>{tags_html(n.get('tags', []))}</td>"
            f"<td>{esc(n.get('context', ''))}</td>"
            f"<td>{len(n.get('links', []))}</td>"
            f"<td>{len(n.get('evolution_history', []))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Statement id</th><th>Content</th><th>Tags</th><th>Context</th>"
        "<th>Links</th><th>Evolutions</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _graph(memories: dict[str, dict]) -> str:
    lines = ["graph LR"]
    for mid, n in memories.items():
        label = textwrap.shorten(n.get("content", mid), 40).replace('"', "'")
        lines.append(f'  {mermaid_id(mid)}["{esc(mid)}: {esc(label)}"]')
    for mid, n in memories.items():
        for target in n.get("links", []):
            if target in memories:
                lines.append(f"  {mermaid_id(mid)} --> {mermaid_id(target)}")
    return '<pre class="mermaid">\n' + "\n".join(lines) + "\n</pre>"


def render(state: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    memories = state.get("memories", {})
    label = state.get("label", state.get("system", "System"))
    body = (
        f"<h1>Memory structure — {esc(label)}</h1>"
        f"<p class='muted'>{len(memories)} memory nodes · params: <code>{esc(state.get('params', {}))}</code></p>"
        "<h2>Nodes</h2>" + _node_table(memories) + "<h2>Graph</h2>" + _graph(memories)
    )
    (out_dir / "index.html").write_text(page(f"Memory — {label}", body))
