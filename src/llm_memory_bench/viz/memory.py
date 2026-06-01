"""Memory-structure visualization: node table + interactive link graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import esc, page, tags_html
from .graph import graph_assets, graph_block

_PLAIN_COLOR = "#60a5fa"
_EVOLVED_COLOR = "#fbbf24"


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


def _note_detail(mid: str, n: dict) -> dict[str, Any]:
    evo = n.get("evolution_history", [])
    rows = [
        {"label": "Content", "value": n.get("content", ""), "type": "text"},
        {"label": "Keywords", "value": n.get("keywords", []), "type": "chips"},
        {"label": "Tags", "value": n.get("tags", []), "type": "chips"},
        {"label": "Context", "value": n.get("context", ""), "type": "text"},
        {"label": "Links", "value": n.get("links", []), "type": "chips"},
    ]
    if evo:
        rows.append(
            {
                "label": f"Evolved {len(evo)}×",
                "value": [
                    f"trigger={e.get('trigger', '?')}, field={e.get('field', '?')}" for e in evo[:5]
                ],
                "type": "list",
            }
        )
    return {"title": mid, "subtitle": n.get("timestamp", ""), "rows": rows}


def _graph(memories: dict[str, dict]) -> str:
    nodes = []
    for mid, n in memories.items():
        nodes.append(
            {
                "id": mid,
                "label": mid,
                "color": _EVOLVED_COLOR if n.get("evolution_history") else _PLAIN_COLOR,
                "shape": "dot",
                "size": 15 + min(len(n.get("links", [])) * 2, 20),
                "_detail": _note_detail(mid, n),
            }
        )

    edges = []
    seen: set[tuple[str, str]] = set()
    for mid, n in memories.items():
        for target in n.get("links", []):
            if target not in memories:
                continue
            pair = tuple(sorted([mid, target]))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append(
                {
                    "from": mid,
                    "to": target,
                    "color": "#9ca3af",
                    "_detail": {
                        "title": "link",
                        "subtitle": "A-Mem link",
                        "rows": [
                            {"label": "From", "value": mid, "type": "code"},
                            {"label": "To", "value": target, "type": "code"},
                        ],
                    },
                }
            )
    return graph_block(nodes, edges, height=600)


def render(state: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    memories = state.get("memories", {})
    label = state.get("label", state.get("system", "System"))
    body = (
        f"<h1>Memory structure — {esc(label)}</h1>"
        f"<p class='muted'>{len(memories)} memory nodes · "
        f"params: <code>{esc(state.get('params', {}))}</code></p>"
        "<h2>Graph</h2>" + _graph(memories) + "<h2>Nodes</h2>" + _node_table(memories)
    )
    (out_dir / "index.html").write_text(page(f"Memory — {label}", body, head=graph_assets()))
