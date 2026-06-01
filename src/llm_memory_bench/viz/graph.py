"""Reusable interactive graph component (vis-network + side detail panel).

``graph_assets()`` returns the one-time ``<head>`` payload (vis-network from a
CDN, styles, and shared helper JS). ``graph_block()`` returns an embeddable
fixed-height graph: a canvas on the left, a detail panel on the right. Clicking
a node or edge populates the panel from its ``_detail`` payload and dims
non-neighbours; clicking empty space clears the selection.

Node shape:  {"id", "label", "color", "shape", "size", "_detail"}
Edge shape:  {"from", "to", "color"?, "width"?, "dashes"?, "label"?, "_detail"?}
Detail shape: {"title", "subtitle"?, "rows": [{"label", "value", "type"}]}
              row types: "text" (default), "chips", "list", "code".
"""

from __future__ import annotations

import itertools
import json
from typing import Any

from .common import esc

_VIS_SRC = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"

_counter = itertools.count(1)

DEFAULT_OPTIONS: dict[str, Any] = {
    "physics": {
        "barnesHut": {
            "gravitationalConstant": -3000,
            "centralGravity": 0.3,
            "springLength": 120,
            "springConstant": 0.04,
        },
        "minVelocity": 0.5,
        "stabilization": {"iterations": 200},
    },
    "interaction": {"hover": False, "tooltipDelay": 99999},
    "edges": {"smooth": False, "color": {"inherit": False}},
}

TRACE_OPTIONS: dict[str, Any] = {
    "physics": {
        "barnesHut": {
            "gravitationalConstant": -2500,
            "centralGravity": 0.4,
            "springLength": 110,
            "springConstant": 0.06,
        },
        "minVelocity": 0.5,
        "stabilization": {"iterations": 150},
    },
    "interaction": {"hover": False, "tooltipDelay": 99999},
    "edges": {"smooth": False, "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}}},
}

_STYLE = """
.graph-wrap { display: flex; border: 1px solid #ddd; border-radius: 4px;
  overflow: hidden; margin: 1rem 0; }
.graph-canvas { flex: 1; min-width: 0; background: #fff; }
.graph-panel { width: 340px; flex-shrink: 0; padding: 0.8em 1em; overflow-y: auto;
  background: #fafafa; border-left: 1px solid #eee; }
.graph-panel .placeholder { color: #9ca3af; font-style: italic; }
.graph-panel h3 { margin: 0 0 0.2em 0; font-size: 1.05em;
  font-family: ui-monospace, monospace; word-break: break-word; }
.graph-panel .sub { color: #6b7280; font-size: 0.85em; margin-bottom: 0.7em; }
.graph-panel .row { margin-bottom: 0.6em; }
.graph-panel .label { font-weight: 600; color: #6b7280; font-size: 0.72em;
  text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.2em; }
.graph-panel .value { white-space: pre-wrap; word-break: break-word; line-height: 1.4; }
.graph-panel code { background: #e5e7eb; padding: 1px 5px; border-radius: 3px;
  font-size: 0.85em; font-family: ui-monospace, monospace;
  display: inline-block; margin: 1px 2px 1px 0; }
.graph-panel ul { padding-left: 1.3em; margin: 0.2em 0; }
"""

_HELPER_JS = """
function lmbEsc(s) {
  return String(s == null ? '' : s)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}
function lmbChips(items) {
  if (!items || !items.length) return '<i class="placeholder">(none)</i>';
  return items.map(s => `<code>${lmbEsc(s)}</code>`).join(' ');
}
function lmbRows(rows) {
  return (rows || []).map(r => {
    let v;
    if (r.type === 'chips') v = lmbChips(r.value);
    else if (r.type === 'list') {
      v = (r.value && r.value.length)
        ? '<ul>' + r.value.map(x => `<li>${lmbEsc(x)}</li>`).join('') + '</ul>'
        : '<i class="placeholder">(none)</i>';
    } else if (r.type === 'code') v = `<code>${lmbEsc(r.value)}</code>`;
    else v = lmbEsc(r.value);
    return `<div class="row"><div class="label">${lmbEsc(r.label)}</div>` +
           `<div class="value">${v}</div></div>`;
  }).join('');
}
function lmbDetail(panel, d, ph) {
  if (!d) { panel.innerHTML = `<p class="placeholder">${lmbEsc(ph)}</p>`; return; }
  const sub = d.subtitle ? `<div class="sub">${lmbEsc(d.subtitle)}</div>` : '';
  panel.innerHTML = `<h3>${lmbEsc(d.title || '')}</h3>${sub}${lmbRows(d.rows)}`;
}
function lmbGraph(gid, pid, nodesRaw, edgesRaw, options, ph) {
  const DEFAULT = {};
  nodesRaw.forEach(n => { DEFAULT[n.id] = n.color; });
  const nodes = new vis.DataSet(nodesRaw);
  const edges = new vis.DataSet(edgesRaw);
  const net = new vis.Network(document.getElementById(gid), { nodes, edges }, options);
  const panel = document.getElementById(pid);
  function highlight(id) {
    if (!id) {
      nodes.update(nodesRaw.map(n => ({ id: n.id, color: DEFAULT[n.id] })));
      return;
    }
    const keep = new Set(net.getConnectedNodes(id));
    keep.add(id);
    nodes.update(nodesRaw.map(n => ({
      id: n.id,
      color: keep.has(n.id) ? DEFAULT[n.id] : 'rgba(200,200,200,0.35)',
    })));
  }
  net.on('selectNode', p => {
    const n = nodes.get(p.nodes[0]);
    lmbDetail(panel, n && n._detail, ph);
    highlight(p.nodes[0]);
  });
  net.on('selectEdge', p => {
    if (p.nodes.length) return;
    const e = edges.get(p.edges[0]);
    lmbDetail(panel, e && e._detail, ph);
  });
  net.on('deselectNode', () => { lmbDetail(panel, null, ph); highlight(null); });
  net.on('click', p => {
    if (!p.nodes.length && !p.edges.length) { lmbDetail(panel, null, ph); highlight(null); }
  });
}
"""


def graph_assets() -> str:
    """One-time ``<head>`` payload: vis-network CDN, styles, helper JS."""
    return f'<script src="{_VIS_SRC}"></script><style>{_STYLE}</style><script>{_HELPER_JS}</script>'


def graph_block(
    nodes: list[dict],
    edges: list[dict],
    *,
    options: dict | None = None,
    height: int = 560,
    placeholder: str = "Click a node or edge to inspect.",
) -> str:
    """Embeddable interactive graph. Requires ``graph_assets()`` in the page head."""
    gid = f"g{next(_counter)}"
    opts = json.dumps(options if options is not None else DEFAULT_OPTIONS)
    return (
        f'<div class="graph-wrap" style="height:{height}px">'
        f'<div id="{gid}_g" class="graph-canvas"></div>'
        f'<div id="{gid}_p" class="graph-panel">'
        f'<p class="placeholder">{esc(placeholder)}</p></div></div>'
        f"<script>lmbGraph("
        f'"{gid}_g","{gid}_p",'
        f"{json.dumps(nodes)},{json.dumps(edges)},{opts},{json.dumps(placeholder)});</script>"
    )
