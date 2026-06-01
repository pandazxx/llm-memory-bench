"""Shared HTML scaffolding for visualizations."""

from __future__ import annotations

import html

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
pre.mermaid {{ background: #fafafa; border: 1px solid #eee; padding: 1rem; }}
</style>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
mermaid.initialize({{ startOnLoad: true, securityLevel: "loose" }});
</script>
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


def page(title: str, body: str) -> str:
    return PAGE.format(title=esc(title), body=body)


def mermaid_id(raw: str) -> str:
    """Safe node id for mermaid."""
    return "n_" + "".join(c if c.isalnum() else "_" for c in str(raw))
