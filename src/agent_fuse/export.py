"""Visualisation exporters for trajectory data.

Three formats are supported:

* **HTML** — single-file, self-contained, inline SVG with clickable nodes.
* **SVG** — raw SVG file, no HTML wrapper.
* **Mermaid** — ``graph TD`` syntax for live rendering in Markdown / GitHub.

All exporters operate on a list of :class:`TrajectoryRecord`. They are
pure functions — no IO happens inside; the caller writes the returned
string to disk.
"""

from __future__ import annotations

import html as _html
from typing import Iterable, List

from .types import TrajectoryRecord


# ---------------------------------------------------------------------------
# shared helpers


_KIND_COLOURS = {
    "progress": "#16a34a",   # green
    "repeat": "#dc2626",     # red
    "stagnate": "#d97706",   # amber
    "default": "#2563eb",    # blue
    "fail": "#b91c1c",       # dark red
}


def _node_colour(record: TrajectoryRecord) -> str:
    if record.is_progress:
        return _KIND_COLOURS["progress"]
    if record.success is False:
        return _KIND_COLOURS["fail"]
    return _KIND_COLOURS["default"]


def _short(text: str, limit: int = 40) -> str:
    text = (text or "").replace("\n", " ").replace("\r", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# HTML + inline SVG (standalone)


def _svg_for_records(records: Iterable[TrajectoryRecord], width: int = 960) -> str:
    rec_list = list(records)
    if not rec_list:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="80" '
            'viewBox="0 0 960 80"><text x="20" y="40" font-family="monospace" '
            'fill="#6b7280">empty trajectory</text></svg>'
        )

    margin_x = 40
    margin_y = 40
    box_w = 180
    box_h = 56
    gap = 24
    total_w = max(width, margin_x * 2 + len(rec_list) * (box_w + gap))
    total_h = margin_y * 2 + box_h

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}" font-family="ui-monospace,Menlo,monospace" '
        f'font-size="12">'
    ]
    parts.append(
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/></marker></defs>'
    )

    for idx, rec in enumerate(rec_list):
        x = margin_x + idx * (box_w + gap)
        y = margin_y
        colour = _node_colour(rec)
        label = f"{idx + 1}. {_short(rec.tool, 18)}"
        sub = _short(rec.args_hash or rec.args_repr or "", 22)
        success = "OK" if rec.success is True else ("FAIL" if rec.success is False else "·")
        title_text = _html.escape(
            f"#{rec.sequence} {rec.tool}\nargs_hash: {rec.args_hash}\n"
            f"args: {_short(rec.args_repr, 200)}\n"
            f"result: {_short(rec.result_repr, 200)}\n"
            f"success: {success}"
        )
        parts.append(
            f'<g><title>{title_text}</title>'
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" '
            f'fill="white" stroke="{colour}" stroke-width="2"/>'
            f'<text x="{x + 10}" y="{y + 20}" fill="#111827" font-weight="600">'
            f'{_html.escape(label)}</text>'
            f'<text x="{x + 10}" y="{y + 38}" fill="#6b7280">'
            f'{_html.escape(sub)}</text>'
            f'<text x="{x + box_w - 10}" y="{y + 20}" fill="{colour}" '
            f'text-anchor="end" font-weight="600">{success}</text>'
            f'</g>'
        )
        if idx + 1 < len(rec_list):
            x1 = x + box_w
            x2 = x + box_w + gap
            cy = y + box_h // 2
            parts.append(
                f'<line x1="{x1}" y1="{cy}" x2="{x2}" y2="{cy}" '
                f'stroke="#6b7280" stroke-width="1.5" marker-end="url(#arr)"/>'
            )

    parts.append("</svg>")
    return "".join(parts)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>agent-fuse trajectory</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; color: #111827; }}
  h1 {{ font-size: 20px; margin: 0 0 8px; }}
  .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; margin-top: 24px; width: 100%; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f9fafb; font-weight: 600; }}
  tr.progress td {{ background: #f0fdf4; }}
  tr.fail td {{ background: #fef2f2; }}
  code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 12px; }}
  .legend span {{ display: inline-block; margin-right: 12px; font-size: 12px; color: #4b5563; }}
  .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>
<h1>agent-fuse trajectory</h1>
<p class="meta">
  run_id <code>{run_id}</code> · {count} record{suffix} · generated by agent-fuse {version}
</p>
<div class="legend">
  <span><span class="swatch" style="background:#16a34a"></span>progress</span>
  <span><span class="swatch" style="background:#b91c1c"></span>failure</span>
  <span><span class="swatch" style="background:#2563eb"></span>ok / unknown</span>
</div>
<div class="graph">{svg}</div>
<table>
  <thead><tr>
    <th>#</th><th>tool</th><th>args_hash</th><th>result</th><th>success</th>
  </tr></thead>
  <tbody>
{rows}
  </tbody>
</table>
</body>
</html>
"""


def render_html(
    records: Iterable[TrajectoryRecord],
    run_id: str = "",
    version: str = "0.1.0",
) -> str:
    """Return a self-contained HTML document with an inline SVG timeline."""
    rec_list = list(records)
    svg = _svg_for_records(rec_list)
    rows: List[str] = []
    for rec in rec_list:
        cls = "progress" if rec.is_progress else ("fail" if rec.success is False else "")
        success = (
            "✓" if rec.success is True
            else ("✗" if rec.success is False else "·")
        )
        rows.append(
            "<tr class='{cls}'>"
            "<td>{seq}</td>"
            "<td>{tool}</td>"
            "<td><code>{args_hash}</code></td>"
            "<td>{result}</td>"
            "<td>{success}</td>"
            "</tr>".format(
                cls=cls,
                seq=rec.sequence,
                tool=_html.escape(rec.tool),
                args_hash=_html.escape(rec.args_hash or "-"),
                result=_html.escape(_short(rec.result_repr, 80)),
                success=success,
            )
        )
    return HTML_TEMPLATE.format(
        run_id=_html.escape(run_id or "(no run id)"),
        count=len(rec_list),
        suffix="" if len(rec_list) == 1 else "s",
        svg=svg,
        rows="\n".join(rows),
        version=_html.escape(version),
    )


def render_svg(records: Iterable[TrajectoryRecord]) -> str:
    """Return a standalone SVG document (no HTML wrapper)."""
    svg = _svg_for_records(records)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
        '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
        + svg
    )


# ---------------------------------------------------------------------------
# Mermaid


def _mermaid_safe(text: str) -> str:
    """Strip characters Mermaid parses (quotes, brackets, pipes)."""
    if not text:
        return ""
    return (
        text.replace('"', "'")
        .replace("[", "(")
        .replace("]", ")")
        .replace("|", "/")
        .replace("\n", " ")
    )


def render_mermaid(records: Iterable[TrajectoryRecord]) -> str:
    """Return a Mermaid ``graph LR`` source string."""
    rec_list = list(records)
    lines: List[str] = ["graph LR"]
    if not rec_list:
        lines.append("  empty[empty trajectory]")
        return "\n".join(lines)

    # Class definitions for the legend.
    lines.append("  classDef progress fill:#16a34a,stroke:#16a34a,color:#fff")
    lines.append("  classDef fail fill:#b91c1c,stroke:#b91c1c,color:#fff")
    lines.append("  classDef ok fill:#2563eb,stroke:#2563eb,color:#fff")

    for idx, rec in enumerate(rec_list):
        node_id = f"n{idx}"
        label = _mermaid_safe(f"{idx + 1}. {rec.tool}")
        if rec.result_repr:
            label += "<br/>" + _mermaid_safe(_short(rec.result_repr, 30))
        if rec.is_progress:
            label += " (progress)"
        elif rec.success is False:
            label += " (fail)"
        # Mermaid node id must not contain special chars; rec.tool is freeform.
        quoted_label = label.replace('"', "'")
        lines.append(f'  {node_id}["{quoted_label}"]')
        if rec.is_progress:
            lines.append(f"  class {node_id} progress")
        elif rec.success is False:
            lines.append(f"  class {node_id} fail")
        else:
            lines.append(f"  class {node_id} ok")
        if idx + 1 < len(rec_list):
            lines.append(f"  {node_id} --> n{idx + 1}")
    return "\n".join(lines) + "\n"