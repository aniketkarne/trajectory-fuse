"""Minimal end-to-end demo of trajectory-fuse.

Run::

    python examples/demo.py

Generates a small synthetic trajectory with a forced deadlock on the 4th
step, then exports the trajectory to HTML / SVG / Mermaid in ``artifacts/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from trajectory_fuse import (
    Action,
    AgentFuse,
    DeadlockDetected,
    DetectionKind,
    FuseConfig,
)
from trajectory_fuse.export import render_html, render_mermaid, render_svg
from trajectory_fuse.loader import load_jsonl


def synthetic_trajectory() -> list:
    """Build a small trajectory with three distinct deadlocks baked in."""
    return [
        {"tool": "search", "args": {"q": "weather"}, "result": "sunny", "success": True},
        {"tool": "summarise", "args": {"text": "sunny"}, "result": "ok", "success": True},
        {"tool": "search", "args": {"q": "weather"}, "result": "sunny", "success": True},
        {"tool": "summarise", "args": {"text": "sunny"}, "result": "ok", "success": True},
        {"tool": "search", "args": {"q": "weather"}, "result": "sunny", "success": True},
        {"tool": "summarise", "args": {"text": "sunny"}, "result": "ok", "success": True},
        # Then 3 identical failing calls — stagnation.
        {"tool": "weather_api", "args": {"city": "nyc"}, "result": "timeout", "success": False},
        {"tool": "weather_api", "args": {"city": "nyc"}, "result": "timeout", "success": False},
        {"tool": "weather_api", "args": {"city": "nyc"}, "result": "timeout", "success": False},
    ]


def main() -> None:
    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    rows = synthetic_trajectory()
    jsonl_path = out / "demo.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {jsonl_path}")

    fuse = AgentFuse(FuseConfig(window=20))
    try:
        for r in rows:
            fuse.observe(
                Action(
                    tool=r["tool"],
                    args=r["args"],
                    result=r["result"],
                    success=r.get("success"),
                )
            )
    except DeadlockDetected as exc:
        print(f"deadlock detected at {exc.detection.kind.value}: {exc.detection.message}")

    # Export via the helpers. Use the loader to normalise raw dicts into
    # TrajectoryRecord instances (which the exporters expect).
    records = load_jsonl(jsonl_path, default_run_id="demo")
    (out / "demo.html").write_text(render_html(records), encoding="utf-8")
    (out / "demo.svg").write_text(render_svg(records), encoding="utf-8")
    (out / "demo.md").write_text(f"```mermaid\n{render_mermaid(records)}```\n", encoding="utf-8")
    print(f"wrote {out}/demo.{{html,svg,md}}")


if __name__ == "__main__":
    main()