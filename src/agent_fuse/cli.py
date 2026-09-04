"""Command-line interface for agent-fuse.

Commands::

    agent-fuse analyse INPUT.jsonl [--db PATH] [--run-id ID]
    agent-fuse export  INPUT.jsonl --out OUT.{html,svg,md} [--run-id ID]
    agent-fuse replay  INPUT.jsonl [--window N] [--direct-repeat N] ...
    agent-fuse stats   INPUT.jsonl

Use ``agent-fuse --help`` for the full flag list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable, List, Optional

from . import __version__
from .export import render_html, render_mermaid, render_svg
from .exceptions import DeadlockDetected
from .guard import AgentFuse, CycleConfig, FuseConfig, StagnationConfig
from .hashing import canonical_hash
from .loader import iter_jsonl, load_jsonl, normalise_row
from .store import TrajectoryStore
from .types import Action


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-fuse",
        description="In-process runtime guard for agent tool-call loops.",
    )
    p.add_argument("--version", action="version", version=f"agent-fuse {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- analyse ---------------------------------------------------------
    a = sub.add_parser(
        "analyse",
        help="Replay a JSONL trajectory against a fresh AgentFuse and report the first deadlock.",
    )
    a.add_argument("input", help="JSONL trajectory file")
    a.add_argument("--db", help="Optional SQLite destination for the replay")
    a.add_argument("--run-id", default="cli", help="Run id for persistence")
    a.add_argument("--window", type=int, default=64)
    a.add_argument("--direct-repeat", type=int, default=3)
    a.add_argument("--cycle-min", type=int, default=2)
    a.add_argument("--cycle-max", type=int, default=6)
    a.add_argument("--stagnation-window", type=int, default=8)
    a.add_argument("--stagnation-threshold", type=float, default=0.9)
    a.add_argument("--stagnation-min-failures", type=int, default=3)
    a.add_argument("--no-stagnation", action="store_true")
    a.add_argument("--allowlist", nargs="*", default=(), help="Tool names to skip")
    a.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    # ---- export ----------------------------------------------------------
    e = sub.add_parser(
        "export",
        help="Render a JSONL trajectory as HTML, SVG, or Mermaid (file extension picks format).",
    )
    e.add_argument("input", help="JSONL trajectory file")
    e.add_argument("--out", required=True, help="Output path (.html / .svg / .md)")
    e.add_argument("--run-id", default="cli")
    e.add_argument("--include-progress", action="store_true", default=True)
    e.add_argument("--no-progress", dest="include_progress", action="store_false")

    # ---- replay ----------------------------------------------------------
    r = sub.add_parser(
        "replay",
        help="Stream JSONL into a live AgentFuse (does not raise; just prints progress).",
    )
    r.add_argument("input", help="JSONL trajectory file")
    r.add_argument("--window", type=int, default=64)
    r.add_argument("--direct-repeat", type=int, default=3)
    r.add_argument("--allowlist", nargs="*", default=())
    r.add_argument("--quiet", action="store_true")

    # ---- stats -----------------------------------------------------------
    s = sub.add_parser("stats", help="Print trajectory statistics (counts, success rate, unique tools).")
    s.add_argument("input", help="JSONL trajectory file")
    s.add_argument("--json", action="store_true")

    # ---- hash ------------------------------------------------------------
    h = sub.add_parser("hash", help="Print the canonical hash for a JSON argument.")
    h.add_argument("arg_json", help="A JSON literal")

    # ---- demo -------------------------------------------------------------
    d = sub.add_parser(
        "demo",
        help="Run the in-process circuit-breaker demo (no network, no LLM).",
    )
    d.add_argument(
        "--path",
        default=None,
        help="Override the demo script path (defaults to examples/demo_circuit_breaker.py).",
    )

    return p


# ---------------------------------------------------------------------------
# analyse


def _analyse(args) -> int:
    cfg = FuseConfig(
        window=args.window,
        cycle=CycleConfig(
            direct_repeat_threshold=args.direct_repeat,
            cycle_min_length=args.cycle_min,
            cycle_max_length=args.cycle_max,
        ),
        stagnation=StagnationConfig(
            enabled=not args.no_stagnation,
            window=args.stagnation_window,
            similarity_threshold=args.stagnation_threshold,
            min_failures=args.stagnation_min_failures,
        ),
        allowlist=tuple(args.allowlist or ()),
    )

    store: Optional[TrajectoryStore] = None
    if args.db:
        store = TrajectoryStore(args.db, run_id=args.run_id)
        cfg.store_factory = lambda: store  # noqa: E731

    fuse = AgentFuse(cfg)
    last_detection = None
    count = 0
    failure = None
    for rec in iter_jsonl(args.input, default_run_id=args.run_id):
        count += 1
        action = Action(
            tool=rec.tool,
            args=rec.args_repr,
            result=rec.result_repr,
            success=rec.success,
        )
        if rec.is_progress:
            fuse.mark_progress(__import__("agent_fuse").types.ProgressSignal(token=rec.args_repr or ""))
            continue
        try:
            fuse.observe(action)
        except DeadlockDetected as exc:
            last_detection = exc
            failure = exc
            break

    out: dict = {
        "input": args.input,
        "run_id": args.run_id,
        "records_replayed": count,
        "deadlock_detected": last_detection is not None,
    }
    if last_detection is not None:
        out["detection"] = {
            "kind": last_detection.detection.kind.value,
            "message": last_detection.detection.message,
            "cycle": last_detection.detection.cycle,
            "similarity": last_detection.detection.similarity,
            "details": last_detection.detection.details,
        }
        out["hint"] = last_detection.steering_hint

    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"records replayed : {count}")
        if last_detection is None:
            print("status           : OK (no deadlock detected)")
        else:
            print(f"status           : DEADLOCK ({out['detection']['kind']})")
            print(f"message          : {out['detection']['message']}")
            if out['detection'].get('cycle'):
                print(f"cycle            : {' -> '.join(out['detection']['cycle'])}")
            if out['detection'].get('similarity') is not None:
                print(f"similarity       : {out['detection']['similarity']:.2f}")
            if out.get("hint"):
                print(f"hint             : {out['hint']}")

    if store is not None:
        store.close()
    return 1 if failure is not None else 0


# ---------------------------------------------------------------------------
# export


def _export(args) -> int:
    records = load_jsonl(args.input, default_run_id=args.run_id)
    if not args.include_progress:
        records = [r for r in records if not r.is_progress]

    lower = args.out.lower()
    if lower.endswith((".html", ".htm")):
        body = render_html(records, run_id=args.run_id, version=__version__)
    elif lower.endswith(".svg"):
        body = render_svg(records)
    elif lower.endswith((".md", ".mmd", ".mermaid")):
        body = (
            "```mermaid\n"
            + render_mermaid(records)
            + "```\n"
        )
    else:
        print(
            f"unknown extension on {args.out!r}; expected .html / .svg / .md",
            file=sys.stderr,
        )
        return 2

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"wrote {args.out} ({len(records)} record(s))")
    return 0


# ---------------------------------------------------------------------------
# replay


def _replay(args) -> int:
    cfg = FuseConfig(
        window=args.window,
        cycle=CycleConfig(direct_repeat_threshold=args.direct_repeat),
        allowlist=tuple(args.allowlist or ()),
    )
    fuse = AgentFuse(cfg)
    seen = 0
    for rec in iter_jsonl(args.input, default_run_id="cli"):
        seen += 1
        if rec.is_progress:
            fuse.mark_progress(__import__("agent_fuse").types.ProgressSignal(token=rec.args_repr or ""))
            if not args.quiet:
                print(f"[{seen}] progress: {rec.args_repr}")
            continue
        action = Action(tool=rec.tool, args=rec.args_repr, result=rec.result_repr, success=rec.success)
        try:
            fuse.observe(action)
            if not args.quiet:
                print(f"[{seen}] ok      {rec.tool} ({rec.args_hash})")
        except DeadlockDetected as exc:
            print(f"[{seen}] DEADLOCK ({exc.detection.kind.value}): {exc.detection.message}")
            return 1
    # Always print a final summary — --quiet only suppresses per-line output.
    print(f"replayed {seen} records; no deadlock detected.")
    return 0


# ---------------------------------------------------------------------------
# stats


def _stats(args) -> int:
    counts = {"total": 0, "ok": 0, "fail": 0, "progress": 0, "tools": {}}
    unique_hashes = set()
    for rec in iter_jsonl(args.input, default_run_id="cli"):
        counts["total"] += 1
        if rec.is_progress:
            counts["progress"] += 1
            continue
        if rec.success is True:
            counts["ok"] += 1
        elif rec.success is False:
            counts["fail"] += 1
        counts["tools"][rec.tool] = counts["tools"].get(rec.tool, 0) + 1
        if rec.args_hash:
            unique_hashes.add(rec.args_hash)

    counts["unique_args"] = len(unique_hashes)
    summary = {
        "total": counts["total"],
        "ok": counts["ok"],
        "fail": counts["fail"],
        "progress_marks": counts["progress"],
        "unique_args": counts["unique_args"],
        "tools": counts["tools"],
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for k in ("total", "ok", "fail", "progress_marks", "unique_args"):
            print(f"{k:<16} {summary[k]}")
        if summary["tools"]:
            print("tools:")
            for tool, n in sorted(summary["tools"].items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {tool:<24} {n}")
    return 0


# ---------------------------------------------------------------------------
# hash


def _hash(args) -> int:
    try:
        payload = json.loads(args.arg_json)
    except json.JSONDecodeError as exc:
        print(f"invalid JSON: {exc}", file=sys.stderr)
        return 2
    print(canonical_hash(payload))
    return 0


# ---------------------------------------------------------------------------
# demo


def _demo(args) -> int:
    """Run the bundled circuit-breaker demo as ``python examples/demo_circuit_breaker.py``.

    The demo is stdlib-only and exits with status 0 on success. We locate
    the script relative to the installed package (or, in a source checkout,
    the repo root) and exec it through ``runpy`` so the user sees exactly
    the same output as a direct ``python examples/demo_circuit_breaker.py``
    invocation.
    """
    import runpy

    # Resolve candidate paths. Prefer (a) an explicit --path, (b) a script
    # shipped alongside the installed package (via wheel data), (c)
    # examples/ in CWD (source checkout).
    candidates: List[str] = []
    if args.path:
        candidates.append(args.path)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    # Wheel data layout: <site-packages>/agent_fuse/share/examples/...
    candidates.append(
        os.path.join(pkg_dir, "share", "examples", "demo_circuit_breaker.py")
    )
    # Source layout: <repo>/src/agent_fuse/cli.py -> <repo>/examples/...
    candidates.append(
        os.path.join(os.path.dirname(pkg_dir), "..", "examples", "demo_circuit_breaker.py")
    )
    candidates.append("examples/demo_circuit_breaker.py")

    chosen: Optional[str] = None
    for c in candidates:
        c_abs = os.path.abspath(c)
        if os.path.isfile(c_abs):
            chosen = c_abs
            break
    if chosen is None:
        print(
            "could not locate examples/demo_circuit_breaker.py — pass --path",
            file=sys.stderr,
        )
        return 2

    print(f"[agent-fuse] running demo: {chosen}")
    try:
        runpy.run_path(chosen, run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


# ---------------------------------------------------------------------------
# entry


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyse":
            return _analyse(args)
        if args.command == "export":
            return _export(args)
        if args.command == "replay":
            return _replay(args)
        if args.command == "stats":
            return _stats(args)
        if args.command == "hash":
            return _hash(args)
        if args.command == "demo":
            return _demo(args)
    except FileNotFoundError as exc:
        print(f"file not found: {exc.filename}", file=sys.stderr)
        return 2
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())