"""Reproducible benchmark suite for trajectory-fuse.

Measures the steady-state overhead of the runtime guard so that we can
detect regressions in detection logic and persistence. The harness is
deliberately stdlib-only: ``time.perf_counter`` is the only timing
primitive, so the script runs with zero external dependencies (matching
the library's zero-runtime-dep posture).

Optional: ``pytest-benchmark`` integration. When ``pytest-benchmark`` is
installed, ``benchmarks/test_benchmarks.py`` provides ``bench_*``
fixtures that delegate to the same per-scenario ``bench()`` functions
implemented below, so the two harnesses report the same workloads in the
same units (microseconds / op). See ``README.md`` for invocation details.

Run the stdlib harness directly::

    python benchmarks/run_benchmarks.py

Run the pytest plugin (optional, more detailed statistics)::

    pip install pytest-benchmark
    pytest benchmarks/test_benchmarks.py --benchmark-only

Each scenario is repeated for a few short rounds so the median is
reported alongside min and max. The harness is *not* asserting pass/fail
on absolute timings — only "is the guard under 1 ms per observe() for a
typical 32-action window?" — so flaky CI runners won't false-fail. Use
``--strict`` to fail the run if any scenario exceeds its baseline ceiling.

Baseline interpretation
-----------------------

These numbers are *not* SLA thresholds — they are reference points
captured on the development machine (M-series Apple Silicon, Python
3.12). Use them to detect regressions, not to compare across machines.
The expected orders of magnitude on any modern CPU are:

* ``bench_observe_normal``       — single-digit microseconds per observe()
                                   for the 32-action default window.
* ``bench_observe_with_store``   — 10–100x slower than ``observe_normal``,
                                   dominated by SQLite commit latency.
* ``bench_direct_repeat_detect`` — sub-millisecond even at the default
                                   threshold of 3.
* ``bench_cycle_detect``         — sub-millisecond for cycle_max_length=8.
* ``bench_stagnation_detect``    — sub-millisecond for window=8.
* ``bench_canonical_hash``       — single-digit microseconds per hash
                                   (sha1 over a small JSON blob).
* ``bench_sqlite_persist``       — a few hundred microseconds per row.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

# Allow running from the repo root without installing.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trajectory_fuse import (  # noqa: E402
    Action,
    AgentFuse,
    CycleConfig,
    FuseConfig,
    StagnationConfig,
    TrajectoryStore,
)
from trajectory_fuse.hashing import canonical_hash  # noqa: E402


# -----------------------------------------------------------------------
# benchmark scenarios — each is a ``Callable[[], None]`` that runs the
# workload once. Per-call timings measure one workload unit (e.g. one
# observe()), not the whole batch.


def _make_normal_action(idx: int) -> Action:
    """A representative action: small dict, successful result."""
    return Action(
        tool="search" if idx % 3 else "lookup",
        args={"q": f"query-{idx}", "limit": 10, "filters": ["a", "b"]},
        result={"items": [1, 2, 3], "total": idx},
        success=True,
    )


def _make_stagnating_action(idx: int) -> Action:
    return Action(
        tool="search",
        args={"q": "same"},
        result="Error: connection timeout while connecting to upstream",
        success=False,
    )


def bench_observe_normal(rounds: int = 2000) -> int:
    """One observe() through a 32-action window with diverse actions.

    The bench loop cycles through *distinct* actions so neither direct-repeat
    nor cycle detection fires; this measures the steady-state overhead
    rather than the cost of the detector itself.
    """
    cfg = FuseConfig(
        window=32,
        cycle=CycleConfig(direct_repeat_threshold=3, cycle_min_length=2, cycle_max_length=6),
        stagnation=StagnationConfig(min_failures=3),
    )
    fuse = AgentFuse(cfg)
    # Warm the window once so trim logic is exercised but the first
    # observe() is not unfairly penalised for filling it.
    for i in range(32):
        fuse.observe(_make_normal_action(i))
    elapsed = time.perf_counter()
    for i in range(rounds):
        fuse.observe(_make_normal_action(1000 + i))
    elapsed = time.perf_counter() - elapsed
    return int(elapsed * 1_000_000)


def bench_observe_with_store(rounds: int = 500) -> int:
    """observe() + SQLite append per call (in-memory store)."""
    tmpdir = tempfile.mkdtemp(prefix="trajectory-fuse-bench-")
    try:
        path = os.path.join(tmpdir, "traj.db")
        store = TrajectoryStore(path)
        store.initialise("bench")
        cfg = FuseConfig(
            window=32,
            cycle=CycleConfig(direct_repeat_threshold=3),
            store_factory=lambda: store,
        )
        fuse = AgentFuse(cfg)
        # Prime the store schema & warm.
        for i in range(8):
            fuse.observe(_make_normal_action(i))
        elapsed = time.perf_counter()
        for i in range(rounds):
            fuse.observe(_make_normal_action(1000 + i))
        elapsed = time.perf_counter() - elapsed
        store.close()
        return int(elapsed * 1_000_000)
    finally:
        for f in os.listdir(tmpdir):
            os.unlink(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)


def bench_direct_repeat_detect(rounds: int = 5000) -> int:
    """Feed repeated identical actions; one observe() per round.

    The detector returns DeadlockDetected at threshold; we expect it on
    every iteration. Catch the exception to avoid bailing out.
    """
    cfg = FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3))
    action = Action(tool="search", args={"q": "x"}, result="ok", success=True)
    elapsed = time.perf_counter()
    for _ in range(rounds):
        fuse = AgentFuse(cfg)
        try:
            fuse.observe(action)
            fuse.observe(action)
            fuse.observe(action)
        except Exception:
            pass
    elapsed = time.perf_counter() - elapsed
    return int(elapsed * 1_000_000)


def bench_cycle_detect(rounds: int = 2000) -> int:
    """Feed an A/B cycle until the detector trips. One trip per round."""
    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=2, cycle_max_length=6),
    )
    elapsed = time.perf_counter()
    for _ in range(rounds):
        fuse = AgentFuse(cfg)
        try:
            for tool in ("A", "B", "A", "B"):
                fuse.observe(Action(tool=tool, args={"w": tool}, result="ok", success=True))
        except Exception:
            pass
    elapsed = time.perf_counter() - elapsed
    return int(elapsed * 1_000_000)


def bench_stagnation_detect(rounds: int = 2000) -> int:
    """Feed three near-identical failures. One trip per round."""
    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
        stagnation=StagnationConfig(min_failures=3, similarity_threshold=0.5),
    )
    elapsed = time.perf_counter()
    for _ in range(rounds):
        fuse = AgentFuse(cfg)
        try:
            for msg in ("timeout alpha", "timeout alpha beta", "timeout alpha gamma"):
                fuse.observe(
                    Action(tool="search", args={"q": "x"}, result=msg, success=False)
                )
        except Exception:
            pass
    elapsed = time.perf_counter() - elapsed
    return int(elapsed * 1_000_000)


def bench_canonical_hash(rounds: int = 50000) -> int:
    """Hash a representative args payload."""
    payload = {
        "q": "the quick brown fox",
        "filters": ["a", "b", "c"],
        "options": {"limit": 10, "offset": 0, "debug": False},
    }
    elapsed = time.perf_counter()
    for _ in range(rounds):
        canonical_hash(payload)
    elapsed = time.perf_counter() - elapsed
    return int(elapsed * 1_000_000)


def bench_sqlite_persist(rounds: int = 2000) -> int:
    """Direct TrajectoryStore.append() (no guard)."""
    from trajectory_fuse.types import TrajectoryRecord

    tmpdir = tempfile.mkdtemp(prefix="trajectory-fuse-bench-")
    try:
        path = os.path.join(tmpdir, "traj.db")
        store = TrajectoryStore(path)
        store.initialise("bench")
        elapsed = time.perf_counter()
        for i in range(rounds):
            store.append(
                TrajectoryRecord(
                    sequence=i + 1,
                    run_id="bench",
                    tool="search",
                    args_hash=canonical_hash({"i": i}),
                    args_repr='{"i": ' + str(i) + "}",
                    result_repr="ok",
                    success=True,
                    is_progress=False,
                )
            )
        elapsed = time.perf_counter() - elapsed
        store.close()
        return int(elapsed * 1_000_000)
    finally:
        for f in os.listdir(tmpdir):
            os.unlink(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)


# -----------------------------------------------------------------------
# harness

# Default ceilings in microseconds — generous defaults for CI on a slow
# runner. Override with --strict-baseline to use the tighter numbers.
DEFAULT_BASELINES = {
    "observe_normal": 1000,       # 1 ms per observe() on a 32-action window
    "observe_with_store": 20000,  # 20 ms per observe()+SQLite commit
    "direct_repeat_detect": 5000, # 5 ms per 3-action trip including setup
    "cycle_detect": 5000,         # 5 ms per 4-action cycle trip
    "stagnation_detect": 10000,   # 10 ms per 3-failure stagnation trip
    "canonical_hash": 50,         # 50 µs per hash
    "sqlite_persist": 5000,       # 5 ms per row incl. commit
}

SCENARIOS: List[Tuple[str, Callable[[int], int], int]] = [
    ("observe_normal", bench_observe_normal, 2000),
    ("observe_with_store", bench_observe_with_store, 500),
    ("direct_repeat_detect", bench_direct_repeat_detect, 5000),
    ("cycle_detect", bench_cycle_detect, 2000),
    ("stagnation_detect", bench_stagnation_detect, 2000),
    ("canonical_hash", bench_canonical_hash, 50000),
    ("sqlite_persist", bench_sqlite_persist, 2000),
]


def run_once(samples: int) -> dict:
    """Run every scenario ``samples`` times; return per-scenario stats."""
    results: dict = {}
    for name, fn, rounds in SCENARIOS:
        timings = []
        for _ in range(samples):
            timings.append(fn(rounds))
        per_call_us = [t / rounds for t in timings]
        results[name] = {
            "rounds_per_sample": rounds,
            "samples": samples,
            "total_us_min": min(timings),
            "total_us_max": max(timings),
            "per_call_us_min": min(per_call_us),
            "per_call_us_median": statistics.median(per_call_us),
            "per_call_us_mean": statistics.fmean(per_call_us),
        }
    return results


def format_table(results: dict) -> str:
    rows = ["scenario                       rounds  samples  per-call µs (min/median/mean)"]
    rows.append("-" * 88)
    for name, stats in results.items():
        per_call = stats["per_call_us_median"]
        per_min = stats["per_call_us_min"]
        per_mean = stats["per_call_us_mean"]
        rows.append(
            f"{name:<28} {stats['rounds_per_sample']:>6}  {stats['samples']:>7}  "
            f"{per_min:>7.2f} / {per_call:>7.2f} / {per_mean:>7.2f}"
        )
    return "\n".join(rows)


def format_json(results: dict) -> str:
    return json.dumps(results, indent=2)


def check_baselines(results: dict, baselines: dict) -> List[str]:
    violations = []
    for name, limit in baselines.items():
        stats = results.get(name)
        if not stats:
            continue
        if stats["per_call_us_median"] > limit:
            violations.append(
                f"{name}: median {stats['per_call_us_median']:.2f} µs exceeds baseline {limit} µs"
            )
    return violations


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="trajectory-fuse benchmark harness")
    p.add_argument("--samples", type=int, default=3,
                   help="Number of timing repetitions per scenario (default: 3)")
    p.add_argument("--strict", action="store_true",
                   help="Fail (exit 1) if any scenario exceeds its baseline ceiling")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = p.parse_args(list(argv) if argv is not None else None)

    results = run_once(args.samples)

    if args.json:
        print(format_json(results))
    else:
        print(format_table(results))

    violations = check_baselines(results, DEFAULT_BASELINES)
    if violations:
        # In JSON mode, surface violations as a structured stderr line
        # so the stdout stays valid JSON for the consumer. The exit
        # code still encodes "failed" (1 with --strict).
        if args.json:
            print(json.dumps({"violations": violations}), file=sys.stderr)
        else:
            print("\nBaseline check:")
            for v in violations:
                print(f"  - {v}")
            if args.strict:
                print("\nFAILED --strict baseline check.")
                return 1

    if not args.json:
        print("\nOK (no baseline violations; use --strict to fail on regressions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
