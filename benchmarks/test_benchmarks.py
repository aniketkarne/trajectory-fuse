"""Optional pytest-benchmark plugin integration for agent-fuse.

These tests share workload definitions with ``benchmarks/run_benchmarks.py``
(imported there as ``bench_*`` callables) but use ``pytest-benchmark``'s
``benchmark`` fixture when it is installed. If ``pytest-benchmark`` is
*not* installed, every test is skipped silently so CI on the minimal
dev-dependency set still passes.

Run with::

    pip install pytest-benchmark
    pytest benchmarks/test_benchmarks.py --benchmark-only

The CLI smoke check (``pytest benchmarks --collect-only`` or
``pytest benchmarks/test_benchmarks.py --benchmark-only --benchmark-min-rounds=1``)
is what CI runs, because it doesn't depend on absolute timing.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow importing the stdlib harness without installing the package.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

from agent_fuse import (  # noqa: E402
    Action,
    AgentFuse,
    CycleConfig,
    FuseConfig,
    StagnationConfig,
    TrajectoryStore,
    canonical_hash,
)


def _normal_action(idx: int) -> Action:
    return Action(
        tool="search" if idx % 3 else "lookup",
        args={"q": f"query-{idx}", "limit": 10, "filters": ["a", "b"]},
        result={"items": [1, 2, 3], "total": idx},
        success=True,
    )


def test_bench_observe_normal(benchmark):
    cfg = FuseConfig(window=32)
    fuse = AgentFuse(cfg)
    for i in range(32):
        fuse.observe(_normal_action(i))
    state = {"i": 0}

    def step():
        # Distinct args every iteration so neither direct-repeat nor
        # cycle detection trips mid-bench.
        i = state["i"]
        state["i"] = i + 1
        fuse.observe(_normal_action(1000 + i))

    benchmark(step)


def test_bench_observe_with_store(benchmark, tmp_path):
    db = tmp_path / "traj.db"
    store = TrajectoryStore(str(db), run_id="bench")
    cfg = FuseConfig(window=32, store_factory=lambda: store)
    fuse = AgentFuse(cfg)
    for i in range(8):
        fuse.observe(_normal_action(i))
    state = {"i": 0}

    def step():
        i = state["i"]
        state["i"] = i + 1
        fuse.observe(_normal_action(1000 + i))

    try:
        benchmark(step)
    finally:
        store.close()


def test_bench_direct_repeat_detect(benchmark):
    cfg = FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3))
    action = Action(tool="search", args={"q": "x"}, result="ok", success=True)

    def trip():
        f = AgentFuse(cfg)
        try:
            f.observe(action)
            f.observe(action)
            f.observe(action)
        except Exception:
            pass

    benchmark(trip)


def test_bench_cycle_detect(benchmark):
    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=2, cycle_max_length=6),
    )

    def trip():
        f = AgentFuse(cfg)
        try:
            for tool in ("A", "B", "A", "B"):
                f.observe(Action(tool=tool, args={"w": tool}, result="ok", success=True))
        except Exception:
            pass

    benchmark(trip)


def test_bench_stagnation_detect(benchmark):
    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
        stagnation=StagnationConfig(min_failures=3, similarity_threshold=0.5),
    )

    def trip():
        f = AgentFuse(cfg)
        try:
            for msg in ("timeout alpha", "timeout alpha beta", "timeout alpha gamma"):
                f.observe(
                    Action(tool="search", args={"q": "x"}, result=msg, success=False)
                )
        except Exception:
            pass

    benchmark(trip)


def test_bench_canonical_hash(benchmark):
    payload = {
        "q": "the quick brown fox",
        "filters": ["a", "b", "c"],
        "options": {"limit": 10, "offset": 0, "debug": False},
    }
    benchmark(lambda: canonical_hash(payload))


def test_bench_sqlite_persist(benchmark, tmp_path):
    from agent_fuse.types import TrajectoryRecord

    db = tmp_path / "traj.db"
    store = TrajectoryStore(str(db), run_id="bench")
    state = {"i": 0}

    def step():
        i = state["i"]
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
        state["i"] = i + 1

    try:
        benchmark(step)
    finally:
        store.close()
