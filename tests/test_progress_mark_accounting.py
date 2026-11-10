"""Regression tests: progress-mark accounting and the wrapper trip contract.

Two failure modes pinned here:

1. Once a ``mark_progress`` anchor slides out of the sliding window, the
   stagnation detector must keep working (every action left in the window
   was observed *after* the mark). Previously the slid-out anchor was
   replaced with a sentinel that shrank the post-mark slice to the single
   newest action, silently disabling semantic-stagnation detection for the
   rest of the run.

2. The ``@fuse.tool`` wrapper advertises a circuit-breaker guarantee: a
   call that the detectors already know would trip the fuse must be
   blocked *before* the tool runs. For tools with an ``allow_repeats``
   override, the pre-call gate used to allow the call unconditionally,
   leaking one tool invocation past the trip threshold.
"""

from __future__ import annotations

import pytest

from trajectory_fuse import (
    Action,
    AgentFuse,
    CycleConfig,
    DeadlockDetected,
    FuseConfig,
    ProgressSignal,
    StagnationConfig,
)
from trajectory_fuse.store import TrajectoryRecord

FAIL_RESULT = "error: connection refused by upstream"


def _stuck_actions(n, tool="fetch", start=0):
    """n failing actions on the same tool with *distinct* args.

    Distinct args keep direct-repeat and cycle detection out of the way so
    only the semantic-stagnation detector can fire.
    """
    return [
        Action(tool=tool, args={"attempt": start + i}, result=FAIL_RESULT, success=False)
        for i in range(n)
    ]


def _fill_actions(n, start=0, success=True):
    return [
        Action(tool=f"step-{start + i}", args={"i": start + i}, result="ok", success=success)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. stagnation after a progress mark slides out of the window
# ---------------------------------------------------------------------------

def test_stagnation_fires_after_mark_slides_out_of_window():
    """A mark that has left the window must not disable stagnation forever."""
    fuse = AgentFuse(
        FuseConfig(
            window=8,
            stagnation=StagnationConfig(min_failures=4, similarity_threshold=0.9),
        )
    )
    fuse.mark_progress(ProgressSignal(token="phase-1", note="sub-task done"))

    # Push the mark out of the window: window=8, so 9+ distinct actions
    # after the mark evict it (and every pre-mark action) entirely.
    for a in _fill_actions(10):
        fuse.observe(a)

    # Now the agent gets stuck: 4 near-identical failures, distinct args.
    with pytest.raises(DeadlockDetected) as exc_info:
        for a in _stuck_actions(4, start=100):
            fuse.observe(a)
    assert exc_info.value.detection.kind.value == "semantic_stagnation"


def test_stagnation_still_fires_with_mark_inside_window():
    """Control: the detector works normally while the mark is in view."""
    fuse = AgentFuse(
        FuseConfig(
            window=32,
            stagnation=StagnationConfig(min_failures=4, similarity_threshold=0.9),
        )
    )
    fuse.mark_progress(ProgressSignal(token="phase-1"))
    for a in _fill_actions(3):
        fuse.observe(a)
    with pytest.raises(DeadlockDetected):
        for a in _stuck_actions(4, start=100):
            fuse.observe(a)


def test_stagnation_suppressed_only_for_pre_mark_failures():
    """Failures recorded *before* the mark stay excluded (the mark's job)."""
    fuse = AgentFuse(
        FuseConfig(
            window=32,
            stagnation=StagnationConfig(min_failures=4, similarity_threshold=0.9),
        )
    )
    # Three identical failures before the mark...
    for a in _stuck_actions(3, start=0):
        fuse.observe(a)
    # ...then a progress mark (new phase)...
    fuse.mark_progress(ProgressSignal(token="recovered"))
    # ...then a single failure: 3 pre-mark + 1 post-mark must NOT fire.
    fuse.observe(_stuck_actions(1, start=50)[0])
    assert fuse.stats.deadlocks == 0


# ---------------------------------------------------------------------------
# 2. wrapper circuit-breaker contract for allow_repeats tools
# ---------------------------------------------------------------------------

def test_wrapper_blocks_before_call_past_allow_repeats_threshold():
    """The (N+1)-th identical call must raise *without* invoking the tool."""
    fuse = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3)))
    calls = []

    @fuse.tool(name="poll", allow_repeats=3)
    def poll(x):
        calls.append(x)
        return "ok"

    raised = False
    for _ in range(6):
        try:
            poll(1)
        except DeadlockDetected:
            raised = True
            break
    assert raised, "fuse never tripped on a runaway poll loop"
    # 3 tolerated identical calls; the 4th must be blocked pre-call.
    assert len(calls) == 3, (
        f"circuit breaker leaked: tool ran {len(calls)} times, expected 3"
    )


def test_wrapper_still_permits_up_to_threshold_identical_calls():
    """No over-blocking: up to the override's worth of repeats run normally."""
    fuse = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3)))
    calls = []

    @fuse.tool(name="poll", allow_repeats=3)
    def poll(x):
        calls.append(x)
        return "ok"

    for _ in range(3):
        poll(1)
    assert len(calls) == 3


def test_wrapper_async_blocks_before_call_past_threshold():
    """Same guarantee on the async wrapper path."""
    import asyncio

    fuse = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3)))
    calls = []

    @fuse.tool(name="poll", allow_repeats=3)
    async def poll(x):
        calls.append(x)
        return "ok"

    async def run():
        raised = False
        for _ in range(6):
            try:
                await poll(1)
            except DeadlockDetected:
                raised = True
                break
        return raised

    assert asyncio.run(run())
    assert len(calls) == 3


def test_plain_repeats_without_override_still_block_pre_call():
    """Tools without an allow_repeats override keep the pre-call guarantee."""
    fuse = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3)))
    calls = []

    @fuse.tool(name="poke")
    def poke(x):
        calls.append(x)
        return "ok"

    raised = False
    for _ in range(6):
        try:
            poke(1)
        except DeadlockDetected:
            raised = True
            break
    assert raised
    assert len(calls) == 2  # 3rd identical call blocked before the tool runs


# ---------------------------------------------------------------------------
# 3. load_from: progress marks must track window trims
# ---------------------------------------------------------------------------

def _record(seq, tool="t", success: "bool | None" = True, is_progress=False, result="ok"):
    return TrajectoryRecord(
        sequence=seq,
        run_id="r",
        tool=tool,
        args_hash="",
        args_repr="{}",
        result_repr=result,
        success=success,
        is_progress=is_progress,
    )


def test_load_from_shifts_progress_marks_on_trim():
    """Replayed marks must track window trims and not mask stagnation."""
    fuse = AgentFuse(
        FuseConfig(
            window=8,
            stagnation=StagnationConfig(min_failures=4, similarity_threshold=0.9),
        )
    )
    # 20 records: 15 successes, a progress marker at sequence 15, then 4
    # failed "fetch" records (16..19). Loading trims to the last 8 actions;
    # the mark must follow the trim so the 4 loaded failures stay in the
    # post-mark slice.
    records = []
    for i in range(20):
        if i == 15:
            records.append(_record(i, tool="<progress>", success=None, is_progress=True))
        elif i > 15:
            records.append(_record(i, tool="fetch", success=False, result=FAIL_RESULT))
        else:
            records.append(_record(i, tool=f"step-{i}"))
    fuse.load_from(records)

    # One more identical failure: the four loaded post-mark failures plus
    # this one must trip stagnation. A stale mark pointing past the loaded
    # actions leaves only the new failure in view and silently misses it.
    with pytest.raises(DeadlockDetected):
        fuse.observe(_stuck_actions(1, start=200)[0])
