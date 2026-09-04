"""Regression tests for the 0.2.0 hardening pass.

Every test in this file exercises a *new* public symbol or behaviour that
ships in 0.2.0 but was not covered by the original smoke / guard / edge-case
suites. They are written to fail loudly if any of the documented behaviour
regresses, and to document the *exact* contract the library honours today.

Do not delete tests here when you "refactor" the implementation — they are
the contract.
"""

from __future__ import annotations

import asyncio
import inspect
import time

import pytest

from agent_fuse import (
    Action,
    AgentFuse,
    BudgetExceeded,
    CycleConfig,
    DeadlockDetected,
    Detection,
    DetectionKind,
    FuseConfig,
    FuseStats,
    ProgressSignal,
    RunBudget,
    StagnationConfig,
    TrajectoryStore,
    fuse_namespace,
)


# ---------------------------------------------------------------------------
# BudgetExceeded / RunBudget
# ---------------------------------------------------------------------------


def _config_with(**budget_kwargs) -> FuseConfig:
    return FuseConfig(budgets=RunBudget(**budget_kwargs))


def _ok(tool="t", args=None, result="ok"):
    # Vary the args by tool so back-to-back identical-arg calls don't trip
    # the default direct-repeat detector before the test gets to its point.
    return Action(tool=tool, args=args if args is not None else {"_uniq": tool}, result=result, success=True)


def _fail(tool="t", args=None, result="err"):
    return Action(
        tool=tool,
        args=args if args is not None else {"_uniq": tool},
        result=result,
        success=False,
    )


def test_budget_max_actions_trips_on_nth_call():
    cfg = _config_with(max_actions=3)
    fuse = AgentFuse(cfg)
    # Vary args so direct-repeat detection doesn't fire first.
    fuse.observe(_ok("a", args=1))
    fuse.observe(_ok("b", args=2))
    fuse.observe(_ok("c", args=3))
    with pytest.raises(BudgetExceeded) as ei:
        fuse.observe(_ok("d", args=4))
    assert ei.value.kind == "actions"
    assert ei.value.limit == 3
    assert ei.value.observed == 3


def test_budget_max_actions_zero_disables():
    cfg = _config_with(max_actions=0)
    fuse = AgentFuse(cfg)
    for i in range(50):
        fuse.observe(_ok("a", args={"i": i}))
    # No exception — disabled.


def test_budget_max_tool_calls_counts_only_non_allowlisted():
    cfg = _config_with(max_tool_calls=2)
    fuse = AgentFuse(FuseConfig(
        budgets=RunBudget(max_tool_calls=2),
        allowlist=["think"],
    ))
    # Allowlisted calls don't count against the tool-call budget.
    for _ in range(5):
        fuse.observe(Action(tool="think", args={}, result=None, success=None))
    fuse.observe(_ok("a"))
    fuse.observe(_ok("b"))
    # Third non-allowlisted call should trip.
    with pytest.raises(BudgetExceeded) as ei:
        fuse.observe(_ok("c"))
    assert ei.value.kind == "tool_calls"
    assert ei.value.limit == 2
    assert ei.value.observed == 2


def test_budget_max_runtime_seconds_trips_after_threshold(monkeypatch):
    cfg = _config_with(max_runtime_seconds=0.05)
    fuse = AgentFuse(cfg)
    # Fake the clock so we don't sleep in tests.
    base = fuse._start_time
    fake_now = base + 1.0
    monkeypatch.setattr(
        "agent_fuse.guard.time.monotonic", lambda: fake_now
    )
    with pytest.raises(BudgetExceeded) as ei:
        fuse.observe(_ok("a", args=1))
    assert ei.value.kind == "runtime"
    assert ei.value.limit == 0.05


def test_budget_max_runtime_none_disables():
    cfg = _config_with(max_runtime_seconds=None)
    fuse = AgentFuse(cfg)
    # Lots of calls — none should trip the runtime budget.
    for i in range(100):
        fuse.observe(_ok("a", args={"i": i}))
    # Force the runtime to be effectively huge.
    fuse._start_time = time.monotonic() - 1000
    fuse.observe(_ok("a", args={"i": 999}))


def test_budget_exception_inherits_runtime_error():
    assert issubclass(BudgetExceeded, RuntimeError)


def test_budget_exceeded_message_contains_kind_and_limit():
    cfg = _config_with(max_actions=7)
    fuse = AgentFuse(cfg)
    for i in range(7):
        fuse.observe(_ok("a", args={"i": i}))
    with pytest.raises(BudgetExceeded) as ei:
        fuse.observe(_ok("a", args={"i": 99}))
    assert "actions" in str(ei.value)
    assert "7" in str(ei.value)


def test_runbudget_defaults_all_disabled():
    b = RunBudget()
    assert b.max_actions == 0
    assert b.max_runtime_seconds is None
    assert b.max_tool_calls == 0
    # None of the defaults should trip with a default fuse.
    fuse = AgentFuse()
    for i in range(20):
        fuse.observe(_ok("a", args={"i": i}))


# ---------------------------------------------------------------------------
# FuseStats
# ---------------------------------------------------------------------------


def test_stats_defaults_zero():
    s = FuseStats()
    assert s.calls_observed == 0
    assert s.calls_allowed == 0
    assert s.calls_blocked == 0
    assert s.deadlocks == 0
    assert s.progress_marks == 0
    assert s.time_avoided_seconds == 0.0
    assert s.first_deadlock_at is None


def test_stats_as_dict_shape():
    s = FuseStats(calls_observed=2, calls_allowed=1, calls_blocked=1)
    d = s.as_dict()
    assert d == {
        "calls_observed": 2,
        "calls_allowed": 1,
        "calls_blocked": 1,
        "deadlocks": 0,
        "progress_marks": 0,
        "time_avoided_seconds": 0.0,
        "first_deadlock_at": None,
    }


def test_stats_reset_zeros_everything():
    s = FuseStats(
        calls_observed=10,
        calls_allowed=8,
        calls_blocked=2,
        deadlocks=2,
        progress_marks=3,
        time_avoided_seconds=1.5,
        first_deadlock_at=4,
    )
    s.reset()
    assert s.calls_observed == 0
    assert s.calls_allowed == 0
    assert s.calls_blocked == 0
    assert s.deadlocks == 0
    assert s.progress_marks == 0
    assert s.time_avoided_seconds == 0.0
    assert s.first_deadlock_at is None


def test_stats_calls_observed_increments_on_observe():
    fuse = AgentFuse()
    fuse.observe(_ok("a"))
    fuse.observe(_ok("b"))
    assert fuse.stats.calls_observed == 2
    assert fuse.stats.calls_allowed == 2
    assert fuse.stats.calls_blocked == 0


def test_stats_calls_allowed_excludes_blocked():
    fuse = AgentFuse(
        FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2))
    )
    fuse.observe(_ok("a", args=1))
    # 2nd identical call trips the fuse.
    with pytest.raises(DeadlockDetected):
        fuse.observe(_ok("a", args=1))
    assert fuse.stats.calls_observed == 2
    assert fuse.stats.calls_allowed == 1
    assert fuse.stats.calls_blocked == 1
    assert fuse.stats.deadlocks == 1
    assert fuse.stats.first_deadlock_at == 1


def test_stats_progress_marks_increments_on_mark_progress():
    fuse = AgentFuse()
    fuse.observe(_ok())
    fuse.mark_progress(ProgressSignal(token="t1"))
    fuse.mark_progress(ProgressSignal(token="t2"))
    assert fuse.stats.progress_marks == 2


def test_stats_reset_clears_window_and_stats():
    fuse = AgentFuse(
        FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2))
    )
    fuse.observe(_ok("a", args=1))
    with pytest.raises(DeadlockDetected):
        fuse.observe(_ok("a", args=1))
    fuse.reset()
    assert fuse.stats.calls_observed == 0
    assert fuse.stats.calls_allowed == 0
    assert fuse.stats.calls_blocked == 0
    assert fuse.stats.deadlocks == 0
    assert fuse.history == []


def test_stats_record_time_avoided_positive_only():
    fuse = AgentFuse()
    fuse.record_time_avoided(2.5)
    assert fuse.stats.time_avoided_seconds == 2.5
    fuse.record_time_avoided(-1.0)
    assert fuse.stats.time_avoided_seconds == 2.5  # unchanged
    fuse.record_time_avoided(0.0)
    assert fuse.stats.time_avoided_seconds == 2.5  # unchanged


# ---------------------------------------------------------------------------
# Preflight: check / check_action
# ---------------------------------------------------------------------------


def test_check_returns_none_when_no_detection():
    fuse = AgentFuse()
    fuse.observe(_ok("a"))
    assert fuse.check("a", {"q": 1}) is None


def test_check_returns_detection_for_pending_deadlock():
    fuse = AgentFuse(
        FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2))
    )
    fuse.observe(_ok("a", args=1))
    # Same args as before — the synthetic action matches the previous one.
    det = fuse.check("a", 1)
    assert det is not None
    assert det.kind == DetectionKind.DIRECT_REPEAT


def test_check_does_not_mutate_window():
    fuse = AgentFuse(
        FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2))
    )
    fuse.observe(_ok("a", args=1))
    size_before = len(fuse.history)
    observed_before = fuse.stats.calls_observed
    fuse.check("a", 1)
    assert len(fuse.history) == size_before
    assert fuse.stats.calls_observed == observed_before


def test_check_does_not_count_against_budgets():
    cfg = FuseConfig(
        budgets=RunBudget(max_actions=2),
        cycle=CycleConfig(direct_repeat_threshold=2),
    )
    fuse = AgentFuse(cfg)
    # 10 preflight calls must NOT consume the actions budget.
    for _ in range(10):
        fuse.check("a")
    assert fuse.stats.calls_observed == 0
    fuse.observe(_ok("a", args=1))
    fuse.observe(_ok("b", args=2))
    with pytest.raises(BudgetExceeded):
        fuse.observe(_ok("c", args=3))


def test_check_action_takes_full_action():
    fuse = AgentFuse(
        FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2))
    )
    fuse.observe(_ok("a", args=1))
    det = fuse.check_action(Action(tool="a", args=1))
    assert det is not None
    assert det.kind == DetectionKind.DIRECT_REPEAT


def test_check_returns_none_for_allowlisted_tool():
    fuse = AgentFuse(
        FuseConfig(
            cycle=CycleConfig(direct_repeat_threshold=2),
            allowlist=["think"],
        )
    )
    for _ in range(10):
        fuse.observe(Action(tool="think", args={}, result=None, success=None))
    assert fuse.check("think", {}) is None


def test_check_detects_pending_cycle():
    """Preflight can detect a cycle that would complete with the next call.

    To trip preflight with a 2-cycle, the synthetic window needs *two*
    full copies already observed — i.e. 3 real observes of a1, b2, a1
    followed by a synthetic b2 [a1,b2,a1,b2_synth] matches a cycle.
    """
    fuse = AgentFuse(
        FuseConfig(cycle=CycleConfig(cycle_min_length=2, cycle_max_length=2))
    )
    fuse.observe(_ok("a", args=1))
    fuse.observe(_ok("b", args=2))
    fuse.observe(_ok("a", args=1))
    # Preflight "b" — synthetic window [a,b,a,b_synth] — has 2 copies of
    # the cycle → preflight detects it.
    det = fuse.check("b", 2)
    assert det is not None
    assert det.kind == DetectionKind.N_CYCLE


def test_check_does_not_mutate_window_after_cycle_preflight():
    """Even if preflight detects a pending cycle, the window is not
    modified — so calling observe() afterwards still uses the original
    state."""
    fuse = AgentFuse(
        FuseConfig(cycle=CycleConfig(cycle_min_length=2, cycle_max_length=2))
    )
    fuse.observe(_ok("a", args=1))
    fuse.observe(_ok("b", args=2))
    fuse.observe(_ok("a", args=1))
    pre_pre = list(fuse.history)
    det = fuse.check("b", 2)
    assert det is not None
    # No mutation from preflight.
    assert fuse.history == pre_pre
    assert fuse.stats.calls_observed == 3


def test_check_does_not_detect_stagnation_with_success_none():
    # Synthetic Action has success=None which stagnation ignores.
    fuse = AgentFuse()
    fuse.observe(_fail("a", result="bad token bad token"))
    det = fuse.check("a", {"args": 1, "kwargs": {}})
    # No detection — synthetic result is None, not a failure.
    assert det is None


# ---------------------------------------------------------------------------
# allow_repeats per-tool override
# ---------------------------------------------------------------------------


def test_allow_repeats_lets_polling_pass():
    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=2),
        allow_repeats={"poll": 50},
    )
    fuse = AgentFuse(cfg)
    for _ in range(20):
        fuse.observe(_ok("poll", args={"q": "status"}))
    # No exception — the 21st repeat is still within tolerance.
    assert fuse.stats.calls_blocked == 0


def test_allow_repeats_threshold_is_inclusive():
    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=2),
        allow_repeats={"poll": 5},
    )
    fuse = AgentFuse(cfg)
    # 5 repeats should pass; 6th should trip.
    for _ in range(5):
        fuse.observe(_ok("poll", args={"q": "status"}))
    with pytest.raises(DeadlockDetected) as ei:
        fuse.observe(_ok("poll", args={"q": "status"}))
    assert ei.value.detection.kind == DetectionKind.DIRECT_REPEAT


def test_allow_repeats_per_tool_threshold_with_global_default():
    """A tool listed in allow_repeats gets its own threshold; tools not
    listed continue to use the global CycleConfig threshold."""
    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=2),
        allow_repeats={"poll": 50},
    )
    fuse = AgentFuse(cfg)
    # "poll" tolerates 50 repeats.
    for _ in range(10):
        fuse.observe(_ok("poll", args={"q": 1}))
    # "other" still trips at the global threshold.
    fuse.observe(_ok("other", args=1))
    with pytest.raises(DeadlockDetected):
        fuse.observe(_ok("other", args=1))


def test_allow_repeats_does_not_apply_to_other_tools():
    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=2),
        allow_repeats={"poll": 50},
    )
    fuse = AgentFuse(cfg)
    fuse.observe(_ok("other", args=1))
    with pytest.raises(DeadlockDetected):
        fuse.observe(_ok("other", args=1))


def test_allow_repeats_different_args_break_run():
    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=2),
        allow_repeats={"poll": 50},
    )
    fuse = AgentFuse(cfg)
    for i in range(10):
        fuse.observe(_ok("poll", args={"q": i}))
    # No detection — args changed each time.


def test_allow_repeats_unset_tools_use_global_threshold():
    cfg = FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2))
    fuse = AgentFuse(cfg)
    fuse.observe(_ok("a", args=1))
    with pytest.raises(DeadlockDetected):
        fuse.observe(_ok("a", args=1))


# ---------------------------------------------------------------------------
# progress_callback (steering without raising)
# ---------------------------------------------------------------------------


FAIL_TEXT = "rate limited 429 please retry again after backoff"


def test_progress_callback_swallows_stagnation():
    calls = []

    def cb(action, detection):
        calls.append((action.tool, detection.kind))

    cfg = FuseConfig(stagnation=StagnationConfig(min_failures=2))
    cfg.progress_callback = cb
    fuse = AgentFuse(cfg)
    # Two consecutive *identical* failed results → Jaccard = 1.0 → stagnation.
    fuse.observe(_fail("api", result=FAIL_TEXT))
    fuse.observe(_fail("api", result=FAIL_TEXT))
    # The callback fired; no exception was raised.
    assert calls == [("api", DetectionKind.SEMANTIC_STAGNATION)]
    assert fuse.stats.calls_allowed == 2
    assert fuse.stats.deadlocks == 0


def test_progress_callback_can_explicitly_raise():
    def cb(action, detection):
        raise DeadlockDetected(
            "explicit abort", detection=detection, trajectory=[]
        )

    cfg = FuseConfig(stagnation=StagnationConfig(min_failures=2))
    cfg.progress_callback = cb
    fuse = AgentFuse(cfg)
    fuse.observe(_fail("api", result=FAIL_TEXT))
    with pytest.raises(DeadlockDetected) as ei:
        fuse.observe(_fail("api", result=FAIL_TEXT))
    assert "explicit abort" in str(ei.value)


def test_progress_callback_exception_other_than_deadlock_swallowed():
    def cb(action, detection):
        raise RuntimeError("boom")

    cfg = FuseConfig(stagnation=StagnationConfig(min_failures=2))
    cfg.progress_callback = cb
    fuse = AgentFuse(cfg)
    fuse.observe(_fail("api", result=FAIL_TEXT))
    # Callback blew up but the loop is preserved.
    fuse.observe(_fail("api", result=FAIL_TEXT))
    assert fuse.stats.deadlocks == 0


def test_progress_callback_does_not_swallow_baseexception():
    """KeyboardInterrupt / SystemExit must NOT be swallowed by the
    progress_callback wrapper — they should propagate so Ctrl-C still works.
    """

    def cb(action, detection):
        raise KeyboardInterrupt()

    cfg = FuseConfig(stagnation=StagnationConfig(min_failures=2))
    cfg.progress_callback = cb
    fuse = AgentFuse(cfg)
    fuse.observe(_fail("api", result=FAIL_TEXT))
    with pytest.raises(KeyboardInterrupt):
        fuse.observe(_fail("api", result=FAIL_TEXT))


# ---------------------------------------------------------------------------
# Ergonomic wrapper: fuse.wrap / fuse.tool / fuse_namespace
# ---------------------------------------------------------------------------


def test_wrap_decorates_function_and_records_success():
    fuse = AgentFuse()
    @fuse.tool(name="search")
    def search(q):
        return f"results for {q}"

    out = search("cats")
    assert out == "results for cats"
    assert fuse.stats.calls_observed == 1
    assert fuse.stats.calls_allowed == 1


def test_wrap_records_exception_as_failure_action():
    fuse = AgentFuse()
    @fuse.tool
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()
    assert fuse.stats.calls_observed == 1
    assert len(fuse.history) == 1
    assert fuse.history[0].success is False


def test_wrap_blocks_on_preflight_before_invoking_tool():
    fuse = AgentFuse(
        FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2))
    )
    call_count = {"n": 0}

    @fuse.tool
    def search(q):
        call_count["n"] += 1
        return q

    search("cats")
    # Second identical call must NOT invoke the underlying fn.
    with pytest.raises(DeadlockDetected):
        search("cats")
    assert call_count["n"] == 1  # never reached the second time


def test_wrap_with_allow_repeats_kwarg():
    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=2),
        allow_repeats={"poll": 50},
    )
    fuse = AgentFuse(cfg)

    @fuse.tool(allow_repeats=50)
    def poll():
        return "ok"

    for _ in range(10):
        poll()
    assert fuse.stats.deadlocks == 0


def test_wrap_attribute_is_same_as_tool():
    fuse = AgentFuse()
    # Both bound to the same function — they should behave identically.
    assert fuse.wrap.__func__ is fuse.tool.__func__


def test_fuse_namespace_callable_decorates():
    fuse = AgentFuse()
    ft = fuse_namespace(fuse)

    @ft(name="search")
    def search(q):
        return q

    assert search("x") == "x"
    assert fuse.stats.calls_observed == 1


def test_fuse_namespace_has_tool_method():
    fuse = AgentFuse()
    ft = fuse_namespace(fuse)

    @ft.tool(name="search")
    def search(q):
        return q

    assert search("x") == "x"


def test_fuse_namespace_preserves_function_metadata():
    fuse = AgentFuse()
    ft = fuse_namespace(fuse)

    @ft
    def my_special_tool(x):
        """docstring"""
        return x

    assert my_special_tool.__name__ == "my_special_tool"
    assert my_special_tool.__doc__ == "docstring"


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


def test_wrap_async_function():
    fuse = AgentFuse()

    @fuse.tool
    async def fetch(url):
        await asyncio.sleep(0)
        return f"body of {url}"

    out = asyncio.run(fetch("http://x"))
    assert out == "body of http://x"
    assert fuse.stats.calls_observed == 1


def test_wrap_async_records_exception():
    fuse = AgentFuse()

    @fuse.tool
    async def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        asyncio.run(boom())
    assert fuse.history[0].success is False


def test_wrap_async_blocks_on_preflight():
    fuse = AgentFuse(
        FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2))
    )

    @fuse.tool
    async def fetch(url):
        return url

    asyncio.run(fetch("a"))
    with pytest.raises(DeadlockDetected):
        asyncio.run(fetch("a"))


def test_wrap_async_iscoroutinefunction_preserved():
    fuse = AgentFuse()

    @fuse.tool
    async def fetch(url):
        return url

    assert inspect.iscoroutinefunction(fetch)


# ---------------------------------------------------------------------------
# Integration: budgets + detection together
# ---------------------------------------------------------------------------


def test_budget_takes_precedence_over_detection():
    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=2),
        budgets=RunBudget(max_actions=2),
    )
    fuse = AgentFuse(cfg)
    fuse.observe(_ok("a", args=1))
    fuse.observe(_ok("a", args=2))  # different args → no detection yet
    # 3rd observe would trip the budget BEFORE detection runs.
    with pytest.raises(BudgetExceeded):
        fuse.observe(_ok("a", args=1))


def test_stats_recorded_consistently_after_reset():
    fuse = AgentFuse()
    fuse.observe(_ok())
    fuse.mark_progress(ProgressSignal(token="t"))
    fuse.reset()
    assert fuse.stats.progress_marks == 0
    assert fuse.stats.calls_observed == 0
    assert fuse.stats.calls_allowed == 0


# ---------------------------------------------------------------------------
# Module-level public surface (regression against accidental re-exports)
# ---------------------------------------------------------------------------


def test_module_reexports_match_dunder_all():
    import agent_fuse

    for name in agent_fuse.__all__:
        assert hasattr(agent_fuse, name), name
