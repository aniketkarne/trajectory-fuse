"""Tests for the recovery hooks API + the integrations package."""

from __future__ import annotations

import pytest

from trajectory_fuse import (
    Action,
    AgentFuse,
    Chain,
    DeadlockDetected,
    DetectionKind,
    FuseConfig,
    RecoveryPolicy,
    on_deadlock,
    on_kind,
    on_loop,
    on_stagnation,
)
from trajectory_fuse.integrations import (
    FrameworkNotInstalled,
    langgraph_middleware,
    openai_agents_middleware,
    pydantic_ai_middleware,
)
from trajectory_fuse.integrations.generic import (
    guard_scope,
    make_observer,
    observe_call,
)

# ---------------------------------------------------------------------------
# RecoveryPolicy / Chain / on_* helpers
# ---------------------------------------------------------------------------


def _action(tool="t", args=None, result="ok", success=None):
    return Action(tool=tool, args=args or {"i": 0}, result=result, success=success)


def test_default_policy_raises_on_direct_repeat():
    fuse = AgentFuse(FuseConfig(cycle=FuseConfig().cycle))  # default threshold 3
    # Three identical calls — third trips.
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    with pytest.raises(DeadlockDetected):
        fuse.observe(_action("t", {"i": 1}, "ok", True))
    assert fuse.stats.deadlocks == 1
    assert fuse.stats.recovery_skips == 0


def test_policy_continue_swallows_deadlock():
    fuse = AgentFuse(
        FuseConfig(),
        recovery_policy=RecoveryPolicy(
            on_direct_repeat=lambda a, d: "continue",
            default_disposition="raise",
        ),
    )
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))  # 3rd call: direct_repeat fires; swallowed
    fuse.observe(_action("t", {"i": 1}, "ok", True))  # 4th call: still direct_repeat; swallowed
    # Every identical call beyond threshold 2 fires direct_repeat, and
    # every fire is swallowed by the recovery policy.
    assert fuse.stats.deadlocks == 2
    assert fuse.stats.recovery_skips == 2
    # No exception escaped.
    assert fuse.stats.calls_allowed == 4


def test_policy_raise_is_default_for_none_return():
    fuse = AgentFuse(
        FuseConfig(),
        recovery_policy=RecoveryPolicy(
            on_direct_repeat=lambda a, d: None,  # no opinion
            default_disposition="raise",
        ),
    )
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    with pytest.raises(DeadlockDetected):
        fuse.observe(_action("t", {"i": 1}, "ok", True))


def test_chain_runs_hooks_in_order():
    calls = []

    def first(a, d):
        calls.append("first")
        return None

    def second(a, d):
        calls.append("second")
        return "continue"

    def third(a, d):
        calls.append("third")  # must NOT be reached
        return "raise"

    fuse = AgentFuse(FuseConfig(), recovery_policy=Chain(hooks=[first, second, third]))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    assert calls == ["first", "second"]
    assert fuse.stats.recovery_skips == 1


def test_chain_default_disposition_continue():
    fuse = AgentFuse(
        FuseConfig(),
        recovery_policy=Chain(hooks=[], default_disposition="continue"),
    )
    # No hooks → falls through to default → continue (no raise).
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    assert fuse.stats.recovery_skips == 1


def test_hook_exception_does_not_break_loop():
    def bad(a, d):
        raise RuntimeError("boom")

    fuse = AgentFuse(
        FuseConfig(),
        recovery_policy=RecoveryPolicy(
            on_direct_repeat=bad,
            default_disposition="raise",
        ),
    )
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    with pytest.raises(DeadlockDetected):
        # Bad hook returns None → falls through to default_disposition=raise.
        fuse.observe(_action("t", {"i": 1}, "ok", True))


def test_on_kind_filters_by_kind():
    calls = []

    def on_direct(a, d):
        calls.append("direct")
        return "continue"

    def on_stag(a, d):
        calls.append("stag")
        return "raise"

    fuse = AgentFuse(
        FuseConfig(),
        recovery_policy=RecoveryPolicy(
            on_direct_repeat=on_kind("direct_repeat", on_direct),
            on_stagnation=on_kind("semantic_stagnation", on_stag),
        ),
    )
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))  # direct_repeat → continue
    assert calls == ["direct"]


def test_on_loop_on_stagnation_on_deadlock_are_identity():
    def hook(a, d):
        return "continue"

    assert on_loop(hook) is hook
    assert on_stagnation(hook) is hook
    assert on_deadlock(hook) is hook


def test_recovery_skips_counted_in_stats():
    fuse = AgentFuse(
        FuseConfig(),
        recovery_policy=RecoveryPolicy(
            on_direct_repeat=lambda a, d: "continue",
        ),
    )
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    fuse.observe(_action("t", {"i": 1}, "ok", True))
    assert fuse.stats.as_dict()["recovery_skips"] == 1
    fuse.stats.reset()
    assert fuse.stats.recovery_skips == 0


# ---------------------------------------------------------------------------
# Integrations — generic
# ---------------------------------------------------------------------------


def test_guard_scope_yields_fuse():
    fuse = AgentFuse(FuseConfig())
    with guard_scope(fuse) as f:
        assert f is fuse
        f.observe(_action("t", {"i": 1}, "ok", True))


def test_guard_scope_records_progress_marks():
    fuse = AgentFuse(FuseConfig())
    with guard_scope(fuse, enter_action="begin", exit_action="end"):
        fuse.observe(_action("t", {"i": 1}, "ok", True))
    assert fuse.stats.progress_marks == 2


def test_make_observer_attaches_tool_name_when_missing():
    fuse = AgentFuse(FuseConfig())
    observer = make_observer(fuse, tool_name="search")
    observer(Action(tool="", args={"q": 1}, result="ok", success=True))
    assert fuse.history[-1].tool == "search"


def test_observe_call_wraps_sync_function():
    fuse = AgentFuse(FuseConfig(allow_repeats={"my_tool": 50}))
    seen = []

    def fn(x):
        seen.append(x)
        return f"out-{x}"

    wrapped = observe_call(fuse, "my_tool", fn)
    assert wrapped(1) == "out-1"
    assert wrapped(2) == "out-2"
    assert seen == [1, 2]
    assert fuse.stats.calls_allowed == 2


def test_observe_call_wraps_async_function():
    import asyncio

    fuse = AgentFuse(FuseConfig(allow_repeats={"my_async": 50}))

    async def fn(x):
        return f"async-{x}"

    wrapped = observe_call(fuse, "my_async", fn)
    result = asyncio.run(wrapped(42))
    assert result == "async-42"
    assert fuse.stats.calls_allowed == 1


def test_observe_call_propagates_exception_with_failure_recorded():
    fuse = AgentFuse(FuseConfig(allow_repeats={"bad": 50}))

    def fn(x):
        raise ValueError("boom")

    wrapped = observe_call(fuse, "bad", fn)
    with pytest.raises(ValueError):
        wrapped(1)
    assert fuse.history[-1].success is False
    assert "boom" in str(fuse.history[-1].result)


# ---------------------------------------------------------------------------
# Integrations — framework adapters (when not installed)
# ---------------------------------------------------------------------------


def test_langgraph_middleware_raises_framework_not_installed(monkeypatch):
    """If langgraph isn't installed, factory raises with a clean message."""
    # Simulate "not installed" by hiding langchain.agents.middleware
    # and langgraph.prebuilt.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("langchain.agents.middleware", "langgraph.prebuilt"):
            raise ImportError("hidden for test")
        if name == "langchain":
            raise ImportError("hidden for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    fuse = AgentFuse(FuseConfig())
    with pytest.raises(FrameworkNotInstalled) as ei:
        langgraph_middleware(fuse)
    assert "langgraph" in str(ei.value)
    assert "pip install" in str(ei.value)


def test_openai_agents_middleware_raises_framework_not_installed(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("agents", "openai_agents"):
            raise ImportError("hidden for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    fuse = AgentFuse(FuseConfig())
    with pytest.raises(FrameworkNotInstalled) as ei:
        openai_agents_middleware(fuse)
    assert "openai-agents" in str(ei.value)


def test_pydantic_ai_middleware_raises_framework_not_installed(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pydantic_ai":
            raise ImportError("hidden for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    fuse = AgentFuse(FuseConfig())
    with pytest.raises(FrameworkNotInstalled) as ei:
        pydantic_ai_middleware(fuse)
    assert "pydantic-ai" in str(ei.value)


def test_framework_not_installed_str():
    err = FrameworkNotInstalled("foo", "pip install foo")
    msg = str(err)
    assert "foo" in msg
    assert "pip install foo" in msg
