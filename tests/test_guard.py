"""Tests for the AgentFuse runtime guard."""

from __future__ import annotations

import pytest

from trajectory_fuse import (
    Action,
    AgentFuse,
    CycleConfig,
    DeadlockDetected,
    DetectionKind,
    FuseConfig,
    ProgressSignal,
    StagnationConfig,
)
from trajectory_fuse.guard import canonical_hash  # re-export via guard module not exposed; use public
from trajectory_fuse.hashing import canonical_hash


def _act(tool, args, result=None, success=None, **meta):
    return Action(tool=tool, args=args, result=result, success=success, metadata=meta)


class TestDirectRepeat:
    def test_no_trigger_below_threshold(self):
        fuse = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3)))
        for _ in range(2):
            fuse.observe(_act("search", {"q": "x"}, "ok", True))
        # No exception.

    def test_triggers_at_threshold(self):
        fuse = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3)))
        fuse.observe(_act("search", {"q": "x"}, "ok", True))
        fuse.observe(_act("search", {"q": "x"}, "ok", True))
        with pytest.raises(DeadlockDetected) as ei:
            fuse.observe(_act("search", {"q": "x"}, "ok", True))
        assert ei.value.detection.kind == DetectionKind.DIRECT_REPEAT
        assert ei.value.detection.details["count"] == 3
        assert ei.value.detection.details["args_hash"] == canonical_hash({"q": "x"})

    def test_different_args_break_run(self):
        fuse = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3)))
        fuse.observe(_act("search", {"q": "a"}, "ok", True))
        fuse.observe(_act("search", {"q": "a"}, "ok", True))
        fuse.observe(_act("search", {"q": "b"}, "ok", True))  # breaks the run
        fuse.observe(_act("search", {"q": "a"}, "ok", True))
        # No exception — never had 3 in a row.

    def test_different_tool_breaks_run(self):
        fuse = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3)))
        fuse.observe(_act("a", {"q": 1}, "ok", True))
        fuse.observe(_act("a", {"q": 1}, "ok", True))
        fuse.observe(_act("b", {"q": 1}, "ok", True))
        fuse.observe(_act("a", {"q": 1}, "ok", True))
        # No exception.

    def test_threshold_one_disables(self):
        fuse = AgentFuse(
            FuseConfig(
                cycle=CycleConfig(
                    direct_repeat_threshold=1,
                    cycle_min_length=99,  # also disable cycle for this test
                )
            )
        )
        for _ in range(5):
            fuse.observe(_act("search", {"q": "x"}, "ok", True))
        # No exception — direct-repeat disabled.

    def test_allowlist_skips(self):
        fuse = AgentFuse(
            FuseConfig(
                cycle=CycleConfig(direct_repeat_threshold=2),
                allowlist=("noop",),
            )
        )
        for _ in range(10):
            fuse.observe(_act("noop", {}, "ok", True))
        # No exception — noop is allowlisted.

    def test_window_eviction(self):
        fuse = AgentFuse(FuseConfig(window=4, cycle=CycleConfig(direct_repeat_threshold=3)))
        # Fill window with non-matching actions.
        fuse.observe(_act("a", 1, "ok", True))
        fuse.observe(_act("a", 2, "ok", True))
        fuse.observe(_act("a", 3, "ok", True))
        # Now this trio of identical calls — but window only has 4 slots.
        fuse.observe(_act("search", {"q": "x"}, "ok", True))
        fuse.observe(_act("search", {"q": "x"}, "ok", True))
        with pytest.raises(DeadlockDetected):
            fuse.observe(_act("search", {"q": "x"}, "ok", True))


class TestNCycle:
    def test_two_state_cycle(self):
        fuse = AgentFuse(
            FuseConfig(
                cycle=CycleConfig(
                    direct_repeat_threshold=99,
                    cycle_min_length=2,
                    cycle_max_length=4,
                )
            )
        )
        # A B A B — at the 4th observation, two full copies of [A,B] exist,
        # so cycle detection fires.
        # Distinct args per tool so direct-repeat does not interfere.
        raised_at = None
        for idx, tool in enumerate(("A", "B", "A", "B"), start=1):
            try:
                fuse.observe(_act(tool, {"who": tool}, "ok", True))
            except DeadlockDetected as ei:
                raised_at = idx
                det = ei.detection
                break
        assert raised_at == 4
        assert det.kind == DetectionKind.N_CYCLE
        assert det.details["length"] == 2
        assert det.cycle == ["A", "B"]

    def test_three_state_cycle(self):
        fuse = AgentFuse(
            FuseConfig(
                cycle=CycleConfig(
                    direct_repeat_threshold=99,
                    cycle_min_length=3,
                    cycle_max_length=4,
                )
            )
        )
        # Three-state cycle fires when two full copies exist (6th observation).
        raised_at = None
        for idx, tool in enumerate(("A", "B", "C", "A", "B", "C"), start=1):
            try:
                fuse.observe(_act(tool, {"who": tool}, "ok", True))
            except DeadlockDetected as ei:
                raised_at = idx
                det = ei.detection
                break
        assert raised_at == 6
        assert det.kind == DetectionKind.N_CYCLE
        assert det.details["length"] == 3

    def test_args_must_match_too(self):
        fuse = AgentFuse(
            FuseConfig(
                cycle=CycleConfig(
                    direct_repeat_threshold=99,
                    cycle_min_length=2,
                    cycle_max_length=4,
                )
            )
        )
        # Different args for same tool break the cycle signature.
        fuse.observe(_act("A", {"v": 1}, "ok", True))
        fuse.observe(_act("B", {"v": 2}, "ok", True))
        fuse.observe(_act("A", {"v": 99}, "ok", True))  # different args
        fuse.observe(_act("B", {"v": 2}, "ok", True))
        # No exception — args differ.

    def test_cycle_min_length_one_disables(self):
        fuse = AgentFuse(
            FuseConfig(
                cycle=CycleConfig(
                    direct_repeat_threshold=99,
                    cycle_min_length=1,
                )
            )
        )
        for tool in ("A", "B") * 5:
            fuse.observe(_act(tool, {"who": tool}, "ok", True))
        # No exception — cycle detection effectively off.

    def test_min_larger_than_max_disables(self):
        fuse = AgentFuse(
            FuseConfig(
                cycle=CycleConfig(
                    direct_repeat_threshold=99,
                    cycle_min_length=5,
                    cycle_max_length=2,
                )
            )
        )
        for tool in ("A", "B") * 5:
            fuse.observe(_act(tool, {"who": tool}, "ok", True))
        # No exception.


class TestSemanticStagnation:
    def test_no_trigger_on_success(self):
        cfg = FuseConfig(
            cycle=CycleConfig(
                direct_repeat_threshold=99,
                cycle_min_length=99,  # disable cycle detection too
            ),
            stagnation=StagnationConfig(min_failures=3, similarity_threshold=0.9),
        )
        fuse = AgentFuse(cfg)
        for _ in range(5):
            fuse.observe(_act("search", {"q": "x"}, "found stuff", True))
        # No exception — no failures.

    def test_no_trigger_with_below_threshold_sim(self):
        cfg = FuseConfig(
            cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
            stagnation=StagnationConfig(min_failures=3, similarity_threshold=0.95),
        )
        fuse = AgentFuse(cfg)
        fuse.observe(_act("search", {"q": "x"}, "alpha bravo charlie", False))
        fuse.observe(_act("search", {"q": "x"}, "delta echo foxtrot", False))
        fuse.observe(_act("search", {"q": "x"}, "golf hotel india", False))
        # No exception — token sets share little.

    def test_triggers_on_repeated_failure(self):
        cfg = FuseConfig(
            cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
            stagnation=StagnationConfig(min_failures=3, similarity_threshold=0.5),
        )
        fuse = AgentFuse(cfg)
        # Three near-identical failure messages (high token overlap).
        fuse.observe(_act("search", {"q": "x"}, "Error: connection timeout", False))
        fuse.observe(_act("search", {"q": "x"}, "Error: connection timeout retry", False))
        with pytest.raises(DeadlockDetected) as ei:
            fuse.observe(_act("search", {"q": "x"}, "Error: connection timeout again", False))
        det = ei.value.detection
        assert det.kind == DetectionKind.SEMANTIC_STAGNATION
        assert det.similarity is not None
        assert det.similarity >= 0.5
        assert det.details["consecutive_failures"] == 3

    def test_success_resets_failure_run(self):
        cfg = FuseConfig(
            cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
            stagnation=StagnationConfig(min_failures=3, similarity_threshold=0.5),
        )
        fuse = AgentFuse(cfg)
        fuse.observe(_act("search", {"q": "x"}, "timeout", False))
        fuse.observe(_act("search", {"q": "x"}, "timeout", False))
        fuse.observe(_act("search", {"q": "x"}, "ok now", True))  # success — reset
        fuse.observe(_act("search", {"q": "x"}, "different error", False))
        fuse.observe(_act("search", {"q": "x"}, "another diff", False))
        # Only 2 consecutive failures — no stagnation.

    def test_different_tool_resets_run(self):
        cfg = FuseConfig(
            cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
            stagnation=StagnationConfig(min_failures=3, similarity_threshold=0.5),
        )
        fuse = AgentFuse(cfg)
        fuse.observe(_act("a", {}, "timeout error", False))
        fuse.observe(_act("a", {}, "timeout error", False))
        fuse.observe(_act("b", {}, "timeout error", False))  # different tool — reset
        fuse.observe(_act("a", {}, "timeout error", False))
        fuse.observe(_act("a", {}, "timeout error", False))
        # No stagnation — only 2 consecutive a-failures after reset.

    def test_progress_mark_resets_window(self):
        cfg = FuseConfig(
            cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
            stagnation=StagnationConfig(min_failures=3, similarity_threshold=0.5),
        )
        fuse = AgentFuse(cfg)
        # Two failures, then a progress mark, then one more — must not fire.
        fuse.observe(_act("search", {"q": "x"}, "timeout", False))
        fuse.observe(_act("search", {"q": "x"}, "timeout", False))
        fuse.mark_progress(ProgressSignal(token="got new state"))
        fuse.observe(_act("search", {"q": "x"}, "timeout", False))
        # No exception — only 1 consecutive failure since mark.

    def test_disabled_via_config(self):
        cfg = FuseConfig(
            cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
            stagnation=StagnationConfig(enabled=False, min_failures=2, similarity_threshold=0.5),
        )
        fuse = AgentFuse(cfg)
        for _ in range(5):
            fuse.observe(_act("search", {"q": "x"}, "timeout", False))
        # No exception — stagnation disabled.

    def test_min_failures_below_two_disables(self):
        cfg = FuseConfig(
            cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
            stagnation=StagnationConfig(min_failures=1, similarity_threshold=0.5),
        )
        fuse = AgentFuse(cfg)
        fuse.observe(_act("search", {"q": "x"}, "timeout", False))
        # No exception — min_failures < 2 disables the detector.


class TestSteeringHook:
    def test_hook_receives_snapshot_and_hint(self):
        seen = {}

        def hook(actions, detection):
            seen["actions"] = list(actions)
            seen["detection"] = detection
            return "consider an alternative approach"

        fuse = AgentFuse(
            FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2)),
            steering_hook=hook,
        )
        fuse.observe(_act("search", {"q": "x"}, "ok", True))
        with pytest.raises(DeadlockDetected) as ei:
            fuse.observe(_act("search", {"q": "x"}, "ok", True))
        assert "actions" in seen
        assert seen["detection"].kind == DetectionKind.DIRECT_REPEAT
        assert ei.value.steering_hint == "consider an alternative approach"
        assert "consider an alternative approach" in str(ei.value)

    def test_hook_exception_swallowed(self):
        def hook(actions, detection):
            raise RuntimeError("hook boom")

        fuse = AgentFuse(
            FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2)),
            steering_hook=hook,
        )
        fuse.observe(_act("search", {"q": "x"}, "ok", True))
        with pytest.raises(DeadlockDetected) as ei:
            fuse.observe(_act("search", {"q": "x"}, "ok", True))
        assert ei.value.steering_hint is None


class TestResetAndLifecycle:
    def test_reset_clears_window(self):
        fuse = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2)))
        fuse.observe(_act("search", {"q": "x"}, "ok", True))
        fuse.reset()
        fuse.observe(_act("search", {"q": "y"}, "ok", True))
        # No exception — reset cleared the duplicate.

    def test_context_manager_closes_store(self):
        from trajectory_fuse.store import TrajectoryStore

        store = TrajectoryStore(":memory:")
        cfg = FuseConfig(store_factory=lambda: store, run_id="ctx")
        with AgentFuse(cfg) as fuse:
            fuse.observe(_act("search", {"q": "x"}, "ok", True))
        # Store should be closed; trying to use raises.
        import sqlite3

        with pytest.raises(sqlite3.ProgrammingError):
            store._conn.execute("SELECT 1")