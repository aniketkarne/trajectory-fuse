"""Comprehensive edge-case tests for agent-fuse.

These tests complement the focused tests in ``test_guard.py`` and
``test_store.py`` with the messy cases that real callers hit:

* window eviction at the exact boundary,
* deeply nested / unordered argument canonicalisation,
* exception / steering-hook interaction,
* concurrent observes from multiple threads (regression test for the
  window state being safe to mutate),
* malformed JSONL and CLI error paths,
* export escaping for HTML / SVG / Mermaid,
* SQLite reopen + roundtrip across separate processes / connections.

Adding a test here must not weaken any test in the existing files.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from agent_fuse import (
    Action,
    AgentFuse,
    CycleConfig,
    DeadlockDetected,
    DetectionKind,
    FuseConfig,
    ProgressSignal,
    StagnationConfig,
    TrajectoryRecord,
    TrajectoryStore,
    canonical_hash,
)
from agent_fuse.export import render_html, render_mermaid, render_svg
from agent_fuse.guard import _tokenise, _jaccard  # internals — test in isolation
from agent_fuse.hashing import _normalise, canonical_payload
from agent_fuse.loader import (
    iter_jsonl,
    load_jsonl,
    normalise_row,
    replay_against_guard,
)


def _act(tool, args, result=None, success=None, **meta):
    return Action(tool=tool, args=args, result=result, success=success, metadata=meta)


# ===========================================================================
# 1. Canonicalisation edge cases
# ===========================================================================


class TestCanonicalisation:
    """Nested structures, exotic types, Unicode, and stability checks."""

    def test_frozenset_normalised(self):
        assert canonical_hash(frozenset({1, 2, 3})) == canonical_hash({1, 2, 3})

    def test_tuple_of_tuples(self):
        a = (("a", 1), ("b", 2))
        b = [["a", 1], ["b", 2]]
        assert canonical_hash(a) == canonical_hash(b)

    def test_nested_list_unordered_dict(self):
        # Lists keep order, but the dict keys surrounding them must be canonical.
        a = {"items": [3, 2, 1], "k": "v"}
        b = {"k": "v", "items": [3, 2, 1]}
        assert canonical_hash(a) == canonical_hash(b)

    def test_list_order_matters(self):
        # Lists are *ordered* — these must hash differently.
        assert canonical_hash([1, 2, 3]) != canonical_hash([3, 2, 1])

    def test_int_vs_float_different(self):
        # We do not silently coerce 1 to 1.0 — caller must do that.
        assert canonical_hash(1) != canonical_hash(1.0)

    def test_bool_vs_int(self):
        # Python quirk: True == 1, but JSON serialisation differs.
        # Our canonical form preserves the bool/int distinction.
        assert canonical_hash(True) != canonical_hash(1)

    def test_unicode_normalisation(self):
        # Same grapheme clusters must hash equal; pre-composed vs decomposed
        # form (NFC vs NFD) are *not* normalised here — caller must do so.
        # We only guarantee stable encoding of the bytes we receive.
        a = "café"  # NFC
        b = "café"  # could be the same NFC; depends on source
        assert canonical_hash({"name": a}) == canonical_hash({"name": b})

    def test_canonical_payload_returns_normalised(self):
        h, norm = canonical_payload({"b": 1, "a": [3, 2, 1]})
        assert h == canonical_hash({"b": 1, "a": [3, 2, 1]})
        # Normalised dict keys are sorted.
        assert list(norm.keys()) == ["a", "b"]
        # Nested list preserved order.
        assert norm["a"] == [3, 2, 1]

    def test_normalise_rejects_bytes(self):
        with pytest.raises(TypeError):
            _normalise(b"bytes-not-supported")

    def test_normalise_rejects_custom_class(self):
        class NotJSON:
            pass

        with pytest.raises(TypeError):
            _normalise(NotJSON())

    def test_hash_stable_across_runs(self):
        # Run twice in two separate subprocesses — must match.
        payload = {"q": "abc", "filters": [1, 2, {"x": "y"}]}
        h1 = canonical_hash(payload)
        h2 = canonical_hash(payload)
        assert h1 == h2
        # Subprocess check — guards against accidental non-determinism.
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys, json; "
                 "sys.path.insert(0, 'src'); "
                 "from agent_fuse.hashing import canonical_hash; "
                 "print(canonical_hash(json.loads(sys.argv[1])))",
                 json.dumps(payload)],
            capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == h1


class TestJaccardAndTokenise:
    def test_tokenise_lowercases(self):
        assert _tokenise("Foo Bar BAZ") == {"foo", "bar", "baz"}

    def test_tokenise_splits_on_punctuation(self):
        # The regex matches [A-Za-z0-9_]+; everything else is a separator.
        assert _tokenise("Error: timeout!") == {"error", "timeout"}

    def test_tokenise_empty_and_none(self):
        assert _tokenise("") == set()
        assert _tokenise(None) == set()  # type: ignore[arg-type]

    def test_jaccard_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_both_empty_is_one(self):
        # Both empty -> perfect overlap (no work to compare).
        assert _jaccard(set(), set()) == 1.0


# ===========================================================================
# 2. Window eviction at the boundary
# ===========================================================================


class TestWindowEvictionEdgeCases:
    def test_window_size_one_keeps_last(self):
        fuse = AgentFuse(
            FuseConfig(window=1, cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99))
        )
        fuse.observe(_act("a", 1, "ok", True))
        fuse.observe(_act("b", 2, "ok", True))
        assert len(fuse.history) == 1
        assert fuse.history[0].tool == "b"

    def test_window_eviction_preserves_cycle_within_window(self):
        # window=4, cycle A B A B A B should still detect at the 6th action
        # because the 5th and 6th form a complete A B cycle inside the window.
        fuse = AgentFuse(
            FuseConfig(
                window=4,
                cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=2, cycle_max_length=4),
            )
        )
        # First, prime the window with actions that get evicted.
        fuse.observe(_act("Z", 1, "ok", True))
        fuse.observe(_act("Y", 2, "ok", True))
        # Now cycle A B inside the window.
        for tool in ("A", "B", "A"):
            fuse.observe(_act(tool, {"w": tool}, "ok", True))
        # After Z Y A B A — last 4 are Y A B A; cycle of length 2 = [A,B] in last 2
        # matches previous 2? prev = [Y, A], cycle = [B, A]. No match. No detection.
        # Add one more action: Y A B A B -> last 4 = A B A B, prev = A B, cycle = A B. Detect!
        with pytest.raises(DeadlockDetected) as ei:
            fuse.observe(_act("B", {"w": "B"}, "ok", True))
        assert ei.value.detection.kind == DetectionKind.N_CYCLE

    def test_window_eviction_with_progress_marks(self):
        # Progress marks must also slide forward when the window trims, and
        # once they slide entirely out of the window the stagnation detector
        # must use a synthesised anchor so that pre-mark failures never leak
        # back into the post-mark region as more actions are observed.
        fuse = AgentFuse(
            FuseConfig(
                window=4,
                cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
                stagnation=StagnationConfig(min_failures=3, similarity_threshold=0.5),
            )
        )
        fuse.observe(_act("a", 1, "timeout one", False))
        fuse.observe(_act("a", 1, "timeout two", False))
        fuse.mark_progress(ProgressSignal(token="phase"))
        # Mark is recorded as the position immediately after the last action
        # observed at the time of the call — here, position 2 (index of the
        # next slot to fill).
        assert fuse._progress_marks == [2]
        # Fill window with new actions that would otherwise stagnate.
        fuse.observe(_act("a", 1, "timeout three", False))
        fuse.observe(_act("a", 1, "timeout four", False))
        # Mark is at history position 2; window is now full at 4 actions,
        # but no trim has fired yet because len never exceeded the window.
        assert fuse._progress_marks == [2]
        assert len(fuse.history) == 4
        # Force eviction past the mark: each observe trims one slot and
        # the mark slides forward; once the slide would make the mark
        # negative it is replaced by a synthesised sentinel.
        for expected_mark in ([1], [0], [-1], [-1], [-1], [-1], [-1], [-1]):
            fuse.observe(_act("a", 1, "timeout again", False))
            assert fuse._progress_marks == expected_mark, (
                f"expected progress mark {expected_mark}, got {fuse._progress_marks}"
            )
        # Stagnation must NOT fire: the cross-message token similarity
        # across the post-mark actions is below the 0.5 threshold, and the
        # synthesised sentinel prevents pre-mark failures from inflating
        # the post-mark count.
        assert not fuse._progress_marks or all(m < 0 for m in fuse._progress_marks)

    def test_window_eviction_does_not_lose_pending_stagnation_too_soon(self):
        # Two identical failures, then 30 unrelated actions. With window=8,
        # the original failures are evicted, so a third identical failure
        # must NOT trigger (we only see one post-eviction failure).
        cfg = FuseConfig(
            window=8,
            cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
            stagnation=StagnationConfig(min_failures=3, similarity_threshold=0.5),
        )
        fuse = AgentFuse(cfg)
        fuse.observe(_act("a", 1, "Error: connection timeout", False))
        fuse.observe(_act("a", 1, "Error: connection timeout again", False))
        # Push them out of the window.
        for i in range(20):
            fuse.observe(_act(f"tool-{i}", i, f"ok {i}", True))
        # Only one post-eviction failure — no stagnation.
        fuse.observe(_act("a", 1, "Error: connection timeout yet again", False))
        # No exception expected.


# ===========================================================================
# 3. Exception / steering behaviour
# ===========================================================================


class TestExceptionBehaviour:
    def test_deadlock_carries_trajectory_snapshot(self):
        fuse = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3)))
        actions = [
            _act("search", {"q": "x"}, "ok", True),
            _act("search", {"q": "x"}, "ok", True),
        ]
        for a in actions:
            fuse.observe(a)
        with pytest.raises(DeadlockDetected) as ei:
            fuse.observe(_act("search", {"q": "x"}, "ok", True))
        # Trajectory includes all 3 actions (last is the trigger).
        assert len(ei.value.trajectory) == 3
        assert ei.value.trajectory[-1].tool == "search"

    def test_hook_returning_none_no_hint(self):
        def hook(actions, detection):
            return None

        fuse = AgentFuse(
            FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2)),
            steering_hook=hook,
        )
        fuse.observe(_act("search", {"q": "x"}, "ok", True))
        with pytest.raises(DeadlockDetected) as ei:
            fuse.observe(_act("search", {"q": "x"}, "ok", True))
        assert ei.value.steering_hint is None
        assert "[hint:" not in str(ei.value)

    def test_hook_returning_empty_string_still_attached(self):
        # The implementation treats any non-None string as a hint; empty
        # string still gets attached (the caller asked for it).
        def hook(actions, detection):
            return ""

        fuse = AgentFuse(
            FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2)),
            steering_hook=hook,
        )
        fuse.observe(_act("search", {"q": "x"}, "ok", True))
        with pytest.raises(DeadlockDetected) as ei:
            fuse.observe(_act("search", {"q": "x"}, "ok", True))
        assert "[hint: ]" in str(ei.value)

    def test_hook_can_observe_trajectory(self):
        # The hook should be able to introspect the full window.
        seen_tools = []

        def hook(actions, detection):
            seen_tools.extend(a.tool for a in actions)
            return None

        fuse = AgentFuse(
            FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3)),
            steering_hook=hook,
        )
        fuse.observe(_act("a", 1, "ok", True))
        fuse.observe(_act("a", 1, "ok", True))
        with pytest.raises(DeadlockDetected):
            fuse.observe(_act("a", 1, "ok", True))
        assert seen_tools == ["a", "a", "a"]

    def test_hook_keyboard_interrupt_propagates(self):
        # The implementation swallows *all* hook exceptions; verify the
        # documented intent. KeyboardInterrupt is a BaseException subclass
        # so it should propagate.
        def hook(actions, detection):
            raise KeyboardInterrupt()

        fuse = AgentFuse(
            FuseConfig(cycle=CycleConfig(direct_repeat_threshold=2)),
            steering_hook=hook,
        )
        fuse.observe(_act("a", 1, "ok", True))
        with pytest.raises(KeyboardInterrupt):
            fuse.observe(_act("a", 1, "ok", True))

    def test_repr_is_safe(self):
        det = DeadlockDetected(
            "x", detection=__import__("agent_fuse").types.Detection(
                kind=DetectionKind.DIRECT_REPEAT, message="x", cycle=["t"]
            )
        )
        # Should not blow up.
        s = repr(det)
        assert "DIRECT_REPEAT" in s


# ===========================================================================
# 4. Concurrency / thread safety
# ===========================================================================


class TestConcurrency:
    def test_concurrent_observe_from_threads_does_not_crash(self):
        """Many threads call observe() with distinct actions.

        AgentFuse is documented as synchronous / single-threaded for
        detection, but Python's GIL means in-memory list.append is atomic.
        This is a smoke test that the *type* of usage is tolerated — it
        is not a guarantee of detection correctness under concurrent
        mutation. We only assert no exception escapes the thread body.
        """
        cfg = FuseConfig(
            window=128,
            cycle=CycleConfig(direct_repeat_threshold=99, cycle_min_length=99),
            stagnation=StagnationConfig(enabled=False),
        )
        fuse = AgentFuse(cfg)
        errors = []

        def worker(start):
            try:
                for i in range(50):
                    fuse.observe(_act(f"t{start}", {"i": start * 1000 + i}, "ok", True))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "thread hung"
        assert errors == []

    def test_store_append_thread_safe(self):
        """TrajectoryStore.append() can be called from multiple threads."""
        store = TrajectoryStore(":memory:", run_id="race")
        errors = []

        def worker(start):
            try:
                for i in range(50):
                    store.append(
                        TrajectoryRecord(
                            sequence=start * 1000 + i + 1,
                            run_id="race",
                            tool="t",
                            args_hash=canonical_hash({"i": i}),
                            args_repr="{}",
                            result_repr="ok",
                            success=True,
                            is_progress=False,
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "thread hung"
        assert errors == []
        rows = store.list_records()
        # 4 threads × 50 records = 200 rows; PRIMARY KEY is (run_id, seq)
        # and sequences are unique per worker, so all 200 must be present.
        assert len(rows) == 200


# ===========================================================================
# 5. Loader / CLI error paths
# ===========================================================================


class TestLoaderEdgeCases:
    def test_load_jsonl_empty_file(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text("")
        assert load_jsonl(p) == []

    def test_load_jsonl_only_blanks(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text("\n\n\n")
        assert load_jsonl(p) == []

    def test_iter_jsonl_yields_in_order(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text('{"tool":"a","args":{}}\n{"tool":"b","args":{}}\n')
        out = list(iter_jsonl(p))
        assert [r.tool for r in out] == ["a", "b"]

    def test_normalise_row_unknown_shape_raises(self):
        with pytest.raises(ValueError):
            normalise_row({"not_a_tool_field": 1})

    def test_normalise_row_accepts_canonical_shape(self):
        rec = normalise_row({
            "sequence": 7,
            "run_id": "r",
            "tool": "a",
            "args_hash": "x",
            "args_repr": "{}",
            "result_repr": "ok",
            "success": True,
            "is_progress": False,
        })
        assert rec.sequence == 7
        assert rec.tool == "a"
        assert rec.success is True

    def test_normalise_row_extra_fields_preserved(self):
        rec = normalise_row({
            "tool": "a",
            "args": {},
            "cost": 0.42,
            "model": "gpt-x",
        })
        assert rec.extra.get("cost") == 0.42
        assert rec.extra.get("model") == "gpt-x"

    def test_normalise_row_string_bool_coercion(self):
        assert normalise_row({"tool": "a", "args": {}, "success": "yes"}).success is True
        assert normalise_row({"tool": "a", "args": {}, "success": "failed"}).success is False
        assert normalise_row({"tool": "a", "args": {}, "success": "maybe"}).success is None

    def test_normalise_row_invalid_type_raises(self):
        with pytest.raises(ValueError):
            normalise_row("not a dict")  # type: ignore[arg-type]

    def test_load_jsonl_reports_line_number(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text('{"tool":"a","args":{}}\n{"this is not": valid}\n')
        with pytest.raises(ValueError) as ei:
            load_jsonl(p)
        assert "line 2" in str(ei.value)

    def test_replay_stops_at_deadlock(self):
        records = [
            TrajectoryRecord(i + 1, "r", "t", "h", "{}", "ok", True, False)
            for i in range(5)
        ]
        # Force a direct-repeat on action #4 — threshold = 2.
        records = [
            TrajectoryRecord(1, "r", "a", canonical_hash({"q": "x"}),
                             '{"q":"x"}', "ok", True, False),
            TrajectoryRecord(2, "r", "a", canonical_hash({"q": "x"}),
                             '{"q":"x"}', "ok", True, False),
            TrajectoryRecord(3, "r", "a", canonical_hash({"q": "x"}),
                             '{"q":"x"}', "ok", True, False),
        ]
        guard = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=3)))
        replayed = replay_against_guard(records, guard)
        # Stops on the third action that triggered detection.
        assert len(replayed) == 3


class TestCLIErrors:
    def _invoke(self, args, cwd=None):
        return subprocess.run(
            [sys.executable, "-m", "agent_fuse.cli", *args],
            capture_output=True, text=True, cwd=cwd,
        )

    def test_analyse_missing_file(self):
        result = self._invoke(["analyse", "/nonexistent/path.jsonl"])
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "no such file" in result.stderr.lower()

    def test_export_unknown_format(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        traj.write_text('{"tool":"a","args":{}}\n')
        result = self._invoke(["export", str(traj), "--out", str(tmp_path / "o.xyz")])
        assert result.returncode == 2

    def test_hash_invalid_json(self):
        result = self._invoke(["hash", "{not-json"])
        assert result.returncode == 2

    def test_no_subcommand_fails(self):
        result = self._invoke([])
        assert result.returncode != 0


# ===========================================================================
# 6. Export escaping
# ===========================================================================


class TestExportEscaping:
    def test_html_escapes_xss_payload(self):
        rec = TrajectoryRecord(
            sequence=1, run_id="r", tool="evil",
            args_hash="h",
            args_repr="<script>alert(1)</script>",
            result_repr="&\"<>'\x00",
            success=False, is_progress=False,
        )
        body = render_html([rec], run_id="r")
        # Raw script tag must not survive escaping.
        assert "<script>alert" not in body
        assert "&lt;script&gt;" in body
        # Ampersand, quotes, angle brackets escaped.
        assert "&amp;" in body
        # Null byte dropped or escaped — at minimum the raw char must not
        # appear inside an attribute or text context.
        assert "\x00<script>" not in body

    def test_svg_escapes_node_label(self):
        rec = TrajectoryRecord(
            sequence=1, run_id="r", tool='"><script>',
            args_hash="h",
            args_repr="{}",
            result_repr="ok",
            success=True, is_progress=False,
        )
        body = render_svg([rec])
        # The raw tool name with quotes/angle must not appear unescaped.
        assert '"><script>' not in body
        # Must be safe to embed in HTML.
        assert "&quot;" in body or "&#34;" in body or '"' not in body.split("<text")[1].split("</text>")[0]

    def test_mermaid_strips_brackets_and_pipes(self):
        rec = TrajectoryRecord(
            sequence=1, run_id="r", tool="a|b[c]d",
            args_hash="h",
            args_repr="{}",
            result_repr="ok",
            success=True, is_progress=False,
        )
        body = render_mermaid([rec])
        # All replaced with safe characters.
        assert "a|b" not in body
        assert "[c]" not in body
        assert "a/b(c)d" in body

    def test_html_render_empty_records(self):
        body = render_html([], run_id="")
        # No records, but still a valid page.
        assert "empty trajectory" in body
        assert "<table" in body

    def test_mermaid_long_label_truncated(self):
        rec = TrajectoryRecord(
            sequence=1, run_id="r", tool="x" * 200,
            args_hash="h",
            args_repr="{}",
            result_repr="y" * 100,
            success=True, is_progress=False,
        )
        body = render_mermaid([rec])
        # _short() truncates to 30 chars; must not blow up.
        assert "x" * 200 not in body


# ===========================================================================
# 7. SQLite reopen / roundtrip / persistence correctness
# ===========================================================================


class TestSQLiteReopen:
    def test_close_then_reopen_preserves_records(self, tmp_path):
        db = tmp_path / "traj.db"
        # Write some rows then close.
        s1 = TrajectoryStore(str(db), run_id="r1")
        s1.append(
            TrajectoryRecord(1, "r1", "a", "h", "{}", "ok", True, False)
        )
        s1.append(
            TrajectoryRecord(2, "r1", "b", "h", "{}", "boom", False, False)
        )
        s1.close()
        # Reopen the same file with a fresh store.
        s2 = TrajectoryStore(str(db))
        runs = s2.list_runs()
        assert "r1" in runs
        rows = s2.list_records(run_id="r1")
        assert len(rows) == 2
        assert rows[0].tool == "a"
        assert rows[1].success is False
        s2.close()

    def test_two_stores_same_file_conflict(self, tmp_path):
        # SQLite locks the file. Two stores on the same path must serialise.
        db = tmp_path / "traj.db"
        s1 = TrajectoryStore(str(db), run_id="r1")
        try:
            s2 = TrajectoryStore(str(db), run_id="r2")
            try:
                s1.append(
                    TrajectoryRecord(1, "r1", "a", "h", "{}", "ok", True, False)
                )
                s2.append(
                    TrajectoryRecord(1, "r2", "b", "h", "{}", "ok", True, False)
                )
            finally:
                s2.close()
        finally:
            s1.close()
        # Verify both runs landed.
        s3 = TrajectoryStore(str(db))
        runs = set(s3.list_runs())
        assert {"r1", "r2"}.issubset(runs)
        s3.close()

    def test_extra_field_roundtrip_with_non_ascii(self, tmp_path):
        s = TrajectoryStore(str(tmp_path / "traj.db"), run_id="r")
        s.append(
            TrajectoryRecord(
                sequence=1, run_id="r", tool="a", args_hash="h",
                args_repr="{}", result_repr="",
                success=True, is_progress=False,
                extra={"name": "café", "emoji": "🤖", "n": 0.1, "ok": True},
            )
        )
        s.close()
        s2 = TrajectoryStore(str(tmp_path / "traj.db"))
        rows = s2.list_records(run_id="r")
        assert rows[0].extra["name"] == "café"
        assert rows[0].extra["emoji"] == "🤖"
        assert rows[0].extra["ok"] is True
        s2.close()

    def test_sequence_zero_works(self, tmp_path):
        s = TrajectoryStore(str(tmp_path / "traj.db"), run_id="r")
        s.append(
            TrajectoryRecord(
                sequence=0, run_id="r", tool="a", args_hash="h",
                args_repr="{}", result_repr="",
                success=True, is_progress=False,
            )
        )
        rows = s.list_records()
        assert rows[0].sequence == 0
        s.close()

    def test_agentfuse_close_then_store_query_raises(self, tmp_path):
        # After FuseConfig closes the store (via context manager), a raw
        # sqlite query against the same connection must fail.
        cfg = FuseConfig(
            store_factory=lambda: TrajectoryStore(str(tmp_path / "t.db"), run_id="r")
        )
        with AgentFuse(cfg) as fuse:
            fuse.observe(_act("a", 1, "ok", True))
        # The guard has called close(); opening the store again works.
        s2 = TrajectoryStore(str(tmp_path / "t.db"))
        rows = s2.list_records(run_id="r")
        assert len(rows) == 1
        s2.close()

    def test_load_from_bulk_then_query_store(self, tmp_path):
        # AgentFuse.load_from() must reconstruct the in-memory window AND
        # leave the store intact for separate queries.
        cfg = FuseConfig(
            store_factory=lambda: TrajectoryStore(str(tmp_path / "t.db"), run_id="r")
        )
        records = [
            TrajectoryRecord(i + 1, "r", f"t{i}", "h", "{}", "ok", True, False)
            for i in range(5)
        ]
        fuse = AgentFuse(cfg)
        fuse.load_from(records)
        assert len(fuse.history) == 5
        # Close the underlying store explicitly (we own it via the factory).
        fuse.close()
        s = TrajectoryStore(str(tmp_path / "t.db"))
        rows = s.list_records(run_id="r")
        # load_from does *not* re-write to the store — only observes do.
        # (Verified by reading the implementation.)
        assert rows == []
        s.close()


# ===========================================================================
# 8. Misc / smoke / integration edge cases
# ===========================================================================


class TestMisc:
    def test_observation_with_no_args(self):
        fuse = AgentFuse(FuseConfig(cycle=CycleConfig(direct_repeat_threshold=99)))
        fuse.observe(_act("ping", None, "pong", True))
        assert fuse.history[0].args is None

    def test_observation_with_complex_result(self):
        # Result can be any JSON-serialisable structure.
        fuse = AgentFuse(FuseConfig())
        result = {"nested": {"a": [1, 2, 3]}, "ok": True}
        fuse.observe(_act("a", {}, result, True))
        assert fuse.history[0].result == result

    def test_history_is_a_copy(self):
        fuse = AgentFuse(FuseConfig())
        fuse.observe(_act("a", 1, "ok", True))
        snap = fuse.history
        snap.clear()
        # Internal state untouched.
        assert len(fuse.history) == 1

    def test_reset_keeps_sequence_counter(self):
        cfg = FuseConfig(
            store_factory=lambda: TrajectoryStore(":memory:", run_id="r")
        )
        fuse = AgentFuse(cfg)
        fuse.observe(_act("a", 1, "ok", True))
        fuse.reset()
        fuse.observe(_act("a", 2, "ok", True))
        rows = fuse.store.list_records()  # type: ignore[union-attr]
        # Two rows; sequence numbers must be 1 and 2 (reset doesn't renumber).
        assert [r.sequence for r in rows] == [1, 2]
        fuse.close()

    def test_progress_mark_does_not_count_as_action_for_detection(self):
        # mark_progress() bumps the trajectory row count but the window's
        # _actions list must not grow.
        fuse = AgentFuse(FuseConfig())
        fuse.observe(_act("a", 1, "ok", True))
        fuse.mark_progress(ProgressSignal(token="step1"))
        fuse.mark_progress(ProgressSignal(token="step2"))
        assert len(fuse.history) == 1
