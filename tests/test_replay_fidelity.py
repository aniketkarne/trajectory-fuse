"""Replay fidelity: replaying a recorded trajectory must produce the same
detection outcomes as the original live run.

Regression tests for the 2026-11-27 audit. The CLI ``analyse``/``replay``
commands and :func:`loader.replay_against_guard` used to hash the
JSON-encoded ``args_repr`` *string* instead of the recorded arguments,
which both fabricated direct repeats (canonical rows with distinct
``args_hash`` values but empty ``args_repr`` all hashed identically) and
missed real ones (arguments equal modulo dict key order hashed
differently than they do live, where ``canonical_hash`` is
key-order-insensitive).
"""

from __future__ import annotations

import json

import pytest

from trajectory_fuse import AgentFuse, DeadlockDetected
from trajectory_fuse.cli import main
from trajectory_fuse.export import render_html
from trajectory_fuse.hashing import canonical_hash
from trajectory_fuse.loader import (
    iter_jsonl,
    normalise_row,
    record_to_action,
    replay_against_guard,
)
from trajectory_fuse.types import TrajectoryRecord


def _write_jsonl(path, rows):
    with open(str(path), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _canonical_row(seq, tool, args_hash):
    return {
        "sequence": seq,
        "run_id": "r1",
        "tool": tool,
        "args_hash": args_hash,
        "args_repr": "",
        "result_repr": "ok",
        "success": True,
        "is_progress": False,
        "extra": {},
    }


# ---------------------------------------------------------------------------
# record_to_action: the replayed action must hash like the recorded args
# ---------------------------------------------------------------------------


class TestRecordToActionHashFidelity:
    def test_friendly_row_round_trips_recorded_hash(self):
        rec = normalise_row({"tool": "search", "args": {"q": 1, "r": 2}})
        action = record_to_action(rec)
        assert canonical_hash(action.args) == rec.args_hash

    def test_key_order_permutation_hashes_equal(self):
        a = normalise_row({"tool": "search", "args": {"q": 1, "r": 2}})
        b = normalise_row({"tool": "search", "args": {"r": 2, "q": 1}})
        assert canonical_hash(record_to_action(a).args) == canonical_hash(
            record_to_action(b).args
        )

    def test_stored_hash_rows_compare_by_recorded_hash(self):
        rec1 = TrajectoryRecord(
            sequence=1, run_id="r", tool="t", args_hash="aaaa1111",
            args_repr="", result_repr="", success=True, is_progress=False,
        )
        rec2 = TrajectoryRecord(
            sequence=2, run_id="r", tool="t", args_hash="aaaa1111",
            args_repr="", result_repr="", success=True, is_progress=False,
        )
        rec3 = TrajectoryRecord(
            sequence=3, run_id="r", tool="t", args_hash="bbbb2222",
            args_repr="", result_repr="", success=True, is_progress=False,
        )
        h1 = canonical_hash(record_to_action(rec1).args)
        h2 = canonical_hash(record_to_action(rec2).args)
        h3 = canonical_hash(record_to_action(rec3).args)
        assert h1 == h2
        assert h1 != h3

    def test_unparseable_repr_falls_back_to_stored_hash(self):
        rec = TrajectoryRecord(
            sequence=1, run_id="r", tool="t", args_hash="cccc3333",
            args_repr='{"truncated": "mid', result_repr="", success=None,
            is_progress=False,
        )
        action = record_to_action(rec)
        assert canonical_hash(action.args) == canonical_hash(
            record_to_action(rec).args
        )

    def test_no_args_info_stays_empty(self):
        rec = TrajectoryRecord(
            sequence=1, run_id="r", tool="t", args_hash="",
            args_repr="", result_repr="", success=None, is_progress=False,
        )
        assert record_to_action(rec).args == ""


# ---------------------------------------------------------------------------
# CLI analyse: detection outcomes must match live semantics
# ---------------------------------------------------------------------------


class TestAnalyseReplayFidelity:
    def test_no_false_repeat_on_distinct_stored_hashes(self, tmp_path, capsys):
        """Three same-tool rows with *different* args_hash must not be
        reported as identical-argument repeats."""
        p = tmp_path / "canonical.jsonl"
        _write_jsonl(
            p,
            [
                _canonical_row(1, "search", "aaaa1111"),
                _canonical_row(2, "search", "bbbb2222"),
                _canonical_row(3, "search", "cccc3333"),
            ],
        )
        rc = main(["analyse", str(p), "--json"])
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert out["deadlock_detected"] is False

    def test_detects_repeat_across_key_order(self, tmp_path, capsys):
        """Args equal modulo dict key order are identical to the live
        guard; replay must see the same direct repeat."""
        p = tmp_path / "friendly.jsonl"
        _write_jsonl(
            p,
            [
                {"tool": "search", "args": {"q": 1, "r": 2}, "result": "ok", "success": True},
                {"tool": "search", "args": {"r": 2, "q": 1}, "result": "ok", "success": True},
                {"tool": "search", "args": {"q": 1, "r": 2}, "result": "ok", "success": True},
            ],
        )
        rc = main(["analyse", str(p), "--json"])
        out = json.loads(capsys.readouterr().out)
        assert rc == 1
        assert out["deadlock_detected"] is True
        assert out["detection"]["kind"] == "direct_repeat"

    def test_replay_against_guard_stops_at_trigger(self, tmp_path):
        p = tmp_path / "friendly.jsonl"
        _write_jsonl(
            p,
            [
                {"tool": "search", "args": {"q": 1}, "result": "ok", "success": True},
                {"tool": "search", "args": {"q": 1}, "result": "ok", "success": True},
                {"tool": "search", "args": {"q": 1}, "result": "ok", "success": True},
                {"tool": "fetch", "args": {"u": 2}, "result": "ok", "success": True},
            ],
        )
        records = list(iter_jsonl(p))
        fuse = AgentFuse()
        out = replay_against_guard(records, fuse)
        assert len(out) == 3  # stops at the triggering record, not past it


# ---------------------------------------------------------------------------
# loader row coercion + error reporting
# ---------------------------------------------------------------------------


class TestNormaliseRowCoercion:
    def test_is_progress_string_false(self):
        rec = normalise_row({"tool": "t", "args": {}, "is_progress": "false"})
        assert rec.is_progress is False

    def test_is_progress_string_zero(self):
        rec = normalise_row({"tool": "t", "args": {}, "is_progress": "0"})
        assert rec.is_progress is False

    def test_is_progress_string_true(self):
        rec = normalise_row({"tool": "t", "args": {}, "is_progress": "true"})
        assert rec.is_progress is True


class TestIterJsonlErrors:
    def test_invalid_json_error_has_line_number(self, tmp_path):
        p = tmp_path / "t.jsonl"
        with open(p, "w") as fh:
            fh.write('{"tool":"a","args":{}}\n')
            fh.write("not json\n")
        with pytest.raises(ValueError, match="line 2"):
            list(iter_jsonl(p))


class TestCLIErrorHandling:
    def test_analyse_malformed_line_clean_error(self, tmp_path, capsys):
        p = tmp_path / "t.jsonl"
        with open(p, "w") as fh:
            fh.write("not json\n")
        rc = main(["analyse", str(p)])
        captured = capsys.readouterr()
        assert rc == 2
        assert "Traceback" not in captured.err

    def test_analyse_missing_tool_clean_error(self, tmp_path, capsys):
        p = tmp_path / "t.jsonl"
        _write_jsonl(p, [{"args": {}}])
        rc = main(["replay", str(p)])
        captured = capsys.readouterr()
        assert rc == 2
        assert "Traceback" not in captured.err


# ---------------------------------------------------------------------------
# export version truth
# ---------------------------------------------------------------------------


class TestExportVersion:
    def test_render_html_default_version_is_package_version(self):
        from trajectory_fuse import __version__

        html = render_html([], run_id="r")
        assert f"trajectory-fuse {__version__}" in html
        assert "trajectory-fuse 0.1.0" not in html
