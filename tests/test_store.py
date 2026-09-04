"""Tests for the SQLite trajectory store."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile

import pytest

from trajectory_fuse import AgentFuse, Action, FuseConfig
from trajectory_fuse.store import TrajectoryStore, open_store
from trajectory_fuse.types import TrajectoryRecord


@pytest.fixture
def tmp_store(tmp_path):
    """Yield an *uninitialised* TrajectoryStore whose path is in tmp_path.

    The fixture does not close the store — the caller / AgentFuse manages
    its lifecycle. This makes it safe to pass the same store into an
    AgentFuse that will close it on exit.
    """
    path = tmp_path / "traj.db"
    store = TrajectoryStore(str(path), run_id="run-1")
    yield store


class TestTrajectoryStoreBasic:
    def test_create_and_append(self, tmp_store):
        rec = TrajectoryRecord(
            sequence=1,
            run_id="run-1",
            tool="search",
            args_hash="abc",
            args_repr='{"q":"x"}',
            result_repr="ok",
            success=True,
            is_progress=False,
        )
        tmp_store.append(rec)
        rows = tmp_store.list_records()
        assert len(rows) == 1
        assert rows[0].tool == "search"
        assert rows[0].success is True

    def test_in_memory_store(self):
        store = TrajectoryStore(":memory:", run_id="r")
        store.append(
            TrajectoryRecord(
                sequence=1,
                run_id="r",
                tool="a",
                args_hash="h",
                args_repr="{}",
                result_repr="",
                success=None,
                is_progress=False,
            )
        )
        rows = store.list_records()
        assert len(rows) == 1
        store.close()

    def test_progress_row_distinguishable(self, tmp_store):
        tmp_store.append(
            TrajectoryRecord(
                sequence=1,
                run_id="run-1",
                tool="<progress>",
                args_hash="",
                args_repr="new state",
                result_repr="",
                success=None,
                is_progress=True,
            )
        )
        tmp_store.append(
            TrajectoryRecord(
                sequence=2,
                run_id="run-1",
                tool="search",
                args_hash="x",
                args_repr="{}",
                result_repr="ok",
                success=True,
                is_progress=False,
            )
        )
        all_rows = tmp_store.list_records()
        assert len(all_rows) == 2
        non_progress = tmp_store.list_records(include_progress=False)
        assert len(non_progress) == 1
        assert non_progress[0].is_progress is False

    def test_extra_json_roundtrip(self, tmp_store):
        tmp_store.append(
            TrajectoryRecord(
                sequence=1,
                run_id="run-1",
                tool="a",
                args_hash="h",
                args_repr="{}",
                result_repr="",
                success=True,
                is_progress=False,
                extra={"cost": 0.01, "model": "gpt-x"},
            )
        )
        row = tmp_store.list_records()[0]
        assert row.extra == {"cost": 0.01, "model": "gpt-x"}

    def test_list_runs(self, tmp_store):
        tmp_store.initialise("run-1")
        tmp_store.append(
            TrajectoryRecord(
                sequence=1,
                run_id="run-1",
                tool="a",
                args_hash="h",
                args_repr="{}",
                result_repr="",
                success=True,
                is_progress=False,
            )
        )
        other = TrajectoryStore(":memory:", run_id="run-2")
        other.append(
            TrajectoryRecord(
                sequence=1,
                run_id="run-2",
                tool="a",
                args_hash="h",
                args_repr="{}",
                result_repr="",
                success=True,
                is_progress=False,
            )
        )
        # tmp_store only sees run-1
        assert "run-1" in tmp_store.list_runs()
        other.close()

    def test_replace_sequence_idempotent(self, tmp_store):
        rec = TrajectoryRecord(
            sequence=1,
            run_id="run-1",
            tool="a",
            args_hash="h",
            args_repr="{}",
            result_repr="",
            success=True,
            is_progress=False,
        )
        tmp_store.append(rec)
        tmp_store.append(rec)  # INSERT OR REPLACE
        assert len(tmp_store.list_records()) == 1


class TestTrajectoryStoreIntegration:
    def test_persists_each_observe(self, tmp_store):
        cfg = FuseConfig(store_factory=lambda: tmp_store, run_id="run-1")
        fuse = AgentFuse(cfg)
        for i in range(5):
            fuse.observe(Action(tool="search", args={"q": i}, result="ok", success=True))
        rows = tmp_store.list_records()
        assert len(rows) == 5
        assert [r.sequence for r in rows] == [1, 2, 3, 4, 5]
        # Args hashes are distinct and deterministic.
        hashes = {r.args_hash for r in rows}
        assert len(hashes) == 5

    def test_persists_progress_marks(self, tmp_store):
        cfg = FuseConfig(store_factory=lambda: tmp_store, run_id="run-1")
        fuse = AgentFuse(cfg)
        fuse.observe(Action(tool="a", args={}, result="ok", success=True))
        fuse.mark_progress(__import__("trajectory_fuse").types.ProgressSignal(token="got new state"))
        rows = tmp_store.list_records()
        assert len(rows) == 2
        assert rows[1].is_progress is True
        assert rows[1].tool == "<progress>"

    def test_open_store_creates_parent_dir(self, tmp_path):
        target = tmp_path / "nested" / "deeper" / "traj.db"
        store = open_store(target)
        store.close()
        assert target.exists()

    def test_initialise_generates_uuid(self, tmp_path):
        store = TrajectoryStore(str(tmp_path / "x.db"))
        rid = store.initialise()
        assert rid and isinstance(rid, str) and len(rid) >= 8
        store.close()