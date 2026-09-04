"""Tests for the CLI and exporters."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trajectory_fuse.cli import main
from trajectory_fuse.export import render_html, render_mermaid, render_svg
from trajectory_fuse.loader import load_jsonl, normalise_row
from trajectory_fuse.types import TrajectoryRecord


def _write_jsonl(path, rows):
    with open(str(path), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _invoke(args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "trajectory_fuse.cli", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


class TestCLIBasic:
    def test_help(self):
        result = _invoke(["--help"])
        assert result.returncode == 0
        assert "trajectory-fuse" in result.stdout

    def test_version(self):
        result = _invoke(["--version"])
        assert result.returncode == 0
        assert "trajectory-fuse" in result.stdout

    def test_hash(self):
        result = _invoke(["hash", '{"a": 1, "b": [1, 2]}'])
        assert result.returncode == 0
        assert len(result.stdout.strip()) == 16


class TestCLILoader:
    def test_normalise_row_minimal(self):
        rec = normalise_row({"tool": "search", "args": {"q": "x"}})
        assert rec.tool == "search"
        assert rec.success is None
        assert rec.is_progress is False
        assert rec.args_hash != ""

    def test_normalise_row_error_implies_failure(self):
        rec = normalise_row({"tool": "search", "args": {}, "error": "boom"})
        assert rec.success is False

    def test_normalise_row_rejects_missing_tool(self):
        with pytest.raises(ValueError):
            normalise_row({"args": {}})

    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        p = tmp_path / "t.jsonl"
        with open(p, "w") as fh:
            fh.write('{"tool":"a","args":{}}\n')
            fh.write("\n")
            fh.write('{"tool":"b","args":{}}\n')
        records = load_jsonl(p)
        assert len(records) == 2

    def test_load_jsonl_invalid_json(self, tmp_path):
        p = tmp_path / "t.jsonl"
        with open(p, "w") as fh:
            fh.write("not json\n")
        with pytest.raises(ValueError):
            load_jsonl(p)


class TestCLIAnalyse:
    def _write_deadlock_trajectory(self, path):
        # 3 identical search calls — should direct-repeat on the 3rd.
        _write_jsonl(
            path,
            [
                {"tool": "search", "args": {"q": "x"}, "result": "ok", "success": True},
                {"tool": "search", "args": {"q": "x"}, "result": "ok", "success": True},
                {"tool": "search", "args": {"q": "x"}, "result": "ok", "success": True},
            ],
        )

    def test_analyses_text(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        self._write_deadlock_trajectory(traj)
        result = _invoke(["analyse", str(traj)])
        assert result.returncode == 1
        assert "DEADLOCK" in result.stdout
        assert "direct_repeat" in result.stdout

    def test_analyses_json(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        self._write_deadlock_trajectory(traj)
        result = _invoke(["analyse", str(traj), "--json"])
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["deadlock_detected"] is True
        assert payload["detection"]["kind"] == "direct_repeat"

    def test_analyses_clean_trajectory(self, tmp_path):
        traj = tmp_path / "clean.jsonl"
        _write_jsonl(
            traj,
            [
                {"tool": "a", "args": {"i": 1}, "result": "ok", "success": True},
                {"tool": "b", "args": {"i": 2}, "result": "ok", "success": True},
                {"tool": "c", "args": {"i": 3}, "result": "ok", "success": True},
            ],
        )
        result = _invoke(["analyse", str(traj)])
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_analyses_persists_to_db(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        self._write_deadlock_trajectory(traj)
        db = tmp_path / "out.db"
        result = _invoke(["analyse", str(traj), "--db", str(db)])
        assert result.returncode == 1
        assert db.exists()


class TestCLIExport:
    def _write_trajectory(self, path):
        _write_jsonl(
            path,
            [
                {"tool": "a", "args": {"i": 1}, "result": "ok", "success": True},
                {"tool": "b", "args": {"i": 2}, "result": "ok", "success": True},
                {"tool": "b", "args": {"i": 2}, "result": "ok", "success": True},
            ],
        )

    def test_export_html(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        self._write_trajectory(traj)
        out = tmp_path / "out.html"
        result = _invoke(["export", str(traj), "--out", str(out)])
        assert result.returncode == 0, result.stderr
        assert out.exists()
        body = out.read_text()
        assert "<svg" in body
        assert "<table" in body
        assert "trajectory-fuse" in body

    def test_export_svg(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        self._write_trajectory(traj)
        out = tmp_path / "out.svg"
        result = _invoke(["export", str(traj), "--out", str(out)])
        assert result.returncode == 0
        assert out.exists()
        body = out.read_text()
        assert body.lstrip().startswith("<?xml")
        assert "<svg" in body

    def test_export_mermaid(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        self._write_trajectory(traj)
        out = tmp_path / "out.md"
        result = _invoke(["export", str(traj), "--out", str(out)])
        assert result.returncode == 0
        body = out.read_text()
        assert "```mermaid" in body
        assert "graph LR" in body

    def test_export_unknown_extension_errors(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        self._write_trajectory(traj)
        out = tmp_path / "out.txt"
        result = _invoke(["export", str(traj), "--out", str(out)])
        assert result.returncode != 0


class TestCLIReplay:
    def test_replay_quiet(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        _write_jsonl(
            traj,
            [
                {"tool": "a", "args": {"i": 1}, "result": "ok", "success": True},
                {"tool": "b", "args": {"i": 2}, "result": "ok", "success": True},
            ],
        )
        result = _invoke(["replay", str(traj), "--quiet"])
        assert result.returncode == 0
        assert "no deadlock" in result.stdout

    def test_replay_verbose(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        _write_jsonl(
            traj,
            [
                {"tool": "a", "args": {"i": 1}, "result": "ok", "success": True},
                {"tool": "a", "args": {"i": 1}, "result": "ok", "success": True},
                {"tool": "a", "args": {"i": 1}, "result": "ok", "success": True},
            ],
        )
        result = _invoke(["replay", str(traj), "--direct-repeat", "2"])
        assert result.returncode == 1
        assert "DEADLOCK" in result.stdout


class TestCLIStats:
    def test_stats_text(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        _write_jsonl(
            traj,
            [
                {"tool": "a", "args": {"i": 1}, "result": "ok", "success": True},
                {"tool": "a", "args": {"i": 2}, "result": "timeout", "success": False},
                {"tool": "b", "args": {}, "result": "ok", "success": True},
                {"tool": "<progress>", "is_progress": True, "args_repr": "ok", "result_repr": "ok"},
            ],
        )
        result = _invoke(["stats", str(traj)])
        assert result.returncode == 0
        assert "total" in result.stdout
        assert "ok" in result.stdout
        assert "fail" in result.stdout

    def test_stats_json(self, tmp_path):
        traj = tmp_path / "t.jsonl"
        _write_jsonl(
            traj,
            [
                {"tool": "a", "args": {"i": 1}, "result": "ok", "success": True},
                {"tool": "a", "args": {"i": 2}, "result": "timeout", "success": False},
            ],
        )
        result = _invoke(["stats", str(traj), "--json"])
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["total"] == 2
        assert payload["ok"] == 1
        assert payload["fail"] == 1
        assert payload["tools"] == {"a": 2}


class TestExportRenderers:
    def test_render_html_empty(self):
        body = render_html([], run_id="x")
        assert "<svg" in body
        assert "empty trajectory" in body

    def test_render_svg_empty(self):
        body = render_svg([])
        assert body.lstrip().startswith("<?xml")

    def test_render_mermaid_empty(self):
        body = render_mermaid([])
        assert "graph LR" in body
        assert "empty" in body

    def test_render_html_escapes_special_chars(self):
        rec = TrajectoryRecord(
            sequence=1,
            run_id="r",
            tool="<script>",
            args_hash="h",
            args_repr="<>",
            result_repr="&\"><",
            success=True,
            is_progress=False,
        )
        body = render_html([rec], run_id="r")
        assert "<script>" not in body  # escaped
        assert "&lt;script&gt;" in body or "&lt;script" in body

    def test_render_mermaid_strips_dangerous_chars(self):
        rec = TrajectoryRecord(
            sequence=1,
            run_id="r",
            tool='a"|b[c]',
            args_hash="h",
            args_repr="{}",
            result_repr="ok",
            success=True,
            is_progress=False,
        )
        body = render_mermaid([rec])
        assert '"|b[c]' not in body  # sanitised

    def test_render_html_failure_and_progress_styled(self):
        records = [
            TrajectoryRecord(1, "r", "a", "h", "{}", "ok", True, False),
            TrajectoryRecord(2, "r", "b", "h", "{}", "boom", False, False),
            TrajectoryRecord(3, "r", "<progress>", "", "step", "", None, True),
        ]
        body = render_html(records, run_id="r")
        assert "fail" in body
        assert "progress" in body