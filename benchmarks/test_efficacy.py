"""Efficacy benchmark test for trajectory-fuse.

Runs the scaled trajectory dataset (450+ trajectories) and asserts
that the fuse:

* trips on every broken trajectory (recall = 1.0),
* does NOT trip on any legitimate trajectory (false-positive rate = 0%),
* still saves a non-trivial fraction of wasted calls in broken scenarios.

These numbers are the *contract* — they are not asserting absolute
performance, just the qualitative property that the fuse gets the
basic shapes right. A regression that breaks the detectors (e.g.
silently disabling direct_repeat) would push either precision or
recall below 1.0 and fail this test loudly.

Run directly::

    pytest benchmarks/test_efficacy.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Allow running from the repo root without installing.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "benchmarks"))

from efficacy_dataset import iter_scaled_dataset  # noqa: E402
from run_efficacy import run_scaled  # noqa: E402

from trajectory_fuse import (  # noqa: E402
    Action,
    AgentFuse,
    DeadlockDetected,
    ProgressSignal,
)
from trajectory_fuse.guard import RunBudget  # noqa: E402

# Acceptable thresholds for the scaled benchmark.
MIN_PRECISION = 0.99
MIN_RECALL = 0.99
MAX_FALSE_POSITIVE_RATE = 0.01
MIN_AVOIDED_FRACTION = 0.05  # the fuse must save at least 5% of broken-trajectory calls


@pytest.fixture(scope="module")
def scaled_result() -> dict:
    return run_scaled()


def test_scaled_dataset_size(scaled_result):
    """Sanity: the scaled dataset must have enough trajectories to be meaningful."""
    assert scaled_result["trajectories"] >= 400, (
        f"only {scaled_result['trajectories']} trajectories in scaled dataset; "
        "expected at least 400 to make recall/precision numbers meaningful"
    )


def test_precision_meets_threshold(scaled_result):
    assert scaled_result["precision"] >= MIN_PRECISION, (
        f"precision {scaled_result['precision']:.3f} below threshold {MIN_PRECISION}; "
        f"false positives: {scaled_result['false_positives']}"
    )


def test_recall_meets_threshold(scaled_result):
    assert scaled_result["recall"] >= MIN_RECALL, (
        f"recall {scaled_result['recall']:.3f} below threshold {MIN_RECALL}; "
        f"false negatives: {scaled_result['false_negatives']}"
    )


def test_false_positive_rate_below_threshold(scaled_result):
    assert scaled_result["false_positive_rate"] <= MAX_FALSE_POSITIVE_RATE, (
        f"false-positive rate {scaled_result['false_positive_rate']:.3f} above "
        f"threshold {MAX_FALSE_POSITIVE_RATE}; "
        f"false positives: {scaled_result['false_positives']}"
    )


def test_call_savings_nontrivial(scaled_result):
    """The fuse must actually save calls in broken scenarios."""
    assert scaled_result["avoided_fraction"] >= MIN_AVOIDED_FRACTION, (
        f"avoided fraction {scaled_result['avoided_fraction']:.3f} below "
        f"threshold {MIN_AVOIDED_FRACTION}; the fuse is not catching broken "
        f"trajectories early enough"
    )


def test_per_family_broken_trajectories_trip():
    """Spot-check that *every family* of broken scenario trips correctly.

    This is more useful than aggregate precision/recall when iterating on
    the dataset: a regression in one detector (e.g. N-cycle) would still
    leave overall precision high if the other detectors catch their
    shapes, but this test would fail loudly.
    """
    families_seen = set()
    for label, actions, expected in iter_scaled_dataset():
        family = label.split("/", 1)[0]
        if "seed=" not in label or family in families_seen:
            continue
        families_seen.add(family)
        if not expected:
            continue
        cfg = __import__("trajectory_fuse.guard", fromlist=["FuseConfig", "CycleConfig", "StagnationConfig"]).FuseConfig(
            cycle=__import__("trajectory_fuse.guard", fromlist=["CycleConfig"]).CycleConfig(
                direct_repeat_threshold=3, cycle_min_length=2, cycle_max_length=6
            ),
            stagnation=__import__("trajectory_fuse.guard", fromlist=["StagnationConfig"]).StagnationConfig(
                enabled=True, window=8, similarity_threshold=0.9, min_failures=4
            ),
            budgets=RunBudget(max_actions=500),
        )
        fuse = AgentFuse(cfg)
        tripped = False
        for a in actions:
            if a.get("is_progress"):
                fuse.mark_progress(
                    ProgressSignal(token=str(a.get("args", "")), note=str(a.get("result", "")))
                )
                continue
            try:
                fuse.observe(
                    Action(
                        tool=a["tool"],
                        args=a["args"],
                        result=a["result"],
                        success=a.get("success"),
                    )
                )
            except DeadlockDetected:
                tripped = True
                break
        assert tripped, (
            f"family {family!r} (label {label!r}) did not trip the fuse "
            "— the detector for this family is broken"
        )


def test_legitimate_trajectories_complete():
    """Spot-check that *every family* of legitimate scenario completes cleanly.

    The fuse must not trip on polling, pagination, retry, long_loop, or
    progress_phase trajectories. A regression that pushes a legitimate
    trajectory over the threshold is a hard fail.
    """
    families_seen = set()
    for label, actions, expected in iter_scaled_dataset():
        family = label.split("/", 1)[0]
        if "seed=" not in label or family in families_seen:
            continue
        families_seen.add(family)
        if expected:
            continue
        cfg = __import__("trajectory_fuse.guard", fromlist=["FuseConfig", "CycleConfig", "StagnationConfig"]).FuseConfig(
            cycle=__import__("trajectory_fuse.guard", fromlist=["CycleConfig"]).CycleConfig(
                direct_repeat_threshold=3, cycle_min_length=2, cycle_max_length=6
            ),
            stagnation=__import__("trajectory_fuse.guard", fromlist=["StagnationConfig"]).StagnationConfig(
                enabled=True, window=8, similarity_threshold=0.9, min_failures=4
            ),
            budgets=RunBudget(max_actions=500),
            allow_repeats={"poll_status": 200, "fetch_page": 200},
        )
        fuse = AgentFuse(cfg)
        tripped = False
        for a in actions:
            if a.get("is_progress"):
                fuse.mark_progress(
                    ProgressSignal(token=str(a.get("args", "")), note=str(a.get("result", "")))
                )
                continue
            try:
                fuse.observe(
                    Action(
                        tool=a["tool"],
                        args=a["args"],
                        result=a["result"],
                        success=a.get("success"),
                    )
                )
            except DeadlockDetected:
                tripped = True
                break
        assert not tripped, (
            f"family {family!r} (label {label!r}) tripped the fuse "
            "— this is a legitimate trajectory; the detector is too aggressive"
        )


def test_dataset_export_roundtrip(tmp_path: Path):
    """The JSONL export roundtrip must produce the same expected outcome."""
    from efficacy_dataset import write_dataset

    written = write_dataset(tmp_path, seed=0)
    assert written > 0
    # Each file is a JSONL trajectory.
    for path in sorted(tmp_path.glob("*.jsonl")):
        lines = list(path.read_text(encoding="utf-8").strip().split("\n"))
        assert lines, f"empty file {path}"
        for line in lines:
            json.loads(line)
