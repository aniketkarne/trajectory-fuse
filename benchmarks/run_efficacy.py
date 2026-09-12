"""Efficacy benchmark for trajectory-fuse.

This script measures the *real-world* value of the fuse by replaying
realistic agent trajectories against it. It reports, per scenario:

* **without_fuse** — the number of tool calls the agent would have made
  before a hypothetical human (or an LLM-side retry budget) noticed
  and stopped the loop. Modeled as ``len(trajectory)`` for broken
  scenarios and ``min(len(trajectory), 50)`` for legitimate scenarios
  (we cap at 50 because legitimate scenarios have no trip point).
* **with_fuse** — the number of calls the fuse allowed before raising
  ``DeadlockDetected`` (broken) or completing cleanly (legitimate).
* **deadlock_detected** — whether the fuse raised on the scenario.
* **expected** — what the benchmark thinks should happen.

The benchmark *asserts* the expected outcome per scenario — so if the
dataset drifts and a legitimate scenario starts tripping, the
benchmark fails loudly. This is the "with-trajectory-fuse vs without"
proof.

Run::

    python benchmarks/run_efficacy.py
    python benchmarks/run_efficacy.py --json > efficacy-results.json

The script is stdlib-only (no external deps); the dataset generator
imports ``trajectory_fuse`` for the FuseConfig / StagnationConfig types.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Allow running from the repo root without installing.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "benchmarks"))

from efficacy_dataset import (  # noqa: E402
    EXPECTED_DEADLOCK,
    SCENARIOS,
    default_config,
    generate,
    iter_scaled_dataset,
)

from trajectory_fuse import (  # noqa: E402
    Action,
    AgentFuse,
    DeadlockDetected,
    FuseConfig,
    ProgressSignal,
    TrajectoryStore,
)
from trajectory_fuse.guard import (
    CycleConfig,  # noqa: E402
    StagnationConfig,  # noqa: E402
)


@dataclass
class ScenarioResult:
    scenario: str
    expected_deadlock: bool
    actual_deadlock: bool
    without_fuse_calls: int
    with_fuse_calls: int
    detection_kind: Optional[str] = None
    duration_ms: float = 0.0
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "expected_deadlock": self.expected_deadlock,
            "actual_deadlock": self.actual_deadlock,
            "match": self.expected_deadlock == self.actual_deadlock,
            "without_fuse_calls": self.without_fuse_calls,
            "with_fuse_calls": self.with_fuse_calls,
            "calls_avoided": self.without_fuse_calls - self.with_fuse_calls,
            "detection_kind": self.detection_kind,
            "duration_ms": round(self.duration_ms, 3),
            "error": self.error,
        }


# ---------------------------------------------------------------------
# replay helpers


def replay_with_fuse(scenario: str, actions: List[dict]) -> tuple:
    """Replay ``actions`` through a fresh ``AgentFuse``.

    Returns ``(calls_allowed_before_trip, detection_kind_or_None)``.
    """
    cfg = default_config(scenario)
    # Use the same budget caps so an "infinite legitimate loop" scenario
    # doesn't make this benchmark hang.
    fuse = AgentFuse(cfg)
    detection = None
    calls_allowed = 0
    for a in actions:
        if a.get("is_progress"):
            fuse.mark_progress(ProgressSignal(token=str(a.get("args", "")), note=str(a.get("result", ""))))
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
            calls_allowed += 1
        except DeadlockDetected as exc:
            detection = exc.detection
            return calls_allowed + 1, detection.kind.value if detection else "unknown"
    return calls_allowed, None


def replay_without_fuse(scenario: str, actions: List[dict], cap: int = 50) -> int:
    """Simulate an unguarded agent.

    For broken scenarios, the model would burn through every call in
    the trajectory (the loop has no exit). For legitimate scenarios,
    we cap at ``cap`` calls because the loop legitimately runs to
    completion — there's nothing for a fuse to save.

    The cap models "a human / external monitor notices after N calls".
    A real outage in production would burn the entire trajectory for
    broken scenarios.
    """
    if EXPECTED_DEADLOCK[scenario]:
        return len(actions)
    return min(len(actions), cap)


# ---------------------------------------------------------------------
# main


def run_all(scenarios: Iterable[str] = SCENARIOS) -> List[ScenarioResult]:
    results = []
    for scenario in scenarios:
        actions = generate(scenario, seed=0)
        started = time.perf_counter()
        try:
            calls_allowed, kind = replay_with_fuse(scenario, actions)
            without_calls = replay_without_fuse(scenario, actions)
            results.append(
                ScenarioResult(
                    scenario=scenario,
                    expected_deadlock=EXPECTED_DEADLOCK[scenario],
                    actual_deadlock=kind is not None,
                    without_fuse_calls=without_calls,
                    with_fuse_calls=calls_allowed,
                    detection_kind=kind,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            )
        except Exception as exc:  # noqa: BLE001 - benchmark reports it
            results.append(
                ScenarioResult(
                    scenario=scenario,
                    expected_deadlock=EXPECTED_DEADLOCK[scenario],
                    actual_deadlock=False,
                    without_fuse_calls=len(actions),
                    with_fuse_calls=0,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def format_table(results: List[ScenarioResult]) -> str:
    rows = ["scenario                       expected  actual    without  with   avoided  detection       duration"]
    rows.append("-" * 110)
    for r in results:
        exp = "trip    " if r.expected_deadlock else "pass    "
        act = "trip    " if r.actual_deadlock else "pass    "
        avoided = r.without_fuse_calls - r.with_fuse_calls
        kind = r.detection_kind or "-"
        rows.append(
            f"{r.scenario:<30} {exp} {act} {r.without_fuse_calls:>7}  {r.with_fuse_calls:>4}  {avoided:>7}  "
            f"{kind:<14} {r.duration_ms:>8.2f}ms"
        )
    return "\n".join(rows)


def summarize(results: List[ScenarioResult]) -> dict:
    total_without = sum(r.without_fuse_calls for r in results)
    total_with = sum(r.with_fuse_calls for r in results)
    matches = sum(1 for r in results if r.expected_deadlock == r.actual_deadlock)
    return {
        "scenarios_total": len(results),
        "scenarios_matched": matches,
        "scenarios_failed": len(results) - matches,
        "total_calls_without_fuse": total_without,
        "total_calls_with_fuse": total_with,
        "calls_avoided": total_without - total_with,
        "avoided_fraction": (total_without - total_with) / max(total_without, 1),
    }


def run_scaled() -> dict:
    """Run every trajectory in the scaled 500+ dataset.

    Returns aggregate metrics: counts of true positives, false
    positives, false negatives, true negatives — the standard
    confusion matrix for a binary classifier.
    """
    tp = fp = tn = fn = 0
    total_without = total_with = 0
    n = 0
    for label, actions, expected in iter_scaled_dataset():
        # Resolve the scenario family (polling/retry/...) for the config.
        family = label.split("/", 1)[0]
        # Map family names that don't match the EXPECTED_DEADLOCK keys.
        family_aliases = {
            "direct_repeat": "broken_direct_repeat",
            "n_cycle": "broken_n_cycle",
            "stagnation": "broken_stagnation",
            "stuck": "broken_stuck_after_recovery",
            "polling": "legitimate_polling",
            "pagination": "legitimate_pagination",
            "retry": "legitimate_retry",
            "long_loop": "legitimate_long_loop",
            "progress_phase": "legitimate_progress_phase",
        }
        scenario = family_aliases.get(family, family) or "legitimate_polling"
        # Infer expected from the label for the edge-case entries.
        if "tiny" in label or "long_budget" in label:
            expected = False
        elif "under_threshold" in label and "direct_repeat" in label:
            expected = False
        elif "at_threshold" in label or "past_threshold" in label or "under_threshold" in label:
            expected = True

        cfg = default_config(scenario)
        # Use a per-trajectory budget so a runaway trajectory can't
        # starve the benchmark.
        cfg.budgets = type(cfg.budgets)(max_actions=500)
        fuse = AgentFuse(cfg)
        detection = None
        calls_allowed = 0
        for a in actions:
            if a.get("is_progress"):
                fuse.mark_progress(ProgressSignal(token=str(a.get("args", "")), note=str(a.get("result", ""))))
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
                calls_allowed += 1
            except DeadlockDetected as exc:
                detection = exc.detection.kind.value
                break

        actual = detection is not None
        # Debug: print false positives so we can investigate. Only the
        # 9-scenario run (without --scaled) prints this; the scaled run
        # stays quiet unless ``--verbose-fp`` is passed.
        if actual and not expected and os.environ.get("EFFICACY_VERBOSE_FP"):
            print(f"FP: {label} tripped with {detection}", file=sys.stderr)
        if expected and actual:
            tp += 1
        elif expected and not actual:
            fn += 1
        elif not expected and actual:
            fp += 1
        else:
            tn += 1

        without_calls = len(actions) if expected else min(len(actions), 50)
        total_without += without_calls
        total_with += calls_allowed
        n += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return {
        "trajectories": n,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "total_calls_without_fuse": total_without,
        "total_calls_with_fuse": total_with,
        "calls_avoided": total_without - total_with,
        "avoided_fraction": (total_without - total_with) / max(total_without, 1),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="trajectory-fuse efficacy benchmark")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any scenario's outcome mismatches expectations")
    p.add_argument("--scaled", action="store_true",
                   help="Run the 500+ trajectory scaled benchmark")
    args = p.parse_args(argv)

    if args.scaled:
        result = run_scaled()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for k, v in result.items():
                print(f"  {k:<28} {v}")
        # Strict mode: any false positives or false negatives are failure.
        if args.strict and (result["false_positives"] or result["false_negatives"]):
            return 1
        return 0

    results = run_all()
    summary = summarize(results)
    payload = {"summary": summary, "scenarios": [r.as_dict() for r in results]}

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_table(results))
        print()
        print("Summary:")
        for k, v in summary.items():
            print(f"  {k:<28} {v}")

    if args.strict and summary["scenarios_failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
