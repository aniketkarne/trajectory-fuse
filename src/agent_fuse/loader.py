"""JSONL trajectory loader + CLI export helpers.

The CLI accepts trajectories in the *TrajectoryRecord* JSON shape::

    {"sequence": 1, "run_id": "abc", "tool": "search",
     "args_hash": "deadbeef", "args_repr": "{...}", "result_repr": "...",
     "success": true, "is_progress": false, "extra": {}}

But to make the CLI ergonomic for humans, we also accept a simpler shape::

    {"tool": "search", "args": {...}, "result": "...", "success": false}

The loader normalises both into :class:`TrajectoryRecord` instances.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Union

from .hashing import canonical_hash
from .types import ProgressSignal, TrajectoryRecord


def _coerce_bool(value, default=None):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "ok", "success", "succeeded"):
            return True
        if v in ("false", "0", "no", "n", "fail", "failed", "error"):
            return False
    return default


def normalise_row(raw: dict, default_run_id: str = "", fallback_seq: int = 0) -> TrajectoryRecord:
    """Convert a raw JSONL row into a :class:`TrajectoryRecord`.

    Accepts both the canonical shape and the friendly shape described in
    the module docstring. Unknown fields are preserved in ``extra``.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"trajectory row must be a JSON object, got {type(raw).__name__}")

    tool = raw.get("tool", "")
    if not tool:
        raise ValueError("trajectory row missing 'tool'")

    raw_args = raw.get("args")
    args_hash = raw.get("args_hash")
    if args_hash is None:
        if raw_args is None:
            args_hash = ""
        else:
            args_hash = canonical_hash(raw_args)
    args_repr = raw.get("args_repr")
    if args_repr is None:
        if raw_args is None:
            args_repr = ""
        else:
            try:
                args_repr = json.dumps(raw_args, default=str, ensure_ascii=False)
            except Exception:
                args_repr = repr(raw_args)

    result = raw.get("result")
    result_repr = raw.get("result_repr")
    if result_repr is None:
        if result is None:
            result_repr = ""
        else:
            if isinstance(result, str):
                result_repr = result
            else:
                try:
                    result_repr = json.dumps(result, default=str, ensure_ascii=False)
                except Exception:
                    result_repr = repr(result)

    success = raw.get("success", None)
    if success is None and "error" in raw:
        # An "error" field implies failure unless caller says otherwise.
        success = False if raw.get("error") else None
    success = _coerce_bool(success, default=None)

    is_progress = bool(raw.get("is_progress", False))

    sequence = raw.get("sequence", fallback_seq)
    if not isinstance(sequence, int):
        try:
            sequence = int(sequence)
        except Exception:
            sequence = fallback_seq

    run_id = raw.get("run_id") or default_run_id

    known = {
        "sequence",
        "run_id",
        "tool",
        "args",
        "args_hash",
        "args_repr",
        "result",
        "result_repr",
        "success",
        "is_progress",
        "extra",
        "error",
    }
    extra = dict(raw.get("extra") or {})
    for k, v in raw.items():
        if k not in known and k not in extra:
            extra[k] = v

    return TrajectoryRecord(
        sequence=sequence,
        run_id=run_id,
        tool=tool,
        args_hash=args_hash,
        args_repr=args_repr,
        result_repr=result_repr,
        success=success,
        is_progress=is_progress,
        extra=extra,
    )


def load_jsonl(path: Union[str, Path], default_run_id: str = "") -> List[TrajectoryRecord]:
    """Load a JSONL file and return its records (in file order)."""
    records: List[TrajectoryRecord] = []
    seq = 0
    with open(str(path), "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
            seq += 1
            records.append(normalise_row(raw, default_run_id=default_run_id, fallback_seq=seq))
    return records


def iter_jsonl(path: Union[str, Path], default_run_id: str = "") -> Iterator[TrajectoryRecord]:
    """Streaming variant of :func:`load_jsonl`."""
    seq = 0
    with open(str(path), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            seq += 1
            yield normalise_row(raw, default_run_id=default_run_id, fallback_seq=seq)


def replay_against_guard(records: Iterable[TrajectoryRecord], guard) -> List[TrajectoryRecord]:
    """Replay ``records`` through ``guard`` (an :class:`AgentFuse` instance).

    Returns the subset of records up to (and including) the one that
    triggered a :class:`DeadlockDetected`, or all records if no deadlock
    was found. Detection exceptions are swallowed at the caller level.
    """
    from .exceptions import DeadlockDetected
    from .types import Action, ProgressSignal

    out: List[TrajectoryRecord] = []
    for rec in records:
        if rec.is_progress:
            guard.mark_progress(ProgressSignal(token=rec.args_repr or ""))
            out.append(rec)
            continue
        action = Action(
            tool=rec.tool,
            args=rec.args_repr,  # already JSON-encoded
            result=rec.result_repr,
            success=rec.success,
        )
        try:
            guard.observe(action)
        except DeadlockDetected:
            out.append(rec)
            return out
        out.append(rec)
    return out