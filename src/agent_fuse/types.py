"""Public type definitions for agent-fuse."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class DetectionKind(str, enum.Enum):
    """Why a deadlock was detected."""

    DIRECT_REPEAT = "direct_repeat"
    """The same (tool, args) appeared N times in a row."""

    N_CYCLE = "n_cycle"
    """A repeating cycle of N distinct actions was found."""

    SEMANTIC_STAGNATION = "semantic_stagnation"
    """Repeated calls to the same tool returned near-identical (failed) tokens."""


@dataclass
class Action:
    """A single tool-call observation.

    Attributes
    ----------
    tool:
        Name of the tool invoked.
    args:
        Tool arguments. Must be JSON-serialisable.
    result:
        Optional textual or structured result. Used for semantic stagnation
        checks. ``None`` means the result is not yet known (record only).
    success:
        Whether the tool call was successful. ``None`` (the default) is treated
        as "not a failure" by the stagnation detector; failures must be marked
        explicitly with ``success=False`` to count.
    timestamp:
        Optional monotonic index or wall-clock timestamp. Currently unused
        by detection logic, preserved for downstream analytics.
    metadata:
        Arbitrary caller-supplied annotations (cost, latency, model used, …).
    """

    tool: str
    args: Any
    result: Optional[Any] = None
    success: Optional[bool] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    """A deadlock detection event."""

    kind: DetectionKind
    message: str
    cycle: List[str] = field(default_factory=list)
    similarity: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressSignal:
    """Caller-supplied progress markers.

    Passing a distinct ``token`` (e.g. a goal-achievement summary, a new
    environment observation, or just an incremented iteration counter)
    tells :class:`AgentFuse` to treat the recent history as making progress.
    The signal is recorded in the trajectory; stagnation analysis ignores
    it when measuring self-similarity.
    """

    token: str
    note: str = ""


#: A steering hook receives the trajectory snapshot and a ``Detection``
#: describing the deadlock. It may return a string that will be attached to
#: the :class:`DeadlockDetected` exception (e.g. a "please reconsider" hint
#: to surface back to the model) or ``None`` to opt out.
SteeringHook = Callable[[List[Action], Detection], Optional[str]]


@dataclass
class TrajectoryRecord:
    """One persisted row in :class:`TrajectoryStore`."""

    sequence: int
    run_id: str
    tool: str
    args_hash: str
    args_repr: str
    result_repr: str
    success: Optional[bool]
    is_progress: bool
    extra: Dict[str, Any] = field(default_factory=dict)