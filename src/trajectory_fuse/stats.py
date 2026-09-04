"""Run statistics for :class:`trajectory_fuse.AgentFuse`.

The stats dataclass is intentionally cheap to read and cheap to update —
every :meth:`AgentFuse.observe` increments a counter or two, and reading
``fuse.stats`` returns the live snapshot. The counts are *not* persisted
to SQLite; they exist only in-process and reset when the fuse is reset
or garbage collected.

Definitions
-----------

* ``calls_observed`` — every action passed to :meth:`observe`, including
  allowlisted ones and the actions that triggered a deadlock.

* ``calls_allowed`` — actions that did **not** trigger a detector. Same
  set as ``calls_observed`` minus ``calls_blocked``.

* ``calls_blocked`` — actions that raised :class:`DeadlockDetected`.
  A blocked call is the *last* call of a (eventually) detected loop; the
  earlier calls in the loop count as ``calls_allowed``.

* ``deadlocks`` — number of unique deadlock events raised. Equal to the
  number of times the caller caught :class:`DeadlockDetected` (we
  cannot know about uncaught exceptions, but in practice one deadlock
  per ``observe`` failure).

* ``time_avoided_seconds`` — wall-clock time the caller saved by
  aborting the loop. This is heuristic: the caller (or a wrapper)
  records ``fuse.record_time_avoided(seconds)`` after catching
  :class:`DeadlockDetected`. We expose it as an explicit field rather
  than guessing.

* ``progress_marks`` — number of :meth:`mark_progress` calls.

* ``first_deadlock_at`` — the position (0-indexed) in the trajectory
  where the first deadlock fired, or ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FuseStats:
    """Lightweight, in-process counters.

    The fields are public and mutable, but the recommended way to read
    them is ``fuse.stats`` which returns the live instance.
    """

    calls_observed: int = 0
    calls_allowed: int = 0
    calls_blocked: int = 0
    deadlocks: int = 0
    progress_marks: int = 0
    time_avoided_seconds: float = 0.0
    first_deadlock_at: Optional[int] = None

    # Internal: not part of the documented contract; used by AgentFuse.
    _last_observed_index: int = field(default=-1, repr=False, compare=False)

    def as_dict(self) -> dict:
        """Return a plain dict suitable for JSON / logging."""
        return {
            "calls_observed": self.calls_observed,
            "calls_allowed": self.calls_allowed,
            "calls_blocked": self.calls_blocked,
            "deadlocks": self.deadlocks,
            "progress_marks": self.progress_marks,
            "time_avoided_seconds": self.time_avoided_seconds,
            "first_deadlock_at": self.first_deadlock_at,
        }

    def reset(self) -> None:
        """Zero out all counters (used by :meth:`AgentFuse.reset`)."""
        self.calls_observed = 0
        self.calls_allowed = 0
        self.calls_blocked = 0
        self.deadlocks = 0
        self.progress_marks = 0
        self.time_avoided_seconds = 0.0
        self.first_deadlock_at = None
        self._last_observed_index = -1
