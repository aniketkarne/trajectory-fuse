"""Exceptions raised by agent-fuse."""

from __future__ import annotations

from typing import List, Optional

from .types import Detection, DetectionKind


class DeadlockDetected(RuntimeError):
    """Raised when :class:`AgentFuse` detects a deadlock.

    Attributes
    ----------
    detection:
        The structured :class:`Detection` describing why the guard fired.
    trajectory:
        Snapshot of the actions that triggered detection (most recent last).
    steering_hint:
        Optional string returned by a steering hook.
    """

    def __init__(
        self,
        message: str,
        detection: Detection,
        trajectory: Optional[List] = None,
        steering_hint: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.detection = detection
        self.trajectory: List = list(trajectory or [])
        self.steering_hint = steering_hint

    @property
    def kind(self) -> DetectionKind:
        """Shortcut for ``self.detection.kind``."""
        return self.detection.kind

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"DeadlockDetected(kind={self.detection.kind.name!r}, "
            f"message={str(self)!r}, hint={self.steering_hint!r})"
        )