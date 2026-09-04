"""agent-fuse: in-process runtime guard for agent tool-call loops.

Detects deadlocks, semantic stagnation, and repeating cycles in agent
tool-call trajectories. Zero external dependencies; Python 3.9+.
"""

from .exceptions import DeadlockDetected
from .guard import AgentFuse, Action, CycleConfig, StagnationConfig, FuseConfig
from .hashing import canonical_hash
from .store import TrajectoryStore
from .types import (
    Detection,
    DetectionKind,
    ProgressSignal,
    SteeringHook,
    TrajectoryRecord,
)

__all__ = [
    "Action",
    "AgentFuse",
    "CycleConfig",
    "DeadlockDetected",
    "Detection",
    "DetectionKind",
    "FuseConfig",
    "ProgressSignal",
    "SteeringHook",
    "StagnationConfig",
    "TrajectoryRecord",
    "TrajectoryStore",
    "canonical_hash",
]

__version__ = "0.1.0"