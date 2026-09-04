"""trajectory-fuse: in-process runtime guard for agent tool-call loops.

Detects deadlocks, semantic stagnation, and repeating cycles in agent
tool-call trajectories. Zero external dependencies; Python 3.9+.
"""

from .exceptions import DeadlockDetected
from .guard import (
    AgentFuse,
    Action,
    BudgetExceeded,
    CycleConfig,
    FuseConfig,
    RunBudget,
    StagnationConfig,
    fuse_namespace,
)
from .hashing import canonical_hash
from .stats import FuseStats
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
    "BudgetExceeded",
    "CycleConfig",
    "DeadlockDetected",
    "Detection",
    "DetectionKind",
    "FuseConfig",
    "FuseStats",
    "ProgressSignal",
    "RunBudget",
    "SteeringHook",
    "StagnationConfig",
    "TrajectoryRecord",
    "TrajectoryStore",
    "canonical_hash",
    "fuse_namespace",
]

__version__ = "0.2.1"
