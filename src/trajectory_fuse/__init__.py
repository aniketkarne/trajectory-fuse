"""trajectory-fuse: in-process runtime guard for agent tool-call loops.

Detects deadlocks, semantic stagnation, and repeating cycles in agent
tool-call trajectories. Zero external dependencies; Python 3.9+.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("trajectory-fuse")
except PackageNotFoundError:  # pragma: no cover - source checkout / not installed
    # The literal here is a fallback for editable installs and source
    # checkouts where importlib.metadata cannot find the distribution.
    # PyPI-published builds always use the value from pyproject.toml via
    # importlib.metadata above. Bump both together when releasing.
    __version__ = "0.3.0+unknown"

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
from .hooks import (
    Chain,
    RecoveryHook,
    RecoveryPolicy,
    on_deadlock,
    on_kind,
    on_loop,
    on_stagnation,
)
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
    "Chain",
    "CycleConfig",
    "DeadlockDetected",
    "Detection",
    "DetectionKind",
    "FuseConfig",
    "FuseStats",
    "ProgressSignal",
    "RecoveryHook",
    "RecoveryPolicy",
    "RunBudget",
    "SteeringHook",
    "StagnationConfig",
    "TrajectoryRecord",
    "TrajectoryStore",
    "canonical_hash",
    "fuse_namespace",
    "on_deadlock",
    "on_kind",
    "on_loop",
    "on_stagnation",
]
