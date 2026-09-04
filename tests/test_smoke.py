"""Smoke tests: package imports & module surface."""

import trajectory_fuse
from trajectory_fuse import (
    Action,
    AgentFuse,
    BudgetExceeded,
    CycleConfig,
    DeadlockDetected,
    Detection,
    DetectionKind,
    FuseConfig,
    FuseStats,
    ProgressSignal,
    RunBudget,
    StagnationConfig,
    SteeringHook,
    TrajectoryRecord,
    TrajectoryStore,
    canonical_hash,
    fuse_namespace,
)


def test_version_present():
    assert hasattr(trajectory_fuse, "__version__")
    assert isinstance(trajectory_fuse.__version__, str)


def test_all_symbols_importable():
    for name in (
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
        "StagnationConfig",
        "SteeringHook",
        "TrajectoryRecord",
        "TrajectoryStore",
        "canonical_hash",
        "fuse_namespace",
    ):
        assert hasattr(trajectory_fuse, name)


def test_detection_kind_values():
    assert DetectionKind.DIRECT_REPEAT.value == "direct_repeat"
    assert DetectionKind.N_CYCLE.value == "n_cycle"
    assert DetectionKind.SEMANTIC_STAGNATION.value == "semantic_stagnation"


def test_deadlock_detected_attach_attrs():
    d = Detection(kind=DetectionKind.DIRECT_REPEAT, message="x", cycle=["t"])
    exc = DeadlockDetected("x", detection=d, trajectory=[], steering_hint=None)
    assert exc.kind == DetectionKind.DIRECT_REPEAT
    assert exc.detection is d
    assert exc.steering_hint is None


def test_module_submodules_present():
    import trajectory_fuse.export
    import trajectory_fuse.guard
    import trajectory_fuse.hashing
    import trajectory_fuse.loader
    import trajectory_fuse.stats
    import trajectory_fuse.store

    assert hasattr(trajectory_fuse.export, "render_html")
    assert hasattr(trajectory_fuse.guard, "AgentFuse")
    assert hasattr(trajectory_fuse.hashing, "canonical_hash")
    assert hasattr(trajectory_fuse.loader, "load_jsonl")
    assert hasattr(trajectory_fuse.stats, "FuseStats")
    assert hasattr(trajectory_fuse.store, "TrajectoryStore")