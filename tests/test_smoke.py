"""Smoke tests: package imports & module surface."""

import agent_fuse
from agent_fuse import (
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
    assert hasattr(agent_fuse, "__version__")
    assert isinstance(agent_fuse.__version__, str)


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
        assert hasattr(agent_fuse, name)


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
    import agent_fuse.export
    import agent_fuse.guard
    import agent_fuse.hashing
    import agent_fuse.loader
    import agent_fuse.stats
    import agent_fuse.store

    assert hasattr(agent_fuse.export, "render_html")
    assert hasattr(agent_fuse.guard, "AgentFuse")
    assert hasattr(agent_fuse.hashing, "canonical_hash")
    assert hasattr(agent_fuse.loader, "load_jsonl")
    assert hasattr(agent_fuse.stats, "FuseStats")
    assert hasattr(agent_fuse.store, "TrajectoryStore")