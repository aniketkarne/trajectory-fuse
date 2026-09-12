"""Recovery hooks for :class:`AgentFuse`.

The fuse raises :class:`DeadlockDetected` (and :class:`BudgetExceeded`)
when something has gone wrong. Sometimes the host application wants to
*observe* that event and decide for itself what to do — log it, send a
"try a different approach" hint back to the model, switch to a fallback
tool, page a human, etc.

The hook API is intentionally tiny:

* :class:`RecoveryHook` — a callable that receives the detection and
  decides whether the fuse should still raise.
* :func:`on_loop` / :func:`on_stagnation` / :func:`on_deadlock` —
  shorthand constructors for the three common shapes.
* :class:`RecoveryPolicy` — bundles multiple hooks with a default
  disposition (``raise`` vs ``continue``).
* :class:`Chain` — runs hooks in order and short-circuits on the first
  one that returns ``"raise"`` / ``"continue"``.

The hook returns one of:

* ``"raise"``     — let the fuse raise the original exception.
* ``"continue"``  — swallow the exception, record a `recovery_skip`
  in stats, return normally.

Returning ``None`` is treated as ``"raise"`` for backward compatibility
with hooks that don't care about the new disposition API.

Example::

    from trajectory_fuse import AgentFuse, FuseConfig
    from trajectory_fuse.hooks import RecoveryPolicy, on_stagnation

    def hint_to_model(action, detection):
        # Pretend we talk to an LLM and ask for a recovery plan.
        plan = ask_llm_for_recovery(detection.message)
        if "switch tool" in plan:
            return "continue"   # let the loop keep going
        return "raise"

    policy = RecoveryPolicy(
        on_stagnation=hook,
        default_disposition="continue",   # be lenient by default
    )
    fuse = AgentFuse(FuseConfig(window=20), recovery_policy=policy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .types import Action, Detection

#: The disposition a hook may return.
#: ``"raise"`` propagates the original :class:`DeadlockDetected`.
#: ``"continue"`` swallows it and lets the loop proceed (recording a
#: ``recovery_skips`` tick in :class:`FuseStats`).
Disposition = str  # "raise" | "continue"


#: A recovery hook receives the action that triggered detection and the
#: :class:`Detection` describing why. Returning ``"raise"`` lets the fuse
#: raise; returning ``"continue"`` swallows the exception.
RecoveryHook = Callable[[Action, Detection], Optional[Disposition]]


# ---------------------------------------------------------------------
# shorthand constructors


def on_loop(fn: Callable[[Action, Detection], Optional[Disposition]]) -> RecoveryHook:
    """Identity wrapper that documents intent.

    Equivalent to passing ``fn`` directly. Use this to make call sites
    self-documenting when multiple hook kinds are present::

        policy = RecoveryPolicy(
            on_loop=on_loop(my_direct_repeat_handler),
            on_stagnation=on_stagnation(my_stagnation_handler),
        )
    """
    return fn


def on_stagnation(fn: Callable[[Action, Detection], Optional[Disposition]]) -> RecoveryHook:
    """Identity wrapper — semantic marker for stagnation-specific hooks."""
    return fn


def on_deadlock(fn: Callable[[Action, Detection], Optional[Disposition]]) -> RecoveryHook:
    """Identity wrapper — semantic marker for *any* deadlock event.

    Distinct from :func:`on_loop` because a deadlock could be a direct
    repeat, an N-cycle, *or* stagnation. Use this when you want one
    hook to handle every kind.
    """
    return fn


def on_kind(kind: str, fn: Callable[[Action, Detection], Optional[Disposition]]) -> RecoveryHook:
    """Factory: invoke ``fn`` only for detections with ``detection.kind.value == kind``.

    For example, :func:`on_kind` ``"direct_repeat"`` lets a hook fire
    exclusively on direct-repeat events::

        policy = RecoveryPolicy(
            on_direct_repeat=on_kind("direct_repeat", my_handler),
        )
    """
    def _filtered(action: Action, detection: Detection) -> Optional[Disposition]:
        if detection.kind.value == kind:
            return fn(action, detection)
        # Pass-through: don't change the disposition.
        return None
    _filtered.__name__ = f"on_kind_{kind}_{getattr(fn, '__name__', 'hook')}"
    return _filtered


# ---------------------------------------------------------------------
# policy


@dataclass
class RecoveryPolicy:
    """Bundle of recovery hooks + a default disposition.

    Attributes
    ----------
    on_direct_repeat:
        Hook fired when :attr:`DetectionKind.DIRECT_REPEAT` fires.
    on_n_cycle:
        Hook fired when :attr:`DetectionKind.N_CYCLE` fires.
    on_stagnation:
        Hook fired when :attr:`DetectionKind.SEMANTIC_STAGNATION` fires.
    on_deadlock:
        Catch-all hook fired for *any* detection (after the kind-specific
        hook above). Useful for cross-cutting telemetry.
    default_disposition:
        ``"raise"`` (default) or ``"continue"``. The disposition used
        when a hook returns ``None`` (i.e. "I have no opinion").
    """

    on_direct_repeat: Optional[RecoveryHook] = None
    on_n_cycle: Optional[RecoveryHook] = None
    on_stagnation: Optional[RecoveryHook] = None
    on_deadlock: Optional[RecoveryHook] = None
    default_disposition: Disposition = "raise"

    def resolve(self, action: Action, detection: Detection) -> Disposition:
        """Return the disposition for ``(action, detection)``.

        Runs the kind-specific hook first, then the catch-all. The first
        non-``None`` return wins. Falls back to :attr:`default_disposition`.
        """
        hook = self._hook_for(detection)
        for fn in (hook, self.on_deadlock):
            if fn is None:
                continue
            try:
                result = fn(action, detection)
            except Exception:
                # A misbehaving hook must not break the loop — same
                # policy as :meth:`AgentFuse`'s progress_callback handling.
                continue
            if result is not None:
                return result
        return self.default_disposition

    def _hook_for(self, detection: Detection) -> Optional[RecoveryHook]:
        from .types import DetectionKind

        if detection.kind == DetectionKind.DIRECT_REPEAT:
            return self.on_direct_repeat
        if detection.kind == DetectionKind.N_CYCLE:
            return self.on_n_cycle
        if detection.kind == DetectionKind.SEMANTIC_STAGNATION:
            return self.on_stagnation
        return None


# ---------------------------------------------------------------------
# chain helper


@dataclass
class Chain(RecoveryPolicy):
    """Run a list of hooks in order; first non-None return wins.

    Use this to compose multiple recovery strategies::

        policy = Chain([
            on_stagnation(expensive_llm_recovery_hook),
            on_deadlock(cheap_telemetry_hook),
        ])
        fuse = AgentFuse(FuseConfig(window=20), recovery_policy=policy)

    :class:`Chain` is a :class:`RecoveryPolicy` subclass — pass it to
    :class:`AgentFuse` exactly where you'd pass a policy.
    """

    hooks: List[RecoveryHook] = field(default_factory=list)
    default_disposition: Disposition = "raise"

    def resolve(self, action: Action, detection: Detection) -> Disposition:
        for fn in self.hooks:
            try:
                result = fn(action, detection)
            except Exception:
                continue
            if result is not None:
                return result
        return self.default_disposition


__all__ = [
    "Chain",
    "RecoveryHook",
    "RecoveryPolicy",
    "on_deadlock",
    "on_kind",
    "on_loop",
    "on_stagnation",
]
