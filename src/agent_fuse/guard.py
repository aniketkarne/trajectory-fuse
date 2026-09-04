"""Core runtime guard for agent tool-call loops.

Usage sketch::

    fuse = AgentFuse(FuseConfig(window=20))
    while not done:
        action = agent.next_action(state)
        try:
            result = tools[action.tool](action.args)
        except Exception as exc:
            result = {"error": str(exc)}
            action.success = False
        action.result = result
        fuse.observe(action)            # raises DeadlockDetected on a loop

The guard is intentionally:

* **synchronous** — agent loops call :meth:`observe` after each action;
* **stateless across instances** — each :class:`AgentFuse` owns its window;
* **side-effect free unless persistence is configured** — see
  :class:`FuseConfig.store_factory`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence

from .exceptions import DeadlockDetected
from .hashing import canonical_hash
from .store import TrajectoryStore
from .types import (
    Action,
    Detection,
    DetectionKind,
    ProgressSignal,
    SteeringHook,
    TrajectoryRecord,
)


# ---------------------------------------------------------------------------
# configuration


@dataclass
class CycleConfig:
    """Cycle-detection thresholds.

    Attributes
    ----------
    direct_repeat_threshold:
        How many *identical* (tool, args-hash) actions in a row constitute a
        direct repeat. Must be ``>= 2``. ``1`` disables direct-repeat
        detection.
    cycle_min_length:
        Smallest cycle size to look for. Must be ``>= 2``.
    cycle_max_length:
        Largest cycle size to look for. Must be ``>= cycle_min_length``.
        Set to ``cycle_min_length`` if you only care about pairs.
    """

    direct_repeat_threshold: int = 3
    cycle_min_length: int = 2
    cycle_max_length: int = 6


@dataclass
class StagnationConfig:
    """Semantic-stagnation thresholds.

    Attributes
    ----------
    enabled:
        Toggle for the whole subsystem.
    window:
        Number of trailing actions to inspect.
    similarity_threshold:
        Jaccard similarity in ``[0, 1]`` above which two consecutive failed
        results count as "near identical". ``1.0`` means *exact* token-set
        equality.
    min_failures:
        Minimum consecutive failures on the same tool before stagnation
        can fire. Avoids false positives on single hiccups.
    """

    enabled: bool = True
    window: int = 8
    similarity_threshold: float = 0.9
    min_failures: int = 3


@dataclass
class FuseConfig:
    """Top-level configuration for :class:`AgentFuse`.

    Attributes
    ----------
    window:
        Maximum number of recent actions to retain in the sliding window.
        Older actions are evicted FIFO.
    cycle:
        See :class:`CycleConfig`.
    stagnation:
        See :class:`StagnationConfig`.
    store_factory:
        Optional callable returning a :class:`TrajectoryStore`. If supplied,
        every observation is appended (and the store is closed automatically
        when the guard is garbage collected). Pass ``None`` to disable
        persistence.
    run_id:
        Identifier attached to every persisted row. Falls back to a UUID4.
    """

    window: int = 32
    cycle: CycleConfig = field(default_factory=CycleConfig)
    stagnation: StagnationConfig = field(default_factory=StagnationConfig)
    store_factory: Optional[Callable[[], TrajectoryStore]] = None
    run_id: Optional[str] = None
    allowlist: Sequence[str] = field(default_factory=tuple)
    """Tools that *never* trigger deadlock detection (e.g. an LLM ``think``).

    Tools whose name is in the allowlist are still recorded but skipped by
    every detector. The allowlist is matched by exact tool name (case
    sensitive)."""


# ---------------------------------------------------------------------------
# helpers


_TOKEN_SPLIT_RE = __import__("re").compile(r"[A-Za-z0-9_]+")


def _tokenise(text: str) -> set:
    if not text:
        return set()
    return {t.lower() for t in _TOKEN_SPLIT_RE.findall(text)}


def _result_repr(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        import json as _json

        return _json.dumps(result, default=str, ensure_ascii=False)
    except Exception:
        return repr(result)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# main class


class AgentFuse:
    """Sliding-window deadlock / stagnation detector."""

    def __init__(
        self,
        config: Optional[FuseConfig] = None,
        steering_hook: Optional[SteeringHook] = None,
    ) -> None:
        self.config: FuseConfig = config or FuseConfig()
        self.steering_hook = steering_hook
        self._actions: List[Action] = []
        self._progress_marks: List[int] = []
        self._sequence = 0
        self._store: Optional[TrajectoryStore] = None
        self._store_owned = False
        if self.config.store_factory is not None:
            self._store = self.config.store_factory()
            # We own the store by default. If the caller already had a store
            # they want to share with other code, they should manage
            # ``store_factory`` themselves and not rely on the guard's
            # ``close()``. The most common pattern is one store per fuse,
            # which is what we treat as "owned".
            self._store_owned = True
            self._store.initialise(self.config.run_id)
        self.run_id: str = (
            self.config.run_id if self.config.run_id else (self._store.run_id if self._store else "")
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying store if we opened it."""
        if self._store_owned and self._store is not None:
            try:
                self._store.close()
            finally:
                self._store = None
                self._store_owned = False

    def __enter__(self) -> "AgentFuse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    # -- public API --------------------------------------------------------

    @property
    def history(self) -> List[Action]:
        """Read-only copy of the current sliding window."""
        return list(self._actions)

    def reset(self) -> None:
        """Clear the sliding window and progress marks.

        Useful when the caller has observed a phase change (e.g. the agent
        has switched to a different sub-task) and wants to restart detection
        cleanly without dropping persistence.
        """
        self._actions.clear()
        self._progress_marks.clear()
        # Sequence counter is *not* reset: persistence stays linear.

    def mark_progress(self, signal: ProgressSignal) -> None:
        """Record an external progress marker.

        This prevents semantic-stagnation detection from including actions
        recorded before the mark in its window. Direct-repeat and cycle
        detection still consider the full window.
        """
        # Position is in *unfiltered* history terms. A new real mark
        # supersedes any synthesised anchor.
        self._progress_marks = [len(self._actions)]
        if self._store is not None:
            self._sequence += 1
            self._store.append(
                TrajectoryRecord(
                    sequence=self._sequence,
                    run_id=self.run_id,
                    tool="<progress>",
                    args_hash="",
                    args_repr=signal.token[:200],
                    result_repr=signal.note[:200],
                    success=None,
                    is_progress=True,
                    extra={"signal_token": signal.token},
                )
            )

    def observe(self, action: Action) -> None:
        """Record an action and possibly raise :class:`DeadlockDetected`.

        Detection order:

        1. direct repeat (cheap);
        2. N-state cycle (small-to-medium);
        3. semantic stagnation.

        Direct repeat and cycle detection are short-circuited for allowlisted
        tools. Stagnation is purely failure-driven and naturally only fires
        on failing actions, so allowlisted tools cannot trigger it.
        """
        self._actions.append(action)
        # Trim window.
        if len(self._actions) > self.config.window:
            had_marks_before_trim = bool(self._progress_marks)
            drop = len(self._actions) - self.config.window
            self._actions = self._actions[drop:]
            # Adjust progress marks accordingly. Synthesised anchors are
            # sticky — they do not shift with subsequent trims — so the
            # buffered actions before the synth cannot leak back into the
            # post-mark region as new failures are observed.
            new_marks: List[int] = []
            for m in self._progress_marks:
                if m < 0:
                    continue  # sentinel (synthesised anchor)
                shifted = m - drop
                if shifted >= 0:
                    new_marks.append(shifted)
            self._progress_marks = new_marks
            # When a mark slides out of the window entirely, record a
            # "synthesised" anchor that the stagnation detector treats as
            # "no post-mark actions yet". Use a negative sentinel so it is
            # distinguishable from real marks and survives future trims.
            if had_marks_before_trim and not self._progress_marks:
                self._progress_marks = [-1]

        # Persist.
        if self._store is not None:
            self._sequence += 1
            self._store.append(
                TrajectoryRecord(
                    sequence=self._sequence,
                    run_id=self.run_id,
                    tool=action.tool,
                    args_hash=canonical_hash(action.args),
                    args_repr=_result_repr(action.args)[:1000],
                    result_repr=_result_repr(action.result)[:1000],
                    success=action.success,
                    is_progress=False,
                    extra={k: action.metadata.get(k) for k in list(action.metadata.keys())[:20]},
                )
            )

        if action.tool in self.config.allowlist:
            return

        detection = (
            self._check_direct_repeat()
            or self._check_cycle()
            or self._check_stagnation()
        )
        if detection is None:
            return

        hint: Optional[str] = None
        if self.steering_hook is not None:
            try:
                hint = self.steering_hook(list(self._actions), detection)
            except Exception:
                # Hooks must not break the loop — swallow + log-style flag.
                hint = None

        msg = detection.message
        if hint is not None:
            msg = f"{msg} [hint: {hint}]"

        raise DeadlockDetected(
            msg,
            detection=detection,
            trajectory=list(self._actions),
            steering_hint=hint,
        )

    # -- persistence helpers ----------------------------------------------

    @property
    def store(self) -> Optional[TrajectoryStore]:
        """The underlying store, or ``None`` if persistence is disabled."""
        return self._store

    def load_from(self, source: Iterable[TrajectoryRecord]) -> None:
        """Bulk-load records from another store (e.g. an analysis run).

        Records are appended in order. Useful for replay / analysis tooling.
        """
        for record in source:
            self._sequence = max(self._sequence, record.sequence)
            action = Action(
                tool=record.tool,
                args=record.args_repr,
                result=record.result_repr,
                success=record.success,
            )
            if record.is_progress:
                self._progress_marks.append(len(self._actions))
            self._actions.append(action)
            if len(self._actions) > self.config.window:
                drop = len(self._actions) - self.config.window
                self._actions = self._actions[drop:]

    # -- detection ---------------------------------------------------------

    def _check_direct_repeat(self) -> Optional[Detection]:
        cfg = self.config.cycle
        if cfg.direct_repeat_threshold < 2:
            return None
        if len(self._actions) < cfg.direct_repeat_threshold:
            return None

        last = self._actions[-1]
        last_hash = canonical_hash(last.args)
        count = 1
        for prev in reversed(self._actions[:-1]):
            if prev.tool != last.tool:
                break
            if canonical_hash(prev.args) != last_hash:
                break
            count += 1
            if count >= cfg.direct_repeat_threshold:
                break
        if count >= cfg.direct_repeat_threshold:
            return Detection(
                kind=DetectionKind.DIRECT_REPEAT,
                message=(
                    f"Tool {last.tool!r} invoked with identical arguments "
                    f"{count} times in a row."
                ),
                cycle=[last.tool] * count,
                details={"count": count, "args_hash": last_hash},
            )
        return None

    def _check_cycle(self) -> Optional[Detection]:
        cfg = self.config.cycle
        # A cycle of length 1 is just a direct repeat; leave that to the
        # direct-repeat detector.
        if cfg.cycle_min_length < 2:
            return None
        if cfg.cycle_max_length < cfg.cycle_min_length:
            return None
        if len(self._actions) < cfg.cycle_min_length * 2:
            return None

        # Sequence must contain at least two full copies of the cycle to
        # declare detection. Build a "signature" per action: (tool, hash).
        sigs = [(a.tool, canonical_hash(a.args)) for a in self._actions]
        max_len = min(cfg.cycle_max_length, len(sigs) // 2)
        for length in range(cfg.cycle_min_length, max_len + 1):
            cycle = sigs[-length:]
            # Check that the previous ``length`` items match the cycle.
            prev = sigs[-(2 * length):-length]
            if len(prev) < length:
                continue
            if prev == cycle:
                tools = [t for t, _ in cycle]
                return Detection(
                    kind=DetectionKind.N_CYCLE,
                    message=(
                        f"Detected repeating cycle of length {length}: "
                        f"{tools}."
                    ),
                    cycle=tools,
                    details={"length": length},
                )
        return None

    def _check_stagnation(self) -> Optional[Detection]:
        cfg = self.config.stagnation
        if not cfg.enabled:
            return None
        if cfg.window < 2 or cfg.min_failures < 2:
            return None

        # Find the most recent progress mark; only consider actions after it.
        start = self._progress_marks[-1] if self._progress_marks else 0
        relevant = self._actions[start:]
        if len(relevant) < cfg.min_failures:
            return None

        tail = relevant[-cfg.window:]
        if len(tail) < cfg.min_failures:
            return None

        last = tail[-1]
        if last.success is not False:
            return None
        if last.tool in self.config.allowlist:
            return None

        # Walk back collecting consecutive failed same-tool results.
        tokens_by_idx: List[set] = []
        for a in tail:
            if a.tool != last.tool:
                tokens_by_idx = []
                continue
            if a.success is not False:
                tokens_by_idx = []
                continue
            tokens_by_idx.append(_tokenise(_result_repr(a.result)))
        if len(tokens_by_idx) < cfg.min_failures:
            return None

        # Check that the trailing *min_failures* failures are all
        # high-similarity to the previous failure.
        recent = tokens_by_idx[-cfg.min_failures:]
        # Jaccard between every consecutive pair.
        sims: List[float] = []
        for i in range(1, len(recent)):
            sims.append(_jaccard(recent[i - 1], recent[i]))
        if not sims:
            return None
        if min(sims) < cfg.similarity_threshold:
            return None
        # Also check the average similarity against the earliest.
        avg_sim = sum(sims) / len(sims)

        return Detection(
            kind=DetectionKind.SEMANTIC_STAGNATION,
            message=(
                f"Tool {last.tool!r} produced {len(tokens_by_idx)} "
                f"consecutive semantically-near-identical failures "
                f"(avg Jaccard {avg_sim:.2f})."
            ),
            cycle=[last.tool] * len(tokens_by_idx),
            similarity=avg_sim,
            details={
                "window": cfg.window,
                "min_failures": cfg.min_failures,
                "similarity_threshold": cfg.similarity_threshold,
                "consecutive_failures": len(tokens_by_idx),
            },
        )