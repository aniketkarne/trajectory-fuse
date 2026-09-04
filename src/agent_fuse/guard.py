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

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Set

from .exceptions import DeadlockDetected
from .hashing import canonical_hash
from .stats import FuseStats
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

    The default thresholds are **conservative on purpose**. Most
    retry-style loops (HTTP 429 / 5xx / transient timeouts) should NOT
    trip the stagnation detector — it exists to surface *real* semantic
    non-progress, not to punish well-behaved retries. If you find
    yourself lowering these values, first verify that your retry layer
    is properly distinguishing transient errors (``success=False`` with
    meaningful new diagnostic tokens each retry) from genuinely
    repetitive failures (the *same* error string over and over).

    Attributes
    ----------
    enabled:
        Toggle for the whole subsystem.
    window:
        Number of trailing actions to inspect.
    similarity_threshold:
        Jaccard similarity in ``[0, 1]`` above which two consecutive failed
        results count as "near identical". ``1.0`` means *exact* token-set
        equality. The default of ``0.9`` requires substantial token-set
        overlap before stagnation fires.
    min_failures:
        Minimum consecutive failures on the same tool before stagnation
        can fire. Default ``4`` (was ``3``) — needs one more corroborating
        failure before declaring stagnation, which is plenty for true
        non-progressing loops but well above the noise floor of
        ordinary retry/backoff sequences.
    """

    enabled: bool = True
    window: int = 8
    similarity_threshold: float = 0.9
    min_failures: int = 4


@dataclass
class RunBudget:
    """Hard ceilings on a single :class:`AgentFuse` run.

    Budgets are checked at the top of :meth:`observe` and (when tripped)
    raise :class:`BudgetExceeded`. Set any field to ``0`` (or ``None``
    for ``max_runtime_seconds``) to disable that specific budget.

    Attributes
    ----------
    max_actions:
        Maximum number of :meth:`observe` calls accepted before the
        budget trips. Counts allowlisted tools too.
    max_runtime_seconds:
        Wall-clock seconds since fuse construction before the budget
        trips. ``None`` disables. The check uses
        ``time.monotonic()`` so it is unaffected by clock adjustments.
    max_tool_calls:
        Maximum number of *non-allowlisted* tool calls before the budget
        trips. Useful when you want to bound effective work rather than
        raw call volume.
    """

    max_actions: int = 0
    max_runtime_seconds: Optional[float] = None
    max_tool_calls: int = 0


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
    budgets:
        See :class:`RunBudget`.
    store_factory:
        Optional callable returning a :class:`TrajectoryStore`. If supplied,
        every observation is appended (and the store is closed automatically
        when the guard is garbage collected). Pass ``None`` to disable
        persistence.
    run_id:
        Identifier attached to every persisted row. Falls back to a UUID4.
    allowlist:
        Tools that *never* trigger deadlock detection (e.g. an LLM ``think``).
        Tools whose name is in the allowlist are still recorded but skipped by
        every detector. The allowlist is matched by exact tool name (case
        sensitive).
    allow_repeats:
        Tools whose direct repeats are *legitimate* (e.g. polling an HTTP
        endpoint for a status). When ``(tool, args-hash)`` repeats N times
        where ``N >= allow_repeat_overrides[tool]``, the repeat is
        accepted without raising. The default override is ``3``.
        Use :attr:`allow_repeats_default` to set the fallback for tools
        not listed here.
    allow_repeats_default:
        Default minimum tolerance for tools not in :attr:`allow_repeats`.
        ``0`` means *no tolerance* — even a single repeat after the same
        action was just observed can be detected. ``3`` (the default)
        matches the standard direct-repeat threshold so existing
        behaviour is unchanged for unlisted tools.
    progress_callback:
        Optional ``Callable[[Action, Detection], None]`` invoked *instead
        of* raising :class:`DeadlockDetected` for a semantic-stagnation
        event on a tool that the caller has marked as making progress.
        The callback receives the action that would have raised and the
        :class:`Detection`. Returning ``None`` from the callback (or
        letting the callback finish without re-raising) suppresses the
        exception. Use this to inject a "different approach" hint to the
        agent without aborting the loop.
    """

    window: int = 32
    cycle: CycleConfig = field(default_factory=CycleConfig)
    stagnation: StagnationConfig = field(default_factory=StagnationConfig)
    budgets: RunBudget = field(default_factory=RunBudget)
    store_factory: Optional[Callable[[], TrajectoryStore]] = None
    run_id: Optional[str] = None
    allowlist: Sequence[str] = field(default_factory=tuple)
    allow_repeats: dict = field(default_factory=dict)
    """Per-tool "this many direct repeats are fine" overrides.

    A repeat is the *N-th* identical call to the same tool with the same
    args. ``allow_repeats[tool] = N`` means a direct-repeat run of up to
    ``N`` repeats is accepted (does not raise); the ``(N+1)``-th repeat
    trips the fuse. Use this for *legitimate* repetition like HTTP
    polling, paginated API walks, or transactional retries where the
    caller is in charge of the loop bounds.

    Unlisted tools fall back to the standard
    :attr:`CycleConfig.direct_repeat_threshold`.

    Examples::

        FuseConfig(
            cycle=CycleConfig(direct_repeat_threshold=3),
            allow_repeats={"poll_status": 50, "fetch_page": 100},
        )
    """
    progress_callback: Optional[Callable[[Action, Detection], None]] = None


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
# budget exception


class BudgetExceeded(RuntimeError):
    """Raised when a hard :class:`RunBudget` is exceeded.

    Attributes
    ----------
    kind:
        Which budget tripped: ``"actions"``, ``"runtime"``, or
        ``"tool_calls"``.
    limit:
        The configured ceiling (for ``actions`` and ``tool_calls``) or
        the elapsed wall-clock seconds (for ``runtime``).
    observed:
        The actual value at the time of trip (raw count or seconds).
    """

    def __init__(self, kind: str, limit, observed) -> None:
        self.kind = kind
        self.limit = limit
        self.observed = observed
        msg = {
            "actions": f"budget: {observed} actions observed (limit {limit})",
            "tool_calls": f"budget: {observed} tool calls observed (limit {limit})",
            "runtime": f"budget: {observed:.2f}s elapsed (limit {limit:.2f}s)",
        }.get(kind, f"budget: {kind} tripped")
        super().__init__(msg)


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
        self._start_time = time.monotonic()
        self._budget_tripped = False
        # Counter of *non-allowlisted* tool calls observed so far. Used by
        # the ``max_tool_calls`` budget. Named ``_tool_call_count`` because
        # it counts every observed tool call — not just "blocked" ones —
        # which is what ``max_tool_calls`` budgets against.
        self._tool_call_count = 0
        self.stats = FuseStats()
        # Resolve allow_repeats into a concrete (tool -> threshold) map.
        self._allow_repeats: dict = dict(self.config.allow_repeats)
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
        self.stats.reset()

    def mark_progress(self, signal: ProgressSignal) -> None:
        """Record an external progress marker.

        This prevents semantic-stagnation detection from including actions
        recorded before the mark in its window. Direct-repeat and cycle
        detection still consider the full window.
        """
        # Position is in *unfiltered* history terms. A new real mark
        # supersedes any synthesised anchor.
        self._progress_marks = [len(self._actions)]
        self.stats.progress_marks += 1
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

    # -- preflight ---------------------------------------------------------

    def check(self, tool: str, args: Any = None) -> Optional[Detection]:
        """Run all detectors *without* recording the action.

        Use this when you want to ask "would this call trip the fuse?" before
        committing to it. The returned :class:`Detection` has the same shape
        as the one carried by :class:`DeadlockDetected`. Returns ``None``
        when no detector would fire.

        The check uses a synthetic :class:`Action` that does **not** mutate
        the sliding window or the stats counters. It is safe to call from
        a hot loop. The args hash is computed once and reused.
        """
        synth = Action(tool=tool, args=args, result=None, success=None)
        return self._run_detectors(synth, record=False)

    def check_action(self, action: Action) -> Optional[Detection]:
        """Run all detectors on a *fully-built* :class:`Action`.

        Like :meth:`check`, but accepts an existing :class:`Action` so
        callers that already have one (e.g. an :class:`AgentFuse.wrap`
        wrapper) can avoid rebuilding it. Like :meth:`check`, this does
        not record the action.
        """
        return self._run_detectors(action, record=False)

    # -- observe -----------------------------------------------------------

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
        # -- budgets --------------------------------------------------------
        cfg_b = self.config.budgets
        if cfg_b.max_actions and self.stats.calls_observed >= cfg_b.max_actions:
            if not self._budget_tripped:
                self._budget_tripped = True
            raise BudgetExceeded("actions", cfg_b.max_actions, self.stats.calls_observed)
        if cfg_b.max_runtime_seconds is not None:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= cfg_b.max_runtime_seconds:
                if not self._budget_tripped:
                    self._budget_tripped = True
                raise BudgetExceeded("runtime", cfg_b.max_runtime_seconds, elapsed)
        is_tool_call = action.tool not in self.config.allowlist
        if cfg_b.max_tool_calls and is_tool_call and self._tool_call_count >= cfg_b.max_tool_calls:
            if not self._budget_tripped:
                self._budget_tripped = True
            raise BudgetExceeded("tool_calls", cfg_b.max_tool_calls, self._tool_call_count)

        self._actions.append(action)
        self.stats.calls_observed += 1
        if is_tool_call:
            self._tool_call_count += 1

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

        detection = self._run_detectors(action, record=True)

        if detection is None:
            self.stats.calls_allowed += 1
            return

        # The action triggered detection. The default action is to raise;
        # ``allow_repeats`` opts out of raising for legitimate direct
        # repeats; the ``progress_callback`` opts out of raising for any
        # semantic-stagnation event without aborting the loop.
        if action.tool in self.config.allowlist:
            self.stats.calls_allowed += 1
            return

        if detection.kind == DetectionKind.DIRECT_REPEAT and self._is_allowed_repeat(action):
            self.stats.calls_allowed += 1
            return

        if detection.kind == DetectionKind.SEMANTIC_STAGNATION and self.config.progress_callback is not None:
            try:
                self.config.progress_callback(action, detection)
            except DeadlockDetected:
                # Caller can still raise explicitly from inside the callback.
                raise
            except Exception:
                # Hooks must not break the loop — swallow the exception and
                # treat the call as allowed. (BaseException — Ctrl-C etc. —
                # is *not* caught here and propagates as expected.)
                self.stats.calls_allowed += 1
                return
            else:
                self.stats.calls_allowed += 1
                return

        self._record_blocked(detection)
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

    # -- stats helpers ----------------------------------------------------

    def record_time_avoided(self, seconds: float) -> None:
        """Add to the ``time_avoided_seconds`` counter.

        Callers should invoke this after catching :class:`DeadlockDetected`
        (or :class:`BudgetExceeded`) with an estimate of the wall-clock
        seconds the abort saved. The library cannot infer this on its own
        because it does not know how long the model would have continued
        looping.
        """
        if seconds > 0:
            self.stats.time_avoided_seconds += float(seconds)

    def _record_blocked(self, detection: Detection) -> None:
        self.stats.calls_blocked += 1
        self.stats.deadlocks += 1
        if self.stats.first_deadlock_at is None:
            self.stats.first_deadlock_at = len(self._actions) - 1

    def _is_allowed_repeat(self, action: Action) -> bool:
        """Return True if this direct repeat is allowed by allow_repeats.

        Per-tool ``allow_repeats[tool]`` sets the *trip threshold* for
        that tool: identical calls past ``N`` are accepted without
        raising (legitimate polling / retry). Tools not listed use
        :attr:`FuseConfig.cycle.direct_repeat_threshold` directly.
        """
        if action.tool not in self._allow_repeats:
            return False
        per_tool_threshold = self._allow_repeats[action.tool]
        if not per_tool_threshold:
            return False
        last = action
        last_hash = canonical_hash(last.args)
        count = 1
        for prev in reversed(self._actions[:-1]):
            if prev.tool != last.tool:
                break
            if canonical_hash(prev.args) != last_hash:
                break
            count += 1
        return count <= per_tool_threshold

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

    def _run_detectors(self, action: Action, record: bool) -> Optional[Detection]:
        """Run direct-repeat → cycle → stagnation in order.

        ``record=True`` mutates the sliding window (called by
        :meth:`observe`). ``record=False`` runs detectors against the
        current window without appending the synthetic action (called by
        :meth:`check` / :meth:`check_action`).
        """
        # When recording, the action is already appended by observe().
        # When not recording, simulate it for the *purpose of the check*
        # by considering what the window would look like with this action
        # appended. We do this by building a local copy of the trailing
        # window — cheap (≤ ``config.window`` entries).
        if record:
            actions = self._actions
        else:
            actions = self._actions + [action]

        if action.tool in self.config.allowlist:
            return None

        detection = (
            self._check_direct_repeat(actions)
            or self._check_cycle(actions)
            or self._check_stagnation(actions)
        )
        return detection

    def _check_direct_repeat(self, actions: List[Action]) -> Optional[Detection]:
        cfg = self.config.cycle
        if cfg.direct_repeat_threshold < 2:
            return None
        if len(actions) < cfg.direct_repeat_threshold:
            return None

        last = actions[-1]
        last_hash = canonical_hash(last.args)
        count = 1
        for prev in reversed(actions[:-1]):
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

    def _check_cycle(self, actions: List[Action]) -> Optional[Detection]:
        cfg = self.config.cycle
        # A cycle of length 1 is just a direct repeat; leave that to the
        # direct-repeat detector.
        if cfg.cycle_min_length < 2:
            return None
        if cfg.cycle_max_length < cfg.cycle_min_length:
            return None
        if len(actions) < cfg.cycle_min_length * 2:
            return None

        # Sequence must contain at least two full copies of the cycle to
        # declare detection. Build a "signature" per action: (tool, hash).
        sigs = [(a.tool, canonical_hash(a.args)) for a in actions]
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

    def _check_stagnation(self, actions: List[Action]) -> Optional[Detection]:
        cfg = self.config.stagnation
        if not cfg.enabled:
            return None
        if cfg.window < 2 or cfg.min_failures < 2:
            return None

        # Find the most recent progress mark; only consider actions after it.
        # NOTE: ``_progress_marks`` is only mutated when ``record=True``;
        # preflight checks use the *full* action list as the relevant slice.
        if self._progress_marks:
            start = max(self._progress_marks)
        else:
            start = 0
        relevant = actions[start:]
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


# ---------------------------------------------------------------------------
# ergonomic wrapper

import functools as _functools
import inspect as _inspect


def _build_tool_decorator(fuse: "AgentFuse"):
    """Return a decorator that registers ``fn`` as a guarded tool.

    The wrapper performs the standard pre-call + post-call sequence:

    1. :meth:`AgentFuse.check` — if any detector would trip, raise
       :class:`DeadlockDetected` *before* invoking the tool. This is the
       "circuit breaker" guarantee: we never call a tool whose call we
       already know to be part of a deadlock.
    2. Invoke ``fn``.
    3. On success/failure, append an :class:`Action` with the result.
    4. :meth:`AgentFuse.observe` (which also performs the same checks for
       redundancy and writes to the persistence store).

    A common idiom::

        fuse = AgentFuse()
        fuse = fuse.with_tool("search", search)
        fuse = fuse.with_tool("summarise", summarise)
    """

    def _wrap(fn: Callable, name: Optional[str] = None, allow_repeats: Optional[int] = None):
        # If ``name`` is None, fall back to fn.__name__; allow_repeats uses
        # the fuse's default if not supplied.
        effective_name = name or getattr(fn, "__name__", "tool")
        # Register on the fuse instance so preflight sees the per-tool
        # allow_repeats override.
        if allow_repeats is not None:
            fuse.config.allow_repeats[effective_name] = allow_repeats
            fuse._allow_repeats[effective_name] = allow_repeats

        if _inspect.iscoroutinefunction(fn):

            @_functools.wraps(fn)
            async def _async_wrapped(*args, **kwargs):
                # Pre-call blocking check.
                det = fuse.check(effective_name, {"args": args, "kwargs": kwargs})
                if det is not None and not _is_allowed(fuse, effective_name, det):
                    raise DeadlockDetected(
                        det.message, detection=det, trajectory=fuse.history, steering_hint=None
                    )
                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    action = Action(
                        tool=effective_name,
                        args={"args": args, "kwargs": kwargs},
                        result={"error": repr(exc)},
                        success=False,
                    )
                    fuse.observe(action)
                    raise
                action = Action(
                    tool=effective_name,
                    args={"args": args, "kwargs": kwargs},
                    result=result,
                    success=True,
                )
                fuse.observe(action)
                return result

            return _async_wrapped

        @_functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            det = fuse.check(effective_name, {"args": args, "kwargs": kwargs})
            if det is not None and not _is_allowed(fuse, effective_name, det):
                raise DeadlockDetected(
                    det.message, detection=det, trajectory=fuse.history, steering_hint=None
                )
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                action = Action(
                    tool=effective_name,
                    args={"args": args, "kwargs": kwargs},
                    result={"error": repr(exc)},
                    success=False,
                )
                fuse.observe(action)
                raise
            action = Action(
                tool=effective_name,
                args={"args": args, "kwargs": kwargs},
                result=result,
                success=True,
            )
            fuse.observe(action)
            return result

        return _wrapped

    def _tool(fn=None, *, name=None, allow_repeats=None):
        # Support both @fuse.tool and @fuse.tool(...) forms.
        if fn is None:
            return lambda f: _wrap(f, name=name, allow_repeats=allow_repeats)
        return _wrap(fn, name=name, allow_repeats=allow_repeats)

    return _tool


def _is_allowed(fuse: "AgentFuse", tool: str, detection: Detection) -> bool:
    """Return True if the per-call detection should NOT raise."""
    if tool in fuse.config.allowlist:
        return True
    if detection.kind == DetectionKind.DIRECT_REPEAT:
        return tool in fuse._allow_repeats and bool(fuse._allow_repeats[tool])
    if detection.kind == DetectionKind.SEMANTIC_STAGNATION:
        return fuse.config.progress_callback is not None
    return False


# Attach the wrapper builders onto the class so users can do:
#   fuse.wrap(fn)
#   fuse.tool(fn) or @fuse.tool
def _wrap_method(self, fn=None, *, name=None, allow_repeats=None):
    deco = _build_tool_decorator(self)
    if fn is None:
        return lambda f: deco(f, name=name, allow_repeats=allow_repeats)
    return deco(fn, name=name, allow_repeats=allow_repeats)


AgentFuse.wrap = _wrap_method  # type: ignore[attr-defined]
AgentFuse.tool = _wrap_method  # type: ignore[attr-defined]


# Also expose a functional shorthand.
class _FuseNamespace:
    """Module-level namespace so users can do ``agent_fuse.wrap(fn)``."""

    def __init__(self, fuse: "AgentFuse"):
        self._fuse = fuse

    def __call__(self, fn=None, *, name=None, allow_repeats=None):
        return _wrap_method(self._fuse, fn=fn, name=name, allow_repeats=allow_repeats)

    def tool(self, fn=None, *, name=None, allow_repeats=None):
        return _wrap_method(self._fuse, fn=fn, name=name, allow_repeats=allow_repeats)


def fuse_namespace(fuse: "AgentFuse") -> _FuseNamespace:
    """Return a callable that wraps tools against ``fuse``.

    Useful when you want a single import::

        from agent_fuse import AgentFuse, fuse_namespace
        fuse = AgentFuse()
        fuse_tool = fuse_namespace(fuse)

        @fuse_tool(name="search")
        def search(...):
            ...
    """
    return _FuseNamespace(fuse)
