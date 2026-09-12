"""Generic Python adapter — zero-dependency integration.

This adapter is the simplest possible wiring: it does not depend on
any external framework. It exposes two helpers on top of
:class:`AgentFuse`:

* :func:`guard_scope` — a context manager that runs an :class:`AgentFuse`
  over a block of code. Useful when you want a guard without decorating
  individual tools (e.g. you drive the loop manually).
* :func:`observe_call` — a single-call wrapper. Returns a callable that
  records the result and forwards to the underlying tool. Useful when
  you want to integrate with a tool dispatcher that hands you a
  ``(tool, args)`` pair but you do not control the call site::

      from trajectory_fuse.integrations.generic import observe_call

      fuse = AgentFuse()
      search = observe_call(fuse, "search", http_get)

      result = search(q="weather")
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable, Optional

from ..guard import AgentFuse
from ..types import Action

__all__ = ["guard_scope", "observe_call", "make_observer"]


@contextmanager
def guard_scope(
    fuse: AgentFuse,
    *,
    enter_action: Optional[str] = None,
    exit_action: Optional[str] = None,
) -> Iterator[AgentFuse]:
    """Context manager around an :class:`AgentFuse` run.

    The block is a no-op if the caller never calls ``fuse.observe``
    inside it. Pass ``enter_action`` / ``exit_action`` to record a
    synthetic boundary on entry and exit (useful for analytics).

    Example::

        with guard_scope(fuse, enter_action="task_start", exit_action="task_end") as f:
            for action in policy_loop(state):
                f.observe(action)
    """
    if enter_action is not None:
        try:
            fuse.mark_progress(_synthetic_signal(enter_action))
        except Exception:
            pass
    try:
        yield fuse
    finally:
        if exit_action is not None:
            try:
                fuse.mark_progress(_synthetic_signal(exit_action))
            except Exception:
                pass


def _synthetic_signal(token: str):
    from ..types import ProgressSignal
    return ProgressSignal(token=token, note="synthetic boundary")


def make_observer(fuse: AgentFuse, tool_name: Optional[str] = None) -> Callable[[Action], None]:
    """Return a zero-arg ``observe`` callable bound to ``fuse``.

    Useful when a framework hands you a callback slot::

        observer = make_observer(fuse, tool_name="search")
        framework.set_observer(observer)

    Each call to ``observer(action)`` is equivalent to
    ``fuse.observe(action)`` (the exception is propagated to the
    caller).
    """
    name = tool_name or "tool"

    def _observer(action: Action) -> None:
        # If the caller passes a bare ``action`` without a tool name,
        # adopt the bound name — but never overwrite an explicit one.
        if not action.tool:
            action.tool = name
        fuse.observe(action)

    return _observer


def observe_call(
    fuse: AgentFuse,
    tool_name: str,
    fn: Callable[..., Any],
    *,
    allow_repeats: Optional[int] = None,
) -> Callable[..., Any]:
    """Return a wrapped callable that records every invocation through ``fuse``.

    The wrapper performs the same preflight + post-call sequence as
    :meth:`AgentFuse.tool` but does **not** mutate ``fuse.config`` — it
    creates a fresh per-call :class:`Action` against the existing fuse.

    Example::

        search = observe_call(fuse, "search", http_get, allow_repeats=100)
        for page in range(50):
            search(q="weather", page=page)   # preflight + observe every time
    """
    if allow_repeats is not None:
        # Mirror @fuse.tool semantics: register the per-tool override so
        # preflight (``fuse.check``) sees the tolerance.
        fuse.config.allow_repeats[tool_name] = allow_repeats
        fuse._allow_repeats[tool_name] = allow_repeats  # noqa: SLF001

    if _is_async(fn):

        @functools.wraps(fn)
        async def _async(*args, **kwargs):
            args_payload = {"args": args, "kwargs": kwargs}
            detection = fuse.check(tool_name, args_payload)
            if detection is not None and tool_name not in fuse.config.allowlist:
                from ..exceptions import DeadlockDetected
                raise DeadlockDetected(
                    detection.message, detection=detection, trajectory=fuse.history
                )
            try:
                result = await fn(*args, **kwargs)
            except Exception as exc:
                fuse.observe(
                    Action(
                        tool=tool_name,
                        args=args_payload,
                        result={"error": repr(exc)},
                        success=False,
                    )
                )
                raise
            fuse.observe(
                Action(tool=tool_name, args=args_payload, result=result, success=True)
            )
            return result

        return _async

    @functools.wraps(fn)
    def _sync(*args, **kwargs):
        args_payload = {"args": args, "kwargs": kwargs}
        detection = fuse.check(tool_name, args_payload)
        if detection is not None and tool_name not in fuse.config.allowlist:
            from ..exceptions import DeadlockDetected
            raise DeadlockDetected(
                detection.message, detection=detection, trajectory=fuse.history
            )
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            fuse.observe(
                Action(
                    tool=tool_name,
                    args=args_payload,
                    result={"error": repr(exc)},
                    success=False,
                )
            )
            raise
        fuse.observe(
            Action(tool=tool_name, args=args_payload, result=result, success=True)
        )
        return result

    return _sync


def _is_async(fn: Callable[..., Any]) -> bool:
    import inspect

    return inspect.iscoroutinefunction(fn)
