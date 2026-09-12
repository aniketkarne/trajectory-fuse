"""LangGraph middleware adapter for trajectory-fuse.

This adapter wraps an :class:`AgentFuse` into a LangGraph-compatible
middleware. It is *optional* — the ``langgraph`` package is imported
lazily inside :func:`langgraph_middleware`. If LangGraph is not
installed, the factory raises :class:`FrameworkNotInstalled`.

We deliberately do not pin a specific LangGraph version. The adapter
tries the modern ``langchain.agents.middleware`` API first (added in
LangGraph 0.2), and falls back to the legacy ``langgraph.prebuilt``
hook signature if available. If neither is importable, the factory
raises a clear ``ImportError`` explaining which version we expected.

Usage::

    from trajectory_fuse import AgentFuse, FuseConfig
    from trajectory_fuse.integrations import langgraph_middleware

    fuse = AgentFuse(FuseConfig(window=20))
    mw = langgraph_middleware(fuse, tool_to_args=lambda node: node.tool_call.args)

    # In your LangGraph agent definition:
    #   graph = builder.compile(middleware=[mw])

Install::

    pip install trajectory-fuse[langgraph]
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Callable, Optional

from ..guard import AgentFuse
from ..types import Action
from . import FrameworkNotInstalled, _hint

__all__ = ["langgraph_middleware"]


def langgraph_middleware(
    fuse: AgentFuse,
    *,
    tool_to_args: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Return a LangGraph middleware wired to ``fuse``.

    Parameters
    ----------
    fuse:
        The :class:`AgentFuse` to feed observations into.
    tool_to_args:
        Optional callable that extracts the ``args`` dict from a LangGraph
        tool-call node. Defaults to a permissive ``dict(node)`` shim that
        works for plain-dict LangGraph state.

    Returns
    -------
    A LangGraph-compatible middleware object. The exact type depends on
    the installed LangGraph version; the returned object exposes a
    ``before_model`` / ``after_model`` pair that drives ``fuse.observe``.

    Raises
    ------
    FrameworkNotInstalled
        If ``langgraph`` (or its successor ``langchain``) is not importable.
    """
    try:
        # LangGraph 0.2+ exposes middleware via langchain.agents.middleware.
        from langchain.agents.middleware import AgentMiddleware  # type: ignore
    except ImportError:
        try:
            # Older LangGraph exposed the same shape under langgraph.
            from langgraph.prebuilt import AgentMiddleware  # type: ignore
        except ImportError:
            raise FrameworkNotInstalled("langgraph", _hint("langgraph"))

    try:
        from langchain_core.messages import AIMessage  # type: ignore  # noqa: F401
    except ImportError:
        try:
            from langgraph.prebuilt import AIMessage  # type: ignore  # noqa: F401
        except ImportError:
            # The middleware framework may still work without AIMessage
            # being importable; surface a soft warning via the install
            # hint rather than failing the whole integration.
            pass

    extractor = tool_to_args or _default_extractor

    class _TrajectoryFuseMiddleware(AgentMiddleware):
        """LangGraph middleware that forwards tool calls to ``fuse``."""

        async def before_model(self, state, runtime=None):  # noqa: D401, ANN001
            # Best-effort: walk the most recent AI message and observe any
            # tool calls it contains. If the LangGraph API is the sync
            # variant we still observe them (just without awaiting).
            for call in _iter_recent_tool_calls(state):
                tool = _call_name(call)
                args = extractor(call)
                detection = fuse.check(tool, args)
                if detection is not None and tool not in fuse.config.allowlist:
                    from ..exceptions import DeadlockDetected

                    raise DeadlockDetected(
                        detection.message, detection=detection, trajectory=fuse.history
                    )

        async def after_model(self, state, runtime=None):  # noqa: D401, ANN001
            # Record each tool call as observed; if a tool raised in the
            # node, surface that as success=False so the stagnation
            # detector sees real failures.
            for call in _iter_recent_tool_calls(state):
                tool = _call_name(call)
                args = extractor(call)
                success = _call_success(call)
                result = _call_result(call)
                fuse.observe(
                    Action(tool=tool, args=args, result=result, success=success)
                )

    return _TrajectoryFuseMiddleware()


# ---------------------------------------------------------------------
# helpers


def _default_extractor(node: Any) -> Any:
    """Best-effort conversion of a LangGraph tool-call node to an args dict."""
    if isinstance(node, dict):
        return node.get("args") or node.get("input") or node
    for attr in ("args", "input", "tool_input"):
        if hasattr(node, attr):
            return getattr(node, attr)
    return node


def _iter_recent_tool_calls(state: Any) -> Iterator:
    """Yield tool-call objects from the most recent AI message in ``state``.

    LangGraph's state shape varies across versions. We accept either:
    * a dict with ``messages`` (modern langchain StateGraph), or
    * an object with a ``messages`` attribute, or
    * a list of messages directly.

    Yields zero elements for any shape we cannot introspect — the
    caller will simply observe no calls, which is a safe no-op.
    """
    messages = None
    if isinstance(state, dict):
        messages = state.get("messages")
    elif hasattr(state, "messages"):
        messages = state.messages

    if not messages:
        return iter(())

    # Walk backwards until we find an AI message with tool_calls.
    for msg in reversed(list(messages)):
        calls = None
        if hasattr(msg, "tool_calls"):
            calls = msg.tool_calls
        elif isinstance(msg, dict):
            calls = msg.get("tool_calls")
        if calls:
            return iter(calls)
    return iter(())


def _call_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or call.get("tool") or "tool")
    return str(getattr(call, "name", None) or getattr(call, "tool", None) or "tool")


def _call_success(call: Any) -> Optional[bool]:
    if isinstance(call, dict):
        if "error" in call or call.get("status") == "error":
            return False
        if call.get("status") == "ok":
            return True
        return None
    err = getattr(call, "error", None)
    if err:
        return False
    return None


def _call_result(call: Any) -> Any:
    if isinstance(call, dict):
        return call.get("output") or call.get("result") or call.get("content")
    return getattr(call, "output", None) or getattr(call, "result", None)
