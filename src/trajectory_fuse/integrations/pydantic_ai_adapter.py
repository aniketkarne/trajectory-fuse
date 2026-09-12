"""PydanticAI middleware adapter for trajectory-fuse.

This adapter exposes a callable that can be passed to
PydanticAI's tool-call instrumentation. The ``pydantic-ai`` package is
imported lazily inside :func:`pydantic_ai_middleware`. If the package
is not installed, the factory raises
:class:`FrameworkNotInstalled`.

Usage::

    from trajectory_fuse import AgentFuse, FuseConfig
    from trajectory_fuse.integrations import pydantic_ai_middleware

    fuse = AgentFuse(FuseConfig(window=20))
    observer = pydantic_ai_middleware(fuse)

    # Pass ``observer`` to your PydanticAI agent via its instrumentation
    # hook (varies by installed version; see the docstring of your
    # pydantic_ai.Agent for the current knob).

Install::

    pip install trajectory-fuse[pydantic-ai]
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..guard import AgentFuse
from ..types import Action
from . import FrameworkNotInstalled, _hint

__all__ = ["pydantic_ai_middleware"]


def pydantic_ai_middleware(
    fuse: AgentFuse,
    *,
    tool_to_args: Optional[Callable[[Any], Any]] = None,
    tool_to_result: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Return a PydanticAI-compatible observer wired to ``fuse``.

    Parameters
    ----------
    fuse:
        The :class:`AgentFuse` to feed observations into.
    tool_to_args:
        Optional callable extracting args from a PydanticAI tool call.
    tool_to_result:
        Optional callable extracting a result/error from a tool call.

    Returns
    -------
    A callable suitable for the PydanticAI agent's tool-call hook. The
    returned object exposes ``.on_tool_call(call)`` and
    ``.on_tool_result(call, result)`` methods, mirroring the agent
    framework's two-phase tool lifecycle.

    Raises
    ------
    FrameworkNotInstalled
        If ``pydantic-ai`` is not importable.
    """
    try:
        import pydantic_ai  # type: ignore  # noqa: F401
    except ImportError:
        try:
            # Older / alternative distribution name.
            import pydantic_ai  # type: ignore  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalled("pydantic-ai", _hint("pydantic-ai"))

    extractor_args = tool_to_args or _default_args_extractor

    class _PydanticAITrajectoryObserver:
        """Two-phase tool-call observer for PydanticAI agents."""

        def on_tool_call(self, call: Any) -> None:
            """Preflight: check the would-be tool call against ``fuse``."""
            tool = _call_name(call)
            args = extractor_args(call)
            detection = fuse.check(tool, args)
            if detection is not None and tool not in fuse.config.allowlist:
                from ..exceptions import DeadlockDetected

                raise DeadlockDetected(
                    detection.message, detection=detection, trajectory=fuse.history
                )

        def on_tool_result(self, call: Any, result: Any) -> None:
            """Post-call: record the observation through ``fuse``."""
            tool = _call_name(call)
            args = extractor_args(call)
            success = _is_failure_result(result)
            payload = result if not success else {"error": repr(result)}
            fuse.observe(Action(tool=tool, args=args, result=payload, success=success))

    return _PydanticAITrajectoryObserver()


# ---------------------------------------------------------------------
# helpers


def _default_args_extractor(call: Any) -> Any:
    # PydanticAI tool calls are normally a (ToolCallPart) with .args as
    # a dict (validated) — but older builds exposed ``arguments``.
    if isinstance(call, dict):
        return call.get("args") or call.get("arguments")
    return getattr(call, "args", None) or getattr(call, "arguments", None)


def _default_result_extractor(call: Any) -> Any:
    return getattr(call, "result", None) if not isinstance(call, dict) else call.get("result")


def _call_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("tool_name") or call.get("name") or "tool")
    return str(getattr(call, "tool_name", None) or getattr(call, "name", None) or "tool")


def _is_failure_result(result: Any) -> bool:
    # Be conservative — only mark failure when the framework hands us a
    # hard exception type. We do not want to false-positive on results
    # that *look* like errors but are domain-correct.
    if isinstance(result, BaseException):
        return True
    if isinstance(result, dict) and result.get("error"):
        return True
    return False
