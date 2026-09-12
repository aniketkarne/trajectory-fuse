"""OpenAI Agents SDK middleware adapter for trajectory-fuse.

This adapter exposes a function-shaped hook that can be plugged into
the OpenAI Agents ``tool_use_behavior`` (or any newer equivalent) to
forward every tool invocation through an :class:`AgentFuse`.

The ``openai-agents`` package is imported lazily inside
:func:`openai_agents_middleware`. If the package is not installed, the
factory raises :class:`FrameworkNotInstalled` with the correct install
hint.

Usage::

    from trajectory_fuse import AgentFuse, FuseConfig
    from trajectory_fuse.integrations import openai_agents_middleware

    fuse = AgentFuse(FuseConfig(window=20))
    hook = openai_agents_middleware(fuse)

    # Pass ``hook`` into your Agent(...) configuration as
    # ``tool_use_behavior`` (or the modern equivalent in your installed
    # version of the OpenAI Agents SDK).

Install::

    pip install trajectory-fuse[openai-agents]
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..guard import AgentFuse
from ..types import Action
from . import FrameworkNotInstalled, _hint

__all__ = ["openai_agents_middleware"]


def openai_agents_middleware(
    fuse: AgentFuse,
    *,
    tool_to_args: Optional[Callable[[Any], Any]] = None,
    tool_to_result: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Return an OpenAI Agents SDK-compatible hook wired to ``fuse``.

    Parameters
    ----------
    fuse:
        The :class:`AgentFuse` to feed observations into.
    tool_to_args:
        Optional callable extracting an args dict from a tool-call object.
    tool_to_result:
        Optional callable extracting a result/error from a tool-call object.

    Returns
    -------
    A callable that accepts a tool-call object and runs it through
    ``fuse`` (preflight + observe). The exact signature matches the
    OpenAI Agents SDK's ``tool_use_behavior`` hook shape.

    Raises
    ------
    FrameworkNotInstalled
        If ``openai-agents`` (or its predecessor ``agents``) is not
        importable.
    """
    try:
        import agents  # type: ignore  # noqa: F401
    except ImportError:
        try:
            import openai_agents  # type: ignore  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalled("openai-agents", _hint("openai-agents"))

    extractor_args = tool_to_args or _default_args_extractor
    extractor_result = tool_to_result or _default_result_extractor

    def _hook(call: Any) -> Any:
        """Observe ``call`` through ``fuse``; reraise on detected deadlock."""
        tool = _call_name(call)
        args = extractor_args(call)
        detection = fuse.check(tool, args)
        if detection is not None and tool not in fuse.config.allowlist:
            from ..exceptions import DeadlockDetected

            raise DeadlockDetected(
                detection.message, detection=detection, trajectory=fuse.history
            )

        result = extractor_result(call)
        success = _call_success(call)
        fuse.observe(Action(tool=tool, args=args, result=result, success=success))
        return call

    return _hook


# ---------------------------------------------------------------------
# helpers


def _default_args_extractor(call: Any) -> Any:
    if isinstance(call, dict):
        return call.get("arguments") or call.get("args") or call.get("input")
    return getattr(call, "arguments", None) or getattr(call, "args", None)


def _default_result_extractor(call: Any) -> Any:
    if isinstance(call, dict):
        return call.get("output") or call.get("result")
    return getattr(call, "output", None) or getattr(call, "result", None)


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
    if getattr(call, "error", None):
        return False
    return None
