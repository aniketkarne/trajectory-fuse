"""Framework adapters for trajectory-fuse.

This package ships *optional* integrations with three popular agent
frameworks — **LangGraph**, the **OpenAI Agents SDK**, and
**PydanticAI** — plus a **generic** adapter that works in any Python
codebase.

Each adapter module follows the same shape:

* The module imports the framework's middleware/hook API inside a
  try/except, so importing ``trajectory_fuse.integrations.langgraph``
  without ``langgraph`` installed does **not** break the host
  application.
* A factory function (``langgraph_middleware``, ``openai_agents_middleware``,
  ``pydantic_ai_middleware``) returns the framework's middleware
  object wired to an :class:`AgentFuse`. Calling the factory when the
  framework is not installed raises :class:`FrameworkNotInstalled`
  with the exact ``pip install`` command that fixes it.
* The **generic** adapter has zero dependencies — it re-exports the
  ergonomic decorators and context-manager helpers so plain Python
  agent loops get the same idioms.

Install an adapter with the matching extra::

    pip install trajectory-fuse[langgraph]
    pip install trajectory-fuse[openai-agents]
    pip install trajectory-fuse[pydantic-ai]

or install them all at once::

    pip install trajectory-fuse[all-integrations]

The framework adapters intentionally *do not* lock in a specific
version range. We treat the framework APIs as best-effort hooks —
when the framework changes, the adapter fails at *import time* with a
clear message, not at agent-execution time.
"""

from __future__ import annotations


class FrameworkNotInstalled(ImportError):
    """Raised when a framework adapter is requested without the framework installed.

    Attributes
    ----------
    framework:
        Display name (e.g. ``"langgraph"``).
    install_hint:
        The exact ``pip install`` line that fixes it.
    """

    framework: str
    install_hint: str

    def __init__(self, framework: str, install_hint: str) -> None:
        self.framework = framework
        self.install_hint = install_hint
        ImportError.__init__(
            self,
            f"the {framework!r} framework is not installed. {install_hint}",
        )


def _hint(extra: str) -> str:
    """Return the canonical install hint for an integration extra."""
    return (
        f"Install it with `pip install trajectory-fuse[{extra}]` "
        f"or `pip install trajectory-fuse[all-integrations]`."
    )


def langgraph_middleware(fuse, *args, **kwargs):  # pragma: no cover - re-exported
    """Factory for the LangGraph middleware. Implemented in :mod:`.langgraph_adapter`."""
    from .langgraph_adapter import langgraph_middleware as _impl

    return _impl(fuse, *args, **kwargs)


def openai_agents_middleware(fuse, *args, **kwargs):  # pragma: no cover - re-exported
    """Factory for the OpenAI Agents middleware. Implemented in :mod:`.openai_agents_adapter`."""
    from .openai_agents_adapter import openai_agents_middleware as _impl

    return _impl(fuse, *args, **kwargs)


def pydantic_ai_middleware(fuse, *args, **kwargs):  # pragma: no cover - re-exported
    """Factory for the PydanticAI middleware. Implemented in :mod:`.pydantic_ai_adapter`."""
    from .pydantic_ai_adapter import pydantic_ai_middleware as _impl

    return _impl(fuse, *args, **kwargs)


__all__ = [
    "FrameworkNotInstalled",
    "langgraph_middleware",
    "openai_agents_middleware",
    "pydantic_ai_middleware",
]
