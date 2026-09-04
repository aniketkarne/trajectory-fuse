"""End-to-end demo of agent-fuse 0.2.0 as a *circuit breaker*.

Run::

    python examples/demo_circuit_breaker.py

Builds a tiny agent loop with three realistic tools — a search API, a
"think" placeholder, and a flaky weather API — and shows three things:

1. The default ``observe()`` detector catches a deadlock on the weather
   endpoint after four identical 429-style failures.
2. The ``@fuse.tool`` decorator *blocks the next call before it is made*
   once a deadlock is imminent (the "circuit breaker" guarantee).
3. The ``allow_repeats`` per-tool override lets the search tool be polled
   repeatedly without tripping the fuse — even after the global threshold.

The script prints a step-by-step trace and exits 0. No network calls.
"""

from __future__ import annotations

import time

from agent_fuse import (
    Action,
    AgentFuse,
    DeadlockDetected,
    DetectionKind,
    FuseConfig,
    RunBudget,
    fuse_namespace,
)


# ---------------------------------------------------------------------------
# fake tools (no network, no LLM)
# ---------------------------------------------------------------------------


def think(prompt: str) -> str:
    """The agent's "inner monologue" — never a real tool call."""
    return f"reasoning about: {prompt}"


def search(query: str) -> str:
    """Pretend web search. Returns deterministic results."""
    return f"<result for {query!r}>"


def weather(city: str) -> str:
    """Pretend flaky weather API. *Always* returns the same 429."""
    return "429 rate limited please retry after backoff"


# ---------------------------------------------------------------------------
# wire tools into the fuse with the @fuse.tool decorator
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = FuseConfig(
        window=32,
        budgets=RunBudget(max_actions=50, max_runtime_seconds=30.0),
        # Polling search is fine; the deadlock we want to catch is the
        # weather endpoint repeatedly failing in the same way.
        allow_repeats={"search": 100},
        # Stagnation is conservative by default — it fires only after
        # *four* identical failures. Lower it for the demo so the circuit
        # trips within a handful of calls.
        stagnation=None,  # set below
    )
    # Replace the None above with an explicit config (kept separate for
    # readability in the print output below).
    from agent_fuse import StagnationConfig

    cfg.stagnation = StagnationConfig(min_failures=2, similarity_threshold=0.8)

    fuse = AgentFuse(cfg)
    fuse_tool = fuse_namespace(fuse)

    @fuse_tool(name="think")
    def _think(prompt: str) -> str:
        return think(prompt)

    @fuse_tool(name="search", allow_repeats=100)
    def _search(query: str) -> str:
        return search(query)

    @fuse_tool(name="weather")
    def _weather(city: str) -> str:
        return weather(city)

    print(f"[init] fuse ready; window={cfg.window}, "
          f"max_actions={cfg.budgets.max_actions}, "
          f"stagnation_min_failures={cfg.stagnation.min_failures}")

    # ------------------------------------------------------------------
    # scenario 1 — legitimate polling of "search" must NOT trip the fuse
    # ------------------------------------------------------------------
    print("\n[scenario 1] polling 'search' 25 times in a row")
    for i in range(25):
        _search(f"q{i}")
    print(f"  ok — {fuse.stats.calls_allowed} calls allowed, "
          f"{fuse.stats.calls_blocked} blocked")

    # ------------------------------------------------------------------
    # scenario 2 — circuit breaker: preflight blocks the call *before*
    # the tool runs once the trajectory makes a deadlock imminent.
    # ------------------------------------------------------------------
    print("\n[scenario 2] weather endpoint failing repeatedly")
    blocked_preflight = 0
    for i in range(6):
        try:
            _weather("nyc")
            print(f"  call {i}: tool ran")
        except DeadlockDetected as exc:
            blocked_preflight += 1
            print(f"  call {i}: BLOCKED by fuse → {exc.detection.kind.value}")
            if exc.detection.kind == DetectionKind.SEMANTIC_STAGNATION:
                break
    print(f"  preflight blocked {blocked_preflight} of 6 calls")

    # ------------------------------------------------------------------
    # scenario 3 — budgets: a fresh fuse with a small action cap
    # ------------------------------------------------------------------
    print("\n[scenario 3] action budget trips after 10 calls")
    fresh = AgentFuse(FuseConfig(budgets=RunBudget(max_actions=10)))
    from agent_fuse import BudgetExceeded

    tripped = False
    for i in range(15):
        try:
            fresh.observe(Action(
                tool="step", args={"i": i}, result="ok", success=True,
            ))
        except BudgetExceeded as exc:
            print(f"  tripped after {exc.observed} actions "
                  f"(kind={exc.kind}, limit={exc.limit})")
            tripped = True
            break
    if not tripped:
        print("  budget did NOT trip — bug?")

    # ------------------------------------------------------------------
    # scenario 4 — wrap stats and record what we avoided
    # ------------------------------------------------------------------
    print(f"\n[final] fuse.stats: {fuse.stats.as_dict()}")
    # Pretend each blocked call saved ~3 seconds of model + tool runtime.
    fuse.record_time_avoided(seconds=fuse.stats.calls_blocked * 3.0)
    print(f"          with time_avoided: {fuse.stats.as_dict()}")


if __name__ == "__main__":
    t0 = time.monotonic()
    main()
    print(f"\n[done] elapsed {time.monotonic() - t0:.3f}s")
