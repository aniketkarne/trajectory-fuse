"""Realistic trajectory generator for the efficacy benchmark.

We model an LLM agent driving a small set of tools. The generator
emits a *trajectory* — a list of :class:`Action`-shaped dicts — for
each of nine scenarios that real agents hit in production:

Scenarios that should NOT trip the fuse (legitimate behaviour)
-----------------------------------------------------------
* ``legitimate_polling``        — repeated identical HTTP poll until status=ok
* ``legitimate_pagination``     — paginated fetch with increasing page indices
* ``legitimate_retry``          — transient 429 with fresh diagnostic tokens
* ``legitimate_long_loop``      — search→summarise repeated with new args
* ``legitimate_progress_phase`` — same tools before/after a mark_progress

Scenarios that MUST trip the fuse (broken behaviour)
----------------------------------------------------
* ``broken_direct_repeat``      — identical calls beyond direct_repeat_threshold
* ``broken_n_cycle``            — A→B→A→B loop
* ``broken_stagnation``         — repeated same-failure strings
* ``broken_stuck_after_recovery`` — a model "tries again" after a fix hint

The generator is deterministic (seeded) so the benchmark numbers are
reproducible across machines.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import List

# Allow running from the repo root without installing the package.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trajectory_fuse.types import TrajectoryRecord  # noqa: E402

# ---------------------------------------------------------------------
# canonical action shape: dicts the fuse can observe.


def _ok(tool: str, args: dict, result, **extra) -> dict:
    return {"tool": tool, "args": args, "result": result, "success": True, **extra}


def _fail(tool: str, args: dict, result, **extra) -> dict:
    return {"tool": tool, "args": args, "result": result, "success": False, **extra}


def _progress(token: str, note: str = "") -> dict:
    return {"tool": "<progress>", "args": token, "result": note, "is_progress": True}


# ---------------------------------------------------------------------
# generators


def legitimate_polling(seed: int = 0, polls: int = 50) -> list:
    """Repeated identical poll until status == "ok".

    The fuse should tolerate this via ``allow_repeats={"poll_status": polls}``.
    Without the fuse the model burns through ``polls`` identical calls;
    with the fuse (correctly configured) it does the same — there is no
    deadlock, the fuse simply doesn't trip.
    """
    out = []
    for i in range(polls):
        if i == polls - 1:
            out.append(_ok("poll_status", {"job_id": "j-42"}, {"status": "ok"}))
        else:
            out.append(
                _ok(
                    "poll_status",
                    {"job_id": "j-42"},
                    {"status": "pending", "attempt": i},
                )
            )
    return out


def legitimate_pagination(seed: int = 0, pages: int = 30) -> List[dict]:
    """Paginated fetch — increasing page indices, never identical."""
    out = []
    for page in range(1, pages + 1):
        out.append(
            _ok(
                "fetch_page",
                {"query": "users", "page": page, "per_page": 50},
                {"rows": [f"u-{page}-{i}" for i in range(50)], "has_more": page < pages},
            )
        )
    return out


def legitimate_retry(seed: int = 0, attempts: int = 5) -> List[dict]:
    """A transient 429 with fresh diagnostic tokens per attempt.

    Each attempt carries an incrementing ``attempt`` field in its args
    so the args-hash changes per call — the *direct-repeat* detector
    does not fire, and the stagnation detector sees the fresh
    diagnostic token in each error message and breaks the similarity
    chain. This is what a well-behaved retry layer actually emits.
    """
    out = []
    for i in range(attempts):
        out.append(
            _fail(
                "http_get",
                {
                    "url": "https://api.example/v1/things",
                    "attempt": i + 1,
                    "backoff_s": i * 2 + 1,
                    "trace_id": f"{seed}-{i}",
                },
                (
                    f"HTTP 429 (attempt {i + 1}): retry-after={i * 2 + 1}s "
                    f"trace-id={seed}-{i} "
                    f"x-request-id={seed:08x}-{i:04x}"
                ),
                error="429",
            )
        )
    out.append(
        _ok(
            "http_get",
            {
                "url": "https://api.example/v1/things",
                "attempt": attempts + 1,
                "backoff_s": 0,
                "trace_id": f"{seed}-{attempts}",
            },
            {"items": [1, 2, 3], "total": 3},
        )
    )
    return out


def legitimate_long_loop(seed: int = 0, rounds: int = 20) -> List[dict]:
    """search→summarise with *different* args each round.

    This is a structural loop the cycle detector must NOT misfire on,
    because each call has new args.
    """
    out = []
    for i in range(rounds):
        out.append(
            _ok(
                "search",
                {"q": f"topic-{i}", "filters": ["a", "b"], "limit": 10},
                {"hits": [f"doc-{i}-{j}" for j in range(3)]},
            )
        )
        out.append(
            _ok(
                "summarise",
                {"text": f"doc-{i}-0 doc-{i}-1 doc-{i}-2", "max_words": 50},
                {"summary": f"round {i} summary"},
            )
        )
    return out


def legitimate_progress_phase(seed: int = 0, per_phase: int = 5) -> List[dict]:
    """Two phases separated by mark_progress.

    Phase 1 — the agent attempts ``weather_api`` five times with an
    increasing backoff. Each call has distinct args (the backoff
    value), so the direct-repeat detector does not fire. The
    stagnation detector also doesn't fire because each error message
    carries a different backoff number.

    The boundary marks a strategy change.

    Phase 2 — successful calls with the same tool.

    Expected outcome: the fuse does NOT trip.
    """
    out = []
    # Phase 1 — five retry attempts with increasing backoff.
    for i in range(per_phase):
        out.append(
            _fail(
                "weather_api",
                {
                    "city": "nyc",
                    "attempt": i + 1,
                    "backoff_s": i * 2 + 1,
                },
                (
                    f"Error: upstream timeout connecting to weather.example "
                    f"(attempt {i + 1}, backoff {i * 2 + 1}s)"
                ),
                error="timeout",
            )
        )
    # Phase boundary.
    out.append(_progress("new_strategy", note="switching to cached fallback"))
    # Phase 2 — successful weather_api calls (distinct args too — different
    # observation timestamps).
    for i in range(per_phase):
        out.append(
            _ok(
                "weather_api",
                {
                    "city": "nyc",
                    "attempt": i + 1,
                    "source": "cache",
                },
                {"temp_f": 72 + i, "conditions": "clear"},
            )
        )
    return out


def broken_direct_repeat(seed: int = 0, repeats: int = 6) -> List[dict]:
    """Identical ``(tool, args)`` past ``direct_repeat_threshold=3``."""
    out = []
    for _ in range(repeats):
        out.append(
            _ok(
                "search",
                {"q": "weather in tokyo"},
                {"hits": ["doc-1"]},
            )
        )
    return out


def broken_n_cycle(seed: int = 0, rounds: int = 4) -> List[dict]:
    """A→B→A→B repeating cycle."""
    out = []
    for _ in range(rounds):
        out.append(_ok("tool_a", {"x": 1}, "ok"))
        out.append(_ok("tool_b", {"y": 2}, "ok"))
    return out


def broken_stagnation(seed: int = 0, fails: int = 6) -> List[dict]:
    """Six consecutive failures with near-identical tokens.

    The stagnation detector must fire on the 4th consecutive failure
    (default ``min_failures=4``).
    """
    out = []
    base = (
        "Error: connection timeout while connecting to upstream "
        "service at 10.0.0.42:443"
    )
    for i in range(fails):
        # Vary one token per attempt but keep 95%+ Jaccard similarity.
        msg = base if i == 0 else base + f" retry={i}"
        out.append(_fail("http_get", {"url": "https://api/x"}, msg))
    return out


def broken_stuck_after_recovery(seed: int = 0, repeats: int = 8) -> List[dict]:
    """A "tries again with same args" loop that survives one hint."""
    out = []
    for _ in range(repeats):
        out.append(
            _fail(
                "weather_api",
                {"city": "tokyo"},
                "timeout waiting for upstream response",
                error="timeout",
            )
        )
    return out


# ---------------------------------------------------------------------
# parametric variants for the scale benchmark


def parametric_retry(seed: int = 0, *, attempts: int) -> List[dict]:
    """Same shape as legitimate_retry but with any attempt count."""
    return legitimate_retry(seed=seed, attempts=attempts)


def parametric_polling(seed: int = 0, *, polls: int) -> List[dict]:
    return legitimate_polling(seed=seed, polls=polls)


def parametric_pagination(seed: int = 0, *, pages: int) -> List[dict]:
    return legitimate_pagination(seed=seed, pages=pages)


def parametric_long_loop(seed: int = 0, *, rounds: int) -> List[dict]:
    return legitimate_long_loop(seed=seed, rounds=rounds)


def parametric_direct_repeat(seed: int = 0, *, repeats: int) -> List[dict]:
    return broken_direct_repeat(seed=seed, repeats=repeats)


def parametric_n_cycle(seed: int = 0, *, rounds: int) -> List[dict]:
    return broken_n_cycle(seed=seed, rounds=rounds)


def parametric_stagnation(seed: int = 0, *, fails: int) -> List[dict]:
    return broken_stagnation(seed=seed, fails=fails)


# ---------------------------------------------------------------------
# scaled dataset (500+ trajectories)


def iter_scaled_dataset() -> Iterator[tuple]:
    """Yield 500+ trajectories spanning legitimate and broken shapes.

    Each tuple is ``(scenario_label, actions, expected_deadlock)``.
    The seed varies so two trajectories of the same shape are not
    identical.
    """
    # Legitimate variants — should never trip.
    legitimate = []
    for seed in range(50):
        legitimate.append((f"polling/seed={seed}", legitimate_polling(seed=seed, polls=50), False))
        legitimate.append((f"pagination/seed={seed}", legitimate_pagination(seed=seed, pages=30), False))
        legitimate.append((f"retry/seed={seed}", legitimate_retry(seed=seed, attempts=5), False))
        legitimate.append((f"long_loop/seed={seed}", legitimate_long_loop(seed=seed, rounds=20), False))
        legitimate.append((f"progress_phase/seed={seed}", legitimate_progress_phase(seed=seed, per_phase=5), False))

    # Broken variants — must trip.
    broken = []
    for seed in range(50):
        broken.append((f"direct_repeat/seed={seed}", broken_direct_repeat(seed=seed, repeats=6), True))
        broken.append((f"n_cycle/seed={seed}", broken_n_cycle(seed=seed, rounds=4), True))
        broken.append((f"stagnation/seed={seed}", broken_stagnation(seed=seed, fails=6), True))
        broken.append((f"stuck/seed={seed}", broken_stuck_after_recovery(seed=seed, repeats=8), True))

    # Edge cases — small variants that push the boundary.
    # IMPORTANT: a 3-call direct-repeat trajectory *will* trip on the
    # 3rd identical call (direct_repeat_threshold=3 means trip on the
    # Nth call). Likewise a 3-fail stagnation trajectory trips on
    # direct_repeat before stagnation has enough samples.
    edges = [
        # Tiny legitimate retries — should not trip even at 2 attempts.
        ("retry/tiny", legitimate_retry(seed=99, attempts=2), False),
        # Retry that hits the budget cap instead of the fuse — the fuse
        # should NOT fire on this; the budget should.
        ("retry/long_budget", legitimate_retry(seed=99, attempts=3), False),
        # Direct repeat exactly at threshold (3) — the 3rd identical call
        # is the trip point, so this trips.
        ("direct_repeat/at_threshold", broken_direct_repeat(seed=99, repeats=3), True),
        # Direct repeat one past threshold (4) — trips.
        ("direct_repeat/past_threshold", broken_direct_repeat(seed=99, repeats=4), True),
        # Direct repeat just under threshold (2) — passes.
        ("direct_repeat/under_threshold", broken_direct_repeat(seed=99, repeats=2), False),
        # Stagnation with only 3 fails — trips on direct_repeat before
        # stagnation has enough samples (so trips, not passes).
        ("stagnation/under_threshold", broken_stagnation(seed=99, fails=3), True),
    ]
    edges = list(edges)

    yield from legitimate
    yield from broken
    yield from edges


def scaled_dataset_size() -> int:
    return sum(1 for _ in iter_scaled_dataset())


# ---------------------------------------------------------------------
# scenario registry


SCENARIOS: Sequence[str] = (
    "legitimate_polling",
    "legitimate_pagination",
    "legitimate_retry",
    "legitimate_long_loop",
    "legitimate_progress_phase",
    "broken_direct_repeat",
    "broken_n_cycle",
    "broken_stagnation",
    "broken_stuck_after_recovery",
)


GENERATORS = {
    "legitimate_polling": legitimate_polling,
    "legitimate_pagination": legitimate_pagination,
    "legitimate_retry": legitimate_retry,
    "legitimate_long_loop": legitimate_long_loop,
    "legitimate_progress_phase": legitimate_progress_phase,
    "broken_direct_repeat": broken_direct_repeat,
    "broken_n_cycle": broken_n_cycle,
    "broken_stagnation": broken_stagnation,
    "broken_stuck_after_recovery": broken_stuck_after_recovery,
}


# Expected outcome for each scenario: True = should trip the fuse.
EXPECTED_DEADLOCK: dict = {
    "legitimate_polling": False,       # allowed via allow_repeats
    "legitimate_pagination": False,    # unique args per call
    "legitimate_retry": False,         # transient, with progress implied
    "legitimate_long_loop": False,     # args change each round
    "legitimate_progress_phase": False,
    "broken_direct_repeat": True,
    "broken_n_cycle": True,
    "broken_stagnation": True,
    "broken_stuck_after_recovery": True,
}


# Default FuseConfig knobs for the benchmark. Each scenario gets its
# own config tweaks where appropriate.
def default_config(scenario: str):
    from trajectory_fuse.guard import CycleConfig, FuseConfig, RunBudget, StagnationConfig

    cfg = FuseConfig(
        cycle=CycleConfig(direct_repeat_threshold=3, cycle_min_length=2, cycle_max_length=6),
        stagnation=StagnationConfig(
            enabled=True, window=8, similarity_threshold=0.9, min_failures=4
        ),
        budgets=RunBudget(max_actions=200),
    )
    if scenario == "legitimate_polling":
        cfg.allow_repeats = {"poll_status": 200}
    elif scenario == "legitimate_pagination":
        cfg.allow_repeats = {"fetch_page": 200}
    elif scenario == "legitimate_retry":
        # http_get failures should not trip stagnation — bump similarity
        # threshold slightly so the fresh per-attempt diagnostic token
        # is enough to break the run.
        cfg.stagnation.similarity_threshold = 0.95
    elif scenario == "broken_stuck_after_recovery":
        # This one SHOULD trip — no allow_repeats override, the failures
        # accumulate to min_failures=4 and stagnation fires.
        pass
    return cfg


# ---------------------------------------------------------------------
# public helpers


def generate(scenario: str, *, seed: int = 0) -> List[dict]:
    """Generate the trajectory for ``scenario``."""
    return GENERATORS[scenario](seed=seed)


def to_records(actions: Iterable[dict], default_run_id: str = "bench") -> List[TrajectoryRecord]:
    """Convert a list of action dicts into :class:`TrajectoryRecord`."""
    out = []
    for i, a in enumerate(actions, 1):
        if a.get("is_progress"):
            out.append(
                TrajectoryRecord(
                    sequence=i,
                    run_id=default_run_id,
                    tool="<progress>",
                    args_hash="",
                    args_repr=str(a.get("args", ""))[:200],
                    result_repr=str(a.get("result", ""))[:200],
                    success=None,
                    is_progress=True,
                    extra={},
                )
            )
            continue
        out.append(
            TrajectoryRecord(
                sequence=i,
                run_id=default_run_id,
                tool=a["tool"],
                args_hash="",  # populated by replay if needed
                args_repr=json.dumps(a["args"], default=str, ensure_ascii=False)[:1000],
                result_repr=json.dumps(a["result"], default=str, ensure_ascii=False)[:1000]
                if not isinstance(a["result"], str)
                else str(a["result"])[:1000],
                success=a.get("success"),
                is_progress=False,
                extra={k: v for k, v in a.items() if k not in {"tool", "args", "result", "success", "is_progress"}},
            )
        )
    return out


def iter_dataset(scenarios: Sequence[str] = SCENARIOS, *, seed: int = 0) -> Iterator[tuple]:
    """Yield ``(scenario, actions, expected_deadlock)`` for each scenario."""
    for s in scenarios:
        yield s, generate(s, seed=seed), EXPECTED_DEADLOCK[s]


def write_dataset(out_path: Path, *, seed: int = 0) -> int:
    """Write every scenario as a separate JSONL file under ``out_path/<scenario>.jsonl``.

    Returns the total number of records written.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    total = 0
    for scenario, actions, _expected in iter_dataset(seed=seed):
        path = out_path / f"{scenario}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for a in actions:
                fh.write(json.dumps(a, default=str, ensure_ascii=False) + "\n")
        total += len(actions)
    return total


__all__ = [
    "EXPECTED_DEADLOCK",
    "GENERATORS",
    "SCENARIOS",
    "default_config",
    "generate",
    "iter_dataset",
    "to_records",
    "write_dataset",
]
