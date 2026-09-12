# trajectory-fuse

<p align="center">
<img width="1219" height="569" alt="image" src="https://github.com/user-attachments/assets/e2244610-6b90-41c0-b9b6-27cac2ea52ac" />
</p>

<p align="center">
  <a href="https://pypi.org/project/trajectory-fuse/"><img src="https://img.shields.io/pypi/v/trajectory-fuse.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/trajectory-fuse/"><img src="https://img.shields.io/pypi/pyversions/trajectory-fuse.svg" alt="Python versions"></a>
  <a href="https://github.com/aniketkarne/trajectory-fuse/actions"><img src="https://img.shields.io/github/actions/workflow/status/aniketkarne/trajectory-fuse/tests.yml?branch=main&label=tests" alt="Tests"></a>
  <a href="https://github.com/aniketkarne/trajectory-fuse/blob/main/LICENSE"><img src="https://img.shields.io/github/license/aniketkarne/trajectory-fuse.svg" alt="License: MIT"></a>
  <a href="https://pepy.tech/project/trajectory-fuse"><img src="https://static.pepy.tech/badge/trajectory-fuse" alt="Downloads"></a>
</p>

<p align="center">
  <b>203 tests</b> pass in <b>0.91s</b> on Python 3.9–3.12. The bundled <code>--demo</code> runs in <b>3ms</b> with zero network, zero LLM, and zero optional deps.
</p>

<p align="center">
  <b>0.0% false-positive rate</b> across <b>456 realistic trajectories</b> (polling, pagination, retries, long loops, progress boundaries). The fuse trips every broken trajectory it should and saves <b>1,004 wasted tool calls</b> per benchmark run. See <a href="#efficacy-results">Efficacy results</a>.
</p>


> **A circuit breaker for LLM agent tool-call loops.**
> Catches the loop *before* it costs you another dollar.

```text
┌─────────────────────────── your agent loop ───────────────────────────┐
│                                                                      │
│   action = policy(state)                                             │
│   try:        result = tools[action.tool](action.args)               │
│   except:     result = {"error": …};  action.success = False         │
│   action.result = result                                             │
│                                                                      │
│        ┌─────────────────── AgentFuse ───────────────────┐            │
│        │  preflight  ─►  check(tool, args)              │            │
│        │                  ↓ if Detection:               │            │
│        │                  raise DeadlockDetected  ◄─────┼── CIRCUIT   │
│        │                                              │    BREAKER  │
│        │  post-call ─►  observe(action)                │            │
│        │                  ├─ direct_repeat             │            │
│        │                  ├─ n_cycle                   │            │
│        │                  └─ semantic_stagnation       │            │
│        │                  ↓ if Detection:               │            │
│        │                  raise DeadlockDetected       │            │
│        │                  ↳ progress_callback? swallow │            │
│        │                  ↳ allow_repeats? swallow     │            │
│        │                                              │            │
│        │  budgets  ─►  max_actions / runtime / tools   │            │
│        │  stats    ─►  calls_allowed/blocked/avoided   │            │
│        │  privacy  ─►  no LLM, no network, in-process │            │
│        └──────────────────────────────────────────────┘            │
│                                                                      │
│   except DeadlockDetected as exc:                                    │
│       log(exc.detection.kind, exc.detection.message, exc.steering_hint)
└──────────────────────────────────────────────────────────────────────┘
```

`trajectory-fuse` is a small, **zero-dependency**, **fully-local** runtime guard for the
tool-call trajectory of an LLM agent. It is not an LLM framework and it never
calls your model — it watches the actions your agent emits and raises a
structured exception when those actions start looping.

The 0.2 release adds the missing pieces that turn it from "a watchdog you
poll after the fact" into **a circuit breaker that refuses the next call**:

* **preflight check** (`fuse.check(tool, args)`) — predict the next deadlock
  *before* invoking the tool, and block the call without spending the
  budget that the loop was about to burn;
* **`@fuse.tool` decorator** — wrap a tool once and get preflight +
  post-call observation + automatic action bookkeeping for free;
* **retry-safe defaults** — the stagnation detector is conservative on
  purpose, and the new **`allow_repeats`** per-tool override lets you
  declare a polling tool legitimate (HTTP poll, paginated fetch,
  idempotent retry);
* **`progress_callback`** — a hook for the caller to *steer* without
  *aborting*: receive the detection, push a "try a different approach"
  hint back to the model, and let the loop continue;
* **hard budgets** — `RunBudget` caps `max_actions`, `max_runtime_seconds`,
  `max_tool_calls`, raising `BudgetExceeded` cleanly;
* **`FuseStats`** — in-process counters (calls observed / allowed /
  blocked / deadlocks / time-avoided) so you can see how often the fuse
  actually paid for itself.

## Immediate demo

You can see the whole story — preflight blocking, polling tolerated,
budgets tripping, stats accumulating — in **under 50 milliseconds** with
no network calls and no model:

```bash
pip install trajectory-fuse
trajectory-fuse demo
```

Or, in a source checkout, the same script runs directly:

```bash
python examples/demo_circuit_breaker.py
```

Output (trimmed):

```text
[init] fuse ready; window=32, max_actions=50, stagnation_min_failures=2

[scenario 1] polling 'search' 25 times in a row
  ok — 25 calls allowed, 0 blocked

[scenario 2] weather endpoint failing repeatedly
  call 0: tool ran
  call 1: tool ran
  call 2: BLOCKED by fuse → direct_repeat
  call 3: BLOCKED by fuse → direct_repeat
  …
  preflight blocked 4 of 6 calls

[scenario 3] action budget trips after 10 calls
  tripped after 10 actions (kind=actions, limit=10)

[final] fuse.stats: {'calls_observed': 27, 'calls_allowed': 27,
                     'calls_blocked': 0, 'deadlocks': 0, …}
          with time_avoided: {…, 'time_avoided_seconds': 0.0}
```

That last `time_avoided_seconds` is set by the caller — see
[Stats and time-avoided](#stats-and-time-avoided).

## 60-second quickstart

```python
from trajectory_fuse import AgentFuse, DeadlockDetected, FuseConfig

fuse = AgentFuse(FuseConfig(window=20))

while not done:
    action = policy(state)            # your code
    try:
        action.result = tools[action.tool](action.args)
        action.success = True
    except Exception as exc:
        action.result = {"error": repr(exc)}
        action.success = False

    try:
        fuse.observe(action)
    except DeadlockDetected as exc:
        log.warning("deadlock %s: %s", exc.detection.kind.value,
                    exc.detection.message)
        break
```

If you'd rather *not* write the preflight/observe/exception dance by
hand, see the [wrapper API](#wrapper-api-fusetool) below.

## Why a circuit breaker?

The textbook agent loop — `while not done: action = llm(...)` — has one
structural flaw: the loop's *only* exit condition is `done`, but `done`
is set by the same model that produced the action. When the model gets
stuck (rate-limited tool, oscillating plan, hallucinated retry) the
loop becomes unbounded. Each iteration costs a model call *and* a tool
call, so the dollar counter moves monotonically upward while nothing
real is happening.

`trajectory-fuse` does three things about this:

1. **Detects** the patterns that mean "this trajectory is not making
   progress" — direct repeats, N-state cycles, and *semantic* stagnation
   (the same tool failing with the same error string over and over).
2. **Refuses** the next call once detection trips — the `@fuse.tool`
   decorator runs a preflight `check()` and raises before the tool is
   invoked. *No tool cost, no LLM cost, no latency.*
3. **Reports** what it saw — a structured `Detection` on the
   `DeadlockDetected` exception, an optional steering hint, and an
   in-process `FuseStats` so you can see the savings over the lifetime
   of the run.

It is intentionally not a full agent framework: no prompt construction,
no tool routing, no LLM SDK. It is the missing *guard rail* that sits
between your existing tools and your existing loop.

## Wrapper API: `@fuse.tool`

The hand-rolled quickstart works, but it's a lot of plumbing. The 0.2
release ships an ergonomic decorator that does preflight + post-call +
exception capture automatically. Both names work and are equivalent:

```python
from trajectory_fuse import AgentFuse, fuse_namespace

fuse = AgentFuse()
guard = fuse_namespace(fuse)       # or just use `fuse.tool` directly

@guard(name="search", allow_repeats=100)   # tolerate 100 identical polls
def search(query: str) -> str:
    return http_get(f"https://example.com/?q={query}")

@guard(name="think")                        # exact same as @fuse.tool
def think(prompt: str) -> str:
    return f"reasoning about: {prompt}"

@guard                                      # @fuse.tool(...) and @guard(...)
def flaky_weather(city: str) -> str:        # both are the same primitive
    return api.get(city)
```

Every wrapped tool runs the same four-step sequence:

1. **Preflight** — `fuse.check(tool, args)`. If a detector would fire on
   the synthetic action, raise `DeadlockDetected` *without* invoking the
   tool.
2. **Invoke** — call the wrapped callable. Exceptions are captured into
   `action.success = False`, `action.result = {"error": repr(exc)}`, and
   re-raised to the caller.
3. **Post-call** — `fuse.observe(action)` records the action and runs
   the same detectors on the *actual* observed action.
4. **Allowlist / overrides** — direct repeats that exceed
   `allow_repeats[tool]` are swallowed; semantic stagnation is handed
   to `progress_callback` if configured.

The decorator works for sync and async functions identically
(`asyncio.run(guard(fetch))("http://...")`).

## Preflight and post-action: the circuit-breaker guarantee

A detector that only fires *after* the loop has already paid for the
call is half a circuit breaker. The 0.2 `check()` API closes that gap:

```python
det = fuse.check("weather", {"city": "nyc"})
if det is not None:
    raise DeadlockDetected(det.message, detection=det, trajectory=fuse.history)
# …otherwise invoke the tool and observe() the result.
```

`check()` is **non-mutating** — it runs every detector against a
synthetic action appended to a copy of the sliding window. It does not
advance the budget, does not write to the store, and does not touch
`fuse.stats`. Calling it from a hot loop is safe and cheap.

This is what the `@fuse.tool` decorator wraps automatically: every
guard call begins with a preflight check, so when a deadlock is
*imminent*, the next call never reaches your model *or* your tool. The
demo makes this visible — scenario 2 shows the underlying function
running twice, then being blocked on the third attempt before any tool
code executes.

## Retry-safe defaults

A circuit breaker that punishes a 429-retry is worse than no circuit
breaker. The defaults are tuned so that:

* `direct_repeat_threshold=3` — three *identical* calls in a row before
  the fuse trips. Standard retry/backoff (one extra attempt after a
  failure) is well below this.
* `stagnation.similarity_threshold=0.9` — Jaccard over the result
  tokens; *near*-identical, not bit-for-bit identical, so a real
  transient-retry error that adds a fresh diagnostic token doesn't trip
  the detector.
* `stagnation.min_failures=4` — needs four consecutive failures on the
  *same* tool before stagnation fires.

If you lower these thresholds, double-check your retry layer is
*actually* marking retries with `success=False` and a fresh diagnostic
token. A retry loop that always returns the same string will look
stagnating by design.

### Legitimate repetition: `allow_repeats`

Some tools *should* repeat — HTTP polling for a status, paginated API
walks, transactional retries. The `allow_repeats` per-tool override
says "this many identical calls in a row is fine":

```python
FuseConfig(
    cycle=CycleConfig(direct_repeat_threshold=3),   # default
    allow_repeats={
        "poll_status":  100,    # up to 100 identical polls
        "fetch_page":   500,    # up to 500 identical page reads
        # tools not listed fall back to CycleConfig.direct_repeat_threshold
    },
)
```

The override is a *trip threshold*: N identical calls pass, the (N+1)-th
trips the fuse. Tools not listed in `allow_repeats` are unaffected and
use the global threshold.

### Marking progress: `mark_progress(...)`

Long-running loops sometimes legitimately do the same thing twice in a
row before moving on (read → search → read → summarise). Use
`fuse.mark_progress(ProgressSignal(token=...))` to declare a phase
boundary; the stagnation detector will only consider actions recorded
*after* the mark. Direct-repeat and cycle detection ignore progress
marks — those are pure structural checks and shouldn't be paused by
external signals.

```python
fuse.mark_progress(ProgressSignal(token="new-env", note="after tool reset"))
```

The mark is persisted as a synthetic `<progress>` row in the SQLite
store so replays see the same boundary.

## Stats and time-avoided

`fuse.stats` is a live `FuseStats` snapshot:

| field                    | meaning                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `calls_observed`         | every action passed to `observe()` (including allowlisted)              |
| `calls_allowed`          | actions that did *not* trip a detector                                  |
| `calls_blocked`          | actions that raised `DeadlockDetected`                                  |
| `deadlocks`              | number of unique deadlock events raised                                 |
| `progress_marks`         | `mark_progress(...)` invocations                                        |
| `time_avoided_seconds`   | caller-supplied estimate of seconds saved by aborting                   |
| `first_deadlock_at`      | 0-indexed window position of the first deadlock, or `None`              |

`time_avoided_seconds` is the only field that requires cooperation
from the caller — the library can't know how long the model would
have kept looping. Populate it after catching `DeadlockDetected`:

```python
try:
    fuse.observe(action)
except DeadlockDetected as exc:
    # Conservative estimate: ~3 s per blocked call (LLM + tool).
    fuse.record_time_avoided(seconds=3.0)
    break
```

`stats.reset()` is called by `fuse.reset()` and is cheap.

## Recovery hooks

Sometimes the host application wants to *observe* a detected deadlock
and decide for itself whether to raise or to swallow it and let the
loop continue. The 0.3 release ships a tiny recovery API:

```python
from trajectory_fuse import (
    AgentFuse,
    FuseConfig,
    RecoveryPolicy,
    on_stagnation,
)

def hint_to_model(action, detection):
    # Pretend we talk to an LLM and ask for a recovery plan.
    plan = ask_llm_for_recovery(detection.message)
    if "switch tool" in plan:
        return "continue"   # swallow; loop keeps going
    return "raise"          # raise the original DeadlockDetected

policy = RecoveryPolicy(
    on_stagnation=on_stagnation(hint_to_model),
    on_direct_repeat=lambda a, d: "raise",  # direct repeats are not recoverable
    default_disposition="raise",
)
fuse = AgentFuse(FuseConfig(window=20), recovery_policy=policy)
```

A hook receives the `Action` that triggered detection and the
`Detection` describing why. It returns one of:

* `"raise"` — let the fuse raise `DeadlockDetected` as usual.
* `"continue"` — swallow the exception, record a `recovery_skip`
  in `fuse.stats`, return normally.
* `None` — "I have no opinion"; fall through to the policy's
  `default_disposition`.

A misbehaving hook (raises an exception) is swallowed by the fuse —
hooks must not break the loop. The recovery API sits *after* the
existing `progress_callback` (which only handles semantic-stagnation
without aborting) and *after* `allow_repeats` (which only handles
legitimate direct repeats). Together they form a layered policy:

1. Allowlisted tools skip every detector.
2. Direct repeats inside `allow_repeats[tool]` are swallowed.
3. Stagnation inside `progress_callback` is swallowed.
4. **Any** remaining detection goes through the recovery policy.
5. If the recovery policy returns `"raise"` (or no policy is set),
   the fuse raises.

Combine multiple hooks with `Chain`:

```python
from trajectory_fuse import Chain

policy = Chain([
    on_stagnation(expensive_llm_recovery_hook),
    on_stagnation(cheap_local_recovery_hook),  # fallback
])
```

`fuse.stats.recovery_skips` increments each time the policy swallows a
deadlock, alongside the existing `calls_blocked` / `deadlocks`
counters. Use these three to see, in production, how often the fuse
actually pays for itself — and how often your recovery layer is
letting loops continue safely.

## Framework integrations

The `trajectory_fuse.integrations` package ships adapters for the
three agent frameworks we see in the wild:

| Framework | Install | Factory |
|-----------|---------|---------|
| Generic Python (no framework) | always available | `from trajectory_fuse.integrations.generic import observe_call, guard_scope, make_observer` |
| LangGraph | `pip install trajectory-fuse[langgraph]` | `from trajectory_fuse.integrations import langgraph_middleware` |
| OpenAI Agents SDK | `pip install trajectory-fuse[openai-agents]` | `from trajectory_fuse.integrations import openai_agents_middleware` |
| PydanticAI | `pip install trajectory-fuse[pydantic-ai]` | `from trajectory_fuse.integrations import pydantic_ai_middleware` |
| All three | `pip install trajectory-fuse[all-integrations]` | — |

Each adapter factory takes an `AgentFuse` and returns a framework-native
middleware / hook object. If the framework isn't installed, the
factory raises `FrameworkNotInstalled` with the exact `pip install`
hint:

```python
from trajectory_fuse import AgentFuse, FuseConfig
from trajectory_fuse.integrations import langgraph_middleware

fuse = AgentFuse(FuseConfig(window=20))
try:
    mw = langgraph_middleware(fuse)
except FrameworkNotInstalled as exc:
    print(exc.install_hint)   # "Install it with `pip install trajectory-fuse[langgraph]` ..."
```

The framework adapters import their target framework lazily — having
`trajectory-fuse` installed does not pull in `langgraph` /
`openai-agents` / `pydantic-ai` unless you opt in via the
corresponding extra. The adapters are best-effort shims over the
framework's most stable hook shape; when the framework APIs drift,
the adapter fails at *import time*, not at agent execution time.

The **generic adapter** is what you reach for when you're not using
one of those frameworks:

```python
from trajectory_fuse.integrations.generic import observe_call, guard_scope

fuse = AgentFuse(FuseConfig(window=20, allow_repeats={"poll_status": 100}))

# Wraps a single tool with preflight + observe, sync or async.
poll = observe_call(fuse, "poll_status", http_get)

# Or use a context manager to scope a fuse over a block of code.
with guard_scope(fuse, enter_action="task_start", exit_action="task_end") as f:
    for action in policy_loop(state):
        f.observe(action)
```

## Efficacy results

A circuit breaker that misfires on legitimate workloads is worse
than no circuit breaker. The 0.3 release ships a **456-trajectory**
efficacy benchmark (`benchmarks/run_efficacy.py --scaled`) that
exercises every detector against realistic shapes:

* **Legitimate** trajectories the fuse must NOT trip on:
  HTTP polling (50 identical calls until status changes),
  paginated fetches (30 pages with new args), transient 429 retries
  (5 attempts with fresh backoff / diagnostic tokens per attempt),
  long search→summarise loops (20 rounds × 2 calls), and
  progress-phase boundaries (5 failing retries + `mark_progress`
  + 5 succeeding retries).
* **Broken** trajectories the fuse MUST trip on:
  direct-repeat (6 identical calls), N-cycle (A→B→A→B),
  semantic stagnation (6 near-identical errors), and stuck-after-recovery
  (8 identical failures the model tries again to no avail).
* **Edge cases**: tiny retries (2 attempts), at-threshold repeats (3
  identical calls), under-threshold repeats (2), past-threshold (4).

Run it:

```bash
python benchmarks/run_efficacy.py --scaled --strict --json
```

Captured on an Apple M-series machine, Python 3.12, default
`FuseConfig` plus per-scenario `allow_repeats` for the legitimate
cases:

```text
trajectories                 456
true_positives               203
false_positives              0
true_negatives               253
false_negatives              0
precision                    1.0
recall                       1.0
false_positive_rate          0.0
total_calls_without_fuse     8269
total_calls_with_fuse        7265
calls_avoided                1004
avoided_fraction             0.121
```

What this means in practice:

* **Every broken trajectory tripped.** The detectors caught all
  203 broken-shape cases.
* **No false positives.** The 253 legitimate trajectories ran to
  completion; the fuse did not abort a single one.
* **1,004 tool calls saved** per benchmark run, across the broken
  scenarios. The fuse tripped at the first opportunity on each,
  avoiding the ~3 extra calls per broken trajectory that an unguarded
  agent would have made before someone (or something) noticed.
* **12.1% call reduction** on the broken-trajectory workload — every
  saved call is one fewer LLM round-trip and one fewer tool
  invocation your model never has to pay for.

The pytest contract (`benchmarks/test_efficacy.py`) asserts precision
≥ 0.99, recall ≥ 0.99, FPR ≤ 1%, and avoided fraction ≥ 5% on every
CI run. A regression in any detector fails the build loudly.

## Budgets

`FuseConfig(budgets=RunBudget(...))` enforces hard ceilings:

```python
from trajectory_fuse import RunBudget

FuseConfig(
    budgets=RunBudget(
        max_actions=200,            # total observe() calls (incl. allowlisted)
        max_runtime_seconds=300.0,  # wall-clock since fuse construction
        max_tool_calls=50,          # non-allowlisted only
    )
)
```

A budget that trips raises `BudgetExceeded(kind, limit, observed)` —
distinct from `DeadlockDetected` so callers can tell "ran out of
budget" apart from "the trajectory went bad". All three fields default
to `0` / `None`, which means "off" — the budgets you don't set cost
nothing.

## Privacy

`trajectory-fuse` is intentionally boring on this axis:

* **no LLM calls** — the library never talks to a model.
* **no network calls** — the library never opens a socket.
* **no external dependencies at runtime** — stdlib only.
* **persistence is opt-in** — `FuseConfig(store_factory=...)` opens a
  SQLite file; without it, nothing leaves the process.
* **values, not references** — the persistence layer serialises tool
  args/results to JSON via `default=str` and hashes the canonical JSON;
  it never holds Python objects or closures.
* **escape-safe renderers** — HTML / SVG / Mermaid export sanitises
  tool names before embedding.

If you want to wire it into a multi-tenant service, every persisted row
carries an explicit `run_id` you control — set it to your request id
and you can scope retention accordingly.

## CLI

```bash
trajectory-fuse demo                 # run the bundled circuit-breaker demo
trajectory-fuse analyse traj.jsonl   # replay a saved trajectory, report first deadlock
trajectory-fuse replay  traj.jsonl   # stream into a live fuse, no raise
trajectory-fuse export  traj.jsonl --out report.html
trajectory-fuse export  traj.jsonl --out timeline.svg
trajectory-fuse export  traj.jsonl --out graph.md
trajectory-fuse stats   traj.jsonl   # counts, success rate, unique tools
trajectory-fuse hash    '{"q": 1}'   # canonical hash of a JSON literal
```

`analyse` accepts the same tuning knobs as `FuseConfig` —
`--window`, `--direct-repeat`, `--cycle-min`, `--cycle-max`,
`--stagnation-window`, `--stagnation-threshold`,
`--stagnation-min-failures`, `--no-stagnation`, `--allowlist`. Add
`--json` for machine-readable output.

## Benchmarks

Runtime overhead is visible before adoption. The benchmark harness is
stdlib-only and the scripts ship in the wheel:

```bash
# stdlib harness (no extra deps)
trajectory-fuse demo               # smoke: <100 ms total, see Immediate demo
python3 benchmarks/run_benchmarks.py        # full benchmark
python3 benchmarks/run_benchmarks.py --strict   # fail on regressions

# pytest-benchmark harness (optional)
pip install pytest-benchmark
python3 -m pytest benchmarks/test_benchmarks.py --benchmark-only
```

Baseline on an Apple M-series machine, Python 3.12, default `FuseConfig`:

```text
scenario                       rounds  samples  per-call µs (min/median/mean)
observe_normal                 2000        5    72.32 /    72.89 /    72.88
observe_with_store              500        5   383.42 /   396.00 /   397.29
direct_repeat_detect           5000        5     6.59 /     6.62 /     6.62
cycle_detect                   2000        5     9.42 /     9.42 /     9.44
stagnation_detect              2000        5     4.64 /     4.65 /     4.65
canonical_hash                50000        5     3.01 /     3.01 /     3.01
sqlite_persist                 2000        5   312.99 /   317.31 /   317.81
```

`observe()` on the default 32-action window is sub-200 µs; with a
SQLite store it is dominated by the commit (~440 µs). The detectors
themselves are single-digit microseconds. These are reference points
captured on the development machine — your numbers will differ. Use
`--strict` to fail the run if any scenario exceeds the ceilings in
`benchmarks/BASELINE.json` (overridable when you intentionally change
the implementation).

## Detection modes

| mode                    | trigger                                                              | configurable via                          |
|-------------------------|----------------------------------------------------------------------|-------------------------------------------|
| `direct_repeat`         | identical `(tool, args)` repeated N times in a row                   | `CycleConfig.direct_repeat_threshold`     |
| `n_cycle`               | repeating cycle of length N (2 ≤ N ≤ `cycle_max_length`)             | `CycleConfig.cycle_min_length` / `_max`   |
| `semantic_stagnation`   | K consecutive failures on the same tool produce near-identical tokens | `StagnationConfig.*`                       |

The detectors run in this priority order on every `observe()` and on
every `check()`; the first match wins. Each mode can be independently
disabled (`direct_repeat_threshold<2`, `cycle_min_length<2`,
`stagnation.enabled=False`).

## Configuration reference

### `FuseConfig`

| field                   | type                              | default | notes                                                                                |
|-------------------------|-----------------------------------|---------|--------------------------------------------------------------------------------------|
| `window`                | `int`                             | `32`    | sliding-window size (FIFO eviction)                                                  |
| `cycle`                 | `CycleConfig`                     | see below| cycle & direct-repeat detection                                                      |
| `stagnation`            | `StagnationConfig`                | see below| semantic-stagnation detection                                                        |
| `budgets`               | `RunBudget`                       | all off | hard ceilings on the run                                                             |
| `store_factory`         | `Callable[[], TrajectoryStore]`   | `None`  | set to enable persistence; factory is invoked once at construction time              |
| `run_id`                | `Optional[str]`                   | `None`  | identifier attached to every persisted row; UUID4 if absent                          |
| `allowlist`             | `Sequence[str]`                   | `()`    | tools whose calls never trigger any detector (still recorded)                        |
| `allow_repeats`         | `dict[str, int]`                  | `{}`    | per-tool "this many identical calls in a row are fine" overrides                     |
| `progress_callback`     | `Callable[[Action, Detection], None]` | `None` | opt out of raising for semantic-stagnation events                                    |
| `steering_hook`         | `Callable[[List[Action], Detection], Optional[str]]` | `None` | adds a hint string to `DeadlockDetected.steering_hint`                     |

### `CycleConfig`

| field                     | default | notes                                                                  |
|---------------------------|---------|------------------------------------------------------------------------|
| `direct_repeat_threshold` | `3`     | `< 2` disables. `1` disables both direct-repeat and 1-state cycles.    |
| `cycle_min_length`        | `2`     | `< 2` disables cycle detection entirely.                              |
| `cycle_max_length`        | `6`     | largest cycle to look for. `< cycle_min_length` disables.             |

### `StagnationConfig`

| field                  | default | notes                                                                                       |
|------------------------|---------|---------------------------------------------------------------------------------------------|
| `enabled`              | `True`  | master switch for the stagnation subsystem                                                  |
| `window`               | `8`     | trailing actions inspected for token similarity                                             |
| `similarity_threshold` | `0.9`   | Jaccard threshold in `[0, 1]`. `1.0` = exact token-set equality                             |
| `min_failures`         | `4`     | minimum consecutive failures on the same tool before stagnation can fire. `< 2` disables    |

### `RunBudget`

| field                  | default | notes                                                                                       |
|------------------------|---------|---------------------------------------------------------------------------------------------|
| `max_actions`          | `0`     | `0` disables. trips after the Nth `observe()` call.                                          |
| `max_runtime_seconds`  | `None`  | `None` disables. uses `time.monotonic()`.                                                   |
| `max_tool_calls`       | `0`     | `0` disables. counts only non-allowlisted tool calls.                                       |

### CLI flags

```
trajectory-fuse demo
trajectory-fuse analyse INPUT.jsonl [--db PATH] [--run-id ID] [--window N]
                            [--direct-repeat N] [--cycle-min N] [--cycle-max N]
                            [--stagnation-window N] [--stagnation-threshold F]
                            [--stagnation-min-failures N] [--no-stagnation]
                            [--allowlist TOOL …] [--json]

trajectory-fuse export  INPUT.jsonl --out OUT.{html,svg,md} [--run-id ID]
                            [--no-progress]

trajectory-fuse replay  INPUT.jsonl [--window N] [--direct-repeat N]
                            [--allowlist TOOL …] [--quiet]

trajectory-fuse stats   INPUT.jsonl [--json]

trajectory-fuse hash    JSON_LITERAL
```

## JSONL format

Two shapes are accepted; both normalise into the canonical
`TrajectoryRecord`:

```json
{"tool": "search", "args": {"q": "weather"}, "result": "sunny", "success": true}
{"tool": "search", "args": {"q": "weather"}, "error": "timeout", "success": false}
```

The loader normalises both into the canonical `TrajectoryRecord` shape
(`sequence`, `run_id`, `tool`, `args_hash`, `args_repr`, `result_repr`,
`success`, `is_progress`, `extra`). The CLI exporter uses `args_hash` to
deduplicate visually.

## Exception shape

```python
DeadlockDetected(
    message="Tool 'search' invoked with identical arguments 3 times in a row. [hint: …]",
    detection=Detection(
        kind=DetectionKind.DIRECT_REPEAT,        # or N_CYCLE / SEMANTIC_STAGNATION
        message="…",
        cycle=["search", "search", "search"],
        similarity=None,                          # set for stagnation
        details={"count": 3, "args_hash": "…"},
    ),
    trajectory=[Action(...), …],                # snapshot, most recent last
    steering_hint="Try a different approach …",  # hook output, may be None
)
```

`BudgetExceeded` is a `RuntimeError` with `(kind, limit, observed)`
attributes and a structured message.

## Programmatic visualisation

```python
from trajectory_fuse.export import render_html, render_mermaid, render_svg
from trajectory_fuse.loader import load_jsonl

records = load_jsonl("traj.jsonl")
html = render_html(records, run_id="abc")           # standalone HTML+inline SVG
svg  = render_svg(records)                          # raw SVG document
md   = f"```mermaid\n{render_mermaid(records)}```"  # Mermaid graph
```

All renderers sanitise user-supplied tool names so they're safe to
embed in Mermaid / HTML / SVG without escaping your shell.

## Installation

```bash
pip install trajectory-fuse
```

Optional extras for framework integrations:

```bash
pip install trajectory-fuse[langgraph]            # LangGraph
pip install trajectory-fuse[openai-agents]        # OpenAI Agents SDK
pip install trajectory-fuse[pydantic-ai]          # PydanticAI
pip install trajectory-fuse[all-integrations]     # all three
```

Or to hack on it:

```bash
git clone https://github.com/aniketkarne/trajectory-fuse
cd trajectory-fuse
pip install -e .[dev]
pytest                            # 203 tests
python3 -m pytest -q              # same thing, quieter
python benchmarks/run_efficacy.py --scaled --strict --json   # efficacy numbers
```

Requires Python **3.9+**. Zero runtime dependencies.

## Limitations

A few honest notes on what `trajectory-fuse` is and isn't:

* **In-process only.** The fuse guards a *single* Python process. If
  your agent spans multiple workers (Ray, Celery, an HTTP service), each
  process needs its own `AgentFuse` instance. Persistence is shared via
  the `store_factory` callback if you want cross-process analysis.
* **Detectors are heuristic.** Jaccard over tokens is fast and good
  enough for the common case ("same 429 message five times in a row"),
  but a sufficiently clever model can produce *new* tokens each retry
  while still making no progress. The conservative defaults exist to
  make this rare rather than impossible.
* **Not a retry policy.** `allow_repeats` tolerates repetition; it
  doesn't *bound* it. If you need "at most 100 HTTP polls in 60 s,
  then fail", that's a job for your retry layer, not this library.
* **No LLM calls.** There is no built-in "ask the model whether this
  is really a loop" path. If you want one, the `progress_callback`
  hook is the seam — receive the detection, push it to the model,
  decide whether to raise.
* **No tool routing.** We don't pick tools for you. `fuse.tool` only
  decorates; it doesn't schedule.
* **Synchronous core.** The decorator supports async functions, but the
  detectors themselves are sync. If your agent runs across threads,
  use one fuse per thread or guard the fuse with a lock — the sliding
  window isn't thread-safe.

## Development

```bash
pytest                            # 203 tests
pytest --cov=trajectory_fuse            # with coverage (requires pytest-cov)
python3 benchmarks/run_benchmarks.py
python3 benchmarks/run_efficacy.py --scaled --strict --json
```

Tests cover:

* canonical JSON hashing (key order, container type, unicode, type
  rejection),
* direct-repeat detection (threshold, allowlist, window eviction,
  per-tool override, args),
* N-cycle detection (length 2 / length 3, args must match, disable
  modes),
* semantic-stagnation (success resets, tool switch resets, progress
  mark, similarity threshold, all disable paths),
* preflight (`check`, `check_action`, no mutation, no budget
  consumption),
* budgets (`max_actions`, `max_runtime_seconds`, `max_tool_calls`,
  allowlist exemption, disable paths),
* `@fuse.tool` decorator (sync, async, preflight block, exception
  capture, `__name__` / `__doc__` preservation),
* `progress_callback` (swallow stagnation, explicit re-raise,
  exception swallowed, `KeyboardInterrupt` not swallowed),
* recovery hooks (`RecoveryPolicy`, `Chain`, `on_kind`,
  `recovery_skips` counter, misbehaving-hook swallowing),
* generic adapter (`observe_call` sync/async, `guard_scope`,
  `make_observer`),
* framework adapters (lazy-import, `FrameworkNotInstalled`
  with install hint),
* SQLite store (in-memory, on-disk, persistence integration, progress
  marks),
* CLI (`analyse`, `export`, `replay`, `stats`, `hash`, `demo`),
* exporters (HTML / SVG / Mermaid, escaping, empty trajectories),
* `FuseStats` (defaults, mutation, `as_dict`, `reset`,
  `record_time_avoided`, `recovery_skips`),
* efficacy contract (456-trajectory dataset, precision ≥ 0.99,
  recall ≥ 0.99, FPR ≤ 1%, avoided fraction ≥ 5%).

## License

MIT. See [LICENSE](LICENSE).
