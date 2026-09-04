# agent-fuse

> **In-process runtime guard for agent tool-call loops.**
> Detects deadlocks, semantic stagnation, and repeating cycles before they cost you another dollar.

```text
       ┌──────────────────────────────────────────────────────────────┐
       │  Agent loop (your code)                                      │
       │                                                              │
       │   while not done:                                            │
       │       action = policy(state)                                 │
       │       result = tools[action.tool](action.args)               │
       │       action.result = result                                 │
       │                                                              │
       │   ┌─────────────────────────────┐                            │
       │   │     AgentFuse.observe()     │                            │
       │   │                             │                            │
       │   │  1. canonical_hash(args)    │──► args_hash (16 hex)       │
       │   │  2. append to sliding win   │                            │
       │   │  3. _check_direct_repeat()  │                            │
       │   │  4. _check_cycle()          │──► (tool, hash) signatures │
       │   │  5. _check_stagnation()     │──► Jaccard over tokens     │
       │   │                             │                            │
       │   │  raise DeadlockDetected     │◄── Detection {kind, …}     │
       │   └─────────────────────────────┘                            │
       │           │                                                  │
       │           ▼                                                  │
       │   optional steering_hook(actions, detection) → hint string   │
       │                                                              │
       │  ┌─────────────────────────────┐    ┌─────────────────────┐  │
       │  │     TrajectoryStore         │    │     CLI             │  │
       │  │     (SQLite, append-only)   │    │  analyse / export / │  │
       │  │     run_id + sequence + …   │    │  replay / stats /   │  │
       │  │                             │    │  hash               │  │
       │  └─────────────────────────────┘    └─────────────────────┘  │
       └──────────────────────────────────────────────────────────────┘
```

## Why?

LLM-driven agent loops frequently:

* **hammer the same tool** with identical arguments until the rate limit trips,
* **oscillate** between two or three tools in a tight cycle (e.g. *search → summarise → search → summarise …*),
* **drown in repeated failures** that look superficially different but actually say the same thing.

`agent-fuse` watches the live trajectory, raises a structured exception when any of these patterns cross your threshold, and persists everything to SQLite so you can analyse it offline.

## Installation

```bash
pip install agent-fuse
```

Or, to hack on it:

```bash
git clone https://github.com/aniketkarne-com/agent-fuse
cd agent-fuse
pip install -e .[dev]
pytest
```

Requires Python **3.9+**. Zero runtime dependencies.

## 3-step quickstart

### 1 — Wrap your agent loop

```python
from agent_fuse import (
    Action,
    AgentFuse,
    DeadlockDetected,
    FuseConfig,
)

fuse = AgentFuse(FuseConfig(window=20))

while not done:
    action = policy(state)              # your code
    result = tools[action.tool](action.args)
    action.result = result
    action.success = True                # mark False on exceptions

    try:
        fuse.observe(action)
    except DeadlockDetected as exc:
        log.error("agent deadlock: %s", exc)
        break
```

### 2 — Add a steering hook (optional but recommended)

```python
def steer(actions, detection):
    return "Try a different approach — last few actions were unproductive."

fuse = AgentFuse(FuseConfig(), steering_hook=steer)
# On detection, DeadlockDetected.steering_hint contains the returned string.
```

### 3 — Persist, then export the trajectory

```python
from agent_fuse.store import TrajectoryStore

store = TrajectoryStore("traj.db")
fuse = AgentFuse(FuseConfig(store_factory=lambda: store))

# ... loop runs ...

store.close()
```

```bash
# Render an interactive timeline as standalone HTML
agent-fuse export traj.jsonl --out report.html

# Or as raw SVG / Mermaid for embedding in Markdown
agent-fuse export traj.jsonl --out timeline.svg
agent-fuse export traj.jsonl --out graph.md

# Replay a saved JSONL and detect deadlocks
agent-fuse analyse traj.jsonl --json
agent-fuse stats traj.jsonl
```

## Detection modes

| Mode                       | Trigger                                                            | Configurable via                      |
|----------------------------|--------------------------------------------------------------------|---------------------------------------|
| `direct_repeat`            | Identical `(tool, args)` repeated N times in a row                 | `CycleConfig.direct_repeat_threshold`  |
| `n_cycle`                  | Repeating cycle of length N detected (2 ≤ N ≤ cycle_max_length)    | `CycleConfig.cycle_min_length / _max` |
| `semantic_stagnation`      | K consecutive failures on the same tool produce near-identical tokens | `StagnationConfig.*`                  |

Each mode can be independently disabled. Detection runs in priority order on every `observe()` call — the first match wins.

## Configuration reference

### `FuseConfig`

| Field            | Type                  | Default                | Notes                                                                                |
|------------------|-----------------------|------------------------|--------------------------------------------------------------------------------------|
| `window`         | `int`                 | `32`                   | Sliding-window size (FIFO eviction).                                                  |
| `cycle`          | `CycleConfig`         | see below              | Cycle & direct-repeat detection.                                                     |
| `stagnation`     | `StagnationConfig`    | see below              | Semantic-stagnation detection.                                                        |
| `store_factory`  | `Callable[[], TrajectoryStore]` | `None`         | Set to enable persistence. The factory is invoked once at fuse construction time.     |
| `run_id`         | `Optional[str]`       | `None`                 | Identifier attached to every persisted row. UUID4 if absent.                         |
| `allowlist`      | `Sequence[str]`       | `()`                   | Tools whose calls never trigger any detector (still recorded).                        |

### `CycleConfig`

| Field                    | Default | Notes                                                                  |
|--------------------------|---------|------------------------------------------------------------------------|
| `direct_repeat_threshold`| `3`     | `< 2` disables. `1` disables both direct-repeat and 1-state cycles.    |
| `cycle_min_length`       | `2`     | `< 2` disables cycle detection entirely.                                |
| `cycle_max_length`       | `6`     | Largest cycle to look for. `< cycle_min_length` disables.               |

### `StagnationConfig`

| Field                  | Default | Notes                                                                                       |
|------------------------|---------|---------------------------------------------------------------------------------------------|
| `enabled`              | `True`  | Master switch for the stagnation subsystem.                                                 |
| `window`               | `8`     | Trailing actions inspected for token similarity.                                            |
| `similarity_threshold` | `0.9`   | Jaccard threshold in `[0, 1]`. `1.0` = exact token-set equality.                            |
| `min_failures`         | `3`     | Minimum consecutive failures on the same tool before stagnation can fire. `< 2` disables.   |

### CLI flags

```
agent-fuse analyse INPUT.jsonl [--db PATH] [--run-id ID] [--window N]
                            [--direct-repeat N] [--cycle-min N] [--cycle-max N]
                            [--stagnation-window N] [--stagnation-threshold F]
                            [--stagnation-min-failures N] [--no-stagnation]
                            [--allowlist TOOL …] [--json]

agent-fuse export  INPUT.jsonl --out OUT.{html,svg,md} [--run-id ID]
                            [--no-progress]

agent-fuse replay  INPUT.jsonl [--window N] [--direct-repeat N]
                            [--allowlist TOOL …] [--quiet]

agent-fuse stats   INPUT.jsonl [--json]

agent-fuse hash    JSON_LITERAL
```

## JSONL format

Two shapes are accepted:

```json
{"tool": "search", "args": {"q": "weather"}, "result": "sunny", "success": true}
{"tool": "search", "args": {"q": "weather"}, "error": "timeout", "success": false}
```

The loader normalises both into the canonical :class:`TrajectoryRecord` shape
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

## Programmatic visualisation

```python
from agent_fuse.export import render_html, render_mermaid, render_svg
from agent_fuse.loader import load_jsonl

records = load_jsonl("traj.jsonl")
html = render_html(records, run_id="abc")           # standalone HTML+inline SVG
svg  = render_svg(records)                          # raw SVG document
md   = f"```mermaid\n{render_mermaid(records)}```"  # Mermaid graph
```

All renderers sanitise user-supplied tool names so they're safe to embed in
Mermaid / HTML / SVG without escaping your shell.

## Development

```bash
pytest                          # 75 tests
pytest --cov=agent_fuse          # with coverage (requires pytest-cov)
```

Tests cover:

* canonical JSON hashing (key-order, container-type, unicode, type rejection),
* direct-repeat detection (threshold, allowlist, window eviction, args),
* N-cycle detection (length 2, length 3, args must match, disable modes),
* semantic-stagnation (success resets, tool switch resets, progress mark,
  similarity threshold, all disable paths),
* steering hook (return-value propagation, exception swallowing),
* SQLite store (in-memory, on-disk, persistence integration, progress marks),
* CLI (`analyse`, `export`, `replay`, `stats`, `hash`),
* exporters (HTML / SVG / Mermaid, escaping, empty trajectories).

## License

MIT. See [LICENSE](LICENSE).