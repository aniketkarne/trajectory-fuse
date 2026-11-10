# Changelog

All notable changes to `trajectory-fuse` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **Stagnation was silently disarmed after a progress mark left the
  window.**
  When a `mark_progress` anchor slid out of the sliding window, it was
  replaced with a sticky sentinel that shrank the post-mark slice to the
  single newest action — silently disabling semantic-stagnation detection
  for the rest of the run. Every action left in the window at that point
  was observed *after* the mark, so the mark is now simply dropped and the
  post-mark slice is the whole window. A stuck loop that starts long after
  a phase boundary is now caught (`tests/test_progress_mark_accounting.py`,
  plus a `stuck_mark_evicted` trajectory in the scaled efficacy corpus —
  it is a false negative against the old code).
- **`@fuse.tool` wrapper leaked one tool call past the `allow_repeats`
  trip threshold.** The pre-call circuit-breaker check consulted
  `allow_repeats` without counting the run length, so the (N+1)-th
  identical call executed before `observe()` raised — breaking the
  documented "never call a tool whose call we already know would trip the
  fuse" guarantee. The pre-call gate now counts the run exactly like
  `observe()` does (sync and async wrappers).
- **`load_from` did not shift progress marks when trimming the window.**
  Replayed marks drifted to stale positions (potentially past the end of
  the window), which masked or delayed stagnation on replayed
  trajectories; marks now track the trim, and a replayed marker points at
  the first post-mark action, matching `mark_progress()` semantics.

### Changed

- The scaled efficacy corpus grows to **457 trajectories** (adds
  `broken_stuck_after_mark_evicted`); measured on Python 3.12:
  precision 1.0, recall 1.0, FPR 0.0%, 1,007 calls avoided (12.1%).
- CI: the `efficacy` job now also runs `pytest benchmarks/test_efficacy.py`
  (per-family trip check + precision/recall/FPR gates) — previously those
  gates existed but were never collected (`testpaths` covers `tests/`).
- CI: `publish.yml` tag trigger accepts bare semver tags (`0.3.1`) as well
  as `v`-prefixed ones — the repo's existing tags are bare semver, so the
  publish-on-tag path had never fired.
- `test_window_eviction_with_progress_marks` rewritten: it pinned the old
  sentinel behaviour (stagnation must not fire post-eviction) where the
  correct contract is the opposite — post-eviction actions are all
  post-mark and three trailing identical failures must trip.

## [0.3.0] — 2026-09-12

### Added

- **Recovery hooks** (`trajectory_fuse.hooks`): pluggable extension point
  that lets the host application decide whether a detected deadlock
  should raise or be swallowed. Ships `RecoveryPolicy`,
  `Chain(RecoveryPolicy)`, and `on_loop` / `on_stagnation` / `on_deadlock` /
  `on_kind` shorthand constructors. Hooks return `"raise"`, `"continue"`,
  or `None`. New `recovery_policy` constructor arg on `AgentFuse`;
  new `recovery_skips` counter on `FuseStats`.
- **Framework integrations** (`trajectory_fuse.integrations`): zero-dep
  generic adapter + lazy-import adapters for **LangGraph**, the
  **OpenAI Agents SDK**, and **PydanticAI**. Each adapter raises a clear
  `FrameworkNotInstalled` with the exact `pip install` hint when its
  framework is missing. Install with
  `pip install trajectory-fuse[langgraph]`,
  `pip install trajectory-fuse[openai-agents]`,
  `pip install trajectory-fuse[pydantic-ai]`, or
  `pip install trajectory-fuse[all-integrations]`.
- **Efficacy benchmark** (`benchmarks/run_efficacy.py`,
  `benchmarks/efficacy_dataset.py`): realistic 456-trajectory dataset
  covering five legitimate shapes (polling, pagination, retry, long
  loop, progress-phase boundary) and four broken shapes (direct
  repeat, N-cycle, stagnation, stuck-after-recovery). Pytest
  assertions: precision ≥ 0.99, recall ≥ 0.99, FPR ≤ 1%, avoided
  fraction ≥ 5% — all met on this release (precision 1.0,
  recall 1.0, FPR 0.0%, 1004 calls saved across broken scenarios).
- **CI**: added `lint` (ruff) and `efficacy` (strict-mode efficacy
  benchmark) jobs to `.github/workflows/tests.yml`.
- **Ruff config** (`ruff.toml`): conservative ruleset scoped to the
  audit's new code paths; existing files exempt from the lint pass.
- **`CONTRIBUTING.md`**: dev setup, PR guidelines, test layout.

### Changed

- **`FuseStats.as_dict()`** now includes `recovery_skips`.
- **`pyproject.toml`** adds `ruff>=0.5` to the `dev` extra and exposes
  four new optional extras for framework integrations.
- **`benchmarks/BASELINE.json`** refreshed against Python 3.12 on
  Apple silicon (`observe_normal` 73 µs, `sqlite_persist` 317 µs).

### Notes

- The framework adapters are best-effort shims — they detect the
  framework on import and wire the most stable hook shape we know
  today. When the framework APIs drift, the adapter fails at
  *import time* with `FrameworkNotInstalled`, not at agent
  execution. Real integration testing requires the framework
  installed; the unit tests cover the not-installed and basic wiring
  paths only.
- The recovery hook and integration modules add **zero** runtime
  dependencies (stdlib only). The four framework integrations are
  opt-in via extras.

## [0.2.2] — 2026-09-05

Renamed `agent-fuse` → `trajectory-fuse` for PyPI release.
Polished badges, DRY'd version handling, added `__main__` shim.

## [0.2.0] — 2026-09-04

Circuit-breaker release. Added preflight `check()` API,
`@fuse.tool` decorator, `allow_repeats`, `progress_callback`,
`RunBudget`, `FuseStats`, and the in-memory + SQLite persistence
story.
