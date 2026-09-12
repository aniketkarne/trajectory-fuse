# Changelog

All notable changes to `trajectory-fuse` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
