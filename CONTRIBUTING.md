# Contributing

Thanks for taking the time to improve `trajectory-fuse`. This is a
small, zero-runtime-dependency library — the goal is to keep it small,
correct, and boring.

## Development setup

```bash
git clone https://github.com/aniketkarne/trajectory-fuse
cd trajectory-fuse
pip install -e ".[dev]"
```

The `dev` extra pulls in `pytest`, `pytest-cov`, and `ruff`.

Optional framework extras (for integration smoke-testing):

```bash
pip install -e ".[langgraph]"
pip install -e ".[openai-agents]"
pip install -e ".[pydantic-ai]"
# or everything at once
pip install -e ".[all-integrations]"
```

## Test layout

* `tests/test_smoke.py` — symbol surface and version.
* `tests/test_guard.py` — direct repeat, N-cycle, stagnation.
* `tests/test_hardening.py` — budgets, hooks, stats, edge cases.
* `tests/test_edge_cases.py` — concurrent observes, JSONL, export.
* `tests/test_cli.py`, `tests/test_store.py`, `tests/test_hashing.py`
  — one file per surface area.
* `tests/test_integrations.py` — recovery hooks + generic + framework
  adapter factories (framework deps mocked as not-installed).
* `benchmarks/test_efficacy.py` — efficacy benchmark contract: runs
  the scaled 456-trajectory dataset and asserts precision, recall,
  FPR, and call-savings thresholds.

Run everything:

```bash
pytest -q
```

Run just the efficacy contract:

```bash
pytest benchmarks/test_efficacy.py -v
```

Run the perf benchmarks:

```bash
python benchmarks/run_benchmarks.py            # table
python benchmarks/run_benchmarks.py --json     # JSON
python benchmarks/run_efficacy.py --scaled --json   # efficacy numbers
```

## Style

The codebase targets **Python 3.9+**, so use `Optional[X]` rather
than `X | None` and `List[X]` rather than `list[X]`. Lint with:

```bash
ruff check src tests benchmarks
```

The `ruff.toml` config enforces only the categories that catch real
bugs (`F`, `S`, `I`); style-level rules are off. Pre-existing files
with stale imports are exempt from the per-file ignore list — clean
them up opportunistically as you touch them, but don't feel obliged
to refactor unrelated code.

## PR guidelines

1. **One change per PR.** New feature, fix, or refactor — not all
   three at once.
2. **Add tests** for any new behaviour. The efficacy contract must
   still pass after your change (precision ≥ 0.99, recall ≥ 0.99,
   FPR ≤ 1%).
3. **Keep the public API stable.** Adding a symbol is fine;
   renaming or removing one needs a deprecation cycle.
4. **No new runtime dependencies.** The library is stdlib-only by
   design. Framework integrations must be opt-in extras, not
   imports.
5. **Run the demo** (`python examples/demo.py`) and verify nothing
   visually changed.

## Reporting issues

Open an issue on GitHub. Include:

* the smallest reproducer you can write,
* the detected `DetectionKind` (if your fuse is raising),
* Python version and OS.

## Code of conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Be patient, be helpful, assume good faith.
