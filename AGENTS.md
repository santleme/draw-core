# Repository guidance

Read `docs/MIGRATION_SPEC.md` before changing anything.

- `src/draw/` is the runtime: only numpy, opencv-headless, onnxruntime (mediapipe optional). Never import torch/pandas.
- Stages exchange frozen dataclasses from `src/draw/contracts.py`, not dicts or tuples. During the migration, modules keep their legacy return shapes (tuples) until the integration PR adapts them to `contracts`; do not widen those shapes.
- Migrate one module per PR, with its tests; the PR must be CI-green. Modules with a hard mutual dependency (for example `geometry` needs `paper`/`config`, the ink engine's parts) may share a PR; say so in the description.
- Every migration PR is based on `main` and must pass CI on its own: no imports from, and no reliance on, other open PRs. Files two PRs must both change (for example `pyproject.toml`) get byte-identical edits.
- Model integrity is fail-closed: loaders take a required keyword `verify` (a callable, or an explicit `None` opt-out). The pipeline injects `draw.infra.manifest.verify_model`.
- Do not add `DRAW_*` env flags; add fields to the config object instead. The only environment read allowed in the runtime is `DRAW_MODEL_STRICT`.
- Models are never committed; `assets/manifest.json` holds sha256/size.
- During iteration, run only the tests covering the touched module — locate them with `grep -rlw "<module>" tests/` (not every file is named `test_<module>.py`; `draw.paper` is covered by `test_paper_formats.py`).
- Before finishing: `ruff check . && ruff format --check . && pytest -q`.

## Test selection while iterating

`pytest-testmon` and `pytest-xdist` are in the `dev` extra. They are optional
helpers for the iteration loop; the final gate above stays the full suite.

- **Scope.** For a change confined to one module, keep using the module-focused
  rule above (`grep -rlw`). Use testmon instead when the change may reach other
  modules (`contracts`, `config`, `ink.config` and the other hubs), or as a
  broader check before the final gate. When in doubt, run the full suite.
- **One map per lane.** Keep the fast and slow histories in separate files.
  Sharing one `.testmondata` is unsafe: a fast pass after an edit drops the
  affected slow tests from the map, and the next slow pass then selects fewer
  tests than it should (measured: 8 instead of 22). The files are already
  covered by `.gitignore` (`.testmondata*`).
  - Build once:
    `TESTMON_DATAFILE=.testmondata-fast pytest --testmon -m "not slow" -n auto --dist worksteal`
    and `TESTMON_DATAFILE=.testmondata-slow pytest --testmon -m slow`.
  - Iterate:
    `TESTMON_DATAFILE=.testmondata-fast pytest --testmon-forceselect -m "not slow" -n auto --dist worksteal`
    and `TESTMON_DATAFILE=.testmondata-slow pytest --testmon-forceselect -m slow`.
  - Plain `-m`, `-k` or a test node id with `--testmon` turns selection off and
    runs everything, without a warning in the summary. `--testmon-forceselect`
    keeps selection and the filter.
- **`slow` runs serialized, never with `-n`.** That is what the marker means.
  Run in parallel, `test_ruling.py::test_cost_per_pass_stays_inside_the_budget`
  failed once on its timing budget (8.11 ms against 8.0) and passes alone.
- Comments never trigger selection; docstring edits do.
- Testmon only sees executed Python. Run the full suite when you change
  `assets/manifest.json`, `run.sh`, `pyproject.toml`, `uv.lock`,
  `tests/conftest.py`, `tests/onnx_fixtures.py`, `tests/rr.py` or `.github/`.
- Code that runs only in a child process can go unnoticed. Verified case:
  changing the message in `src/draw/_hvcam_mjpeg.py` broke three tests in
  `tests/test_hvcam.py` and testmon selected none. If you touch that file, run
  all of `tests/test_hvcam.py`. Other child-process code was not checked.

## Change philosophy

The guiding style draws on *The Tao of Programming*, KISS, and YAGNI: make
behavior, ownership, and invariants easy to see, and choose the smallest
complete design. These are simplicity heuristics, not a contest to minimize
line count.

- **No code golf.** A short diff is useful when it reduces concepts and makes
  the design easier to understand. Deleting newlines, compressing branches, or
  hiding control flow does not help.
- **Avoid routine docs-only and whitespace-only changes.** A requested update
  to repository instructions is a valid standalone maintenance change when it
  materially improves how the project is developed or reviewed.
- **Benchmark performance claims.** Measure old and new behavior on a
  representative workload. Report the setup, baseline, result, and relevant
  variance or latency percentiles. Profile before optimizing. Weigh even a
  marginal speedup against its cost in maintainability, memory, and latency.
- **Keep changes reviewable and independently useful.** Give each PR one clear
  purpose and keep it based on `main`, CI-green, and independent of other open
  PRs. Keep the one-module-per-migration rule above. A prerequisite refactor is
  welcome when it is a clear improvement by itself and makes the later feature
  easier to review; turning a feature into a three-line change is a good result
  when the prerequisite work is valuable on its own.
- Code outside the core (`src/draw/`) is not well tested: don't touch it unless
  it's actually broken.

What's welcome:

- **Bug fixes** with a regression test that fails before the fix and passes
  afterward.
- **Features** with regression coverage and a usefulness case proportional to
  their implementation and maintenance cost. All new code must be high quality
  and well tested. Prefer familiar NumPy API conventions when they fit the
  runtime and the contracts in this repository.
- **Refactors** that clearly improve readability, ownership, or invariants.
  Whitespace changes or moving functions without improving the design are not
  wins. Behavior-preserving refactors must pass process replay and relevant
  tests.
- **Non-brittle tests and fuzzers.** Prefer deterministic reproductions. An
  `@unittest.expectedFailure` marker must identify a tracked defect and have a
  clear condition for removal.
- **Dead-code removal** from the core (`src/draw/`) after checking supported
  callers and entry points.

## State and concurrency changes

Before changing shared state or thread boundaries, write down the state owner,
readers and writers, object lifetime, mutable-array aliases, invalidation epochs,
commit points, lock order, expected lock hold time, progress requirements, and
exception policy. Keep this proportional to the change; the checklist is not a
reason to add locks or counters without need.

- A stale-work check at publication does not undo mutations made earlier.
  Validate at the boundary that permits the mutation, or keep work private
  until it can be committed.
- Make concurrency regressions deterministic. Use events or barriers to pause
  immediately before the shared mutation, not after the real mutation has
  already happened. Assert both authoritative state and the public result; do
  not use sleeps to manufacture an interleaving.
- Do not rely on the GIL to protect NumPy buffers. Frozen dataclasses and
  read-only array views do not make the underlying data deeply immutable. State
  whether inputs are copied, transferred, or promised not to change while in
  use.
- Give every generation or revision token a distinct invalidation meaning. Do
  not add a counter when an existing generation or epoch already covers that
  change.
- Extract a collaborator only when its inputs, outputs, and state ownership
  become clearer. A helper that still reaches into and mutates its parent
  object's state has not created a useful boundary.
