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

## Change philosophy

- No code golf: a low line count is a guiding light, but the real goal is reducing complexity and increasing readability. Deleting `\n`s does nothing to help.
- No docs-only or whitespace-only changes: they are useless and carry a risk of introducing bugs.
- Anything claimed as a "speedup" must be benchmarked. Even a marginal speedup must be weighed against maintainability and readability — the goal is simplicity.
- Code outside the core (`src/draw/`) is not well tested: don't touch it unless it's actually broken.
- A PR that looks complex, is a big diff, or adds lots of lines won't be reviewed or merged. Break work into smaller PRs that are each a clear win. A common pattern: prerequisite refactors until the feature becomes a 3-line change that's easy to review.

What's welcome:

- Bug fixes, always with a regression test.
- Features, always with a regression test, weighing the line tradeoff: a 3-line change has a lower usefulness bar than a 30- or 300-line one. With no other constraints, a feature's API should match torch or numpy. All new code must be high quality and well tested.
- Refactors that are clear wins — e.g. two 100-line functions that become one parametrized 110-line function while also improving readability. Moving whitespace or functions around is useless. Refactors must pass process replay / the existing test suite.
- Non-brittle tests and fuzzers. Finding bugs is progress — even a broken test that should pass is valuable, marked `@unittest.expectedFailure`.
- Dead code removal from the core (`src/draw/`).
