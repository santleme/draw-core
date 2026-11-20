# Migration spec: santleme/draw -> draw-core

Status: draft for review. Legacy repo: `santleme/draw` (kept intact as the
historical lab; tag `legacy-final` marks the last state).

## Goal

A clean production repo for the real-time Draw engine: modular, typed
contracts between stages, fast test suite, and a boundary clean enough
that the hot path can later move to C++ without touching the rest.

## Rules for what gets copied

1. **Runtime imports no training stack.** `src/draw/` depends only on
   `numpy`, `opencv-python-headless`, `onnxruntime` (+ optional
   `mediapipe` extra). No `torch`, no `pandas`.
2. **Stages end on `contracts.py` dataclasses.** The final stage
   boundary is the frozen dataclasses, with no loose dicts or
   positional tuples between modules. During the migration each
   module keeps its legacy return shape (tuples included) and the
   integration PR adapts them to `contracts`; do not widen those
   shapes meanwhile.
3. **Test-driven migration.** One module per PR, together with its
   tests. A module counts as migrated only when `pytest` is green in
   this repo. Modules with a hard mutual dependency may share a PR,
   saying so in the description.
4. **Every PR stands alone.** Each PR is based on `main`, small,
   reviewable, and CI-green on its own: no imports from, and no
   reliance on, other open PRs. Files two PRs must both change get
   byte-identical edits.
5. **No dead flags.** A `DRAW_*` flag survives only if it is on in the
   production configuration. Rejected or default-off experiments stay in
   legacy (see Flags).

## Target layout

```
draw-core/
├── pyproject.toml
├── assets/manifest.json        # sha256 of production models (not the binaries)
├── src/draw/
│   ├── contracts.py            # FramePacket, SheetQuad, InkState, PageIdentity
│   ├── infra/                  # safe_io, manifest verification, config, telemetry
│   ├── camera.py               # capture + latest-wins queue + FPS control
│   ├── perception.py           # ONNX sessions (sheet + ink), unified preprocess
│   ├── geometry.py             # corner fit, warp, optional ECC refine
│   ├── tracker.py              # SheetTracker + F-11 page identity state machine
│   ├── accumulator.py          # confirmed/provisional ink, votes, registration
│   ├── hand.py                 # optional hand presence/occlusion
│   ├── exporters.py            # SVG, DXF (locked scale), CAD bridge
│   └── pipeline.py             # frame -> events orchestration (threads)
├── lab/                        # replay, simulator, judges (not runtime)
├── site/                       # React site (moved as-is, own CI job)
├── bridge/                     # Windows camera helper
└── tests/
```

Separate repo `draw-training` (not created yet): `training/`, dataset
tools, labeling, Kaggle runner, `draw/selflearn/`, `gpu_augmenter.py`.
It delivers only approved `.onnx` files + hashes to `assets/manifest.json`.

## Contracts (first sketch, to be refined in the contracts PR)

- `FramePacket`: image (H,W,3 uint8), monotonic timestamp, sequence id.
- `SheetQuad`: 4x2 float32 corners in camera space, confidence,
  detector id, `present: bool`.
- `PageIdentity`: generation id, decision (`same` / `new_page` / `defer`),
  recall/containment evidence.
- `InkState`: confirmed mask, provisional mask, generation id, anchor corners.
- `Config`: immutable settings resolved once from env/CLI (replaces scattered
  `os.environ.get` calls; legacy has ~40 `DRAW_*` reads).

## Legacy inventory

| Legacy | Action | Destination |
|---|---|---|
| `draw/ink_layers.py` (4001 lines) | split | `accumulator.py` (+ `votes`, `registration`, `page_clean_worker`, `render`) |
| `draw/app.py`, `draw/loop.py` | split | `pipeline.py` + thin CLI; drop dead `ink_model` path and `legacy_ink` |
| `draw/neural.py` | split | `perception.py` (ONNX) + `geometry.py` (corner fit/refit) |
| `draw/sheet_track.py` | migrate | `tracker.py` |
| `draw/capture.py` | migrate | `camera.py` |
| `draw/paper.py`, `page_clean.py`, `view.py`, `lk_propagate.py` | migrate | engine |
| `draw/hand.py`, `hand_presence.py` | migrate | `hand.py` (mediapipe extra) |
| `draw/manifest.py`, `safe_io.py`, `config.py`, `telemetry.py`, `net_policy.py`, `session_report.py` | migrate | `infra/` |
| `draw/ink.py` | migrate `render_canvas` only; drop legacy ink stack | see section below |
| `draw/fusion.py` | not runtime | `lab/` or archive |
| `draw/gpu_augmenter.py`, `draw/selflearn/` | archive | `draw-training` |
| `tools/` replay, sim_*, judge*, eval_*, falsify, seg_metrics | migrate | `lab/` |
| `tools/` dataset, label, kaggle, runner, export/merge | archive | `draw-training` |
| `tools/replay_reregister*.py` | archive | hash-anchored; bytes must not change |
| `training/` | archive | `draw-training` |
| `site/`, `bridge/` | migrate | same names (monorepo) |
| `docs/` | keep README, SIMULATOR, SETUP_WINDOWS, a trimmed FALHAS | dated pre-registrations stay in legacy |

Tests: ~60 module tests migrate with their module; sim/judge/replay tests go
to `lab/`; training and selflearn tests go to `draw-training`; pre-registration
and doc-lint tests stay in legacy.

## Legacy loose files (decided)

| File | Decision | Why |
|---|---|---|
| `diag_live*.py` (8) | archive in legacy | one-off scripts with hardcoded `~/draw` paths; superseded by replay lab |
| `diag_live_desk.py` | ported to `lab/desk_capture.py` | headless real-desk driver; `--source`/`--out`/`--fps` flags replace the hardcoded paths |
| `diag_quickdraw_*` | archive in legacy | depends on external `~/draw-lab/ink-data/quickdraw` corpus |
| `perf_sampler.py`, `summarize_perf.py` | migrate to `lab/perf/` (late) | per-thread CPU profiling is required before any C++ decision |
| `install-v6-swap.py`, `prove-v6-on-desk.py` | archive | one-time swap with hardcoded paths; result is recorded in the manifest |
| `r`, `pr76_comments.txt` | drop | run output / review dump, no lasting value |
| `draw-hack/` | drop (superseded by `site/`) | same pages/components as `site/`; extra README/asset only |
| `site/public/images/vision/sailor-moon-hair-wind.png` | exclude until rights confirmed | possible copyright issue |
| `models/lab/*.onnx`, `models/*.bak` | not migrated | experimental/rollback binaries; hashes of former production are in the manifest |

## Production models (assets/manifest.json)

Source of truth is the legacy manifest at `1fa3d76+3`:
sheet detector T16 `5e20a89f...`, page-clean v6 `613e250f...`, MediaPipe
hand landmarker (unmodified, Apache-2.0). Former production (`e20a89f0`,
`a470bd28`) is recorded only as rollback references.

## Flags policy

Legacy has ~40 `DRAW_*` variables and **every behavior flag defaults to
off**: the legacy default is not the configuration that was validated. The
validated set comes from legacy commit `a95eec8` (`docs/SIMULATOR.md`,
bench of the T16 model, 2026-11-13) and `docs/FALHAS.md`.

Tier A, production preset (become defaults in `Config`, env var removed):
`DRAW_INK_FOLLOW_MOVE`, `DRAW_SHEET_COVERAGE`, `DRAW_CANVAS_PERSIST`,
`DRAW_INK_SKIP_STALE`, `DRAW_INK_REMOVAL`, `DRAW_SEG_HAND_OCCLUSION`,
`DRAW_ANCHOR_ARBITRATION`, `DRAW_ANCHOR_FOLLOW`.

Tier B, approved in replay but never enabled (kept as typed `Config` fields,
default off, one small PR each; enabling needs a desk test, per FALHAS):
`DRAW_REG_DEADBAND`, `DRAW_REG_ANCHOR_MEDIAN3`.

Tier C, not documented as validated anywhere (not in docs, not in the T16
set): `DRAW_INK_VOTE_E3`, `DRAW_INK_VOTE_E4`, `DRAW_INK_VOTE_FAST`,
`DRAW_INK_ACTIVE_CADENCE`, `DRAW_FIRST_CONFIRM_GATE`,
`DRAW_SHEET_LK_PROPAGATE`, `DRAW_CAPTURE_DRAIN_LATEST`,
`DRAW_SHEET_DETECT_SETTLED_EVERY`, `DRAW_PAGE_CLEAN_BATCH`. Not copied in the
first pass; revisit one by one when the owning module is migrated (the vote
rules live in `ink_layers.py`, ~570 lines).

Tier D, refuted, vetoed or diagnostic-only (stay in legacy):
`DRAW_ECC_REFINE` (refuted in P2; its `ecc_rejected` quarantine goes with
it, the `deferred_occlusion` quarantine stays), `DRAW_ANCHOR_MIGRATION` (v1
vetoed), `DRAW_IDENTITY_WATCH`, `DRAW_PAGE_TURN_EVENT` (hotfix-disabled),
`DRAW_IDENTITY_CHAMFER` (diagnostic), `DRAW_REG_CONTENT_GUARD`, and
`DRAW_ANCHOR_MOTION_ABSTAIN` (reverted).

Real configuration, not flags: model paths, `DRAW_FORMAT`, ORT/OpenCV
thread caps. These moved to explicit constructor/`Config` arguments —
the `DRAW_ORT_SPIN`, `DRAW_ORT_THREADS`, `DRAW_PAGE_CLEAN_THREADS`,
`DRAW_HAND_MODEL`, `DRAW_PAGE_CLEAN_MODEL` and `DRAW_NEURAL_MODEL` env
reads are gone (spin is always disabled). `DRAW_MODEL_STRICT` stays as
the single env read the runtime keeps.

Consequence for parity: legacy must be run **with the Tier A env set** when
used as the reference.

Parity evidence so far:

- **Fake-worker parity** (pipeline design PR): 10 scenarios, 720 frames,
  zero behavior divergences; one 1-ULP timestamp difference.
- **End-to-end parity** (`lab/parity_e2e.py`): real neural detector,
  tracker, `PageCleanInk` and ink worker on both sides, same model
  binaries, synthetic 68-frame scenario — zero divergences (identical
  per-frame canvas masks, identical presence, corners within 0.5 px).

## PR sequence

Every item is its own PR against `main`. Modules with a hard mutual
dependency shipped grouped — those groupings are marked.

1. Scaffold: pyproject, ruff, CI, `.gitattributes`, manifest, this spec.
2. `contracts.py` + tests (PR #2).
3. `infra/`: safe_io, manifest verification, config (PR #3).
4. `tracker.py` (+ sheet tracker, page identity, loss clock tests) (PR #4).
5. `camera.py` (+ capture tests) (PR #5).
6. `geometry.py` + `perception.py` in one PR (PR #6): the corner fit
   consumes the ONNX session output directly and they cannot pass CI
   apart.
7. `page_clean.py` (PR #7), `infra` telemetry/session report (PR #8).
8. `ink-engine` (PR #9): the `ink_layers.py` runtime (accumulator,
   votes, registration/identity, worker) in one PR — the pieces share
   worker state and cannot pass CI apart.
9. `hand.py` (PR #10).
10. `lab/` in two PRs: simulator (`lab-sim`, PR #11) and
    metrics/vectorize (`lab-metrics-vectorize`, PR #12).
11. `site/` (PR #13), `bridge/` (PR #14).
12. `pipeline.py` + CLI (`app.py`, `view.py`): done, evolved through the
    opt-in feature PRs (#45-#57: coverage floor, auto gain, strict vote,
    dense warp, ruling removal, hysteresis). `exporters/` carries
    `export_vectors` and `vectorize`; nothing beyond that is planned.
13. Parity gate: `lab/parity_e2e.py` (legacy in a subprocess, same model
    binaries, Tier A flags on) plus `tests/test_parity_e2e.py` and the
    fake-worker scenarios in `tests/test_pipeline*`. Open: re-run
    `parity_e2e` on a real recorded session (not only the synthetic
    68-frame one) before retiring the legacy repo; the opt-in features
    are off by default and outside the parity claim.

## Invariants to preserve

- Page identity (F-11): same page survives re-entry; distinct content is rejected.
- Export scale locked for DXF.
- GT end-of-video is `ink_final` (post-erase), per-phase pose via `quad_end`,
  dense scene->page maps when the page curls.
- Model sha256/size verified at load; mismatch refused unless explicitly relaxed.
- Latest-wins capture; worker rest/wake cadence.

## C++ (deferred)

Not in scope now. Revisit only after per-stage profiling (`lab/perf/`) shows a
stage dominated by non-ONNX Python time. Today ~12 FPS is dominated by ONNX
inference, which is already native. Boundary for a future port: the contract
types in `contracts.py`.

## Legacy baseline (measured 2026-11-16, legacy `main`)

`pytest -m "not lento"`: 979 passed, 27 skipped, 1 failed, plus a hang.

- **Failure** `test_sim_writer::test_generate_scenario_corpus_contract`:
  needs real background frames in `~/workspace/draw-lab/...` that are not in
  the repo. Rule: lab tests that need external data must `skip` when it is
  absent, so CI is hermetic.
- **Hang** (found with `faulthandler_timeout`): the full run deadlocks in
  `HandLandmarker.__del__` (mediapipe `serial_dispatcher`) when GC finalizes a
  leaked landmarker on the main thread, e.g. during
  `test_simulator::test_real_hand_adapters_feed_visibility_and_activity`;
  it also leaks a `selflearn.SessionRecorder` worker thread. Rules: `hand.py`
  closes the landmarker deterministically (context manager / `close()`), never
  relies on `__del__`; tests close what they open; CI runs pytest with
  `faulthandler_timeout`.

## `ink.py` and `fusion.py` (decided from the code)

- `draw/fusion.py`: nothing in `draw/` imports it; only `tools/cadence_fusion.py`
  and its test use it. Not runtime -> `lab/` or archive.
- `draw/ink.py`: still imported by `ink_layers`, `loop`, `app`, `view`, but
  mostly for `render_canvas`. `extract_ink`, `InkAccumulator` and
  `SessionBaselineInk` serve only the `--legacy-ink` / `--markers` paths.
  Migrate only `render_canvas` + paper constants; drop the legacy ink stack,
  `--legacy-ink`, `--markers`, `--ink-model` (the `ink_model` module does not
  even exist).

## Intentional deviations from legacy

Behavior that differs from legacy on purpose (bugs found by automated review
or by the migration). Parity runs must account for them.

- Tracker: absence before the first confirmation discards the pending
  `blank_maybe` candidate (legacy kept it and could confirm a returning blank
  immediately or hand stale corners to a new sheet).
- Sheet detection: an edge with no reliable measurement and no continuity
  stores "absent" in the edge memory (legacy inherited the old line, pulling a
  new sheet toward the previous one).
- Ink worker: rotated landings enter ink rest (legacy compared positional
  quads with a rolled anchor and never rested).
- Pipeline: the detector's hand mask feeds `seg_hand_occlusion` regardless
  of `coverage` (legacy tied both detection extras to the same flag).
- Pipeline: with `canvas_persist` off, a tracker drop also clears the
  rendered canvas and the provisional layer (legacy cleared only the
  confirmed mask). Neither change alters the production preset.
- Camera: drain mode is not migrated (Tier C); `PngWriter` survives a failing
  `on_done`; a failed recovery reopen raises the normal opening error.
- Integrity is fail-closed (`verify` required); `DRAW_ORT_SPIN`,
  `DRAW_ORT_THREADS`, `DRAW_PAGE_CLEAN_THREADS`, `DRAW_HAND_MODEL`,
  `DRAW_PAGE_CLEAN_MODEL`, `DRAW_NEURAL_MODEL` env reads are gone (explicit
  arguments; spin always disabled).
- `HandMasker.close()` is lock-serialized and deterministic (legacy relied on
  mediapipe's `__del__`, which could deadlock).
- Safe I/O: JSON integers past Python's digit limit raise `InputLimitError`;
  `ensure_private_dir` appends a catch-all to an existing `.gitignore`.
- Session reports: unique, atomic writes; same-second sessions no longer
  collide; `latest.json` is replaced atomically.
- Contracts: array fields are read-only views/copies; `SheetQuad` corners and
  verdict must agree.
- Lab: unique photo ids, rotation ceiling respected, rigid 180 degree returns,
  blur from per-corner motion, thickness-aware erased-stroke labels
  respecting the exclusive erase border; `lab` is dev tooling run from
  the repo root, not part of the installed package; `judge_candidate` resolves relative
  candidate paths, reports `INCONCLUSIVO` when no gated judge ran, and has no
  `--with-replay`; segment-based Douglas-Peucker; `cldice` of a missed dot is 0.
- Site: reduced-motion progress follows scroll, instant programmatic scrolls,
  long strokes continue past the point cap, webcam card can be dismissed,
  graphite strokes follow the theme.
- Bridge: helper stays resident across client disconnects, total handshake
  deadline, wildcard host aliases need the explicit opt-in, timestamps are
  process-relative as the protocol states.

## Open questions

- Release hosting for model binaries (Release in this repo vs legacy
  `models-2026-11-06`)?
- Rights for the sailor-moon image before it enters the site.
- Confirm Tier A matches what is run at the desk today (the repo evidence says
  yes; only the owner can confirm).
