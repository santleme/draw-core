# draw-core

Real-time bridge from paper to digital ink: webcam -> rectified sheet -> ink.
Clean production repo; the experimental history lives in `santleme/draw`.

See [`docs/MIGRATION_SPEC.md`](docs/MIGRATION_SPEC.md) for structure and plan.

## Run the live app

One command sets up the environment, downloads any missing official model
binaries from the `models-v3` GitHub Release (verified against
[`assets/manifest.json`](assets/manifest.json)), starts `hvcam-host.exe`
automatically on WSL2 (downloading the verified Windows binary if needed),
picks an available webcam, and opens Draw:

```sh
./run.sh
```

Any CLI flags pass straight through (`./run.sh --check`, `./run.sh --fullscreen`,
`./run.sh --source 1`). `./run.sh` is the portable entry point that selects
the correct environment across Linux, WSL, and Windows, including a separate
environment for hand tracking if the launcher set one up.

For a direct launch with `uv run --no-sync draw`, set
`UV_PROJECT_ENVIRONMENT` to the prepared environment you want to use.
For example, in a shared checkout with a Windows `.venv`, WSL uses
`.venv-linux` for the base installation. If a later MediaPipe retry succeeded
in `.venv-linux-hand`, launch with hand tracking using:

```sh
UV_PROJECT_ENVIRONMENT=.venv-linux-hand uv run --no-sync draw
```

The same `-hand` suffix applies to the other base directories (`.venv` and
`.venv-win`). If MediaPipe is already installed in the base environment, use
that directory directly.

No flags selects the production preset (`InkConfig.production()` plus
`PipelineConfig()` defaults): digital gain, strict vote `0.85`, ruling
removal, hysteresis, motion veto, `0.15 s` ink interval and Detect-Wake.
`--source` selects the camera (`0` by default); `--ink-vote` and
`--ink-interval` tune the numbers. Known trade-offs: the strict vote thins
strokes (it relies on the gain), and ruling removal can suppress fine pencil
laid over printed lines (`--no-ink-ruling` turns just that off).
`--legacy-ink` restores the pre-preset behaviour for comparison. Streaming
still requires an explicit port and token file; model verification stays on.

```sh
pip install -e ".[dev]"
ruff check . && pytest -q
```

For a quick local check, run the module's tests directly, then use Testmon for
changes that can affect other modules.

```sh
pytest -q tests/test_sheet_tracker.py tests/test_tracker_loss_clock.py
pytest -q --testmon -n 4
```

Seed `.testmondata` with a serial `pytest -q --testmon` on a fresh worktree.
To keep `slow` tests serial, add `--testmon-forceselect -m 'not slow'` to
the parallel command and run `pytest -q -m slow` separately. Use
`--testmon-forceselect` whenever combining impact selection with `-m`.

Run the corresponding tests directly for scripts and subprocess entry
points, such as `tests/test_run_sh.py` after changing `run.sh` or
`tests/test_hvcam.py` after editing the MJPEG worker. CI runs the full
Python suite.
