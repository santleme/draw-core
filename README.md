# draw-core

Real-time bridge from paper to digital ink: webcam -> rectified sheet -> ink.
Clean production repo; the experimental history lives in `santleme/draw`.

See [`docs/MIGRATION_SPEC.md`](docs/MIGRATION_SPEC.md) for structure and plan.

The website (`site/` plus the published research drafts in `docs/research/`)
lives in its own repository, [`GUZZBR1/-draw-site-`](https://github.com/GUZZBR1/-draw-site-).

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

On Linux the locked wheels need glibc 2.28 or newer (onnxruntime ships
manylinux_2_28 binaries): Ubuntu 20.04+, Debian 11+, Fedora 34+ or WSL2 —
older glibc releases cannot run the runtime.

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
uv pip install -e ".[dev]"
ruff check . && pytest -q
```

`draw-core` depends on `draw-dral` (`packages/draw-dral`: the DRAL ink
wire plus the shared token handshake). Inside the repo, uv resolves it
from the workspace via `[tool.uv.sources]` — plain `pip install -e .`
cannot, since pip does not read that configuration. `draw-dral` is
versioned together with `draw-core` (same version, same commit); an
install straight from Git therefore pins both at the same SHA:

```sh
pip install \
  "draw-dral @ git+https://github.com/santleme/draw-core@<SHA>#subdirectory=packages/draw-dral" \
  "draw-core @ git+https://github.com/santleme/draw-core@<SHA>"
```

If `draw-dral` is later published to an index, plain `draw-core` resolves
normally and these extra pins become unnecessary.

Two notes for consumers that pin to a commit. First, a same-version pin
(like `0.1.0` today) does not by itself upgrade an environment that
already has the dist installed — pip may keep the existing copy. Update
flows should reinstall both Git pins in one `pip install --force-reinstall`
command or use a fresh venv; reinstalling only `draw-core` can make pip look
for the unpublished `draw-dral` on an index. Second, drawlink only stops
pulling `draw-core`'s heavy deps when it drops its remaining `draw.*` imports
(`draw.config`, `draw.exporters`) — repointing the wire imports alone is not
enough.

For a quick local check, run the module's tests directly. For changes that can
affect other modules, use the separate parallel and serialized Testmon lanes in
`AGENTS.md`.

```sh
pytest -q tests/test_sheet_tracker.py tests/test_tracker_loss_clock.py
```

Do not run an unfiltered `pytest --testmon -n 4`: it can send selected
`slow`/`timing` tests to xdist and a shared Testmon data file can lose history
for tests excluded by a marker filter. The lane setup and commands in
`AGENTS.md` keep those histories separate and run the serialized lane without
xdist.

Run the corresponding tests directly for scripts and subprocess entry
points, such as `tests/test_run_sh.py` after changing `run.sh` or
`tests/test_hvcam.py` after editing the MJPEG worker. CI runs the full
Python suite.
