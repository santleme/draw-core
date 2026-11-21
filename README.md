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
the correct environment across Linux, WSL, and Windows. In shared checkouts
with isolated environments (e.g. `.venv-linux` under WSL alongside Windows
`.venv`), set `UV_PROJECT_ENVIRONMENT=.venv-linux` if invoking `uv run --no-sync draw` directly.

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
