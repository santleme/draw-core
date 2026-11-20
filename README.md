# draw-core

Real-time bridge from paper to digital ink: webcam -> rectified sheet -> ink.
Clean production repo; the experimental history lives in `santleme/draw`.

See [`docs/MIGRATION_SPEC.md`](docs/MIGRATION_SPEC.md) for structure and plan.

## Run the live app

Place the model binaries verified by [`assets/manifest.json`](assets/manifest.json)
(`draw_seg_v3_dyn.onnx`, `page_clean_unet.onnx`, and optionally
`hand_landmarker.task`) in `models/`. Because the base runtime depends on
`opencv-contrib-python-headless`, replace it with the GUI wheel in your
virtualenv before opening the live window:

```sh
uv sync --extra gui --extra hand
uv pip uninstall opencv-contrib-python-headless && uv pip install --reinstall "opencv-contrib-python>=4.8,<5.1"
uv run --no-sync draw --check
uv run --no-sync draw
```

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
