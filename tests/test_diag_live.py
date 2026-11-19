"""Ferramenta de diagnóstico ao vivo (lab/diag_live.py): partes puras."""

import json

import cv2
import numpy as np
import pytest
from lab import diag_live

from draw import config


def test_quad_aspect_is_width_over_height():
    quad = [[0, 0], [200, 0], [200, 100], [0, 100]]
    assert diag_live.quad_aspect(quad) == pytest.approx(2.0)
    tall = [[0, 0], [100, 0], [100, 300], [0, 300]]
    assert diag_live.quad_aspect(tall) == pytest.approx(1 / 3)


def test_aspect_grid_keeps_quad_proportion_on_the_long_side():
    wide = [[0, 0], [300, 0], [300, 100], [0, 100]]  # 3:1
    assert diag_live.aspect_grid(wide, long_side=900) == (900, 300)
    tall = [[0, 0], [100, 0], [100, 300], [0, 300]]
    assert diag_live.aspect_grid(tall, long_side=900) == (300, 900)
    config.apply_format("a4")
    assert max(diag_live.aspect_grid(wide)) == max(config.PAPER_W, config.PAPER_H)


def test_luminance_splits_quad_left_right():
    frame = np.full((100, 200, 3), 200, np.uint8)
    frame[:, :100] = 40  # metade esquerda escura
    quad = [[10, 10], [190, 10], [190, 90], [10, 90]]
    stats = diag_live.luminance_stats(frame, quad)
    assert stats["quad_left_mean"] < 60 < 180 < stats["quad_right_mean"]
    assert stats["frame_mean"] == pytest.approx(120, abs=1)
    assert "quad_left_mean" not in diag_live.luminance_stats(frame)


def test_rotate_frame_180_moves_the_corner():
    frame = np.full((40, 60, 3), 230, np.uint8)
    frame[2:8, 3:9] = 20
    out = diag_live.rotate_frame(frame, 180)
    assert out[32:38, 51:57].mean() < 60
    assert diag_live.rotate_frame(frame, 0) is frame


def test_warp_to_returns_requested_grid():
    frame = np.zeros((120, 160, 3), np.uint8)
    quad = [[10, 10], [150, 10], [150, 110], [10, 110]]
    assert diag_live.warp_to(frame, quad, (70, 50)).shape == (50, 70, 3)


def test_diagnose_frame_writes_artifacts_and_summary(tmp_path):
    """Roda o fluxo inteiro com detector e tinta de mentira."""
    config.apply_format("a4")
    frame = np.full((240, 320, 3), 200, np.uint8)
    quad = np.float32([[20, 20], [300, 20], [300, 220], [20, 220]])

    class _Detector:
        last_hand_mask = None

        def __call__(self, image, *, return_mask=True):
            return quad, "paper", np.full(image.shape[:2], 255, np.uint8)

    class _Ink:
        threshold = 0.9

        def ink_prob(self, warped):
            return np.full(warped.shape[:2], 0.95, np.float32)

    record = diag_live.diagnose_frame(frame, _Detector(), _Ink(), tmp_path, 0)
    for name in (
        "000_raw.png",
        "000_seg.png",
        "000_warp.png",
        "000_warp_aspect.png",
        "000_prob.png",
        "000_mask.png",
        "000_mask_aspect.png",
    ):
        assert cv2.imread(str(tmp_path / name)) is not None, name
    assert record["verdict"] == "paper"
    assert record["quad_aspect"] == pytest.approx(280 / 200)
    assert record["ink_px"]["warp"]["0.85"] > 0
    json.dumps(record)  # serializável para o summary.json


def test_diagnose_frame_without_quad_records_luminance_only(tmp_path):
    class _Detector:
        last_hand_mask = None

        def __call__(self, image, *, return_mask=True):
            return None, None, None

    record = diag_live.diagnose_frame(
        np.full((60, 80, 3), 100, np.uint8), _Detector(), object(), tmp_path, 3
    )
    assert record["corners"] is None
    assert record["luminance"]["frame_mean"] == pytest.approx(100)
    assert (tmp_path / "003_raw.png").exists()
