"""Testes das partes puras do lab/replay_lab (corpus, sinais,
seleção dos piores frames, prancha). O replay em si precisa dos
modelos reais; o caminho ponta a ponta do Pipeline já é coberto
por tests/test_eval_capture.py."""

import cv2
import numpy as np
import pytest
from lab.replay_lab import (
    _rank,
    build_panel,
    compute_signals,
    list_corpus_frames,
    quad_area,
    quad_jump,
    select_worst,
)


def test_list_corpus_frames_prefere_raw(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "a.txt").write_text("x")
    cv2.imwrite(str(tmp_path / "raw" / "f2.png"), np.zeros((4, 4, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "raw" / "f1.png"), np.zeros((4, 4, 3), np.uint8))
    assert [p.name for p in list_corpus_frames(tmp_path)] == ["f1.png", "f2.png"]


def test_list_corpus_frames_vazio_falha(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_corpus_frames(tmp_path)


def test_quad_area_e_jump():
    q = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert quad_area(q) == pytest.approx(100.0)
    assert quad_area(None) is None
    q2 = [[0, 0], [13, 0], [10, 10], [0, 10]]
    assert quad_jump(q, q2) == pytest.approx(3.0)
    assert quad_jump(q, None) is None


def test_compute_signals_flip_e_crescimento():
    recs = [
        {
            "frame": 0,
            "present": True,
            "quad": [[0, 0], [9, 0], [9, 9], [0, 9]],
            "det_quad": None,
            "canvas_px": 10,
        },
        {"frame": 1, "present": False, "quad": None, "det_quad": None, "canvas_px": 10},
        {
            "frame": 2,
            "present": True,
            "quad": [[0, 0], [9, 0], [9, 9], [0, 9]],
            "det_quad": None,
            "canvas_px": 30,
        },
    ]
    compute_signals(recs)
    assert recs[1]["sig_presence_flip"] == 1
    assert recs[2]["sig_canvas_growth_px"] == 20
    assert recs[1]["sig_quad_aspect"] is None


def test_rank_empate_divide_posto_medio():
    out = _rank([0.0, 5.0, 5.0, 9.0])
    assert out[0] == 0.0
    assert out[1] == out[2] == pytest.approx((1 + 2) / 2 / 3)
    assert out[3] == 1.0


def test_select_worst_respeita_topk_e_dedupe():
    recs = [
        {
            "frame": i,
            "present": True,
            "quad": [[0, 0], [10 + i * 20, 0], [10, 10], [0, 10]],
            "det_quad": None,
            "canvas_px": 0,
        }
        for i in range(10)
    ]
    compute_signals(recs)
    worst = select_worst(recs, 3)
    assert len(worst) <= 3
    idxs = [w["index"] for w in worst]
    assert all(abs(a - b) >= 2 for a in idxs for b in idxs if a != b)


def test_build_panel_layout():
    frame = np.full((60, 80, 3), 255, np.uint8)
    warp = np.zeros((40, 30, 3), np.uint8)
    mask = np.zeros((40, 30), np.uint8)
    mask[10:20, 10:20] = 255
    panel = build_panel(frame, [[5, 5], [60, 5], [60, 50], [5, 50]], warp, mask)
    assert panel.shape == (60, 80, 3)


def test_build_panel_sem_warp():
    frame = np.full((60, 80, 3), 255, np.uint8)
    panel = build_panel(frame, None, None, None)
    assert panel.shape == (60, 80, 3)
