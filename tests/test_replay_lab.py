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


def test_select_worst_top_k_zero_devolve_vazio():
    from lab.replay_lab import select_worst

    records = [{"frame": i, "sig_quad_jump_px": 10.0} for i in range(3)]
    assert select_worst(records, 0) == []
    assert select_worst(records, -1) == []


def test_select_worst_frame_eh_indice_do_arquivo():
    # Registro pulado (frame ilegível não entra): posição e frame
    # divergem — a seleção tem de devolver o índice do arquivo.
    from lab.replay_lab import select_worst

    records = [{"frame": f, "sig_quad_jump_px": 0.0} for f in (0, 1, 3, 4)]
    records[2]["sig_quad_jump_px"] = 99.0
    worst = select_worst(records, 1, min_gap=0)
    assert worst[0]["frame"] == 3


def test_run_replay_pareia_deteccao_consumida(tmp_path):
    """Pipeline.step consulta o resultado anterior antes de submeter:
    o det_* do registro é a detecção CONSUMIDA (produzida pelo frame
    anterior), não a recém-submetida."""
    from lab.replay_lab import run_replay

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i}.png"), np.zeros((64, 64, 3), np.uint8))
    quads = [
        np.float32([[x, 10], [x + 20, 10], [x + 20, 30], [x, 30]]) for x in (5, 15, 25)
    ]
    submitted = []

    class FakeDetector:
        last_presence_logit = None

        def __call__(self, frame, return_mask=False):
            idx = len(submitted)
            submitted.append(idx)
            return (quads[idx], "paper", np.ones(frame.shape[:2], np.uint8))

    class FakeInk:
        threshold = 0.5

        def ink_mask(self, warp):
            return np.zeros(warp.shape[:2], np.uint8)

    records = run_replay(
        tmp_path,
        tmp_path / "out",
        detector=FakeDetector(),
        page_clean=FakeInk(),
        verify=None,
    )
    assert len(records) == 3
    # Frame 0: nenhum resultado pendente — nada foi consumido.
    assert records[0]["det_frame"] is None
    assert records[0]["det_quad"] is None
    # Frames seguintes: det_* descreve a detecção do frame anterior —
    # a que o rastreador acabou de consumir neste step.
    assert records[1]["det_frame"] == 0
    assert np.allclose(records[1]["det_quad"], quads[0])
    assert records[2]["det_frame"] == 1
    assert np.allclose(records[2]["det_quad"], quads[1])


def test_build_panel_sobrepoe_canvas_confirmado():
    frame = np.full((60, 80, 3), 255, np.uint8)
    warp = np.full((40, 30, 3), 128, np.uint8)
    ink = np.zeros((40, 30), np.uint8)
    canvas = np.zeros((40, 30), np.uint8)
    # O rótulo cobre as ~26 primeiras linhas do ladrilho 30×40 —
    # o confirmado tem de cair na faixa visível de baixo (warp
    # rows ≥34 → ladrilho rows ≥26) para a prova não se satisfazer
    # com o branco do texto.
    canvas[34:39, 2:28] = 255
    panel = build_panel(frame, None, warp, ink, canvas)
    # O ladrilho warp+tinta ocupa o quadrante superior-direito; a
    # faixa de baixo, livre do rótulo, tem de ficar ciana — B e G
    # altos e R baixo (o texto branco do rótulo teria R=255).
    strip = panel[27:30, 41:79]
    # Blend ciano sobre warp 128: B/G ≈ 0.55·128+0.45·255 ≈ 185,
    # R ≈ 70 — branco do rótulo teria os três em 255.
    assert strip[..., 0].max() > 150 and strip[..., 1].max() > 150
    assert strip[..., 2].mean() < 120


def test_run_replay_panel_overlay_honors_ink_divnorm(tmp_path):
    """Review #48: com ink_divnorm=1 o overlay por quadro da prancha
    infere sobre a página normalizada, como o canvas — nunca sobre o
    warp cru."""
    from lab.replay_lab import run_replay

    from draw.paper import warp_sheet

    grad = np.tile(np.linspace(60, 230, 64, dtype=np.float32), (64, 1))
    frame = np.repeat(grad[..., None], 3, axis=2).astype(np.uint8)
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i}.png"), frame)
    quad = np.float32([[2, 2], [60, 2], [60, 60], [2, 60]])
    raw_warp = warp_sheet(frame, quad)
    seen = []

    class FakeDetector:
        last_presence_logit = None

        def __call__(self, frame, return_mask=False):
            return (quad.copy(), "paper", np.ones(frame.shape[:2], np.uint8))

    class FakeInk:
        threshold = 0.5

        def ink_prob(self, warp):
            seen.append(np.array(warp, copy=True))
            return np.zeros(warp.shape[:2], np.float32)

        def ink_mask(self, warp):
            return np.zeros(warp.shape[:2], np.uint8)

    run_replay(
        tmp_path,
        tmp_path / "out",
        detector=FakeDetector(),
        page_clean=FakeInk(),
        flags=["ink_divnorm=1"],
        panel_frames={1, 2},
        verify=None,
    )
    assert seen, "a prancha não rodou inferência"
    assert not any(np.array_equal(x, raw_warp) for x in seen)
