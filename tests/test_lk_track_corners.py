"""Contrato do tracking optico dos cantos (lk_track_corners).

O propagador Lucas-Kanade do quad de EXIBICAO nao tinha nenhum
teste direto: nas bancadas so o caminho de rejeicao rodava, porque
quadros uniformes perdem todos os pontos no minEigThreshold do
OpenCV. Estes testes pinam o contrato inteiro — a ida confiavel
num deslocamento pequeno e cada guarda de rejeicao (ponto perdido,
ida-e-volta inconsistente, passo maior que _LK_MAX_STEP_PX).
"""

import cv2
import numpy as np

from draw import tracker

# Quadro 240x320 com um quad ~200x140 a pelo menos 40 px das bordas:
# a janela 31x31 do LK nunca sai do quadro ao redor destes cantos.
SHAPE = (240, 320)

CORNERS = np.array(
    [[60.0, 50.0], [260.0, 55.0], [265.0, 190.0], [55.0, 185.0]],
    dtype=np.float32,
)


def _textured_gray(seed=7):
    """Ruido denso: toda janela 31x31 e localmente unica no quadro."""
    return np.random.default_rng(seed).integers(0, 256, size=SHAPE, dtype=np.uint8)


def _install_fake_lk(monkeypatch, forward_shift, backward_shift):
    """LK falso com um deslocamento fixo por chamada (ida, depois volta).

    Mesma assinatura que o modulo usa em cv2.calcOpticalFlowPyrLK —
    (prev, next, points, None) + kwargs — devolvendo
    (pontos (N,1,2) f32, status (N,1), erro (N,1)).
    """
    shifts = iter([np.float32(forward_shift), np.float32(backward_shift)])
    calls = []

    def fake(prev, nxt, points, _none, **_kwargs):
        calls.append(points.copy())
        moved = np.asarray(points, dtype=np.float32) + next(shifts)
        status = np.ones((moved.shape[0], 1), dtype=np.uint8)
        err = np.zeros((moved.shape[0], 1), dtype=np.float32)
        return moved, status, err

    fake.calls = calls
    monkeypatch.setattr(tracker.cv2, "calcOpticalFlowPyrLK", fake)
    return fake


def test_tracks_a_small_shift_within_half_a_pixel():
    """Deslocamento de (4, 3) px: LK real devolve os 4 cantos seguidos."""
    prev = _textured_gray()
    nxt = cv2.warpAffine(
        prev,
        np.float32([[1.0, 0.0, 4.0], [0.0, 1.0, 3.0]]),
        (SHAPE[1], SHAPE[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    tracked = tracker.lk_track_corners(prev, nxt, CORNERS)
    assert tracked is not None
    assert tracked.shape == (4, 2)
    error = np.linalg.norm(tracked - (CORNERS + np.float32([4.0, 3.0])), axis=1)
    assert float(error.max()) < 0.5


def test_returns_none_when_a_point_is_lost():
    """Quadro sem textura: o OpenCV marca todos os pontos perdidos."""
    blank = np.full(SHAPE, 128, dtype=np.uint8)
    assert tracker.lk_track_corners(blank, blank.copy(), CORNERS) is None


def test_step_above_max_step_px_returns_none(monkeypatch):
    """Passo > _LK_MAX_STEP_PX e rejeitado mesmo com volta perfeita.

    A volta cai exatamente sobre os cantos de origem (erro ida-e-volta
    zero), entao a unica guarda que pode rejeitar e a do passo.
    """
    fake = _install_fake_lk(monkeypatch, (100.0, 0.0), (-100.0, 0.0))
    assert (
        tracker.lk_track_corners(_textured_gray(), _textured_gray(8), CORNERS) is None
    )
    assert len(fake.calls) == 2


def test_fb_error_above_max_fb_error_px_returns_none(monkeypatch):
    """Ida curta (4 px < 80) com volta errando 2 px (> 1,5): rejeitado."""
    fake = _install_fake_lk(monkeypatch, (4.0, 0.0), (-2.0, 0.0))
    assert (
        tracker.lk_track_corners(_textured_gray(), _textured_gray(8), CORNERS) is None
    )
    assert len(fake.calls) == 2
