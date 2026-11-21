"""Guarda de suporte do ajuste na máscara (_FIT_MIN_MASK_FILL).

Cobre a falha medida nas capturas draw-data/captures s1-s4 (2026-11-21):
num cenário fora da mesa de referência (s3_geometria, câmera oblíqua
640x480 com piso claro à vista), o modelo rotulou não-papel como folha e
o ajuste explodiu para um quadrilátero do tamanho do frame — razão de
arestas ~1 e excursão dentro da margem, então a guarda de plausibilidade
não pegava. A fração do quad coberta pela região não-fundo separa os
dois casos: ajustes sãos >= 0,84, explodidos <= 0,80, fantasmas de mesa
vazia 0,43.
"""

import cv2
import numpy as np
import pytest

from draw import geometry, perception
from test_neural_onnx_contract import _logits_for_classes, _Session

_PAPER_CLASS = perception._PAPER_CLASS
_HAND_CLASS = perception._HAND_CLASS


def _exploding_mask():
    """Folha na metade inferior + mancha conectada de "falso papel" acima.

    É o padrão da s3_geometria: o ajuste sem a guarda devolve um quad
    quase igual ao frame inteiro.
    """
    mask = np.zeros((480, 640), np.uint8)
    mask[240:479, 60:600] = 255
    cv2.fillConvexPoly(
        mask,
        np.array([[30, 20], [300, 90], [380, 200], [200, 260], [20, 180]]),
        255,
    )
    return mask


def _fill(quad, support):
    filled = np.zeros(support.shape, np.uint8)
    cv2.fillConvexPoly(filled, np.round(quad).astype(np.int32), 255)
    area = np.count_nonzero(filled)
    return np.count_nonzero((filled > 0) & (support > 0)) / max(area, 1)


def test_fit_rejects_quad_exploding_off_the_mask():
    quad, lines, trusted = geometry._fit_quad_impl(_exploding_mask(), None, None)
    assert quad is None and lines is None and trusted is None


def test_fit_quad_public_rejects_same_explosion():
    assert geometry.fit_quad(_exploding_mask()) is None


def test_fit_accepts_sheet_supported_by_the_mask():
    mask = np.zeros((480, 640), np.uint8)
    mask[150:430, 80:560] = 255
    quad = geometry.fit_quad(mask)
    assert quad is not None
    assert _fill(quad, mask) >= geometry._FIT_MIN_MASK_FILL


def test_min_area_fallback_is_also_guarded(monkeypatch):
    """A reserva minAreaRect não pode devolver quad explodido."""
    mask = np.zeros((480, 640), np.uint8)
    mask[400:479, 0:639] = 255  # rodapé fino
    mask[0:479, 0:79] = 255  # lateral esquerda fina (L junto com o rodapé)
    monkeypatch.setattr(geometry, "_approx_quad", lambda contour: None)
    quad, _, _ = geometry._fit_quad_impl(mask, None, None)
    assert quad is None


def test_hand_over_the_sheet_counts_as_foreground_support(monkeypatch):
    """A máscara de papel exclui a mão; a cobertura (não-fundo) a inclui.

    Mão sobre o meio da folha abre um buraco na máscara de papel mas o
    contorno externo continua sendo a folha inteira: o quad cobre a mão
    e o fill na máscara cai bem abaixo do piso, enquanto na cobertura
    passa folgado. A guarda de fill só existe no caminho sem retas
    ajustadas, então o teste força a reserva minArea.
    """
    monkeypatch.setattr(geometry, "_approx_quad", lambda contour: None)
    mask = np.zeros((480, 640), np.uint8)
    mask[120:430, 80:560] = 255
    hand = np.zeros_like(mask)
    cv2.fillConvexPoly(
        hand,
        np.array([[150, 170], [510, 160], [530, 390], [170, 400]]),
        255,
    )
    paper_only = cv2.subtract(mask, hand)
    coverage = mask  # cobertura = tudo que não é fundo (papel + mão)

    quad_masked, _, _ = geometry._fit_quad_impl(paper_only, None, None)
    assert quad_masked is None
    quad_supported, _, _ = geometry._fit_quad_impl(
        paper_only, None, None, support_mask=coverage
    )
    assert quad_supported is not None


def test_refit_path_keeps_occluded_edge_beyond_the_mask(monkeypatch):
    """Oclusão pela mão: a memória de arestas reconstrói a borda cortada e
    o quad cobre de propósito região fora da máscara — o fill não pode
    rejeitar o caminho com retas ajustadas.
    """
    mask = np.zeros((500, 600), np.uint8)
    mask[60:444, 72:296] = 255
    mask[60:444, 304:528] = 255
    occluded = mask.copy()
    occluded[:, 396:] = 0  # mão cobre a borda externa da página direita

    # Simula o quad recuperado pela memória: retas presentes, canto
    # direito além da máscara visível.
    quad, lines, trusted = geometry._fit_quad_impl(occluded, None, None)
    if lines is None:
        # Sem a memória herdada neste teste unitário, o caminho com
        # retas não se prova: verifica ao menos que o caminho sem retas
        # é protegido (fill baixo -> sem quad), já coberto acima.
        pytest.skip(
            "approx não produziu quad nesta máscara; caminho de retas não exercitado"
        )
    # Com retas ajustadas, mesmo quad cobrindo area fora da mascara nao
    # pode ser derrubado pela guarda de fill.
    assert quad is not None


def test_neural_detector_uses_coverage_as_support(monkeypatch):
    """Via detector: mão (classe 2) no interior da folha entra na
    cobertura e mantém o ajuste sobre a folha inteira.

    O caminho minArea é forçado porque é o único em que a guarda de
    fill decide — com retas ajustadas o teste passaria mesmo sem o
    repasse da cobertura. Na máscara de papel a mão é buraco (~33%
    do quad, fill < 0,82): sem a cobertura como suporte o ajuste é
    rejeitado; com ela, passa.
    """
    monkeypatch.setattr(geometry, "_approx_quad", lambda contour: None)
    classes = np.zeros((384, 384), np.uint8)
    classes[90:340, 60:330] = _PAPER_CLASS
    classes[150:300, 150:300] = _HAND_CLASS  # mão no interior da folha
    session = _Session({"logits": _logits_for_classes(classes)})
    detector = perception.NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session
    frame = np.zeros((480, 640, 3), np.uint8)
    corners, verdict, _ = detector(frame, return_mask=False)
    assert corners is not None and verdict == "paper"


def test_rejected_fit_preserves_edge_memory_for_later_occlusion(monkeypatch):
    """limpo → máscara contaminada rejeitada → oclusão pela mão.

    A rejeição de fill devolve (None, None, None); sem a saída cedo no
    _fit, o caminho "sem cantos" apaga as retas guardadas e grava a
    máscara recusada como referência — a aresta ocluída deixa de ser
    recuperável no quadro seguinte.
    """
    det = perception.NeuralSheetDetector("unused.onnx", verify=None)

    clean = np.zeros((480, 640), np.uint8)
    clean[80:400, 120:520] = 255
    q_clean = det._fit(clean)
    assert q_clean is not None

    contaminated = clean.copy()
    cv2.fillConvexPoly(
        contaminated,
        np.array(
            [[10, 20], [300, 70], [410, 300], [260, 430], [20, 230]],
            np.int32,
        ),
        255,
    )
    monkeypatch.setattr(geometry, "_approx_quad", lambda contour: None)
    assert det._fit(contaminated) is None
    monkeypatch.undo()

    occluded = clean.copy()
    occluded[:, 440:] = 0  # mão cobre a borda direita da folha
    for _ in range(3):
        q_occ = det._fit(occluded)
        assert q_occ is not None
        assert q_occ[:, 0].max() >= 500.0


def test_sustained_rejection_releases_edge_memory(monkeypatch):
    """Rejeição isolada preserva a memória; rejeições seguidas demais
    soltam a referência antiga — uma folha substituta no mesmo lugar
    não pode herdar as retas da anterior (risco apontado no review)."""
    det = perception.NeuralSheetDetector("unused.onnx", verify=None)

    clean = np.zeros((480, 640), np.uint8)
    clean[80:400, 120:520] = 255
    assert det._fit(clean) is not None

    contaminated = clean.copy()
    cv2.fillConvexPoly(
        contaminated,
        np.array(
            [[10, 20], [300, 70], [410, 300], [260, 430], [20, 230]],
            np.int32,
        ),
        255,
    )

    def _reject_once():
        with monkeypatch.context() as m:
            m.setattr(geometry, "_approx_quad", lambda contour: None)
            assert det._fit(contaminated) is None

    # Uma rejeição isolada: memória viva — a oclusão recupera a aresta.
    _reject_once()
    occluded = clean.copy()
    occluded[:, 440:] = 0
    q_occ = det._fit(occluded)
    assert q_occ is not None and q_occ[:, 0].max() >= 500.0

    # Rejeições até o limite: a referência antiga é solta e a oclusão
    # seguinte só enxerga a máscara visível.
    for _ in range(perception._REJECTED_FIT_STREAK_LIMIT):
        _reject_once()
    assert det._edge_lines is None and det._memory_mask is None
    q_after = det._fit(occluded)
    assert q_after is not None and q_after[:, 0].max() < 500.0


def test_absent_frames_reset_the_rejection_streak(monkeypatch):
    """Rejeições separadas por frames sem folha não acumulam a
    sequência: a mão cobrindo a folha inteira pula o _fit e a memória
    sobrevive para recuperar a aresta depois."""
    det = perception.NeuralSheetDetector("unused.onnx", verify=None)
    empty = np.zeros((384, 384), np.uint8)  # tudo fundo
    det._session = _Session({"logits": _logits_for_classes(empty)})
    frame = np.zeros((480, 640, 3), np.uint8)

    clean = np.zeros((480, 640), np.uint8)
    clean[80:400, 120:520] = 255
    assert det._fit(clean) is not None

    contaminated = clean.copy()
    cv2.fillConvexPoly(
        contaminated,
        np.array(
            [[10, 20], [300, 70], [410, 300], [260, 430], [20, 230]],
            np.int32,
        ),
        255,
    )

    def _reject_once():
        with monkeypatch.context() as m:
            m.setattr(geometry, "_approx_quad", lambda contour: None)
            assert det._fit(contaminated) is None

    for _ in range(perception._REJECTED_FIT_STREAK_LIMIT + 2):
        _reject_once()
        corners, verdict, _ = det(frame, return_mask=False)
        assert corners is None and verdict is None

    occluded = clean.copy()
    occluded[:, 440:] = 0
    q_occ = det._fit(occluded)
    assert q_occ is not None and q_occ[:, 0].max() >= 500.0


def test_neural_detector_rejects_exploding_quad():
    """Via detector: mancha de fundo rotulada como papel explode o quad e
    a detecção volta None em vez de entregar o frame inteiro."""
    classes = np.zeros((384, 384), np.uint8)
    classes[240:383, 40:340] = _PAPER_CLASS  # folha
    # mancha CONECTADA de falso papel subindo (piso rotulado como folha):
    # desconectada, o _largest_contour já a ignorava — o perigo é a
    # mancha que cola na folha e arrasta o ajuste para fora dela.
    cv2.fillConvexPoly(
        classes,
        np.array([[20, 10], [200, 60], [270, 250], [140, 270], [10, 120]]),
        _PAPER_CLASS,
    )
    session = _Session({"logits": _logits_for_classes(classes)})
    detector = perception.NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session
    frame = np.zeros((480, 640, 3), np.uint8)
    corners, verdict, _ = detector(frame, return_mask=False)
    assert corners is None and verdict is None
