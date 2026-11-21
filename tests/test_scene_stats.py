"""Sinais de qualidade de cena do detector neural (issue #106).

O sinal de presença da segmentação não separa "papel detectado" de
"cena estranha detectada": na captura da issue (piso claro fora da
mesa de referência) a lajota virou papel e a mancha colou no blob da
folha. predict publica last_scene_stats — o dict de _scene_stats —
e o rastro do combinado leva essas medidas à telemetria em todos os
desfechos neurais, inclusive nos sem quadrilátero, que são os da
cena fora da mesa.
"""

import numpy as np
import pytest

from draw import app as app_module
from draw import paper as paper_module
from draw.perception import NeuralSheetDetector, _scene_stats
from test_neural_onnx_contract import _logits_for_classes, _Session


def _detector(outputs):
    """NeuralSheetDetector sobre sessão falsa — sem disco nem ORT."""
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = _Session(outputs)
    return detector


def _frame(h=16, w=16):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ------------------------------------------------------- _scene_stats
def test_empty_mask_is_all_zeros():
    stats = _scene_stats(np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8))
    assert stats["coverage_ratio"] == 0.0
    assert stats["components"] == 0
    assert stats["largest_share"] == 0.0
    assert stats["border_ratio"] == 0.0
    assert stats["hand_ratio"] == 0.0


def test_single_central_blob_is_one_full_component():
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = 1  # 16 px, sem tocar a borda
    stats = _scene_stats(mask, None)
    assert stats["components"] == 1
    assert stats["largest_share"] == 1.0
    assert stats["coverage_ratio"] == 16 / 64
    assert stats["border_ratio"] == 0.0
    assert stats["hand_ratio"] is None


def test_two_separated_blobs_share_the_non_background():
    mask = np.zeros((8, 8), np.uint8)
    mask[1:4, 1:3] = 1  # 6 px
    mask[4:7, 5:8] = 1  # 9 px
    stats = _scene_stats(mask, None)
    assert stats["components"] == 2
    # o maior componente leva 9 dos 15 px de não-fundo
    assert stats["largest_share"] == 9 / 15


def test_blob_glued_to_the_border_marks_border_ratio():
    mask = np.zeros((8, 8), np.uint8)
    mask[0:4, 0:4] = 1  # toca a linha de cima e a coluna da esquerda
    stats = _scene_stats(mask, None)
    # perímetro de 8x8: 2*8 + 2*(8-2) = 28 px; o blob cobre 7 deles
    # (4 na linha de cima + 3 na coluna, o canto já contado na linha)
    assert stats["border_ratio"] == 7 / 28
    assert stats["components"] == 1
    assert stats["largest_share"] == 1.0


def test_noise_speckle_below_min_area_does_not_count():
    # Frame de 100x100: o piso de 0,5% pede ≥50 px; o salpico de 4 px
    # não conta — o filtro existe para ignorar exatamente isso.
    mask = np.zeros((100, 100), np.uint8)
    mask[50:52, 50:52] = 1
    stats = _scene_stats(mask, None)
    assert stats["components"] == 0
    # Mas a fatia do maior blob é medida sobre TODOS os componentes:
    # ali todo o não-fundo é o salpico — 1,0. Quem marca a ruindade
    # da cena é components=0 com coverage_ratio ≈ 0.
    assert stats["largest_share"] == 1.0
    assert stats["coverage_ratio"] == 4 / 10000


def test_uncounted_speckle_does_not_shrink_the_largest_share():
    # Blob contado + salpico: components filtra o salpico e a fatia
    # do maior segue medida em pixels reais (400 dos 404 não-fundo).
    mask = np.zeros((100, 100), np.uint8)
    mask[10:30, 10:30] = 1  # 400 px, conta
    mask[80:82, 80:82] = 1  # 4 px < 50, não conta
    stats = _scene_stats(mask, None)
    assert stats["components"] == 1
    assert stats["largest_share"] == 400 / 404


def test_thin_long_component_counts_by_pixels_not_polygon_area():
    # Linha de 1 px de altura: a área de polígono do contorno seria
    # zero e o piso a descartaria, mas em pixels ela passa do piso —
    # o componente existe e conta.
    mask = np.zeros((100, 100), np.uint8)
    mask[50, 10:70] = 1  # 60 px ≥ 50
    stats = _scene_stats(mask, None)
    assert stats["components"] == 1
    assert stats["largest_share"] == 1.0


# --------------------------------------------- predict publica o dict
def test_four_class_predict_exposes_stats_with_hand_ratio():
    classes = np.zeros((8, 8), np.uint8)
    classes[1:5, 1:5] = 1  # papel, 16 px
    classes[5, 5] = 2  # 1 px de mão
    detector = _detector({"logits": _logits_for_classes(classes)})
    assert detector.last_scene_stats is None

    detector.predict(_frame())

    stats = detector.last_scene_stats
    assert stats["coverage_ratio"] == 17 / 64  # cobertura inclui a mão
    assert stats["hand_ratio"] == 1 / 64
    assert stats["components"] == 1


def test_two_channel_contract_has_no_hand_ratio():
    logits = np.full((1, 2, 8, 8), -4.0, np.float32)
    logits[0, 1, 1:7, 1:7] = 4.0
    detector = _detector(
        {
            "mask_logits": logits,
            "presence_logit": np.array([2.5], np.float32),
        }
    )
    detector.predict(_frame())

    stats = detector.last_scene_stats
    assert stats["hand_ratio"] is None
    assert stats["coverage_ratio"] == 36 / 64


def test_empty_frame_clears_scene_stats():
    classes = np.zeros((8, 8), np.uint8)
    classes[1:5, 1:5] = 1
    detector = _detector({"logits": _logits_for_classes(classes)})
    detector.predict(_frame())
    assert detector.last_scene_stats is not None

    assert detector(np.zeros((0, 0, 3), np.uint8)) == (None, None, None)
    assert detector.last_scene_stats is None
    assert detector.last_coverage_raw is None


def test_last_coverage_raw_keeps_the_enclosed_background_hole():
    """O mapa cru conserva o buraco que a máscara de voto preenche.

    predict publica duas coberturas: last_coverage_mask remenda o
    fundo cercado para o voto de tinta; last_coverage_raw é o argmax
    como saiu — o que _scene_stats mede e a auditoria de cena
    reporta (achado da revisão: o fill do quad precisa enxergar o
    buraco).
    """
    classes = np.zeros((8, 8), np.uint8)
    classes[1:7, 1:7] = 1  # anel de papel
    classes[3:5, 3:5] = 0  # buraco de fundo cercado
    detector = _detector({"logits": _logits_for_classes(classes)})
    detector.predict(_frame())

    assert detector.last_coverage_raw is not None
    assert detector.last_coverage_mask is not None
    # O buraco segue fundo no mapa cru e virou não-fundo na máscara
    # de voto:
    assert detector.last_coverage_raw[3:5, 3:5].max() == 0
    assert detector.last_coverage_mask[3:5, 3:5].min() == 255
    # Fora do buraco os dois mapas concordam:
    assert (
        detector.last_coverage_raw[1:3, 1:7] == detector.last_coverage_mask[1:3, 1:7]
    ).all()


# ---------------------------- o rastro do combinado leva o dict à telemetria
class _StubNeural:
    """Dublê de detector antigo: sem o atributo last_scene_stats."""

    def __init__(self, out, logit):
        self._out = out
        self.last_presence_logit = logit

    def __call__(self, frame, *, return_mask=True):
        return self._out


@pytest.fixture
def classic(monkeypatch):
    """Clássico sempre negativo: o desfecho fica nas mãos da neural."""
    monkeypatch.setattr(
        paper_module, "detect_sheet_with_confidence", lambda _frame: (None, None)
    )


def test_trace_carries_scene_stats_in_every_neural_outcome(classic):
    frame = _frame(32, 48)
    # Desfecho sem quadrilátero fechado (presence_only ou neural).
    classes = np.zeros((8, 8), np.uint8)
    classes[0:4, 0:4] = 1  # mancha colada no canto superior esquerdo
    detector = _detector({"logits": _logits_for_classes(classes)})
    _, _, trace, _ = app_module._detect_sheet_combined_traced(frame, detector)
    assert trace["scene_stats"] == detector.last_scene_stats
    assert trace["scene_stats"]["coverage_ratio"] == 16 / 64
    assert trace["scene_stats"]["border_ratio"] == 7 / 28
    # E o veto de presença: a cena estranha sem quad da issue, que é
    # exatamente onde as medidas precisam chegar à telemetria.
    empty = _detector({"logits": _logits_for_classes(np.zeros((8, 8), np.uint8))})
    _, _, trace, _ = app_module._detect_sheet_combined_traced(frame, empty)
    assert trace["source"] == "veto"
    assert trace["scene_stats"]["coverage_ratio"] == 0.0


def test_trace_stub_without_the_attribute_gets_none(classic):
    stub = _StubNeural((None, None, None), logit=1.0)
    _, _, trace, _ = app_module._detect_sheet_combined_traced(_frame(), stub)
    assert trace["source"] == "presence_only"
    assert trace["scene_stats"] is None
