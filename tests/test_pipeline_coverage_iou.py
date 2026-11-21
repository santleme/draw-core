"""IoU de cobertura por componente conexo (issue #106).

A união do IoU quad×cobertura somava TODO o não-fundo do frame: a
mancha de piso claro que o segmentador rotula como papel fora da mesa
de referência entrava inteira na união e o IoU do quad certo caía
abaixo do piso — falso negativo que vetava passadas de tinta válidas.
Estes testes fixam a medida nova (_coverage_iou): só os componentes
conexos que tocam o quad (dilatado pelo anel de rasterização) entram
na união.
"""

import numpy as np
import pytest

from draw.pipeline import (
    _COVERAGE_IOU_RING_FRAC,
    _COVERAGE_IOU_RING_MIN_PX,
    _coverage_iou,
)
from test_pipeline import QUAD, _drive, _make_loop, _mask_block

SHAPE = (720, 1280)


def _ring_px(shape=SHAPE):
    """O anel que _coverage_iou dilata o quad nesta grade."""
    return max(_COVERAGE_IOU_RING_MIN_PX, round(_COVERAGE_IOU_RING_FRAC * min(shape)))


def _old_union_iou(mask):
    """A conta antiga, para referência: união com todo o não-fundo."""
    quad = np.zeros(mask.shape, bool)
    # fillConvexPoly inclui a borda: o QUAD rasterizado cobre
    # x em [100, 500] e y em [100, 400] — 401x301 px.
    quad[100:401, 100:501] = True
    mask_u = mask > 0
    return float((quad & mask_u).sum() / (quad | mask_u).sum())


def test_disjoint_false_blob_leaves_the_union():
    """O caso da issue: quad certo na folha + mancha falsa disjunta.

    O piso claro lido como papel cobre ~57% do frame a 50 px do quad:
    fora da união, o IoU volta ao da folha sozinha (~0,99). Pela conta
    antiga a mancha entrava inteira e o IoU caía a ~0,19, abaixo do
    piso 0,2 — a passada de tinta válida era vetada.
    """
    sheet = _mask_block(100, 400, 100, 500)  # coincide com o QUAD
    mask = sheet.copy()
    mask[:, 550:] = 255  # mancha falsa do outro lado do frame
    clean = _coverage_iou(QUAD, sheet)
    assert clean > 0.9
    assert _coverage_iou(QUAD, mask) == pytest.approx(clean)
    assert _old_union_iou(mask) < 0.2  # a medida antiga vetava o quad certo


def test_quad_inside_a_giant_blob_still_measures_low():
    """Modo preservado: o quad dentro de um blob falso gigante.

    O componente toca o quad e entra inteiro na união — é a única
    guarda de runtime desse modo: IoU = |quad|/|blob| (~0,13 aqui),
    abaixo do piso.
    """
    mask = np.full(SHAPE, 255, np.uint8)  # frame inteiro lido como papel
    iou = _coverage_iou(QUAD, mask)
    assert 0.0 < iou < 0.2


def test_empty_desk_quad_far_from_mask_measures_zero():
    """Mesa vazia: máscara longe do quad, interseção nula → 0."""
    mask = _mask_block(50, 400, 900, 1100)
    assert _coverage_iou(QUAD, mask) == 0.0


def test_blob_inside_the_ring_counts_and_beyond_does_not():
    """A margem de rasterização: a < ring_px do quad o blob conta na
    união — a folha real com a borda rasterizada a poucos px do
    polígono não pode desconectar —; a > ring_px, não."""
    sheet = _mask_block(100, 400, 100, 500)
    clean = _coverage_iou(QUAD, sheet)
    near = sheet.copy()
    near[100:400, 500 + _ring_px() // 2 : 900] = 255  # a ~ring/2 px do quad
    assert _coverage_iou(QUAD, near) < clean
    far = sheet.copy()
    far[100:400, 500 + 4 * _ring_px() : 900] = 255  # a ~4*ring px do quad
    assert _coverage_iou(QUAD, far) == pytest.approx(clean)


def test_empty_coverage_mask_measures_zero():
    """Máscara toda fundo: IoU 0 sem erro."""
    assert _coverage_iou(QUAD, np.zeros(SHAPE, np.uint8)) == 0.0


def test_method_returns_none_without_mask_and_delegates_with_it():
    """O método segue None sem _coverage_mask e delega com ela."""
    loop, _det, _ink = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    assert loop._coverage_mask is None
    assert loop._quad_coverage_iou(QUAD) is None
    mask = _mask_block(100, 400, 100, 500)
    loop._coverage_mask = mask
    assert loop._quad_coverage_iou(QUAD) == _coverage_iou(QUAD, mask)


def test_detached_hand_still_counts_against_the_quad():
    """A mão desconectada segue na união: o piso 0,2 depende dela.

    Reproduz o caso da revisão: quad falso com sobreposição local
    alta + mão gigante desconectada além do anel. O piso foi
    calibrado sobre a união total (papel+mão) — a mesa fantasma só
    afunda por causa dos pixels de mão fora do quad. Excluir a mão
    por desconexão devolve o fantasma à passada de tinta; a exclusão
    por componente vale só para o não-fundo sem classe própria.
    """
    blob = _mask_block(150, 400, 150, 500)  # ~88k px de falso papel sob o quad
    hand = _mask_block(0, 720, 640, 1280)  # mão grande do outro lado
    coverage = ((blob | hand) > 0).astype(np.uint8) * 255
    # Sem o termo mão, o fantasma cruza o piso — o falso positivo
    # que a união total sempre vetou:
    assert _coverage_iou(QUAD, coverage) > 0.2
    assert _coverage_iou(QUAD, coverage, hand) < 0.2


def test_hand_bridge_does_not_pull_false_paper_into_the_union():
    """Mão entre a folha e a mancha falsa não arrasta a mancha.

    A mão entra inteira na união, mas a componentização roda sobre o
    não-fundo SEM a classe mão (achado da revisão): uma mancha de
    falso papel que só alcança o quad através de pixels de mão fica
    fora — a mão não pode virar ponte para um blob espúrio.
    """
    sheet = _mask_block(100, 400, 100, 500)  # sob o QUAD
    bridge = _mask_block(150, 350, 500, 560)  # mão colada na folha
    false_paper = _mask_block(150, 350, 560, 800)  # só alcança via mão
    coverage = ((sheet | bridge | false_paper) > 0).astype(np.uint8) * 255
    # Sem máscara de mão a ponte cola tudo num componente só e a
    # mancha entra na união — é o limite do modelo, não da medida:
    assert _coverage_iou(QUAD, coverage) < _coverage_iou(QUAD, sheet)
    with_hand = _coverage_iou(QUAD, coverage, bridge)
    # A mancha atrás da ponte some da união: mesmo IoU de quando ela
    # nem existia na cobertura.
    without_blob = ((sheet | bridge) > 0).astype(np.uint8) * 255
    assert with_hand == pytest.approx(_coverage_iou(QUAD, without_blob, bridge))
    assert with_hand > 0.8


def test_filled_gap_still_bridges_known_limitation():
    """LIMITAÇÃO conhecida: o vão preenchido reconecta a mancha.

    A cobertura do pipeline é a máscara de voto — buracos internos
    já preenchidos. Um anel de dedos ao redor do vão entre folha e
    mancha faz o preenchimento marcar o vão como não-fundo: os
    pixels remendados não são classe mão e sobrevivem à remoção —
    a mancha volta a contar (fixar o comportamento residual, não
    o ideal; o mapa cru não atravessa o contrato DetectionResult).
    """
    sheet = _mask_block(100, 400, 100, 500)
    gap = _mask_block(150, 350, 500, 560)  # vão que a mão cerrou e o fill marcou
    false_paper = _mask_block(150, 350, 560, 800)
    # Dedos em faixas acima e abaixo do vão: fecham o anel com a
    # folha à esquerda e a mancha à direita SEM sobrepor o vão nem a
    # mancha — os pixels remendados do vão não são classe mão e
    # seguem ligando a mancha à folha depois da remoção.
    hand = _mask_block(140, 150, 495, 575) | _mask_block(350, 360, 495, 575)
    coverage = ((sheet | gap | false_paper | hand) > 0).astype(np.uint8) * 255
    bridged = _coverage_iou(QUAD, coverage, hand)
    assert bridged < _coverage_iou(QUAD, sheet, hand)
    assert bridged < 0.7  # a mancha voltou a pesar — resíduo medido


def test_ink_pass_vetoed_when_distant_hand_fills_the_scene():
    """Ponta a ponta: o fantasma com mão desconectada não sobe.

    Mesma montagem do teste acima no laço real — coverage_mask
    carrega o blob local e a mão (não-fundo), hand_mask carrega a
    classe mão, como o detector combinado publica. A mão está fora
    do quad, então a oclusão semântica passa limpa: o veto vem do
    piso de IoU, não da oclusão.
    """
    blob = _mask_block(150, 400, 150, 500)
    hand = _mask_block(0, 720, 640, 1280)
    coverage = ((blob | hand) > 0).astype(np.uint8) * 255
    loop, _det, ink = _make_loop(
        lambda _i: (
            QUAD.copy(),
            "paper",
            {"coverage_mask": coverage, "hand_mask": hand},
        )
    )
    _drive(loop, 40)
    assert loop.tracker.corners is not None
    assert ink.submitted == []


def test_ink_pass_submits_with_disjoint_false_blob():
    """Ponta a ponta no laço: a mancha disjunta não derruba a passada.

    Reproduz o falso negativo da issue no caminho real — mesma máscara
    (folha sob o quad + mancha de ~57% do frame a 50 px) e piso 0,2:
    a passada de tinta volta a subir.
    """
    mask = _mask_block(100, 400, 100, 500)
    mask[:, 550:] = 255
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": mask})
    )
    _drive(loop, 40)
    assert loop.tracker.corners is not None
    assert ink.submitted, "o quad certo voltou a passar no piso de IoU"
