"""Razao entre arestas opostas x folha clipada pelo frame (issue 104).

Medida sintetica de 2026-11-21: uma A4 projetada numa camera pinhole
obliqua e cortada pela borda inferior do quadro era recusada a cada
frame pela guarda de plausibilidade — a razao entre arestas opostas
(3,1 a 7,3 nos casos medidos) media o recorte do frame, nao a
pagina. Nos pares reprovados a aresta-corte e sempre a LONGA, com os
dois vertices a ~2 px da borda; a aresta curta do par e a aresta
distante de verdade, interior. Ja a familia de colapso que a guarda
caca (razao 3,8-4,7, mesa de referencia de 2026-10-30) tem todos os
vertices interiores: o contato com a borda e o sinal que separa os
dois casos. A razao so se aplica a um par oposto em que as duas
arestas sao interiores — nenhum dos 4 vertices envolvidos a
_FIT_EDGE_BORDER_FRAC * min(H,W) da borda nem fora do quadro.
"""

import cv2
import numpy as np
import pytest

from draw import geometry
from draw.geometry import fit_plausible

# Frame de referencia 1080x1920 alimentado por mascara 384x384, o
# contrato do detector neural (scale_x=5,0 e scale_y=2,8125).
_REF_SHAPE = (1080, 1920)
_MASK_SIZE = 384
_SCALE_X = _REF_SHAPE[1] / _MASK_SIZE
_SCALE_Y = _REF_SHAPE[0] / _MASK_SIZE


def _poly_mask(shape, pts):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


def _detector_chain(mask):
    """O caminho do detector: _fit_quad_impl + fit_plausible, com o
    shape de referencia reconstruido da mascara e da escala."""
    corners, _lines, _trusted = geometry._fit_quad_impl(
        mask, None, None, _SCALE_X, _SCALE_Y
    )
    if corners is None:
        return None
    ref_shape = (mask.shape[0] * _SCALE_Y, mask.shape[1] * _SCALE_X)
    if not geometry.fit_plausible(corners, ref_shape):
        return None
    return corners


def _mask_iou(quad, mask):
    """IoU do quadrilatero ajustado (coords do frame) contra a mascara."""
    small = quad / np.array([_SCALE_X, _SCALE_Y], dtype=np.float64)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(filled, np.round(small).astype(np.int32), 255)
    inter = int(np.logical_and(filled > 0, mask > 0).sum())
    union = int(np.logical_or(filled > 0, mask > 0).sum())
    return inter / union


# ------------------------------------------------------------ unidade
def test_interior_quad_with_collapsed_edge_ratio_is_refused():
    """Razao > 3 com os 4 vertices interiores e colapso: a familia
    medida na mesa de referencia segue recusada."""
    quad = np.array(
        [[300.0, 60.0], [1500.0, 400.0], [1500.0, 640.0], [320.0, 1040.0]],
        dtype=np.float32,
    )
    assert fit_plausible(quad, _REF_SHAPE) is False


def test_ratio_with_long_edge_glued_to_bottom_border_is_accepted():
    """O caso medido da issue: a aresta LONGA do par e o corte do
    frame (ambos os vertices a ~2 px de y=1079); a curta, interior,
    e a aresta distante real da folha."""
    quad = np.array(
        [[850.0, 300.0], [1150.0, 320.0], [1500.0, 1078.5], [400.0, 1078.0]],
        dtype=np.float32,
    )
    assert fit_plausible(quad, _REF_SHAPE) is True


def test_ratio_with_long_edge_past_the_frame_is_accepted():
    """Vertice fora do quadro tambem conta como contato de borda:
    a aresta pode continuar alem do recorte."""
    quad = np.array(
        [[850.0, 300.0], [1150.0, 320.0], [1520.0, 1096.0], [390.0, 1092.0]],
        dtype=np.float32,
    )
    assert fit_plausible(quad, _REF_SHAPE) is True


def test_ratio_with_short_edge_vertex_on_border_is_accepted():
    """Um unico vertice na borda — mesmo da aresta curta — ja
    invalida o par: a aresta curta pode ser ela mesma o recorte."""
    quad = np.array(
        [[300.0, 60.0], [1918.5, 400.0], [1917.0, 640.0], [320.0, 1040.0]],
        dtype=np.float32,
    )
    assert fit_plausible(quad, _REF_SHAPE) is True


@pytest.mark.parametrize(
    "quad",
    [
        # A familia de colapso medida em 2026-10-30 no frame
        # 1289x710: esquerda 980-1.025 px contra direita 213-269 px
        # (razao 3,8-4,7), todos os vertices interiores.
        np.array(
            [[70.0, 20.0], [1160.0, 330.0], [1190.0, 585.0], [780.0, 690.0]],
            dtype=np.float32,
        ),
        np.array(
            [[60.0, 15.0], [1150.0, 300.0], [1180.0, 530.0], [800.0, 695.0]],
            dtype=np.float32,
        ),
        np.array(
            [[60.0, 12.0], [1170.0, 340.0], [1195.0, 557.0], [822.0, 697.0]],
            dtype=np.float32,
        ),
    ],
    ids=["razao-3.8", "razao-4.3", "razao-4.7"],
)
def test_measured_collapse_family_stays_refused(quad):
    assert fit_plausible(quad, (710, 1289)) is False


def test_plausible_sheet_and_invalid_input():
    quad = np.array(
        [[300.0, 200.0], [1000.0, 180.0], [980.0, 650.0], [320.0, 660.0]],
        dtype=np.float32,
    )
    assert fit_plausible(quad, _REF_SHAPE) is True
    assert fit_plausible(np.full((4, 2), np.nan), _REF_SHAPE) is False
    assert fit_plausible(np.zeros((3, 2)), _REF_SHAPE) is False


# --------------------------------------------------------------- chain
@pytest.mark.parametrize(
    "visible_polygon",
    [
        # Regiao visivel da A4 clipada em camera obliqua: trapezio
        # alto ocupando o quadro, aresta superior curta e interior,
        # base colada na borda inferior da mascara (y=383 -> 1,8 px
        # da borda do frame). Razoes medidas no quad ajustado:
        # ~7,4 / ~3,2 / ~3,1.
        [[167, 110], [217, 110], [376, 383], [8, 383]],
        [[136, 130], [248, 130], [372, 383], [12, 383]],
        [[150, 140], [262, 140], [380, 383], [30, 383]],
    ],
    ids=["clip-forte-7x", "clip-3x", "clip-3x-obliquo"],
)
def test_chain_accepts_sheet_clipped_by_frame(visible_polygon):
    """Os tres casos reais da issue: ajuste bom (IoU >= 0,6 com a
    regiao visivel) que so a razao derrubava."""
    mask = _poly_mask((_MASK_SIZE, _MASK_SIZE), visible_polygon)
    quad = _detector_chain(mask)
    assert quad is not None, "chain recusou folha real clipada pelo quadro"
    assert _mask_iou(quad, mask) >= 0.6


def test_chain_still_refuses_fully_interior_collapse():
    """Controle: mascara cujo quad ajustado e todo interior com
    razao > 3 segue recusado — ali a razao mede arestas de folha."""
    mask = _poly_mask(
        (_MASK_SIZE, _MASK_SIZE), [[80, 30], [290, 150], [300, 230], [110, 340]]
    )
    assert _detector_chain(mask) is None
