"""Histerese espaco-temporal da tinta: poeira isolada nao confirma nem aparece.

Duas operacoes sobre mascaras de pagina inteira, ambas por componentes
8-conectados (cv2.connectedComponents), restritas a caixa envolvente do que
importa para ficar baratas (a tinta nova ocupa uma fracao pequena da folha):

- keep_anchored: dos candidatos fracos, so os que tocam (por 8-vizinhanca,
  direta ou por outros fracos) uma ancora forte;
- filter_provisional: da camada provisoria exibida, so componentes com
  tamanho minimo e uma ancora (evidencia forte nesta passada, ou colados a
  tinta confirmada).
"""

import cv2
import numpy as np

_KERNEL = np.ones((3, 3), np.uint8)

# Folga das comparacoes de limiar sobre mapas float32 (0.9 em float32 e
# 0.89999998): um valor exatamente no limiar tem que contar como atingido.
FLOAT_EPS = 1e-6


def _bbox(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def keep_anchored(weak, anchors):
    """Subconjunto de `weak` cujos componentes 8-conectados tocam uma ancora."""
    out = np.zeros_like(weak)
    if not weak.any():
        return out
    near = cv2.dilate(weak.view(np.uint8), _KERNEL) > 0
    work = weak | (anchors & near)
    y0, y1, x0, x1 = _bbox(work)
    roi_work = work[y0:y1, x0:x1].view(np.uint8)
    _, labels = cv2.connectedComponents(roi_work, connectivity=8)
    anchored = np.unique(labels[(anchors[y0:y1, x0:x1]) & (roi_work > 0)])
    anchored = anchored[anchored > 0]
    keep = np.isin(labels, anchored)
    out[y0:y1, x0:x1] = keep & weak[y0:y1, x0:x1]
    return out


def filter_provisional(provisional, evidence, confirmed, *, t_high, min_px):
    """Componentes provisorios com >= min_px pixels e uma ancora."""
    out = np.zeros_like(provisional)
    if not provisional.any():
        return out
    y0, y1, x0, x1 = _bbox(provisional)
    roi = provisional[y0:y1, x0:x1].view(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if n <= 1:
        return out
    strong = evidence[y0:y1, x0:x1] >= t_high - FLOAT_EPS
    # A tinta confirmada colada fica FORA da caixa do provisorio: dilata com 1
    # px de folga e recorta (senao so contaria a que cai dentro da caixa).
    py0, py1 = max(0, y0 - 1), min(provisional.shape[0], y1 + 1)
    px0, px1 = max(0, x0 - 1), min(provisional.shape[1], x1 + 1)
    grown = cv2.dilate(confirmed[py0:py1, px0:px1].view(np.uint8), _KERNEL) > 0
    touching = grown[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0]
    anchored = np.unique(labels[(strong | touching) & (roi > 0)])
    good = np.zeros(n, bool)
    good[anchored] = True
    good &= stats[:, cv2.CC_STAT_AREA] >= min_px
    good[0] = False
    out[y0:y1, x0:x1] = good[labels]
    return out
