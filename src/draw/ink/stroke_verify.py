"""(comeco, fim) por esqueleto: sob `stroke_endpoint_certainty`, so pixels de
componentes com par verificado sobem a confirmado; rejeitados seguem na
provisoria e tentam de novo na proxima passada.

Veredito por componente 8-conectado sobre o esqueleto de 1 px (Zhang-Suen +
8-minimal, os mesmos da exportacao vetorial), contando pontas (grau 1):

- 2+ pontas: traco aberto ou cruzamento — as duas extremidades existem;
- 0 pontas: laco fechado, comeco == fim (comprimento minimo ja conferido);
- 1 ponta: extensao real so quando a outra extremidade ancora em tinta
  confirmada; sem contato e um stub nao verificavel.

Esporas — ramos-folha curtos perto da largura local do traco — sao podadas
antes da contagem para nao inflarem o numero de pontas.
"""

import cv2
import numpy as np

from ..exporters import export_vectors as _ev
from . import hysteresis

# Teto de candidatos por passada: acima dele (adocao em massa da primeira
# passada sobre uma folha ja desenhada) entra tudo — o afinamento numpy
# nunca pode travar uma passada.
_VERIFY_MAX_CANDIDATE_PX = 150_000

_KERNEL3 = np.ones((3, 3), np.uint8)


def filter_verified(new_confirmed, confirmed_before, *, min_length_px, spur_ratio):
    """Subconjunto de `new_confirmed` a promover (mesma forma, bool)."""
    new_confirmed = np.asarray(new_confirmed, dtype=bool)
    confirmed_before = np.asarray(confirmed_before, dtype=bool)
    if not new_confirmed.any():
        return new_confirmed
    if np.count_nonzero(new_confirmed) > _VERIFY_MAX_CANDIDATE_PX:
        return new_confirmed
    y0, y1, x0, x1 = hysteresis._bbox(new_confirmed)
    # Folga de 4 px: o contato por dilatacao e o esqueleto nao morrem na borda.
    y0, x0 = max(0, y0 - 4), max(0, x0 - 4)
    y1 = min(new_confirmed.shape[0], y1 + 4)
    x1 = min(new_confirmed.shape[1], x1 + 4)
    roi_new = new_confirmed[y0:y1, x0:x1]
    roi_conf = confirmed_before[y0:y1, x0:x1]
    n, labels = cv2.connectedComponents(roi_new.view(np.uint8), connectivity=8)
    out = np.zeros_like(new_confirmed)
    roi_out = out[y0:y1, x0:x1]
    for i in range(1, n):
        comp = labels == i
        if _verified_component(comp, roi_conf, min_length_px, spur_ratio):
            roi_out[comp] = True
    return out


def _verified_component(comp, confirmed, min_length_px, spur_ratio):
    """True quando o esqueleto do componente fecha um par (comeco, fim)."""
    skel = _ev.make_8_minimal(_ev.zhang_suen_thin(comp))
    if spur_ratio > 0:
        skel = _prune_spurs(skel, comp, spur_ratio)
    if np.count_nonzero(skel) < min_length_px:
        return False  # Poeira/mancha: arco curto demais para ser traco.
    deg = _ev._degree(skel)
    ends = skel & (deg == 1)
    # Pontas 8-coladas (quina de traco espesso) contam como uma extremidade.
    n_ends = cv2.connectedComponents(ends.view(np.uint8), connectivity=8)[0] - 1
    if n_ends != 1:
        return True  # 2+: traco aberto/cruzamento; 0: laco fechado.
    # Uma ponta livre so vale quando a outra extremidade ancora no confirmado.
    near = cv2.dilate(comp.view(np.uint8), _KERNEL3) > 0
    return bool((near & confirmed).any())


def _prune_spurs(skel, comp, spur_ratio):
    """Esqueleto novo, sem ramos-folha curtos demais para a largura local.

    Quinas e cruzamentos ganham esporas falsas no esqueleto: cadeias de
    grau 2 que saem de uma juncao (grau >= 3) e morrem numa ponta — curtas
    perto da meia-largura local (transformada de distancia do componente).
    Traco de verdade e comprido e fica. Dois passes bastam: cortar uma
    espora pode revelar outra logo atras.
    """
    dist = cv2.distanceTransform(comp.view(np.uint8), cv2.DIST_L2, 3)
    h, w = skel.shape
    out = skel.copy()
    for _ in range(2):
        deg = _ev._degree(out)
        removed = False
        for ey, ex in np.argwhere(out & (deg == 1)):
            if not out[ey, ex]:
                continue  # Ja cortado por outra cadeia neste passe.
            chain = [(ey, ex)]
            py = px = jy = jx = -1
            y, x = ey, ex
            while True:
                ny = nx = -1
                for dy, dx in _ev._NEIGHBORS:
                    cy, cx = y + dy, x + dx
                    inside = 0 <= cy < h and 0 <= cx < w
                    if inside and (cy, cx) != (py, px) and out[cy, cx]:
                        ny, nx = cy, cx
                        break
                if ny < 0:
                    break  # Sem vizinho novo: cadeia morre longe de juncao.
                py, px, y, x = y, x, ny, nx
                if deg[y, x] != 2:
                    if deg[y, x] >= 3:
                        jy, jx = y, x
                    break
                chain.append((y, x))
            if jy < 0:
                continue  # Morreu em outra ponta: segmento isolado, nao espora.
            if len(chain) < spur_ratio * max(1.0, dist[jy, jx]):
                for cy, cx in chain:
                    out[cy, cx] = False
                removed = True
        if not removed:
            break
    return out
