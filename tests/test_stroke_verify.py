"""Testes deterministicos de ink/stroke_verify.py (mascaras sinteticas).

filter_verified() decide por componente 8-conexo: esqueletiza
(Zhang-Suen + 8-minimal), poda esporas curtas (< spur_ratio x largura
local na juncao) e conta px de arco e pontas de grau 1:

- arco < min_length_px            -> rejeitado (po/mancha)
- >= 2 pontas livres              -> aceito (traco aberto; X/T tem 3-4)
- 0 pontas                        -> aceito (laco fechado) se arco >= min
- exatamente 1 ponta              -> aceito so se toca confirmed_before

O componente aceito e mantido com os pixels ORIGINAIS (nao o esqueleto).
Acima de _VERIFY_MAX_CANDIDATE_PX ha passthrough integral.

Geometrias validadas com o afinamento do proprio repo
(export_vectors.zhang_suen_thin + make_8_minimal): arcos e contagens de
pontas comentados por teste.
"""

import cv2
import numpy as np

from draw.ink import stroke_verify as sv

_SHAPE = (64, 64)
_MIN_LEN = 12
_SPUR_RATIO = 1.5


def _mask(shape=_SHAPE):
    return np.zeros(shape, dtype=bool)


def _line(m, p0, p1, thickness=3):
    """cv2 nao desenha em bool: desenha 1s na view uint8 do proprio mask."""
    cv2.line(m.view(np.uint8), p0, p1, 1, thickness)


def _lollipop():
    """Anel (0 pontas) + cauda: exatamente 1 ponta de grau 1, arco ~45 px.

    Componente aciclico sempre tem >= 2 pontas; a unica forma
    deterministica de obter exatamente 1 e um ciclo com uma cauda.
    """
    m = _mask()
    m[20, 20:29] = True
    m[28, 20:29] = True
    m[20:29, 20] = True
    m[20:29, 28] = True
    m[24, 29:46] = True  # cauda saindo do lado direito do anel
    return m


def test_straight_stroke_is_verified():
    # Barra de 40 px -> arco ~36 px, 2 pontas.
    new = _mask()
    _line(new, (10, 32), (49, 32), 3)
    new_in, before_in = new.copy(), _mask()
    out = sv.filter_verified(
        new, before_in, min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert out.dtype == bool and out.shape == new.shape
    assert np.array_equal(out, new)  # componente aceito: pixels originais
    assert np.array_equal(new, new_in)  # entradas nao sao modificadas
    assert np.array_equal(before_in, _mask())


def test_curved_stroke_is_verified():
    # Cotovelo em L -> arco ~63 px, 2 pontas.
    new = _mask()
    _line(new, (10, 10), (10, 44), 3)
    _line(new, (10, 44), (44, 44), 3)
    out = sv.filter_verified(
        new, _mask(), min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert np.array_equal(out, new)


def test_x_crossing_is_verified():
    # Duas diagonais cruzando -> juncao central, 4 pontas.
    new = _mask()
    _line(new, (12, 12), (51, 51), 2)
    _line(new, (51, 12), (12, 51), 2)
    out = sv.filter_verified(
        new, _mask(), min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert np.array_equal(out, new)


def test_t_junction_is_verified():
    # Haste terminando no meio da barra -> 3 pontas.
    new = _mask()
    _line(new, (32, 10), (32, 30), 3)
    _line(new, (12, 30), (52, 30), 3)
    out = sv.filter_verified(
        new, _mask(), min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert np.array_equal(out, new)


def test_closed_loop_is_verified():
    # Contorno de circulo r14 -> esqueleto em anel, 0 pontas, arco ~76.
    new = _mask()
    cv2.circle(new.view(np.uint8), (32, 32), 14, 1, 1)
    out = sv.filter_verified(
        new, _mask(), min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert np.array_equal(out, new)


def test_extension_of_confirmed_stroke_is_verified():
    # Continuacao encostada na tinta confirmada (sobreposicao de ~23 px).
    before = _mask()
    _line(before, (10, 32), (35, 32), 3)
    new = _mask()
    _line(new, (33, 32), (55, 32), 3)
    out = sv.filter_verified(
        new, before, min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert np.array_equal(out, new)


def test_anchored_single_tip_is_verified():
    # O mesmo "pirulito" de 1 ponta, com a ponta da cauda mergulhada na
    # tinta confirmada: a ancoragem e o que muda o veredito.
    new = _lollipop()
    before = _mask()
    before[24, 44:60] = True  # sobrepoe a ponta da cauda (cols 44-45)
    out = sv.filter_verified(
        new, before, min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert np.array_equal(out, new)


def test_dust_component_is_rejected():
    # Mancha de 6 px -> arco esqueletico ~1 px < min_length_px.
    new = _mask()
    new[10:12, 20:23] = True
    out = sv.filter_verified(
        new, _mask(), min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert not out.any()


def test_single_free_tip_stub_is_rejected():
    # Laco+cauda: exatamente 1 ponta livre e arco >= min -> so passa
    # ancorado. Sem before, ou com before que NAO toca, e rejeitado.
    new = _lollipop()
    out = sv.filter_verified(
        new, _mask(), min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert not out.any()

    far_before = _mask()
    far_before[50:53, 5:20] = True  # tinta confirmada longe do componente
    out = sv.filter_verified(
        new, far_before, min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert not out.any()


def test_spur_pruning_recovers_a_stroke():
    # Barra de 5 px com espora lateral de 2 px: a cadeia da espora fica
    # abaixo de spur_ratio x largura local na juncao e e podada — no
    # esqueleto podado o traco volta a ter 2 pontas (veredito de traco
    # limpo). O componente inteiro passa: pixels da espora tambem ficam.
    new = _mask()
    new[29:34, 10:51] = True
    new[27:29, 30] = True  # espora de 2 px
    out = sv.filter_verified(
        new, _mask(), min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert np.array_equal(out, new)


def test_mixed_components_filter_independently():
    # Traco valido + po no mesmo frame: so o po sai.
    stroke = _mask()
    _line(stroke, (10, 32), (49, 32), 3)
    new = stroke.copy()
    new[5:7, 50:53] = True  # po de 6 px
    out = sv.filter_verified(
        new, _mask(), min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert np.array_equal(out, stroke)


def test_empty_candidate_returns_empty():
    new = _mask()
    out = sv.filter_verified(
        new, _mask(), min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert out.dtype == bool
    assert np.array_equal(out, new)


def test_passthrough_over_candidate_cap(monkeypatch):
    # Acima do teto o filtro devolve o candidato intacto — inclusive
    # componente que sozinho seria rejeitado (bloco de 100 px, arco ~1).
    monkeypatch.setattr(sv, "_VERIFY_MAX_CANDIDATE_PX", 50)
    new = _mask()
    new[5:15, 5:15] = True
    out = sv.filter_verified(
        new, _mask(), min_length_px=_MIN_LEN, spur_ratio=_SPUR_RATIO
    )
    assert np.array_equal(out, new)


def test_min_length_boundary():
    # Linha de 1 px ja e o proprio esqueleto: arco == comprimento.
    # 19 px < min 20 sai; 20 px >= min 20 fica.
    short = _mask()
    _line(short, (5, 20), (23, 20), 1)
    long = _mask()
    _line(long, (5, 40), (24, 40), 1)
    out_short = sv.filter_verified(
        short, _mask(), min_length_px=20, spur_ratio=_SPUR_RATIO
    )
    out_long = sv.filter_verified(
        long, _mask(), min_length_px=20, spur_ratio=_SPUR_RATIO
    )
    assert not out_short.any()
    assert np.array_equal(out_long, long)
