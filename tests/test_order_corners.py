"""Propriedades de paper._order_corners, sem teste direto ate aqui.

Todo quad do sistema nasce ordenado por esta funcao, e a cobertura
dela era so indireta (via fit_quad/deteccao). Os invariantes sao
baratos e pegam qualquer troca de convencao — a classe de bug que ja
custou a cadeia de PRs da auto-rotacao: a saida e uma permutacao dos
4 pontos de entrada, o primeiro e o de menor soma (o TL) e ordenar
de novo nao muda nada.
"""

import itertools

import numpy as np

from draw.paper import _order_corners

_SEED = 20261122


def _convex_quad(rng):
    """Quad convexo aleatorio: centro + angulos ordenados + raios."""
    center = rng.uniform(50.0, 150.0, size=2)
    angles = np.sort(rng.uniform(-np.pi, np.pi, size=4))
    radii = rng.uniform(20.0, 80.0, size=4)
    pts = center + np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=1)
    return pts.astype(np.float32)


def test_order_corners_invariants_on_random_quads():
    rng = np.random.default_rng(_SEED)
    for _ in range(500):
        quad = _convex_quad(rng)
        shuffled = quad[rng.permutation(4)]
        ordered = _order_corners(shuffled)
        assert ordered.shape == (4, 2)
        assert ordered.dtype == np.float32
        # Sai uma permutacao dos 4 pontos de entrada.
        assert sorted(map(tuple, ordered.tolist())) == sorted(map(tuple, quad.tolist()))
        # O primeiro e o de menor soma (o TL).
        assert ordered[0].sum() == quad.sum(axis=1).min()
        # Ordenar de novo nao muda nada.
        assert np.array_equal(_order_corners(ordered), ordered)


def test_order_corners_pins_tl_tr_br_bl_direction():
    # Os invariantes acima nao distinguem a convencao espelhada
    # (TL, BL, BR, TR): este caso conhecido fixa a direcao.
    quad = np.array([[10.0, 10.0], [0.0, 10.0], [10.0, 0.0], [0.0, 0.0]], np.float32)
    expected = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], np.float32
    )
    assert np.array_equal(_order_corners(quad), expected)


def test_order_corners_same_for_every_input_permutation():
    rng = np.random.default_rng(_SEED + 1)
    for _ in range(50):
        quad = _convex_quad(rng)
        expected = _order_corners(quad)
        for perm in itertools.permutations(range(4)):
            assert np.array_equal(_order_corners(quad[list(perm)]), expected)
