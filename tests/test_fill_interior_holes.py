"""Propriedade de geometry.fill_interior_holes contra uma referencia BFS.

A funcao nao tinha nenhum teste direto: a semantica dela e sutil e
central (fundo encerrado por folha volta pra cobertura; fundo ligado
a borda e mesa e segue fora), e uma regressao mudaria a cobertura em
silencio. A referencia aqui e a definicao escrita a mao: BFS a partir
da borda sobre o fundo (4-vizinhanca, como o floodFill da funcao);
fundo que a BFS nao alcanca e buraco e fecha.
"""

from collections import deque

import numpy as np

from draw.geometry import fill_interior_holes

_SEED = 20261122


def _reference(mask):
    """mask com os buracos fechados, por BFS da borda sobre o fundo."""
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    queue = deque()
    for x in range(w):
        for y in (0, h - 1):
            if mask[y, x] == 0 and not outside[y, x]:
                outside[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if mask[y, x] == 0 and not outside[y, x]:
                outside[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= ny < h
                and 0 <= nx < w
                and mask[ny, nx] == 0
                and not outside[ny, nx]
            ):
                outside[ny, nx] = True
                queue.append((ny, nx))
    return (mask != 0) | ~outside


def test_fill_interior_holes_matches_border_bfs():
    rng = np.random.default_rng(_SEED)
    for _ in range(300):
        h = int(rng.integers(1, 33))
        w = int(rng.integers(1, 33))
        density = float(rng.uniform(0.1, 0.9))
        mask = (rng.random((h, w)) < density).astype(np.uint8)
        expected = _reference(mask).astype(np.uint8)
        result = fill_interior_holes(mask)
        assert result.dtype == np.uint8
        assert np.array_equal(result, expected)


def test_fill_interior_holes_is_idempotent():
    rng = np.random.default_rng(_SEED + 1)
    for _ in range(50):
        mask = (rng.random((16, 16)) < 0.5).astype(np.uint8)
        once = fill_interior_holes(mask)
        assert np.array_equal(fill_interior_holes(once), once)


def test_fill_interior_holes_empty_and_full():
    empty = np.zeros((8, 8), np.uint8)
    assert np.array_equal(fill_interior_holes(empty), empty)
    full = np.ones((8, 8), np.uint8)
    assert np.array_equal(fill_interior_holes(full), full)


def test_fill_interior_holes_closes_enclosed_ring_only():
    mask = np.zeros((7, 7), np.uint8)
    mask[2:5, 2:5] = 1
    mask[3, 3] = 0
    result = fill_interior_holes(mask)
    assert result[3, 3] == 1
    assert result[0, 0] == 0
