"""Propriedades de `_shift_bool_mask` e `_mask_bbox` (draw.ink.worker).

Funções puras do caminho de registro da tinta — mover o caderno é
só re-registro, e é esse deslocamento que elas calculam — sem
cobertura direta até aqui. A referência é força-bruta escrita no
próprio teste; a grade é aleatória com seed fixa, então o teste é
determinístico e não-frágil.
"""

import numpy as np

from draw.ink.worker import _mask_bbox, _shift_bool_mask


def _ref_shift(mask, dy, dx):
    out = np.zeros_like(mask)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x] and 0 <= y + dy < h and 0 <= x + dx < w:
                out[y + dy, x + dx] = True
    return out


def _ref_bbox(mask):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _random_masks(rng, count):
    for _ in range(count):
        h = int(rng.integers(1, 25))
        w = int(rng.integers(1, 25))
        yield rng.random((h, w)) < float(rng.uniform(0.0, 0.6))


def test_shift_bool_mask_matches_brute_force():
    rng = np.random.default_rng(20261122)
    for mask in _random_masks(rng, 300):
        h, w = mask.shape
        shifts = [(int(rng.integers(-14, 15)), int(rng.integers(-14, 15)))]
        shifts += [(0, 0), (h + 2, 0), (0, -(w + 1)), (-h, w)]
        for dy, dx in shifts:
            got = _shift_bool_mask(mask, dy, dx)
            assert got.dtype == bool and got.shape == mask.shape
            assert np.array_equal(got, _ref_shift(mask, dy, dx))


def test_mask_bbox_matches_brute_force():
    rng = np.random.default_rng(20261122)
    for mask in _random_masks(rng, 200):
        assert _mask_bbox(mask) == _ref_bbox(mask)


def test_mask_bbox_empty_and_single_pixel():
    assert _mask_bbox(np.zeros((5, 7), bool)) is None
    mask = np.zeros((5, 7), bool)
    mask[2, 3] = True
    assert _mask_bbox(mask) == (2, 3, 3, 4)
