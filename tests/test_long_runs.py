"""Forca-bruta de ruling._long_runs, sem teste direto ate aqui.

A regra da borda e o detalhe que se perde num refactor do ruling: um
trecho que encosta na divisa da trilha vale com edge_length (o resto
dele esta na outra trilha); trecho interno precisa de length inteiro.
A referencia percorre cada trecho a mao e aplica a regra escrita.
"""

import numpy as np

from draw.ink.ruling import _long_runs

_SEED = 20261122


def _reference(mask, length, edge_length):
    out = np.zeros_like(mask)
    rows, cols = mask.shape
    for r in range(rows):
        x = 0
        while x < cols:
            if not mask[r, x]:
                x += 1
                continue
            start = x
            while x < cols and mask[r, x]:
                x += 1
            need = edge_length if start == 0 or x == cols else length
            if x - start >= need:
                out[r, start:x] = True
    return out


def test_long_runs_matches_brute_force():
    rng = np.random.default_rng(_SEED)
    for _ in range(300):
        rows = int(rng.integers(1, 7))
        cols = int(rng.integers(1, 41))
        density = float(rng.uniform(0.2, 0.8))
        mask = rng.random((rows, cols)) < density
        length = int(rng.integers(1, 8))
        edge_length = int(rng.integers(1, length + 1))
        result = _long_runs(mask, length, edge_length)
        assert np.array_equal(result, _reference(mask, length, edge_length))


def test_long_runs_edge_rule_cases():
    mask = np.zeros((3, 10), dtype=bool)
    mask[0, 0:2] = True  # encosta na esquerda: vale com edge_length=2
    mask[1, 8:10] = True  # encosta na direita: idem
    mask[2, 4:6] = True  # interno de 2: nao vale com length=3
    result = _long_runs(mask, 3, 2)
    assert result[0].sum() == 2
    assert result[1].sum() == 2
    assert result[2].sum() == 0
