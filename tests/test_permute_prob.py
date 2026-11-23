"""Propriedades de registration._permute_prob, sem teste direto ate aqui.

A docstring da funcao registra um fato medido (frentes 23/24): a
permutacao certa e rot90 + resize de volta ao canonico, NAO o registro
com cantos rolados (escores 0,005 contra 0,23-0,27). Este teste fixa o
fato em codigo: k=0 e identidade exata, k=2 e rot90 exato, k impar
preserva a forma canonica e k vale modulo 4.
"""

import numpy as np

from draw.ink.registration import _permute_prob

_SEED = 20261122


def _probs(rng, count):
    for _ in range(count):
        h = int(rng.integers(4, 40))
        w = int(rng.integers(4, 40))
        yield rng.random((h, w)).astype(np.float32)


def test_permute_prob_k0_is_identity():
    rng = np.random.default_rng(_SEED)
    for prob in _probs(rng, 100):
        assert np.array_equal(_permute_prob(prob, 0), prob)


def test_permute_prob_k2_is_exact_rot90():
    rng = np.random.default_rng(_SEED + 1)
    for prob in _probs(rng, 100):
        assert np.array_equal(_permute_prob(prob, 2), np.rot90(prob, 2))


def test_permute_prob_odd_k_preserves_canonical_shape():
    rng = np.random.default_rng(_SEED + 2)
    for prob in _probs(rng, 100):
        for k in (1, 3):
            assert _permute_prob(prob, k).shape == prob.shape


def test_permute_prob_k_is_mod_4_and_composes():
    rng = np.random.default_rng(_SEED + 3)
    for prob in _probs(rng, 100):
        assert np.array_equal(_permute_prob(prob, 4), _permute_prob(prob, 0))
        assert np.array_equal(_permute_prob(prob, 6), _permute_prob(prob, 2))
        assert np.array_equal(_permute_prob(_permute_prob(prob, 2), 2), prob)
