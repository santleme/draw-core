"""Division normalization do runtime: sombra some, traço fica."""

import numpy as np

from draw import photometry


def _scene(h=400, w=400):
    """Papel com sombra suave à esquerda + traço escuro no centro."""
    x = np.linspace(0.55, 1.0, w, dtype=np.float32)
    illum = np.repeat(x[None, :], h, axis=0)
    img = (illum[..., None] * np.array([200, 210, 230], np.float32)).astype(np.uint8)
    img[:, w // 2 - 2 : w // 2 + 2] = (30, 60, 90)
    return img


def test_flat_paper_goes_to_white():
    out = photometry.division_normalize(np.full((300, 300, 3), 190, np.uint8))
    mid = out[10:-10, 10:-10]
    assert abs(float(mid.mean()) - 235.0) < 8.0
    assert mid.std() < 4.0


def test_shadow_is_suppressed():
    out = photometry.division_normalize(_scene())
    left = float(np.median(out[:, :100, 0]))
    right = float(np.median(out[:, -100:, 0]))
    assert abs(left - right) < 25.0


def test_stroke_survives():
    img = _scene()
    out = photometry.division_normalize(img)
    stroke = float(out[:, img.shape[1] // 2, 0].mean())
    paper = float(np.median(out[:, img.shape[1] // 2 + 40, 0]))
    assert stroke < paper - 30


def test_deterministic_uint8_same_shape():
    img = _scene()
    a, b = photometry.division_normalize(img), photometry.division_normalize(img)
    assert a.dtype == np.uint8 and a.shape == img.shape
    np.testing.assert_array_equal(a, b)
