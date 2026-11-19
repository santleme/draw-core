"""Ganho digital da página (draw.exposure)."""

import numpy as np
import pytest

from draw import exposure


def _page(paper, ink=20, shape=(120, 90)):
    img = np.full((*shape, 3), paper, np.uint8)
    img[40:50, 10:80] = ink
    return img


def test_dark_page_is_lifted_to_target():
    out, gain = exposure.apply_paper_gain(_page(80))
    assert gain == pytest.approx(200 / 80)
    assert float(np.percentile(out, 90)) == pytest.approx(200, abs=1)


def test_bright_page_passes_untouched():
    page = _page(230)
    out, gain = exposure.apply_paper_gain(page)
    assert gain == 1.0
    assert out is page  # nem cópia: cena bem exposta não muda


def test_gain_is_capped_for_near_black_frames():
    out, gain = exposure.apply_paper_gain(_page(10, ink=2))
    assert gain == exposure._MAX_GAIN
    assert out.max() <= 30


def test_gain_preserves_level_order():
    page = _page(90, ink=30)
    out, _ = exposure.apply_paper_gain(page)
    assert out[45, 20, 0] < out[5, 5, 0]
    assert out.dtype == np.uint8 and out.shape == page.shape
