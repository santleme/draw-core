"""Ganho digital da página (draw.exposure)."""

import time

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


def _dark_paper_with_bright_desk(desk_fraction=0.14, shape=(1200, 850)):
    h, w = shape
    img = np.full((h, w, 3), 80, np.uint8)
    desk_rows = int(h * desk_fraction)
    img[:desk_rows] = 225  # mesa clara no topo do crop
    paper = np.ones((h, w), bool)
    paper[:desk_rows] = False
    return img, paper


def test_bright_desk_does_not_cancel_the_paper_gain():
    """Review #52: com >10% de mesa clara no crop o p90 da imagem inteira
    virava ~225 e o ganho 1,0, deixando o papel escuro como estava."""
    img, paper = _dark_paper_with_bright_desk()
    assert exposure.paper_gain(img) == 1.0  # sem máscara, a mesa vence
    assert exposure.paper_gain(img, paper) == pytest.approx(200 / 80, rel=0.01)
    out, gain = exposure.apply_paper_gain(img, paper)
    assert gain == pytest.approx(2.5, rel=0.01)
    assert out[800, 400, 0] == pytest.approx(200, abs=2)


def test_empty_mask_falls_back_to_the_whole_image():
    img = np.full((120, 90, 3), 80, np.uint8)
    nothing = np.zeros((120, 90), bool)
    assert exposure.paper_gain(img, nothing) == pytest.approx(2.5, rel=0.01)


def test_lut_gain_matches_the_naive_per_pixel_result():
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(90, 30, (240, 200, 3)), 0, 255).astype(np.uint8)
    out, gain = exposure.apply_paper_gain(img)
    naive = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(out, naive)


def test_gain_is_much_cheaper_than_the_naive_full_frame_version():
    """Roda no laço de UI (33 ms a 30 fps): medido ~24 ms ingênuo contra
    ~1 ms agora. Compara com a versão ingênua na MESMA máquina."""
    rng = np.random.default_rng(2)
    img = np.clip(rng.normal(80, 12, (1200, 850, 3)), 0, 255).astype(np.uint8)

    def naive():
        gray = img.mean(axis=2)
        gain = float(np.clip(200.0 / np.percentile(gray, 90), 1.0, 3.0))
        return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    def best(fn, runs=7):
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    assert best(lambda: exposure.apply_paper_gain(img)) < 0.35 * best(naive)


@pytest.mark.parametrize("period", [24, 28, 32, 40])
def test_sparse_estimate_tracks_the_full_resolution_gain_on_ruled_pages(period):
    """Review #52: o percentil sai de uma amostra com passo fixo de 8; em
    pauta de período múltiplo do passo, a amostra poderia cair sempre na
    linha (ou sempre fora dela). Compara com o p90 de resolução cheia em
    todos os deslocamentos da pauta."""
    h, w = 600, 420
    for paper in (80, 110):
        for offset in range(period):
            img = np.full((h, w, 3), paper, np.uint8)
            for y in range(offset, h, period):
                img[y : y + 2] = 25  # linha de pauta escura de 2 px
            img[200:260, 100:300] = 30  # uma mancha de tinta
            full = float(np.clip(200.0 / np.percentile(img.mean(axis=2), 90), 1.0, 3.0))
            sampled = exposure.paper_gain(img)
            assert sampled == pytest.approx(full, rel=0.02), (period, offset, paper)
