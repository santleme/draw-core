"""ink_divnorm na trabalhadora (issue #101).

O blur de pagina inteira da division_normalize custava ~42 ms no
laco principal. O flag agora diz para a TRABALHADORA aplicar a
transformacao na passada — que tem folga de cadencia — e o laco
sobe o warp cru. A imagem que o modelo ve deve continuar
bit-identica a receita antiga do laco: fora-da-folha := media do
papel ANTES da divisao (a mesa clara nao escurece a borda do
papel) e de novo DEPOIS, sobre o normalizado.
"""

import numpy as np

from draw import photometry
from draw.contracts import WarpedPage
from draw.ink import worker as worker_mod

H, W = 60, 80
Q0 = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)


class _RecordingInk:
    """page_clean falso que guarda a imagem vista pelo modelo."""

    threshold = 0.7

    def __init__(self):
        self.seen = []

    def ink_prob(self, warped):
        self.seen.append(warped.copy())
        return np.zeros((H, W), dtype=np.float32)


def _run_one_pass(worker, page):
    worker.submit(page)
    with worker._condition:
        job = worker._take_pending_locked()
    assert worker._run_pass(job) is True


def _legacy_recipe(raw, coverage):
    """O que o laco submetia ate a #101:
    fill -> division_normalize -> fill (medias por fancy-indexing)."""
    filled = raw.copy()
    filled[~coverage] = raw[coverage].mean(axis=0).astype(np.uint8)
    normalized = photometry.division_normalize(filled)
    out = normalized.copy()
    out[~coverage] = normalized[coverage].mean(axis=0).astype(np.uint8)
    return out


def _raw_page():
    raw = np.full((H, W, 3), 220, np.uint8)  # mesa clara
    raw[5:55, 10:60] = 100  # papel mais escuro que a mesa
    coverage = np.zeros((H, W), dtype=bool)
    coverage[8:52, 12:58] = True
    return raw, coverage


def test_divnorm_image_is_bit_identical_to_the_legacy_loop_recipe():
    ink = _RecordingInk()
    worker = worker_mod.PageCleanWorker(ink, ink_divnorm=True)
    generation = worker.new_generation()
    raw, coverage = _raw_page()
    _run_one_pass(worker, WarpedPage(raw, generation, Q0, coverage=coverage))
    np.testing.assert_array_equal(ink.seen[0], _legacy_recipe(raw, coverage))


def test_divnorm_disabled_passes_the_raw_warp_untouched():
    ink = _RecordingInk()
    worker = worker_mod.PageCleanWorker(ink)
    generation = worker.new_generation()
    raw, coverage = _raw_page()
    _run_one_pass(worker, WarpedPage(raw, generation, Q0, coverage=coverage))
    np.testing.assert_array_equal(ink.seen[0], raw)


def test_divnorm_without_coverage_normalizes_the_plain_warp():
    ink = _RecordingInk()
    worker = worker_mod.PageCleanWorker(ink, ink_divnorm=True)
    generation = worker.new_generation()
    raw, _coverage = _raw_page()
    _run_one_pass(worker, WarpedPage(raw, generation, Q0))
    np.testing.assert_array_equal(ink.seen[0], photometry.division_normalize(raw))
