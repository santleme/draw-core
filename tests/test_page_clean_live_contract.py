"""Contrato vivo do caminho page-clean: tinta nova chega ao
canvas publicado em ate 3 passadas.

O canvas publicado aqui e a camada confirmada, que os
consumidores leem pelo poll(). A confirmacao voltou como voto
entre passadas (ver tests/test_provisional_ink.py), mas fora do
caminho de exibicao: a tela mostra a tinta na primeira passada
como provisoria. O que este teste pina e o teto do confirmado:
com tinta forte estavel, o voto fecha na 2a passada, dentro das
3 do contrato. A regua de replay
(draw-lab/replay-ruler-2026-11-01) mede o mesmo contrato sobre
warps reais: evento pequeno e evento f05 em 3 passadas.
"""

import time

import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import worker as worker_mod

H, W = 120, 160
BASE_RECT = (30, 40, 30, 60)
NEW_RECT = (70, 100, 100, 130)


def _mask(rects):
    mask = np.zeros((H, W), dtype=np.uint8)
    for y0, y1, x0, x1 in rects:
        mask[y0:y1, x0:x1] = 255
    return mask


class _FakeInk:
    threshold = 0.7

    def __init__(self):
        self.calls = 0
        self.extra = False

    def ink_mask(self, _warped):
        self.calls += 1
        rects = [BASE_RECT]
        if self.extra:
            rects.append(NEW_RECT)
        return _mask(rects)


@pytest.mark.slow
def test_new_ink_reaches_published_canvas_within_3_passes(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    # A sonda ociosa de producao (1 s) nao pode vazar para o arnes.
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    fake = _FakeInk()
    worker = worker_mod.PageCleanWorker(fake).start()
    try:
        warp = np.zeros((H, W, 3), dtype=np.uint8)
        generation = worker.new_generation()
        version = 0
        mask = None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            worker.submit(WarpedPage(warp, generation))
            got = worker.poll(generation, version)
            if got is not None:
                version, mask = got.version, got.confirmed
                if int(np.count_nonzero(mask)) == 300:
                    break
            time.sleep(0.005)
        assert mask is not None and int(np.count_nonzero(mask)) == 300

        calls_at_event = fake.calls
        fake.extra = True
        new_ref = _mask([NEW_RECT]) > 0
        entered = False
        while time.monotonic() < deadline:
            worker.submit(WarpedPage(warp, generation))
            got = worker.poll(generation, version)
            if got is not None:
                version, mask = got.version, got.confirmed
                hit = int(np.count_nonzero((mask > 0) & new_ref))
                if hit >= 0.9 * int(np.count_nonzero(new_ref)):
                    entered = True
                    break
            time.sleep(0.005)
        assert entered, "a tinta nova nunca chegou ao canvas publicado"
        passes_used = fake.calls - calls_at_event
        assert passes_used <= 3, (
            f"tinta nova levou {passes_used} passadas ate publicar; "
            "o contrato vivo e ate 3"
        )
    finally:
        worker.stop()
