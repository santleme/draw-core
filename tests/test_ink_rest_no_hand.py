"""Descanso da tinta sem sinal de mao: cadencia cai com a tinta parada.

Clone sem MediaPipe nao tem porta de ociosidade: o estado de mao
fica None para sempre e a trabalhadora rodaria na cadencia ativa
sem fim. O contrato novo:

1. Sem sinal e com a tinta mudando (provisoria muda a cada
   passada), a cadencia e a normal — nada de perder traco.
2. Sem sinal e com a tinta parada (acumulador quieto por
   _NO_HAND_REST_AFTER_STILL_PASSES passadas), a cadencia cai
   para a da sonda ociosa.
3. A tinta volta a mudar e a cadencia ativa volta junto.
"""

import numpy as np

from draw.contracts import WarpedPage
from draw.ink import worker as worker_mod

H, W = 40, 60


class _ShiftingInk:
    """Tinta falsa: parada ou com um bloco que anda a cada passada."""

    threshold = 0.7

    def __init__(self):
        self.calls = 0
        self.moving = False

    def ink_prob(self, _warped):
        self.calls += 1
        prob = np.zeros((H, W), dtype=np.float32)
        if self.moving:
            col = (self.calls * 7) % (W - 10)
            prob[5:15, col : col + 5] = 0.9
        return prob


def _pump_for(worker, generation, now, seconds, step=0.01):
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    end = now[0] + seconds
    while now[0] < end:
        worker.submit(WarpedPage(warp, generation))
        worker.pump()
        now[0] += step


def test_rest_without_hand_signal(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.30)
    monkeypatch.setattr(worker_mod, "_NO_HAND_REST_AFTER_STILL_PASSES", 3)
    now = [1000.0]
    fake = _ShiftingInk()
    worker = worker_mod.PageCleanWorker(fake, clock=lambda: now[0])
    generation = worker.new_generation()
    # Tinta parada desde o inicio: aquece na cadencia ativa e
    # depois descansa — poucas passadas na janela longa.
    _pump_for(worker, generation, now, 0.35)
    warm = fake.calls
    assert warm >= 3, "o aquecimento roda na cadencia ativa"
    before = fake.calls
    _pump_for(worker, generation, now, 0.65)
    rested = fake.calls - before
    assert rested <= 3, f"tinta parada sem mao deveria sondar devagar: {rested}"

    # Tinta andando: a cadencia ativa volta.
    fake.moving = True
    before = fake.calls
    _pump_for(worker, generation, now, 0.65)
    active = fake.calls - before
    assert active >= 6, f"tinta mudando deveria voltar a cadencia ativa: {active}"
