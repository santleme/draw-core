"""Cadencia por atividade da tinta (frente 92): a sonda ociosa nao
governa enquanto a saida esta mudando.

O defeito medido (sessao viva 1794443693): com o sinal de presenca
dizendo "sem mao", _pass_interval devolvia _IDLE_PROBE_SECONDS em
93 de 102 intervalos — 1 passada/s — enquanto a confirmada crescia
em quase toda passada. A sonda pressupoe "sem mao, sem tinta
nova"; a saida desmentia o pressuposto e o escalonador nao ouvia a
saida.

1. Passada cuja geracao trocou entre a retirada e a execucao
   morre antes do modelo (o descarte tardio ja a jogava fora
   depois de pagar o modelo inteiro).
2. O registro de telemetria carrega o intervalo que governou a
   espera, o estado de mao na retirada e a quietude da saida.
"""

import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import worker as worker_mod

H, W = 40, 60


class _ShiftingInk:
    """Tinta falsa que imita o desenho real: cada passada ve tudo
    que ja foi desenhado mais uma celula nova (a tinta fica no
    papel), entao a confirmada cresce passada a passada — o regime
    da sessao medida. moving=False congela o mapa."""

    threshold = 0.7

    _CELL = 4
    _COLS = W // _CELL

    def __init__(self):
        self.calls = 0
        self.moving = True
        self._cells = 0

    def ink_prob(self, _warped):
        self.calls += 1
        if self.moving:
            self._cells += 1
        prob = np.zeros((H, W), dtype=np.float32)
        for index in range(self._cells):
            row, col = divmod(index, self._COLS)
            y, x = row * self._CELL, col * self._CELL
            prob[y : y + self._CELL, x : x + self._CELL] = 0.9
        return prob


@pytest.mark.slow
def test_stale_generation_pass_skips_model():
    fake = _ShiftingInk()
    worker = worker_mod.PageCleanWorker(fake)
    generation = worker.new_generation()
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    worker.submit(WarpedPage(warp, generation))
    with worker._condition:
        job = worker._take_pending_locked()
    worker.new_generation()
    assert worker._run_pass(job) is True
    assert fake.calls == 0, "passada de geracao morta nao paga o modelo"
    snapshot = worker.telemetry_snapshot()
    assert snapshot["passes_skipped_stale"] == 1
    assert snapshot["passes"] == 0


def test_telemetry_record_carries_interval_and_hand_state(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.25)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 1.0)
    now = [1000.0]
    fake = _ShiftingInk()
    worker = worker_mod.PageCleanWorker(fake, clock=lambda: now[0])
    generation = worker.new_generation()
    warp = np.zeros((H, W, 3), dtype=np.uint8)

    worker.submit(WarpedPage(warp, generation))
    assert worker.pump() is True
    worker.submit(WarpedPage(warp, generation))
    now[0] += 0.10
    assert worker.pump() is False, "a cadencia segura a segunda passada"
    now[0] += 0.20
    assert worker.pump() is True

    records = worker.telemetry_snapshot()["records"]
    assert len(records) == 2
    assert records[0]["interval_s"] is None, "primeira passada nao tem espera"
    assert records[0]["hand_state"] is None
    assert records[1]["interval_s"] == 0.25
    assert records[1]["hand_state"] is None
    assert records[1]["queue_wait_s"] == pytest.approx(0.30)
    assert "activity_quiet_passes" in records[1]
