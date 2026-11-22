"""Descanso total da tinta COM a mao presente e parada.

Decisao do Lucas (2026-11-03): o descanso total so existia sem
a mao; ele segurando o caderno parado, a cadencia de 250 ms
seguia eterna. O contrato novo:

1. Mao presente, saida confirmada quieta (mesma maquinaria do
   descanso sem mao) e mao parada alem da janela de quietude:
   descanso total — ZERO passadas, worker_mode "rest".
2. A mao se mexe (sinal de atividade da assinatura da mascara,
   set_hand_presence(..., moved=True)): acorda na hora e a
   cadencia ativa volta. Sem sonda periodica de tinta.
3. Desenho ativo nunca descansa: enquanto a saida confirmada
   muda, a quietude nao fecha — mesmo sem nenhum reporte de
   movimento, e a primeira tranca.
4. Reportes de movimento continuos impedem o descanso mesmo
   com a tinta congelada: a mao em atividade nao e mao parada.
"""

import threading
import time

import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import worker as worker_mod

H, W = 40, 60


class _SettlingInk:
    """Tinta que desliza (confirma sempre) ou congela (voto fecha)."""

    threshold = 0.7

    def __init__(self):
        self.calls = 0
        self.frozen = True

    def ink_prob(self, _warped):
        self.calls += 1
        prob = np.zeros((H, W), dtype=np.float32)
        col = 40 if self.frozen else 5 + (self.calls % 30)
        prob[5:15, col : col + 8] = 0.95
        return prob


def _pump_for(worker, generation, warp, now, seconds, step=0.01):
    end = now[0] + seconds
    while now[0] < end:
        worker.submit(WarpedPage(warp, generation))
        worker.pump()
        now[0] += step


def _wait_resting(worker, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.stats()["resting"]:
            return True
        time.sleep(0.005)
    return False


def _sync_rig(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.05)
    monkeypatch.setattr(worker_mod, "_HAND_REST_STILL_SECONDS", 0.3)
    now = [1000.0]
    fake = _SettlingInk()
    worker = worker_mod.PageCleanWorker(fake, clock=lambda: now[0])
    generation = worker.new_generation()
    return fake, worker, generation, now


def test_total_rest_with_hand_present_and_still(monkeypatch):
    fake, worker, generation, now = _sync_rig(monkeypatch)
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    worker.set_hand_presence(True)
    _pump_for(worker, generation, warp, now, 0.25)
    assert fake.calls >= 4, "aquecimento na cadencia ativa"

    # Tinta congelada + mao parada: o descanso total entra
    # mesmo com a mao no quadro.
    _pump_for(worker, generation, warp, now, 0.35)
    assert worker.stats()["resting"], "o descanso com a mao presente nao entrou"
    assert worker.stats()["worker_mode"] == "rest"
    before = fake.calls
    _pump_for(worker, generation, warp, now, 0.5)
    assert fake.calls == before, (
        f"em descanso com a mao presente, {fake.calls - before} passadas rodaram"
    )

    # A mao se mexe: acorda e a cadencia ativa volta.
    # (A espera fica bem abaixo da janela de quietude
    # de 0,3 s: o descanso nao pode reengatar antes da
    # leitura.)
    worker.set_hand_presence(True, moved=True)
    _pump_for(worker, generation, warp, now, 0.15)
    assert fake.calls - before >= 4, "o movimento nao acordou"
    assert not worker.stats()["resting"]


class _GrowingInk:
    """Barra que cresce 1 coluna por passada: tinta nova sempre.

    Desenho de verdade nao retraca o que ja confirmou — produz
    tinta nova sem parar, e a frente de confirmacao do voto
    anda junto. (Um bloco que desliza e volta por cima do que
    ja confirmou NAO serve aqui: sobre o traco confirmado ele
    nao confirma nada novo, e a saida fica quieta de verdade.)
    """

    threshold = 0.7

    def __init__(self):
        self.calls = 0

    def ink_prob(self, _warped):
        self.calls += 1
        prob = np.zeros((H, 240), dtype=np.float32)
        prob[5:15, 5 : 5 + min(self.calls, 220)] = 0.95
        return prob


def test_active_drawing_never_rests_even_without_motion_reports(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.05)
    monkeypatch.setattr(worker_mod, "_HAND_REST_STILL_SECONDS", 0.3)
    now = [1000.0]
    fake = _GrowingInk()
    worker = worker_mod.PageCleanWorker(fake, clock=lambda: now[0])
    generation = worker.new_generation()
    worker.set_hand_presence(True)
    warp = np.zeros((H, 240, 3), dtype=np.uint8)
    # Bem alem da janela de quietude da mao, sem nenhum
    # reporte de movimento: a saida mudando segura acordado.
    _pump_for(worker, generation, warp, now, 1.2)
    assert not worker.stats()["resting"], "descansou no meio do desenho ativo"
    before = fake.calls
    _pump_for(worker, generation, warp, now, 0.3)
    assert fake.calls - before >= 4, "a cadencia ativa caiu desenhando"


def test_motion_reports_block_rest_with_frozen_ink(monkeypatch):
    fake, worker, generation, now = _sync_rig(monkeypatch)
    worker.set_hand_presence(True)
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    # Tinta congelada, mas a mao segue se mexendo: reportes
    # de movimento a cada 0,1 s por 1 s inteiro.
    for _ in range(10):
        worker.submit(WarpedPage(warp, generation))
        worker.pump()
        worker.set_hand_presence(True, moved=True)
        now[0] += 0.1
    assert not worker.stats()["resting"], "descansou com a mao em atividade"
    # Os reportes param: a quietude fecha e o descanso entra.
    _pump_for(worker, generation, warp, now, 0.4)
    assert worker.stats()["resting"], "sem movimento, o descanso nao entrou"


@pytest.mark.slow
def test_total_rest_with_hand_present_wakes_on_real_thread(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_HAND_REST_STILL_SECONDS", 0.04)
    fake = _SettlingInk()
    worker = worker_mod.PageCleanWorker(fake).start()
    generation = worker.new_generation()
    stop = threading.Event()

    def feeder():
        warp = np.zeros((H, W, 3), dtype=np.uint8)
        while not stop.is_set():
            worker.submit(WarpedPage(warp, generation))
            time.sleep(0.002)

    thread = threading.Thread(target=feeder, daemon=True)
    try:
        worker.set_hand_presence(True)
        thread.start()
        assert _wait_resting(worker), "o descanso com a mao presente nao entrou"
        assert worker.stats()["worker_mode"] == "rest"
        before = fake.calls
        time.sleep(0.04)
        assert fake.calls == before
        worker.set_hand_presence(True, moved=True)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and (
            fake.calls < before + 2 or worker.stats()["resting"]
        ):
            time.sleep(0.005)
        assert fake.calls >= before + 2
        assert not worker.stats()["resting"]
    finally:
        stop.set()
        thread.join(timeout=2.0)
        worker.stop()


class _SettableMasker:
    def __init__(self):
        self.current = None

    def mask(self, _frame, timestamp_ms=None):
        return self.current


def _block(frame_shape, top, left, size):
    mask = np.zeros(frame_shape, dtype=np.uint8)
    mask[top : top + size, left : left + size] = 255
    return mask
