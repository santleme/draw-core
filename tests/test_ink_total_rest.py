"""Descanso total da tinta: assentou sem mao, zero passadas.

Especificacao do Lucas (2026-11-01, repetida 2026-11-03): com
a folha parada, o app repassava tinta sobre o desenho ja
registrado sem fim. A sonda de 1 Hz da porta de ociosidade
seguia alimentando o voto. O contrato novo, dirigido com uma
tinta falsa contavel:

1. Sem mao e com a saida confirmada ainda mudando, a
   trabalhadora sonda na cadencia ociosa — o voto fecha.
2. A saida confirmada para de mudar (janela da geracao
   completa, passadas quietas bastantes): descanso total,
   ZERO passadas mesmo com submits continuos, e a telemetria
   diz resting/worker_mode "rest".
3. A mao volta e a cadencia ativa volta junto, na hora.
"""

import threading
import time

import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import worker as worker_mod

H, W = 40, 60


class _SettlingInk:
    """Tinta que confirma algo novo a cada passada ate congelar.

    Deslizando, um bloco forte anda 1 coluna por passada: a
    intersecao das 3 posicoes da janela confirma a cada passada
    (a saida confirmada muda sempre). Congelada, devolve o mesmo
    mapa: o voto fecha em ate 2 passadas e a saida para.
    """

    threshold = 0.7

    def __init__(self):
        self.calls = 0
        self.frozen = False

    def ink_prob(self, _warped):
        self.calls += 1
        prob = np.zeros((H, W), dtype=np.float32)
        col = 40 if self.frozen else 5 + (self.calls % 30)
        prob[5:15, col : col + 8] = 0.95
        return prob


def _pump_for(worker, generation, now, seconds, step=0.01):
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    end = now[0] + seconds
    while now[0] < end:
        worker.submit(WarpedPage(warp, generation))
        worker.pump()
        now[0] += step


class _Feeder:
    """Submit continuo em thread, para o teste poder esperar."""

    def __init__(self, worker, generation):
        self._worker = worker
        self._generation = generation
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        warp = np.zeros((H, W, 3), dtype=np.uint8)
        while not self._stop:
            self._worker.submit(WarpedPage(warp, self._generation))
            time.sleep(0.002)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop = True
        self._thread.join(timeout=2.0)


def _wait_resting(worker, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.stats()["resting"]:
            return True
        time.sleep(0.005)
    return False


def test_total_rest_after_settle_without_hand(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.05)
    now = [1000.0]
    fake = _SettlingInk()
    worker = worker_mod.PageCleanWorker(fake, clock=lambda: now[0])
    generation = worker.new_generation()
    _pump_for(worker, generation, now, 0.3)
    assert fake.calls >= 4, "aquecimento na cadencia ativa"

    # A mao sai e a tinta ainda muda: sonda, nao descansa.
    worker.set_hand_presence(False)
    before = fake.calls
    _pump_for(worker, generation, now, 0.35)
    probing = fake.calls - before
    assert probing >= 3, f"saida mudando sem mao deveria sondar: {probing}"
    assert not worker.stats()["resting"]

    # A tinta congela: o voto fecha e o descanso total entra.
    fake.frozen = True
    _pump_for(worker, generation, now, 0.35)
    assert worker.stats()["resting"], "o descanso total nao entrou"
    assert worker.stats()["worker_mode"] == "rest"
    before = fake.calls
    _pump_for(worker, generation, now, 0.4)
    assert fake.calls == before, (
        f"em descanso, {fake.calls - before} passadas rodaram sobre a pagina parada"
    )

    # A mao volta: acorda na hora e a cadencia ativa volta.
    worker.set_hand_presence(True)
    _pump_for(worker, generation, now, 0.3)
    assert fake.calls - before >= 4, "a mao de volta reativa"
    assert not worker.stats()["resting"]


@pytest.mark.slow
def test_total_rest_without_hand_wakes_on_real_thread(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.01)
    fake = _SettlingInk()
    fake.frozen = True
    worker = worker_mod.PageCleanWorker(fake).start()
    try:
        generation = worker.new_generation()
        worker.set_hand_presence(False)
        with _Feeder(worker, generation):
            assert _wait_resting(worker), "o descanso total nao entrou"
            assert worker.stats()["worker_mode"] == "rest"
            before = fake.calls
            time.sleep(0.04)
            assert fake.calls == before
            worker.set_hand_presence(True)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and (
                fake.calls < before + 2 or worker.stats()["resting"]
            ):
                time.sleep(0.005)
            assert fake.calls >= before + 2
            assert not worker.stats()["resting"]
    finally:
        worker.stop()


def test_production_vote_confirms_late_stroke_after_hand_exit_and_then_rests():
    """Regressao (review #74): numa pagina madura e quieta, um traco forte
    escrito logo antes da mao sair (visto 1x antes da saida e o resto apos
    a saida) precisa fechar as 3 leituras do InkConfig.production() antes
    do descanso total e depois entrar em descanso sem novo movimento."""
    from draw.ink.config import InkConfig

    class _LateDashInk:
        threshold = 0.7

        def __init__(self):
            self.calls = 0
            self.add_dash = False

        def ink_prob(self, _warped):
            self.calls += 1
            prob = np.zeros((H, W), dtype=np.float32)
            # Bloco base sempre presente na pagina.
            prob[5:15, 5:20] = 0.98
            if self.add_dash:
                prob[25:33, 35:50] = 0.98
            return prob

    now = [1000.0]
    fake = _LateDashInk()
    cfg = InkConfig.production()
    worker = worker_mod.PageCleanWorker(fake, clock=lambda: now[0], config=cfg)
    generation = worker.new_generation()
    worker.set_hand_presence(True, moved=True)
    # 1) Aquece e confirma o bloco base ate acumular passadas quietas (>= 2).
    _pump_for(worker, generation, now, 1.0)
    base_snap = worker.poll(generation, 0)
    assert base_snap is not None
    assert int(np.count_nonzero(base_snap.confirmed[5:15, 5:20])) == 10 * 15
    assert int(np.count_nonzero(base_snap.confirmed[25:33, 35:50])) == 0
    assert worker._generation_passes >= worker_mod._REST_MIN_GENERATION_PASSES
    assert worker._quiet_passes >= worker_mod._REST_QUIET_PASSES

    # 2) Usuario risca um traco curto: modelo ve 1x com a mao presente, e logo
    #    em seguida a mao sai da folha.
    fake.add_dash = True
    worker.set_hand_presence(True, moved=True)
    _pump_for(worker, generation, now, 0.16)
    calls_before_exit = fake.calls
    assert (
        int(np.count_nonzero(worker.poll(generation, 0).confirmed[25:33, 35:50])) == 0
    )
    worker.set_hand_presence(False)

    # 3) Sem nenhum novo movimento da mao, a janela pos-saida fecha as 3
    #    leituras na cadencia ativa, confirma o traco e depois descansa.
    _pump_for(worker, generation, now, 0.5)
    after_snap = worker.poll(generation, 0)
    assert int(np.count_nonzero(after_snap.confirmed[25:33, 35:50])) == 8 * 15
    assert fake.calls >= calls_before_exit + 2
    _pump_for(worker, generation, now, 2.5)
    assert worker.stats()["resting"] is True
    calls_at_rest = fake.calls
    _pump_for(worker, generation, now, 0.5)
    assert fake.calls == calls_at_rest


def test_queued_masked_frame_on_exit_legacy_vs_production():
    """Regressao (review #74): com um quadro mascarado ja na fila quando a mao
    sai numa pagina madura, o legado (InkConfig()) gasta sua unica passada
    pos-saida e descansa sem confirmar um traco visto so 1x antes da oclusao,
    enquanto o preset de producao re-arma na primeira submissao limpa e fecha
    as 3 leituras antes de descansar."""
    from draw.ink.config import InkConfig

    class _DashInk:
        threshold = 0.7

        def __init__(self):
            self.calls = 0
            self.add_dash = False

        def ink_prob(self, _warped):
            self.calls += 1
            prob = np.zeros((H, W), dtype=np.float32)
            prob[5:15, 5:20] = 0.98
            if self.add_dash:
                prob[25:33, 35:50] = 0.98
            return prob

    warp = np.zeros((H, W, 3), dtype=np.uint8)
    masked_vis = np.ones((H, W), dtype=bool)
    masked_vis[25:33, 35:50] = False

    def _run_case(cfg):
        now = [1000.0]
        fake = _DashInk()
        worker = worker_mod.PageCleanWorker(fake, clock=lambda: now[0], config=cfg)
        gen = worker.new_generation()
        worker.set_hand_presence(True, moved=True)
        _pump_for(worker, gen, now, 1.0)
        assert worker._quiet_passes >= worker_mod._REST_QUIET_PASSES

        # 1 leitura clara do traco novo com a mao presente.
        fake.add_dash = True
        worker.set_hand_presence(True, moved=True)
        now[0] += 0.3
        worker.submit(WarpedPage(warp, gen))
        assert worker.pump() is True

        # Quadros mascarados esgotam tanto os skips mascarados quanto a
        # janela inicial pos-saida, levando a trabalhadora ao descanso total
        # antes de a mascara de mao limpar.
        worker.submit(WarpedPage(warp, gen, visible=masked_vis))
        worker.set_hand_presence(False)
        for _ in range(6):
            now[0] += 0.3
            worker.submit(WarpedPage(warp, gen, visible=masked_vis))
            worker.pump()
        assert worker.stats()["resting"] is True

        # Submissoes limpas seguintes sem novo movimento da mao: no preset de
        # producao, a 1a submissao limpa re-arma a janela de 3 passadas e
        # recupera a confirmacao; no legado (1 passada), permanece em descanso.
        for _ in range(5):
            now[0] += 1.05
            worker.submit(WarpedPage(warp, gen))
            worker.pump()
        snap = worker.poll(gen, 0)
        dash_px = int(np.count_nonzero(snap.confirmed[25:33, 35:50]))
        return dash_px, worker

    legacy_px, legacy_worker = _run_case(InkConfig())
    assert legacy_px == 0
    assert legacy_worker.stats()["resting"] is True

    prod_px, prod_worker = _run_case(InkConfig.production())
    assert prod_px == 8 * 15
    assert prod_worker.stats()["resting"] is True
