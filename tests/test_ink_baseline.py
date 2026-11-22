"""Referencia inicial da tinta (#130) sobre o PageModel: prontidao e re-zero.

A supressao do conteudo pre-existente e o PageModel (draw.ink.page_model,
coberta em tests/test_page_model.py). Aqui: o estado que a interface le e
a tecla z, que faz o conteudo a vista virar a nova referencia.
"""

import time

import numpy as np

from draw.contracts import InkBaselineState, WarpedPage
from draw.ink import occlusion
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig

H, W = 96, 128
PAPER = 235.0
INK = 40.0
Q0 = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)


class _ScriptInk:
    """Tinta alta onde o warp esta escuro (par imagem/prob consistente)."""

    threshold = 0.7

    def ink_prob(self, warped):
        gray = warped if warped.ndim == 2 else warped.mean(axis=2)
        return (gray < 100).astype(np.float32) * 0.95


def _bgr(gray):
    return np.stack([gray] * 3, axis=-1).astype(np.uint8)


def _drive(worker, generation, image, visible=None, corners=Q0):
    before = worker.stats()["passes_total"]
    worker.submit(WarpedPage(_bgr(image), generation, corners, visible))
    deadline = time.monotonic() + 5.0
    while worker.stats()["passes_total"] <= before:
        if time.monotonic() > deadline:
            raise AssertionError("a passada nao executou")
        time.sleep(0.001)


def _confirmed(worker, generation):
    snap = worker.poll(generation, 0)
    return np.zeros((H, W), dtype=np.uint8) if snap is None else snap.confirmed


def _paper():
    return np.full((H, W), PAPER, dtype=np.float32)


def test_baseline_disabled_without_page_model():
    worker = worker_mod.PageCleanWorker(_ScriptInk(), config=InkConfig())
    assert worker.baseline_state() == InkBaselineState(False, False, 0.0, 0, "disabled")
    assert worker.rezero_baseline() is False


def test_baseline_state_goes_from_seeding_to_ready(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = worker_mod.PageCleanWorker(
        _ScriptInk(), config=InkConfig(page_model=True)
    ).start()
    try:
        generation = worker.new_generation()
        assert worker.baseline_state().phase == "seeding"
        assert not worker.baseline_state().ready
        for _ in range(4):
            _drive(worker, generation, _paper())
        state = worker.baseline_state()
        assert state.phase == "ready"
        assert state.ready and state.pending_px == 0
        assert state.fraction == 1.0
    finally:
        worker.stop()


def test_baseline_state_is_partial_while_visible_area_is_pending(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = worker_mod.PageCleanWorker(
        _ScriptInk(), config=InkConfig(page_model=True)
    ).start()
    try:
        generation = worker.new_generation()
        covered = np.ones((H, W), dtype=bool)
        covered[32:64, 32:64] = False  # mao sobre um bloco
        for _ in range(4):
            _drive(worker, generation, _paper(), covered)
        # Area coberta nao semeada: nao esta pronto, mesmo sem nada
        # visivel pendente — desenhar ali viraria referencia.
        state = worker.baseline_state()
        assert state.phase == "ready_partial"
        assert state.ready and state.pending_px > 0 and state.fraction < 1.0
        # A mao sai e o bloco semeia: so entao "ready".
        for _ in range(4):
            _drive(worker, generation, _paper())
        state = worker.baseline_state()
        assert state.phase == "ready" and state.pending_px == 0
    finally:
        worker.stop()


def test_rezero_clears_canvas_and_absorbs_current_content(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = worker_mod.PageCleanWorker(
        _ScriptInk(), config=InkConfig(page_model=True)
    ).start()
    try:
        generation = worker.new_generation()
        image = _paper()
        for _ in range(3):
            _drive(worker, generation, image)
        image[50:60, 50:60] = INK  # traco novo depois da referencia
        for _ in range(4):
            _drive(worker, generation, image)
        assert np.count_nonzero(_confirmed(worker, generation)) == 100

        assert worker.rezero_baseline()
        assert worker.baseline_state().phase == "recapturing"
        _drive(worker, generation, image)
        # O canvas sai limpo na passada do re-zero...
        assert np.count_nonzero(_confirmed(worker, generation)) == 0
        assert worker.baseline_state().phase in ("seeding", "ready")
        for _ in range(6):
            _drive(worker, generation, image)
        # ...e o traco, agora parte da referencia, nao volta como tinta.
        assert np.count_nonzero(_confirmed(worker, generation)) == 0
        assert worker.baseline_state().phase == "ready"

        image[10:20, 10:20] = INK  # tinta nova depois do re-zero
        for _ in range(4):
            _drive(worker, generation, image)
        canvas = _confirmed(worker, generation)
        assert np.count_nonzero(canvas) == 100
        assert np.all(canvas[10:20, 10:20])
    finally:
        worker.stop()


def _worker_pm(**overrides):
    return worker_mod.PageCleanWorker(
        _ScriptInk(), config=InkConfig(page_model=True, **overrides)
    ).start()


def test_rezero_keeps_canvas_until_a_pass_has_usable_paper(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        image = _paper()
        for _ in range(3):
            _drive(worker, generation, image)
        image[50:60, 50:60] = INK
        for _ in range(4):
            _drive(worker, generation, image)
        before = _confirmed(worker, generation).copy()
        assert np.count_nonzero(before) == 100

        assert worker.rezero_baseline()
        blind = np.zeros((H, W), dtype=bool)  # nada a vista (sombra/mao)
        for _ in range(3):
            _drive(worker, generation, image, blind)
        # Sem papel utilizavel: o pedido segue pendente e o canvas fica.
        assert worker.baseline_state().phase == "recapturing"
        assert np.array_equal(_confirmed(worker, generation), before)

        _drive(worker, generation, image)
        assert np.count_nonzero(_confirmed(worker, generation)) == 0
    finally:
        worker.stop()


def test_worker_stays_awake_until_rezero_seeding_finishes(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        for _ in range(8):
            _drive(worker, generation, _paper())
        assert worker.rezero_baseline()
        assert not worker._at_rest()
        _drive(worker, generation, _paper())
        # Modelo novo, ainda sem semear: a pagina parada nao pode dormir.
        assert worker._pm_rezero_seeding
        assert not worker._at_rest()
        for _ in range(4):
            _drive(worker, generation, _paper())
        assert not worker._pm_rezero_seeding
        assert worker.baseline_state().phase == "ready"
    finally:
        worker.stop()


def test_new_generation_cancels_a_pending_rezero(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, _paper())
        assert worker.rezero_baseline()
        generation = worker.new_generation()
        assert not worker._pm_rezero_requested
        assert worker.baseline_state().phase != "recapturing"
        # A folha nova nao e re-zerada por um pedido da anterior.
        for _ in range(5):
            _drive(worker, generation, _paper())
        assert worker.baseline_state().phase == "ready"
    finally:
        worker.stop()


def _scribble(image, value=INK, rows=range(40, 80, 8), cols=(10, 110)):
    """Rabisco de tracos finos (2 px, passo 8): o guarda de papel trata mancha
    escura densa na janela de 32 px como mao/sombra (passo 4 ou bloco solido
    nunca confirma), entao a testemunha de producao vem de tracos esparsos.
    5 x 2 x 100 = 1000 px."""
    for row in rows:
        image[row : row + 2, cols[0] : cols[1]] = value
    return image


SCRIBBLE_PX = 1000


def _print_guide():
    printed = _paper()
    printed[20:30, 20:60] = 90.0  # guia impressa que ainda pontua como tinta
    return printed


def _dense_print():
    """Impresso denso (>40% do tile vira tinta): a guarda nunca aprende
    as marcas dele — textura exige >=60% de amostras de papel por tile."""
    printed = _paper()
    printed[24:34, 24:48] = 90.0
    return printed


def _stroke_over_print_with_witness():
    """Guia impressa + traco escuro cruzando + desenho grande separado.

    O rabisco separado (1000 px) e a testemunha de producao da remocao:
    retido >= 65% do canvas, sem baixar o piso de testemunha.
    """
    drawn = _print_guide()
    drawn[24:26, 20:60] = 20.0
    return _scribble(drawn, 20.0)


def test_erased_stroke_over_print_is_removed_from_canvas(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        for _ in range(4):
            _drive(worker, generation, _print_guide())
        assert np.count_nonzero(_confirmed(worker, generation)) == 0
        drawn = _stroke_over_print_with_witness()
        for _ in range(5):
            _drive(worker, generation, drawn)
        canvas = _confirmed(worker, generation)
        assert np.all(canvas[24:26, 20:60])
        assert np.count_nonzero(canvas[40:80]) == SCRIBBLE_PX

        # Apaga so o traco: a guia volta a aparencia semeada.
        erased = _scribble(_print_guide(), 20.0)
        for _ in range(14):
            _drive(worker, generation, erased)
        canvas = _confirmed(worker, generation)
        assert not np.any(canvas[24:26, 20:60])
        assert np.count_nonzero(canvas[40:80]) == SCRIBBLE_PX
    finally:
        worker.stop()


def test_moved_page_projects_the_occluder_veto_into_anchor_space(monkeypatch):
    """Regua imitando a guia sobre o traco com a pagina deslocada.

    A guarda mede `anomalous_px`/`suspect_ink` no warp da passada, mas
    `returned_seen` e as mascaras de baseline vivem na grade da ancora:
    sem a projecao, o veto fotometrico cobre os pixels errados e o
    retorno forjado apaga a tinta confirmada (Devin Review do PR #144).
    As mascaras sao injetadas na grade do warp (linhas 40-46); com o
    quad deslocado +4 px elas so vetam a linha 48 do traco depois de
    projetadas para a ancora.
    """
    fake_anom = np.zeros((H, W), dtype=bool)
    fake_anom[40:47, 20:60] = True
    monkeypatch.setattr(
        occlusion.PaperExposureGuard,
        "anomalous_px",
        property(lambda self: self.__dict__.get("_fake_anom", self._anomalous_px)),
    )
    monkeypatch.setattr(
        occlusion.PaperExposureGuard,
        "suspect_ink",
        property(lambda self: self.__dict__.get("_fake_sus", self._suspect_ink)),
    )
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        guide = _print_guide()
        guide[48:50, 20:60] = 90.0  # linha impressa no topo do tile 2
        for _ in range(4):
            _drive(worker, generation, guide)
        drawn = _scribble(guide, 20.0, rows=range(56, 88, 8), cols=(88, 112))
        drawn[48:50, 20:60] = 20.0  # traco sobre a linha impressa
        for _ in range(5):
            _drive(worker, generation, drawn)
        assert np.all(_confirmed(worker, generation)[48:50, 20:60])

        # A regua cobre a linha impressa (mesma luma da semente: o
        # retorno dispara) e o quad reportado desloca +4 px — o warp
        # chega deslocado -4, como num erro de deteccao.
        worker._paper_exposure._fake_anom = fake_anom
        worker._paper_exposure._fake_sus = None
        ruled = _scribble(guide, 20.0, rows=range(56, 88, 8), cols=(88, 112))
        ruled[44:64, 20:60] = 90.0
        ruled[44:64, 28:60:16] = 20.0  # marcas escuras da regua
        shifted = np.full((H, W), PAPER, dtype=np.float32)
        shifted[:-4] = ruled[4:]
        moved = Q0 + np.array([0, 4], dtype=np.float32)
        for _ in range(7):
            _drive(worker, generation, shifted, corners=moved)
        canvas = _confirmed(worker, generation)
        assert np.all(canvas[48:50, 26:58])
    finally:
        worker.stop()


def test_erased_stroke_over_dense_print_is_removed_from_canvas(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        for _ in range(4):
            _drive(worker, generation, _dense_print())
        drawn = _scribble(_dense_print(), 20.0)
        drawn[26:28, 24:48] = 20.0  # traco sobre a impressao densa
        for _ in range(5):
            _drive(worker, generation, drawn)
        canvas = _confirmed(worker, generation)
        assert np.all(canvas[26:28, 24:48])
        assert np.count_nonzero(canvas[40:80]) == SCRIBBLE_PX

        # Apaga so o traco: o impresso denso volta a aparencia semeada
        # mas segue pontuando como tinta em posicao nunca aprendida.
        erased = _scribble(_dense_print(), 20.0)
        for _ in range(14):
            _drive(worker, generation, erased)
        canvas = _confirmed(worker, generation)
        assert not np.any(canvas[26:28, 24:48])
        assert np.count_nonzero(canvas[40:80]) == SCRIBBLE_PX
    finally:
        worker.stop()


def test_erasure_needs_the_production_removal_witness(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        printed = _scribble(_paper(), 90.0, rows=range(10, 50, 8))  # impresso
        for _ in range(4):
            _drive(worker, generation, printed)
        drawn = _scribble(printed.copy(), 20.0, rows=range(10, 50, 8))
        drawn[60:70, 60:70] = 20.0  # desenho minusculo, testemunha fraca
        for _ in range(5):
            _drive(worker, generation, drawn)
        before = _confirmed(worker, generation).copy()
        assert np.count_nonzero(before[10:50]) == SCRIBBLE_PX
        assert np.count_nonzero(before[60:70]) == 100

        erased = printed.copy()
        erased[60:70, 60:70] = 20.0
        for _ in range(14):
            _drive(worker, generation, erased)
        # Retido 100 px << 65% do canvas e << 4000 px: sem testemunha a
        # remocao nao age, e o zeramento da evidencia nao pode apagar.
        assert np.array_equal(_confirmed(worker, generation), before)
    finally:
        worker.stop()


def test_stroke_that_only_lost_contrast_is_not_erased(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        for _ in range(4):
            _drive(worker, generation, _print_guide())
        drawn = _stroke_over_print_with_witness()
        drawn[24:26, 20:60] = 50.0
        for _ in range(5):
            _drive(worker, generation, drawn)
        assert np.all(_confirmed(worker, generation)[24:26, 20:60])

        # Reflexo: o traco clareia para 70, ainda 20 abaixo da guia (90) —
        # nao e "novo" (limiar 24) nem "de volta" (tolerancia 8), e o
        # detector ainda o ve. Com testemunha de producao presente.
        glare = drawn.copy()
        glare[24:26, 20:60] = 70.0
        for _ in range(14):
            _drive(worker, generation, glare)
        assert np.all(_confirmed(worker, generation)[24:26, 20:60])
    finally:
        worker.stop()


def test_dark_print_that_never_seeds_does_not_block_ready(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        image = _paper()
        image[8:72, 8:72] = 10.0  # logo preto maior que a janela da guarda
        for _ in range(8):
            _drive(worker, generation, image)
        state = worker.baseline_state()
        assert state.ready and state.phase == "ready", state
        assert state.pending_px == 0
    finally:
        worker.stop()


def test_guard_blocked_area_counts_as_pending_until_it_clears(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        image = _paper()
        image[8:72, 8:72] = 10.0  # mancha negada so pela guarda (mao "invisivel")
        for _ in range(5):
            _drive(worker, generation, image)
        # Os pixels negados recentemente nunca foram observados: contam
        # como pendentes enquanto o bloqueio e recente — "ready" so
        # depois que a mancha sai e a regiao semeia (Devin Review do
        # PR #144).
        state = worker.baseline_state()
        assert state.pending_px > 0, state
        image = _paper()
        for _ in range(6):
            _drive(worker, generation, image)
        state = worker.baseline_state()
        assert state.phase == "ready" and state.pending_px == 0
    finally:
        worker.stop()


def test_rezero_survives_identity_recovery_of_the_same_page(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        image = _paper()
        for _ in range(3):
            _drive(worker, generation, image)
        _scribble(image)
        for _ in range(4):
            _drive(worker, generation, image)
        assert np.count_nonzero(_confirmed(worker, generation)) == SCRIBBLE_PX

        assert worker.rezero_baseline()
        recovered = worker.new_generation(preserve_identity=True)
        # Queda de tracking com identidade preservada: o pedido fica.
        assert worker._pm_rezero_requested
        for _ in range(8):
            _drive(worker, recovered, image)
        decision = worker.telemetry_snapshot()["page_identity_events"][-1]["decision"]
        assert decision == "same_page"
        # Mesma folha: o re-zero aceito antes da queda roda depois dela.
        assert not worker._pm_rezero_requested
        assert np.count_nonzero(_confirmed(worker, recovered)) == 0
    finally:
        worker.stop()


def test_rezero_is_cancelled_when_the_returning_page_is_another(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker_pm()
    try:
        generation = worker.new_generation()
        image = _paper()
        for _ in range(3):
            _drive(worker, generation, image)
        _scribble(image)
        for _ in range(4):
            _drive(worker, generation, image)
        assert worker.rezero_baseline()
        recovered = worker.new_generation(preserve_identity=True)
        other = _paper()
        other[10:30, 10:30] = INK  # outra folha, sem o desenho antigo
        for _ in range(10):
            _drive(worker, recovered, other)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
        assert not worker._pm_rezero_requested
        assert worker.baseline_state().phase != "recapturing"
    finally:
        worker.stop()
