"""Certeza da tinta confirmada (InkConfig.confirmed_immutable).

Dentro da geracao a tinta confirmada nao sai por nada: o voto de
remocao nem conta (acumulador), o timeout de quarentena nao existe
(identidade — perda de lock nao e prova de pagina nova) e a queda
transitoria do tracker nao descarta o canvas antes do veredito
(laco). So a virada comprovada limpa. Os contrastes com o regime
legado provam que e a bandeira, nao a maquinaria, que mudou.
"""

import numpy as np

import test_page_identity as pgid
from draw.ink import accumulator as accumulator_mod
from draw.ink import identity as identity_mod
from draw.ink import registration as registration_mod
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig
from draw.settings import PipelineConfig
from test_pipeline import QUAD, _drive, _make_loop

WITNESS = (slice(15, 45), slice(15, 55))  # testemunha: 1200 px sempre lidos
TARGET = (slice(65, 75), slice(65, 85))  # alvo: 200 px que a borracha quer


def _ink():
    prob = np.zeros((100, 120), np.float32)
    prob[WITNESS] = 0.95
    prob[TARGET] = 0.95
    return prob


def _start_worker(monkeypatch, config):
    """PageCleanWorker real, passadas sem espera (mesmo arnes da
    suíte de identidade de página)."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0)
    return worker_mod.PageCleanWorker(pgid.Model(), config=config).start()


def test_confirmed_ink_is_never_removed_under_immutable():
    """Sob certeza o voto de remocao nem conta: a testemunha segue
    viva e a guarda prova papel, mas `_confirmed` nao perde pixel."""
    acc = accumulator_mod.LayeredInkAccumulator(
        config=InkConfig(confirmed_immutable=True)
    )
    prob = _ink()
    visible = np.ones(prob.shape, bool)
    for _ in range(3):
        acc.add(prob, visible, valid_mask=visible, paper_exposed=visible)
    confirmed_px = int(np.count_nonzero(acc._confirmed))
    assert confirmed_px > 0
    erased = prob.copy()
    erased[TARGET] = 0.0
    for _ in range(registration_mod._REMOVAL_PASSES + 2):
        result = acc.add(erased, visible, valid_mask=visible, paper_exposed=visible)
        assert result["removed_confirmed_px"] == 0
    assert np.count_nonzero(acc._confirmed) == confirmed_px
    assert np.all(acc._confirmed[TARGET] == 255)


def test_removal_still_works_with_legacy_config():
    """Contraste legado: o mesmo drive apaga o traco sem a bandeira —
    e a porta, nao a maquinaria, que mudou."""
    acc = accumulator_mod.LayeredInkAccumulator(config=InkConfig())
    prob = _ink()
    visible = np.ones(prob.shape, bool)
    for _ in range(3):
        acc.add(prob, visible, valid_mask=visible, paper_exposed=visible)
    erased = prob.copy()
    erased[TARGET] = 0.0
    for _ in range(registration_mod._REMOVAL_PASSES - 1):
        result = acc.add(erased, visible, valid_mask=visible, paper_exposed=visible)
        assert result["removed_confirmed_px"] == 0
    result = acc.add(erased, visible, valid_mask=visible, paper_exposed=visible)
    assert result["removed_confirmed_px"] == 200
    assert not acc._confirmed[TARGET].any()
    assert np.all(acc._confirmed[WITNESS] == 255)


def test_seeded_streak_cannot_commit_under_immutable():
    """O commit fica selado com o voto: nem um streak armado antes
    da politica (ou semeado) tira pixel da camada confirmada."""
    acc = accumulator_mod.LayeredInkAccumulator(
        config=InkConfig(confirmed_immutable=True)
    )
    prob = _ink()
    visible = np.ones(prob.shape, bool)
    for _ in range(3):
        acc.add(prob, visible, valid_mask=visible, paper_exposed=visible)
    confirmed_px = int(np.count_nonzero(acc._confirmed))
    assert confirmed_px > 0
    acc._removal_streak[TARGET] = registration_mod._REMOVAL_PASSES
    result = acc.add(prob, visible, valid_mask=visible, paper_exposed=visible)
    assert result["removed_confirmed_px"] == 0
    assert np.count_nonzero(acc._confirmed) == confirmed_px
    assert np.all(acc._confirmed[TARGET] == 255)


def test_provisional_still_retracts_under_immutable():
    """A camada provisoria nao e confirmada: o pixel visto uma vez e
    depois ausente retrata como sempre — a bandeira so protege o
    confirmado."""
    acc = accumulator_mod.LayeredInkAccumulator(
        config=InkConfig(confirmed_immutable=True)
    )
    prob = np.zeros((60, 80), np.float32)
    prob[40:50, 40:70] = 0.85
    first = acc.add(prob)
    assert np.count_nonzero(first["provisional"]) == 300
    result = first
    for _ in range(3):
        result = acc.add(np.zeros_like(prob))
    assert result["retracted_px_total"] == 300
    assert not acc._provisional.any()
    assert not acc._confirmed.any()


def test_quarantine_never_times_out_under_immutable(monkeypatch):
    """A quarentena adia ate a evidencia limpa decidir: a referencia
    nao morre por prazo e a publicacao congela com o desenho."""
    monkeypatch.setattr(identity_mod, "_PAGE_IDENTITY_MAX_DEFERRALS", 2)
    worker = _start_worker(monkeypatch, InkConfig(confirmed_immutable=True))
    try:
        canvas = pgid.settle(worker)
        worker.set_hand_presence(False)
        generation = worker.new_generation(preserve_identity=True)
        occluded = np.zeros(pgid.SHAPE, bool)
        for count in range(1, identity_mod._PAGE_IDENTITY_MAX_DEFERRALS + 3):
            pgid.drive(worker, generation, pgid.page(other=True), visible=occluded)
            event = pgid.last_event(worker)
            assert event["decision"] == "deferred_occlusion"
            assert event["quarantine_frames"] == count
            np.testing.assert_array_equal(worker._identity_reference, canvas)
            # Publicacao congelada ate o veredito: nem o quadro vazio
            # da quarentena sai pelo canal dos consumidores.
            assert worker.poll(generation, 0) is None
        decisions = {
            e["decision"] for e in worker.telemetry_snapshot()["page_identity_events"]
        }
        assert "quarantine_timeout" not in decisions
        assert worker._identity_recovery_requested
        # A mesma folha de volta resolve: same_page descongela e o
        # desenho volta inteiro na publicacao da decisao.
        pgid.drive(worker, generation)
        assert pgid.last_event(worker)["decision"] == "same_page"
        assert worker._identity_reference is None
        np.testing.assert_array_equal(worker.poll(generation, 0).confirmed, canvas)
    finally:
        worker.stop()


def test_quarantine_timeout_still_fires_without_immutable(monkeypatch):
    """Contraste legado: o mesmo cenario expira na quarentena e a
    referencia morre — e a bandeira, nao a contagem, que mudou."""
    monkeypatch.setattr(identity_mod, "_PAGE_IDENTITY_MAX_DEFERRALS", 2)
    worker = _start_worker(monkeypatch, InkConfig())
    try:
        pgid.settle(worker)
        worker.set_hand_presence(False)
        generation = worker.new_generation(preserve_identity=True)
        occluded = np.zeros(pgid.SHAPE, bool)
        for _ in range(identity_mod._PAGE_IDENTITY_MAX_DEFERRALS + 1):
            pgid.drive(worker, generation, pgid.page(other=True), visible=occluded)
        assert pgid.last_event(worker)["decision"] == "quarantine_timeout"
        assert worker._identity_reference is None
    finally:
        worker.stop()


def _reopen_after_drop(config):
    """Folha vista, drop real do tracker e reabertura: devolve o laco
    depois da reabertura, com o canvas armado antes da queda."""
    state = {"lost": False}

    def script(_i):
        if state["lost"]:
            return None, None, None
        return QUAD.copy(), "paper", None

    loop, _det, ink = _make_loop(script, config=config)
    _drive(loop, 8)
    canvas = np.ones((100, 100), np.uint8) * 255
    loop.clean_mask = canvas.copy()
    loop.clean_provisional_mask = np.ones((100, 100), np.uint8) * 128
    loop.accumulated_canvas = np.ones((100, 100, 3), np.uint8) * 200
    state["lost"] = True
    _drive(loop, 80)
    assert loop.generation_events["drop"] >= 1
    assert not loop.ink_session_open
    state["lost"] = False
    _drive(loop, 30)
    assert loop.ink_session_open
    # A reabertura pediu preservacao: a recuperacao estava pendente.
    assert ink.new_generation_calls[-1] is True
    return loop, canvas


def test_reopen_keeps_canvas_when_recovery_pending_and_immutable():
    """Sob certeza a queda transitoria nao descarta o canvas antes do
    veredito da identidade: a mascara exibida segue na tela."""
    loop, canvas = _reopen_after_drop(
        PipelineConfig(
            ink=InkConfig.production(),
            ink_auto_gain=False,
            detect_wake=False,
        )
    )
    np.testing.assert_array_equal(loop.clean_mask, canvas)
    assert loop.clean_provisional_mask is not None
    assert loop.accumulated_canvas is not None


def test_reopen_clears_canvas_when_recovery_pending_legacy():
    """Contraste legado: sem a bandeira a reabertura zera o canvas,
    como sempre fez."""
    loop, _canvas = _reopen_after_drop(PipelineConfig.legacy(paper_guard=False))
    assert loop.clean_mask is None
    assert loop.clean_provisional_mask is None
    assert loop.accumulated_canvas is None
