"""Cadencia da tinta configuravel (InkConfig.pass_interval_s, --ink-interval).

O piso de 0,25 s entre passadas foi medido com o page_clean fp32 em
mosaico (~170-230 ms por passada). Com o INT8 de pagina inteira (~90 ms)
o piso passou a ser quem segura a latencia; o campo deixa validar uma
cadencia menor ao vivo sem mudar o padrao.
"""

import sys

import pytest

from draw import app as app_module
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig


class _NoModel:
    threshold = 0.7

    def ink_prob(self, warped):  # pragma: no cover - nunca chamado aqui
        raise AssertionError


def _interval(config):
    worker = worker_mod.PageCleanWorker(_NoModel(), config=config)
    worker.set_hand_presence(True)  # mao ativa: a cadencia de escrita
    return worker._pass_interval()


def test_default_keeps_the_module_floor():
    assert InkConfig().pass_interval_s is None
    assert _interval(InkConfig()) == worker_mod._PAGE_CLEAN_INTERVAL_SECONDS == 0.25


def test_configured_interval_replaces_the_floor():
    assert _interval(InkConfig(pass_interval_s=0.15)) == 0.15


@pytest.mark.parametrize("hand", [True, False, None])
def test_only_the_floor_changes_idle_and_rest_rules_stay(hand):
    # Em qualquer estado da mao, o campo so substitui o piso: onde o
    # padrao devolve a sonda ociosa ou o descanso, o configurado tambem.
    now = [100.0]
    default = worker_mod.PageCleanWorker(
        _NoModel(), config=InkConfig(), clock=lambda: now[0]
    )
    tuned = worker_mod.PageCleanWorker(
        _NoModel(), config=InkConfig(pass_interval_s=0.15), clock=lambda: now[0]
    )
    for worker in (default, tuned):
        worker.set_hand_presence(hand)
    assert tuned._pass_interval() == 0.15
    assert default._pass_interval() == worker_mod._PAGE_CLEAN_INTERVAL_SECONDS

    # Sonda ociosa (mao fora sem confirmacao pendente, ou sem sinal apos quietude)
    for worker in (default, tuned):
        if hand is False:
            worker._confirm_once = False
        elif hand is None:
            worker._still_passes = worker_mod._NO_HAND_REST_AFTER_STILL_PASSES
    if hand in (False, None):
        assert default._pass_interval() == worker_mod._IDLE_PROBE_SECONDS
        assert tuned._pass_interval() == worker_mod._IDLE_PROBE_SECONDS

    # Descanso total: quietude + geracao madura (+ mao parada se hand=True)
    for worker in (default, tuned):
        worker._generation_passes = worker_mod._REST_MIN_GENERATION_PASSES
        worker._quiet_passes = worker_mod._REST_QUIET_PASSES
    now[0] += worker_mod._HAND_REST_STILL_SECONDS + 1.0
    if hand in (True, False):
        assert default._pass_interval() is None
        assert tuned._pass_interval() is None


def test_slow_cadence_movement_resets_quiet_streak():
    # Com intervalo maior que _HAND_REST_STILL_SECONDS (ex.: 2 s), um
    # movimento breve da mao precisa zerar _quiet_passes para o worker
    # nao voltar ao descanso antes da proxima passada agendada.
    now = [100.0]
    worker = worker_mod.PageCleanWorker(
        _NoModel(), config=InkConfig(pass_interval_s=2.0), clock=lambda: now[0]
    )
    worker.set_hand_presence(True, moved=True)
    worker._generation_passes = worker_mod._REST_MIN_GENERATION_PASSES
    worker._quiet_passes = worker_mod._REST_QUIET_PASSES
    now[0] += worker_mod._HAND_REST_STILL_SECONDS + 0.5
    assert worker._at_rest()

    # Mao faz um traco curto e para: mesmo apos _HAND_REST_STILL_SECONDS,
    # o worker segue acordado ate novas passadas reestabelecerem quietude.
    worker.set_hand_presence(True, moved=True)
    now[0] += worker_mod._HAND_REST_STILL_SECONDS + 0.5
    assert not worker._at_rest()
    assert worker._pass_interval() == 2.0


def test_motion_gate_latches_reference_after_blank_start():
    # Se a sessao comeca em folha em branco, o portao nao pode travar a
    # pagina vazia como referencia: assim que surge escrita na mesma
    # ancora, a referencia trava e movimentos seguintes sao medidos.
    import numpy as np

    from draw.ink.motion import MotionGate

    gate = MotionGate()
    anchor = np.float32([[0, 0], [850, 0], [850, 1200], [0, 1200]])
    blank = np.full((1200, 850), 235, dtype=np.uint8)
    vis = np.ones((1200, 850), dtype=np.uint8)
    assert gate.measure(blank, None, vis, anchor) is None

    rng = np.random.default_rng(7)
    written = blank.copy()
    for _ in range(180):
        y = int(rng.integers(40, 1160))
        x = int(rng.integers(40, 810))
        written[y - 2 : y + 3, x - 2 : x + 3] = 25
    # Primeira passada com textura trava a referencia (ainda sem movimento)
    assert gate.measure(written, None, vis, anchor) is None
    # Folha deslocada 2,5 px na mesma ancora agora e medida pelo portao
    shifted = np.roll(written, 3, axis=1)
    motion = gate.measure(shifted, None, vis, anchor)
    assert motion is not None
    assert motion.max_px >= 2.0


@pytest.mark.parametrize("bad", [0.0, -0.1, 5.5])
def test_config_validates_the_interval(bad):
    with pytest.raises(ValueError):
        InkConfig(pass_interval_s=bad)


def test_cli_flag_sets_it_and_markers_mode_rejects(monkeypatch):
    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    monkeypatch.setattr(sys, "argv", ["draw", "--ink-interval", "0.15", "--ink-ruling"])
    app_module.main()
    ink = seen[-1]["pipeline_config"].ink
    assert ink.pass_interval_s == 0.15 and ink.ruling_removal
    for argv in (
        ["draw", "--markers", "--ink-interval", "0.15"],
        ["draw", "--ink-interval", "0"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as error:
            app_module.main()
        assert error.value.code == 2


def test_full_live_stack_flags_combine(monkeypatch):
    # A pilha do teste de mesa: ganho, pauta, histerese, portao e cadencia.
    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "draw",
            "--ink-gain",
            "--ink-ruling",
            "--ink-hysteresis",
            "--ink-motion-veto",
            "--ink-interval",
            "0.15",
        ],
    )
    app_module.main()
    config = seen[-1]["pipeline_config"]
    ink = config.ink
    assert config.ink_auto_gain
    assert ink.ruling_removal and ink.hysteresis and ink.motion_veto
    assert ink.pass_interval_s == 0.15
    monkeypatch.setattr(sys, "argv", ["draw", "--markers", "--ink-motion-veto"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2
