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
    default = worker_mod.PageCleanWorker(_NoModel(), config=InkConfig())
    tuned = worker_mod.PageCleanWorker(
        _NoModel(), config=InkConfig(pass_interval_s=0.15)
    )
    for worker in (default, tuned):
        worker.set_hand_presence(hand)
    want = default._pass_interval()
    got = tuned._pass_interval()
    if want == worker_mod._PAGE_CLEAN_INTERVAL_SECONDS:
        assert got == 0.15
    else:
        assert got == want


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
