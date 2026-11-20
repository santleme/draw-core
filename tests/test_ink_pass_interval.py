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


@pytest.mark.parametrize("interval", [0.9, 2.0])
def test_slow_cadence_movement_resets_quiet_streak(interval):
    # Com intervalo mais lento que o piso de 0,25 s (ex.: 0,9 s ou 2 s), um
    # movimento breve da mao precisa zerar _quiet_passes para o worker
    # nao voltar ao descanso antes de confirmar um traco curto.
    now = [100.0]
    worker = worker_mod.PageCleanWorker(
        _NoModel(), config=InkConfig(pass_interval_s=interval), clock=lambda: now[0]
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
    assert worker._pass_interval() == interval


def test_explicit_default_cadence_preserves_rest_after_move_and_exit():
    # --ink-interval 0.25 precisa preservar exatamente o comportamento de
    # descanso do padrao (None): mover a mao e sair nao exige passadas
    # quietas extras alem da passada de confirmacao.
    now = [100.0]
    default = worker_mod.PageCleanWorker(
        _NoModel(), config=InkConfig(), clock=lambda: now[0]
    )
    explicit = worker_mod.PageCleanWorker(
        _NoModel(), config=InkConfig(pass_interval_s=0.25), clock=lambda: now[0]
    )
    for worker in (default, explicit):
        worker.set_hand_presence(True, moved=True)
        worker._generation_passes = worker_mod._REST_MIN_GENERATION_PASSES
        worker._quiet_passes = worker_mod._REST_QUIET_PASSES
        worker.set_hand_presence(True, moved=True)
        worker.set_hand_presence(False)
        worker._confirm_once = False
    assert default._at_rest() is True
    assert explicit._at_rest() is True


def test_late_reference_aligns_to_confirmed_ink_when_sheet_returns():
    # Regressao: folha comeca com tinta esparsa (< 6 blocos com textura),
    # confirma essa tinta, depois escorrega 3 px enquanto ganha o resto da
    # textura que qualifica a referencia tardia. Quando a folha volta para
    # a posicao original, o portao nao pode empurrar a tinta para +3 px e
    # duplicar os tracos ja confirmados.
    import cv2
    import numpy as np

    from draw.contracts import WarpedPage

    class _DarkIsInk:
        threshold = 0.5

        def ink_prob(self, warped):
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            return ((255.0 - gray) / 255.0).astype(np.float32)

    def _draw_block_texture(img, y0, x0, seed):
        rng = np.random.default_rng(seed)
        for _ in range(18):
            y = int(rng.integers(y0 + 20, y0 + 180))
            x = int(rng.integers(x0 + 20, x0 + 180))
            img[y - 2 : y + 3, x - 2 : x + 3] = 25

    base = np.full((1200, 850), 240, dtype=np.uint8)
    # 4 blocos texturizados (< 6): confirma tinta sem armar o portao
    sparse_blocks = [(0, 0), (0, 200), (200, 0), (200, 200)]
    for idx, (by, bx) in enumerate(sparse_blocks):
        _draw_block_texture(base, by, bx, seed=10 + idx)

    full = base.copy()
    # +6 blocos extras: agora qualifica (>= 6 blocos)
    extra_blocks = [
        (400, 0),
        (400, 200),
        (400, 400),
        (600, 0),
        (600, 200),
        (600, 400),
    ]
    for idx, (by, bx) in enumerate(extra_blocks):
        _draw_block_texture(full, by, bx, seed=50 + idx)

    corners = np.float32([[100, 100], [500, 100], [500, 660], [100, 660]])
    now = [1000.0]
    worker = worker_mod.PageCleanWorker(
        _DarkIsInk(),
        config=InkConfig(motion_veto=True, anchor_follow=False),
        clock=lambda: now[0],
    )
    worker.set_hand_presence(True)
    generation = worker.new_generation()

    def _bgr(gray):
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 1) Confirma tinta esparsa na posicao (0, 0)
    for _ in range(4):
        worker.submit(WarpedPage(_bgr(base), generation, corners))
        assert worker.pump()
        now[0] += 1.0
    before = worker.poll_display(generation, 0).confirmed > 0
    assert before.any()
    assert worker._motion._reference is None  # ainda nao armou (< 6 blocos)

    # 2) Folha desloca +3 px sob cantos fixos e ganha blocos que armam o portao
    shifted_full = np.roll(full, 3, axis=1)
    shifted_full[:, :3] = 240
    worker.submit(WarpedPage(_bgr(shifted_full), generation, corners))
    assert worker.pump()
    now[0] += 1.0
    assert worker._motion._reference is not None

    # 3) Folha volta para a posicao original (0, 0): nao pode duplicar tinta
    for _ in range(4):
        worker.set_hand_presence(True, moved=True)
        worker.submit(WarpedPage(_bgr(full), generation, corners))
        assert worker.pump()
        now[0] += 1.0
    after = worker.poll_display(generation, 0).confirmed > 0
    # Na regiao dos 4 blocos iniciais, nao pode ter copia deslocada fora da
    # tinta original (margem de 1 px para interpolacao).
    orig_region = before[:400, :400]
    orig_dilated = (
        cv2.dilate(orig_region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    )
    displaced_px = int(np.count_nonzero(after[:400, :400] & ~orig_dilated))
    assert displaced_px == 0


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


def test_cli_default_is_production_and_flag_overrides(monkeypatch):
    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    monkeypatch.setattr(sys, "argv", ["draw"])
    app_module.main()
    assert seen[-1]["pipeline_config"].ink.pass_interval_s == 0.15
    monkeypatch.setattr(sys, "argv", ["draw", "--ink-interval", "0.4"])
    app_module.main()
    assert seen[-1]["pipeline_config"].ink.pass_interval_s == 0.4
    monkeypatch.setattr(sys, "argv", ["draw", "--legacy-ink"])
    app_module.main()
    assert seen[-1]["pipeline_config"].ink.pass_interval_s is None


def test_default_is_the_full_live_stack(monkeypatch):
    # A pilha do teste de mesa: ganho, pauta, histerese, portao e cadencia.
    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    monkeypatch.setattr(sys, "argv", ["draw"])
    app_module.main()
    config = seen[-1]["pipeline_config"]
    ink = config.ink
    assert config.ink_auto_gain
    assert ink.ruling_removal and ink.hysteresis and ink.motion_veto
    assert ink.pass_interval_s == 0.15
