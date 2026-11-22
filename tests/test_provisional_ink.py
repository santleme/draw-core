"""Tinta provisória: exibição na 1ª passada, canvas só com voto.

O contrato tem três pernas, todas observáveis sem câmera:

1. A primeira passada publica a tinta nova na camada provisória
   (o canal da tela), e o canal dos consumidores (poll) segue
   sem ela.
2. A tinta estável confirma pelas regras de voto (forte na 2ª
   passada, fraca consistente na 3ª) e aí entra no poll.
3. A tinta que aparece e some (mobiliário, tremor) é retratada
   da camada provisória, entra na contagem de retratados e nunca
   toca o canvas confirmado.
"""

import time

import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import accumulator as accumulator_mod
from draw.ink import worker as worker_mod

H, W = 60, 80
STRONG = (slice(5, 15), slice(5, 25))
WEAK = (slice(25, 35), slice(5, 25))
FLICKER = (slice(40, 50), slice(40, 70))


def _prob(rects):
    prob = np.zeros((H, W), dtype=np.float32)
    for rect, value in rects:
        prob[rect] = value
    return prob


def _px(mask, rect):
    return int(np.count_nonzero(mask[rect]))


def test_first_pass_is_provisional_only_and_second_confirms_strong():
    acc = accumulator_mod.LayeredInkAccumulator(threshold=0.7)
    first = acc.add(_prob([(STRONG, 0.95)]))
    assert _px(first["provisional"], STRONG) == 200
    assert _px(first["confirmed"], STRONG) == 0
    second = acc.add(_prob([(STRONG, 0.95)]))
    assert _px(second["confirmed"], STRONG) == 200
    assert _px(second["provisional"], STRONG) == 0


def test_weak_consistent_ink_confirms_on_third_pass():
    acc = accumulator_mod.LayeredInkAccumulator(threshold=0.7)
    acc.add(_prob([(WEAK, 0.55)]))
    acc.add(_prob([(WEAK, 0.55)]))
    third = acc.add(_prob([(WEAK, 0.55)]))
    assert _px(third["confirmed"], WEAK) == 200


def test_flicker_is_retracted_and_never_confirms():
    acc = accumulator_mod.LayeredInkAccumulator(threshold=0.7)
    first = acc.add(_prob([(FLICKER, 0.85)]))
    assert _px(first["provisional"], FLICKER) == 300
    result = first
    for _ in range(3):
        result = acc.add(_prob([]))
    assert _px(result["provisional"], FLICKER) == 0
    assert _px(result["confirmed"], FLICKER) == 0
    assert result["retracted_px_total"] == 300
    assert result["provisional_ever_px"] == 300


def test_confirmed_ink_is_never_retracted():
    acc = accumulator_mod.LayeredInkAccumulator(threshold=0.7)
    acc.add(_prob([(STRONG, 0.95)]))
    acc.add(_prob([(STRONG, 0.95)]))
    result = None
    for _ in range(4):
        result = acc.add(_prob([]))
    assert _px(result["confirmed"], STRONG) == 200
    assert result["retracted_px_total"] == 0


def test_guard_excluded_provisional_is_not_retraction():
    acc = accumulator_mod.LayeredInkAccumulator(threshold=0.7)
    first = acc.add(_prob([(FLICKER, 0.85), (WEAK, 0.85)]))
    assert _px(first["provisional"], FLICKER) == 300
    assert _px(first["provisional"], WEAK) == 200
    # A re-ancoragem alargou a guarda sobre a faixa do provisorio:
    # a passada deixou de inspecionar ali — a tinta nao sumiu e
    # nao pode contar como retracao. Dentro da guarda a retracao
    # segue valendo.
    guard = np.ones((H, W), dtype=bool)
    guard[FLICKER] = False
    result = None
    for _ in range(3):
        result = acc.add(_prob([]), valid_mask=guard)
    assert _px(result["provisional"], FLICKER) == 0
    assert _px(result["provisional"], WEAK) == 0
    assert result["retracted_px_total"] == 200


class _ScriptedProbInk:
    threshold = 0.7

    def __init__(self, maps):
        self._maps = maps
        self.calls = 0

    def ink_prob(self, _warped):
        index = min(self.calls, len(self._maps) - 1)
        self.calls += 1
        return self._maps[index]


def _drive_until(worker, generation, predicate, timeout=5.0):
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    deadline = time.monotonic() + timeout
    display_version = 0
    while time.monotonic() < deadline:
        worker.submit(WarpedPage(warp, generation))
        got = worker.poll_display(generation, display_version)
        if got is not None:
            display_version = got.version
            if predicate(got.confirmed, got.provisional):
                return got
        time.sleep(0.005)
    raise AssertionError("a condição nunca chegou na exibição")


@pytest.mark.slow
def test_worker_display_shows_provisional_before_poll_confirms(
    monkeypatch,
):
    # pass_interval_s=0,3s: a passada mede ~27ms, entao a versao
    # provisoria vive ~270ms — margem segura para o poll de 5ms;
    # com 0,01s a v1 morria entre dois polls e o teste oscilava.
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.05)
    # A sonda ociosa de producao (1 s) nao pode vazar para o arnes, mas
    # tambem nao pode ficar abaixo da cadencia configurada: quando a
    # sonda governa (pagina assentada), o provisorio precisa dos mesmos
    # ~270 ms de vida entre passadas que pass_interval_s garante.
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.3)
    strong_twice = _prob([(STRONG, 0.95)])
    fake = _ScriptedProbInk([strong_twice])
    config = worker_mod.InkConfig(pass_interval_s=0.3)
    worker = worker_mod.PageCleanWorker(fake, config=config).start()
    try:
        generation = worker.new_generation()
        got = _drive_until(
            worker,
            generation,
            lambda confirmed, provisional: _px(provisional, STRONG) == 200,
        )
        assert _px(got.confirmed, STRONG) == 0
        confirmed_poll = worker.poll(generation, 0)
        assert confirmed_poll is None or _px(confirmed_poll.confirmed, STRONG) == 0
        _drive_until(
            worker,
            generation,
            lambda confirmed, provisional: _px(confirmed, STRONG) == 200,
        )
        _snap = worker.poll(generation, 0)
        _version, canvas = _snap.version, _snap.confirmed
        assert _px(canvas, STRONG) == 200
    finally:
        worker.stop()
