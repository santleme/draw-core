"""Rebase do quad congelado na ancora-mediana (interacao das flags do #57).

A prova composta sobre a main mergeada (draw-lab/live-audit-2026-11-10/
17-prova-composta-flags.md) isolou o mecanismo: com
reg_anchor_median e reg_deadband juntos, 100% do descarte
do braco composto nasce ate a passada 5 — a ancora anda nas passadas
1-3 (mediana assentando) enquanto o quad congelado da banda morta
ficou preso na amostra 1, e as primeiras passadas registram contra um
congelado que a propria ancora ja abandonou. Pedagio unico, nao
vazamento: da passada 6 a 20 o descarte adicional e zero.

Correcao: enquanto a janela de amostras da mediana esta aberta (a
ancora acabou de ser recalculada nesta passada), o congelado e
rebaseado para a ancora corrente e o contador de liberacao recomeca.
Fechada a janela, a semantica e exatamente a da banda morta do #57.
Com qualquer uma das flags desligada, nada muda.
"""

import time

import numpy as np
import pytest

import rr
from draw.contracts import WarpedPage
from draw.ink import registration as registration_mod
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig

SHAPE = (120, 85)
BASE = rr.base_quad(SHAPE)


class _ProbIsWarp:
    """page_clean falso: o "warp" submetido ja e o mapa."""

    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.005)
    return monkeypatch


@pytest.fixture
def spy(monkeypatch):
    """Registra (ancora, quad de registro) que cada passada entregou."""
    calls = []
    real = registration_mod._register_ink_probability

    def wrapped(
        prob, anchor, corners, visible=None, visible_cam=None, pass_corners=None
    ):
        calls.append(
            (
                None if anchor is None else np.array(anchor, copy=True),
                None if corners is None else np.array(corners, copy=True),
            )
        )
        return real(
            prob,
            anchor,
            corners,
            visible,
            visible_cam=visible_cam,
            pass_corners=pass_corners,
        )

    monkeypatch.setattr(registration_mod, "_register_ink_probability", wrapped)
    return calls


def _page():
    _drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=3)
    return settled[0]


def _pass(page, warp_corners):
    corners = np.asarray(warp_corners, np.float32)
    return rr.render_pass(page, BASE, corners), corners


def _shifted(dx, dy=0.0, corner=0):
    quad = BASE.copy()
    quad[corner] += np.float32([dx, dy])
    return quad


def _drive(worker, generation, sequence):
    for prob, corners in sequence:
        before = worker.stats()["passes_total"]
        worker.submit(WarpedPage(prob, generation, corners))
        deadline = time.monotonic() + 5.0
        while worker.stats()["passes_total"] <= before:
            assert time.monotonic() < deadline, "a passada nao executou"
            time.sleep(0.001)


def _run(sequence, config=None):
    worker = worker_mod.PageCleanWorker(_ProbIsWarp(), config=config).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, sequence)
        _disp = worker.poll_display(generation, 0)
        _version, confirmed, provisional = (
            _disp.version,
            _disp.confirmed,
            _disp.provisional,
        )
        return worker, confirmed, provisional
    finally:
        worker.stop()


BOTH_ON = InkConfig(reg_anchor_median=True, reg_deadband=True)


@pytest.mark.slow
def test_frozen_follows_median_anchor_while_window_open(fast, spy):
    page = _page()
    s1 = BASE.copy()
    s2 = _shifted(12.0)  # TL +12 em x
    s3 = _shifted(4.0)  # TL +4 em x
    sequence = [_pass(page, s1), _pass(page, s2), _pass(page, s3)]
    worker, _confirmed, _provisional = _run(sequence, config=BOTH_ON)
    median2 = np.median(np.stack([s1, s2]), axis=0)
    median3 = np.median(np.stack([s1, s2, s3]), axis=0)
    # Passada 1: congelado = ancora = amostra 1.
    np.testing.assert_allclose(spy[0][1], s1)
    # Passadas 2 e 3: o registro usa a ancora-mediana CORRENTE (o
    # congelado rebaseado), nao a amostra 1 abandonada.
    np.testing.assert_allclose(spy[1][1], median2)
    np.testing.assert_allclose(spy[2][1], median3)
    np.testing.assert_allclose(worker._reg_frozen_corners, median3)
    np.testing.assert_allclose(worker._anchor_corners, median3)
    records = worker.telemetry_snapshot()["records"]
    rebased = [bool(r.get("reg_deadband_rebased", False)) for r in records[:3]]
    assert rebased == [False, True, True]
    assert worker.telemetry_snapshot()["reg_deadband_releases"] == 0


@pytest.mark.slow
def test_sustained_step_releases_after_window_closes(fast, spy):
    page = _page()
    step = BASE + np.float32([120.0, 0.0])
    sequence = [
        _pass(page, BASE),
        _pass(page, BASE),
        _pass(page, BASE),
        _pass(page, step),
        _pass(page, step),
    ]
    worker, _confirmed, _provisional = _run(sequence, config=BOTH_ON)
    assert worker.telemetry_snapshot()["reg_deadband_releases"] == 1
    # A primeira passada do degrau segura (over=1); a segunda libera e
    # o registro passa a usar o quad do degrau.
    np.testing.assert_allclose(spy[3][1], BASE)
    np.testing.assert_allclose(spy[4][1], step)
    records = worker.telemetry_snapshot()["records"]
    assert records[3]["reg_deadband_released"] is False
    assert records[4]["reg_deadband_released"] is True


@pytest.mark.slow
def test_spike_inside_window_corrupts_neither_anchor_nor_frozen(fast, spy):
    page = _page()
    spike = _shifted(200.0)
    s3 = _shifted(4.0)
    sequence = [
        _pass(page, BASE),
        _pass(page, spike),
        _pass(page, s3),
        _pass(page, s3),
    ]
    worker, _confirmed, _provisional = _run(sequence, config=BOTH_ON)
    expected = np.median(np.stack([BASE, spike, s3]), axis=0)
    # Mediana de (0, 200, 4) no TL = 4: o spike nao vira ancora nem
    # congelado, e a passada quieta seguinte registra contra eles.
    np.testing.assert_allclose(worker._anchor_corners, expected)
    np.testing.assert_allclose(worker._reg_frozen_corners, expected)
    np.testing.assert_allclose(spy[3][1], expected)
    assert worker.telemetry_snapshot()["reg_deadband_releases"] == 0


@pytest.mark.slow
def test_deadband_alone_has_no_rebase(fast, spy):
    page = _page()
    near = _shifted(6.0)  # 6 px < tau: segura no congelado da amostra 1
    sequence = [_pass(page, BASE), _pass(page, near), _pass(page, near)]
    worker, _confirmed, _provisional = _run(
        sequence, config=InkConfig(reg_deadband=True)
    )
    for _anchor, registration in spy:
        np.testing.assert_allclose(registration, BASE)
    records = worker.telemetry_snapshot()["records"]
    assert all("reg_deadband_rebased" not in r for r in records)


@pytest.mark.slow
def test_median_alone_registers_with_pass_corners(fast, spy):
    page = _page()
    s2 = _shifted(12.0)
    s3 = _shifted(4.0)
    sequence = [_pass(page, BASE), _pass(page, s2), _pass(page, s3)]
    worker, _confirmed, _provisional = _run(
        sequence, config=InkConfig(reg_anchor_median=True)
    )
    # Sem banda morta nao ha congelado: o registro usa o quad da
    # propria passada, e a ancora assenta na mediana mesmo assim.
    for (_anchor, registration), (_prob, corners) in zip(spy, sequence, strict=False):
        np.testing.assert_allclose(registration, corners)
    np.testing.assert_allclose(
        worker._anchor_corners,
        np.median(np.stack([BASE, s2, s3]), axis=0),
    )
