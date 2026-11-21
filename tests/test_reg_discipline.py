"""Disciplina de registro: ancora por mediana e banda morta (M1).

O defeito (auditoria de 2026-11-10, provas P2/P3a/P3b em
draw-lab/live-audit-2026-11-10/): o voto de tinta e destruido pelo
jitter do quad de registro, e o jitter real e evento discreto (flip
de modo do detector, pouso de salto). Duas correcoes, atras de flags
desligadas por padrao, so sobre o quad que alimenta
_registration_matrix:

1. InkConfig.reg_anchor_median=True: a ancora da geracao e a
   mediana por componente dos cantos das 3 primeiras passadas com
   quad valido (mediana das que houver antes disso), nao a 1a
   amostra.
2. InkConfig.reg_deadband=True: quad de registro congelado por
   geracao; desvio maximo de canto < _REG_DEADBAND_TAU_PX usa o
   congelado; >= tau por 2 passadas seguidas libera (o congelado
   salta para o quad da passada) e conta reg_deadband_releases.
   Spike de 1 passada nunca libera.
3. Com as duas desligadas, o canvas e byte-identico ao baseline.
"""

import hashlib
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
# Quad espurio da passada 0 (modo do detector trocado): 126-331 px
# por canto, o regime medido na P3b.
OUTLIER = BASE + np.float32([(150, -126), (331, 40), (200, 180), (-140, 250)])


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
    """Registra o quad que cada passada entregou ao registro."""
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


def _pass(page, warp_corners):
    """A pagina parada em BASE, retificada com o quad do detector."""
    corners = np.asarray(warp_corners, np.float32)
    return rr.render_pass(page, BASE, corners), corners


def _iou(a, b):
    a, b = a > 0, b > 0
    union = np.count_nonzero(a | b)
    return 1.0 if union == 0 else np.count_nonzero(a & b) / union


def _jitter(rng, amplitude):
    return rng.uniform(-amplitude, amplitude, size=(4, 2)).astype(np.float32)


def test_deadband_tau_is_the_approved_named_constant():
    assert registration_mod._REG_DEADBAND_TAU_PX == 10
    assert registration_mod._REG_DEADBAND_RELEASE_PASSES == 2


def test_deadband_step_holds_below_tau_and_releases_on_second_pass():
    step = registration_mod._deadband_step
    frozen = BASE.copy()
    near = BASE + np.float32([6.0, 7.9])  # 9,9 px < tau
    quad, frozen, over, released, shift = step(frozen, near, 0)
    assert np.array_equal(quad, BASE) and over == 0 and not released
    assert shift < 10
    edge = BASE + np.float32([6.0, 8.0])  # exatamente tau: conta como >= tau
    quad, frozen, over, released, _ = step(frozen, edge, 0)
    assert np.array_equal(quad, BASE) and over == 1 and not released
    quad, frozen, over, released, _ = step(frozen, edge, over)
    assert released and over == 0
    assert np.array_equal(quad, edge) and np.array_equal(frozen, edge)


# --- (a) ancora por mediana -------------------------------------------


def _outlier_generation(page, rng, passes=8):
    sequence = [_pass(page, OUTLIER)]
    for _ in range(passes - 1):
        sequence.append(_pass(page, BASE + _jitter(rng, 1.0)))
    return sequence


@pytest.mark.slow
def test_anchor_median_rejects_first_pass_outlier(fast, spy):
    page = _page()
    sequence = _outlier_generation(page, np.random.default_rng(1))
    worker, confirmed, _provisional = _run(
        sequence, config=InkConfig(reg_anchor_median=True)
    )
    anchor = worker._anchor_corners
    agreeing = np.stack([c for _p, c in sequence[1:3]])
    expected = np.median(np.stack([sequence[0][1], *agreeing]), axis=0)
    np.testing.assert_allclose(anchor, expected)
    assert np.max(np.linalg.norm(anchor - BASE, axis=1)) <= 2.0
    # Antes de completar 3 a ancora e a mediana das que houver (2 =
    # media), sem bloquear passada: toda submissao rodou e registrou.
    assert len(spy) == len(sequence)
    np.testing.assert_allclose(
        spy[1][0], np.median(np.stack([sequence[0][1], sequence[1][1]]), axis=0)
    )
    for anchor_used, _corners in spy[2:]:
        np.testing.assert_allclose(anchor_used, expected)
    records = worker.telemetry_snapshot()["records"]
    assert [r["reg_anchor_samples"] for r in records] == [1, 2] + [3] * (
        len(sequence) - 2
    )
    _w, truth, _p = _run([_pass(page, BASE) for _ in sequence])
    assert _iou(confirmed, truth) > 0.4


@pytest.mark.slow
def test_anchor_is_first_sample_with_median_off(fast, spy):
    page = _page()
    sequence = _outlier_generation(page, np.random.default_rng(1))
    worker, confirmed, _provisional = _run(
        sequence,
        config=InkConfig(anchor_arbitration=False, anchor_follow=False),
    )
    assert np.array_equal(worker._anchor_corners, sequence[0][1])
    for anchor_used, _corners in spy:
        assert np.array_equal(anchor_used, OUTLIER)
    assert all(
        "reg_anchor_samples" not in r for r in worker.telemetry_snapshot()["records"]
    )
    _w, truth, _p = _run([_pass(page, BASE) for _ in sequence])
    assert _iou(confirmed, truth) < 0.1, "a falha silenciosa que a flag elimina"


# --- (b) banda morta --------------------------------------------------


@pytest.mark.slow
def test_deadband_settled_jitter_never_releases(fast, spy):
    page = _page()
    rng = np.random.default_rng(2)
    sequence = [_pass(page, BASE + _jitter(rng, 4.0)) for _ in range(10)]
    worker, _confirmed, _provisional = _run(
        sequence, config=InkConfig(reg_deadband=True)
    )
    frozen = sequence[0][1]
    assert worker.telemetry_snapshot()["reg_deadband_releases"] == 0
    assert len(spy) == 10
    for _anchor, corners in spy:
        assert np.array_equal(corners, frozen)
    assert all(r["reg_deadband_held"] for r in worker.telemetry_snapshot()["records"])


@pytest.mark.slow
def test_deadband_single_spike_never_releases(fast, spy):
    page = _page()
    rng = np.random.default_rng(3)
    corners = [BASE + _jitter(rng, 2.0) for _ in range(9)]
    corners[4] = BASE + np.float32([200.0, 0.0])
    sequence = [_pass(page, c) for c in corners]
    worker, _confirmed, _provisional = _run(
        sequence, config=InkConfig(reg_deadband=True)
    )
    assert worker.telemetry_snapshot()["reg_deadband_releases"] == 0
    for _anchor, used in spy:
        assert np.array_equal(used, corners[0])
    records = worker.telemetry_snapshot()["records"]
    assert [r["reg_deadband_pending"] for r in records] == [0] * 4 + [1] + [0] * 4


@pytest.mark.slow
def test_deadband_sustained_move_releases_once_and_converges(fast, spy):
    page = _page()
    moved = BASE + np.float32([120.0, 0.0])
    corners = [BASE] * 6 + [moved] * 2 + [moved] * 4
    sequence = [_pass(page, c) for c in corners]
    worker, confirmed, _provisional = _run(
        sequence, config=InkConfig(reg_deadband=True)
    )
    snapshot = worker.telemetry_snapshot()
    assert snapshot["reg_deadband_releases"] == 1
    released = [r["reg_deadband_released"] for r in snapshot["records"]]
    assert released == [False] * 7 + [True] + [False] * 4
    # Ate a liberacao o registro e contra o congelado; dali em diante
    # converge para o quad da passada (a matriz do baseline).
    for _anchor, used in spy[:7]:
        assert np.array_equal(used, BASE)
    for _anchor, used in spy[7:]:
        assert np.array_equal(used, moved)
    assert np.array_equal(worker._reg_frozen_corners, moved)
    _w, baseline, _p = _run(sequence)
    assert np.array_equal(confirmed, baseline), "a passada segurada nao confirma"


@pytest.mark.slow
def test_deadband_state_is_per_generation(fast, spy):
    page = _page()
    worker = worker_mod.PageCleanWorker(
        _ProbIsWarp(), config=InkConfig(reg_deadband=True)
    ).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, [_pass(page, BASE)] * 2)
        other = BASE + np.float32([40.0, 30.0])
        generation = worker.new_generation()
        _drive(worker, generation, [_pass(page, other)] * 2)
        assert np.array_equal(spy[-1][1], other)
        assert worker.telemetry_snapshot()["reg_deadband_releases"] == 0
    finally:
        worker.stop()


# --- (c) paridade com as flags desligadas -----------------------------

# sha256 de confirmado + provisorio publicados, gerados na main
# 8dcce18 (antes das flags), no replay sintetico de
# tools/replay_reregister.py com tremor de 4 px e recuperacao.
GOLDEN = {
    "jitter": "f5c9b83c2462b5e2d8608a824d24e8b6b5c28b391129e9388684eab5e4f1afbf",
    "recover": "42a9851ef5e027e0b868ae8f75f86d548da49b1dde3a2426dbb772405c7a151b",
}


def _parity_canvas(scenario, config=None):
    drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=3)
    rng = np.random.default_rng(0)
    sequence = []
    for stage in drawing + [settled]:
        for k in range(4):
            corners = (BASE + rng.normal(0, 4.0, size=(4, 2))).astype(np.float32)
            sequence.append(
                (rr.render_pass(stage[k % len(stage)], BASE, corners), corners)
            )
    return_at = None
    if scenario == "recover":
        return_at = len(sequence)
        moved = BASE + np.float32([6.0, -4.0])
        sequence += [(rr.render_pass(settled[0], BASE, moved), moved)] * 3
    worker = worker_mod.PageCleanWorker(_ProbIsWarp(), config=config).start()
    try:
        generation = worker.new_generation()
        for i, (prob, corners) in enumerate(sequence):
            if i == return_at:
                generation = worker.new_generation(preserve_identity=True)
            _drive(worker, generation, [(prob, corners)])
        _disp = worker.poll_display(generation, 0)
        _version, confirmed, provisional = (
            _disp.version,
            _disp.confirmed,
            _disp.provisional,
        )
        snapshot = worker.telemetry_snapshot()
    finally:
        worker.stop()
    digest = hashlib.sha256(confirmed.tobytes() + provisional.tobytes()).hexdigest()
    return digest, snapshot


@pytest.mark.slow
@pytest.mark.parametrize("scenario", sorted(GOLDEN))
def test_reg_fields_off_canvas_is_byte_identical_to_baseline(fast, scenario):
    digest, snapshot = _parity_canvas(
        scenario, config=InkConfig(reg_deadband=False, reg_anchor_median=False)
    )
    assert digest == GOLDEN[scenario]
    assert "reg_deadband_releases" not in snapshot
    assert not any(k.startswith("reg_") for r in snapshot["records"] for k in r)
