"""Veto de passada por movimento residual (InkConfig.motion_veto).

Medido no replay da escrita-folha-movida (2026-11-20): a folha deslizou
1,9-3,3 px sob um quad parado e essas passadas carimbaram 39% da tinta
nova do clipe ao lado da tinta assentada. O portao mede o deslocamento
da pagina registrada contra a referencia da geracao: veta enquanto a
folha anda e, quando o mesmo deslocamento se repete (folha parada fora
do lugar), corrige pela translacao em vez de vetar para sempre.
"""

import time

import cv2
import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import motion
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig

SHAPE = (600, 450)
BASE = np.float32([(0, 0), (450, 0), (450, 600), (0, 600)])


def _page(seed=0, strokes=70, shape=SHAPE, thickness=3):
    """Mapa de tinta com tracos em angulos variados (textura 2D)."""
    rng = np.random.default_rng(seed)
    page = np.zeros(shape, np.float32)
    h, w = shape
    for _ in range(strokes):
        x0, y0 = rng.integers(20, w - 20), rng.integers(20, h - 20)
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(10, 40)
        x1 = int(np.clip(x0 + length * np.cos(ang), 0, w - 1))
        y1 = int(np.clip(y0 + length * np.sin(ang), 0, h - 1))
        cv2.line(page, (int(x0), int(y0)), (x1, y1), 1.0, thickness=thickness)
    return page


def _bgr(prob):
    gray = (255 - prob * 200).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --- medida e decisao ------------------------------------------------------


def test_measure_reads_static_page_as_still_and_shift_as_shift():
    # Pagina sintetica binaria e o pior caso do estimador sub-pixel
    # (pico estreito): erro ate ~0,55 px. Na pagina retificada real
    # (e5, mesmos deslocamentos) o erro medido foi <= 0,22 px.
    page = _page(shape=(850, 1200), strokes=150, thickness=5)
    gate = motion.MotionGate()
    assert gate.measure(_bgr(page), None, None, BASE) is None  # vira referencia
    still = gate.measure(_bgr(page), None, None, BASE)
    assert np.hypot(still[0], still[1]) < 0.2 and still[2] < 0.2
    dx, dy, _half_max, spread = gate.measure(
        _bgr(motion.translate(page, 2.5, -1.0)), None, None, BASE
    )
    assert np.hypot(dx - 2.5, dy + 1.0) < 0.6
    # Translacao pura: as metades concordam. Na pagina real (15
    # paginas dos clipes, ate 3,5 px) o spread medido ficou em p90
    # 0,24 / max 0,29 px, contra MOTION_RIGID_TOL_PX 1,5; a sintetica
    # binaria e o pior caso do estimador (~1 px).
    assert spread < motion.MOTION_RIGID_TOL_PX


def test_measure_sees_rotation_about_the_center():
    # Rotacao em torno do centro: o global fica perto de zero, mas as
    # metades andam — o residuo tem de passar do limiar e o
    # movimento nao pode passar por translacao (nao corrige).
    page = _page(shape=(850, 1200), strokes=150, thickness=5)
    gate = motion.MotionGate()
    gate.measure(_bgr(page), None, None, BASE)
    rot = cv2.getRotationMatrix2D((600, 425), 0.8, 1.0)
    turned = cv2.warpAffine(page, rot, (1200, 850))
    measured = gate.measure(_bgr(turned), None, None, BASE)
    dx, dy, half_max, spread = measured
    assert np.hypot(dx, dy) < motion.MOTION_VETO_PX
    assert half_max > motion.MOTION_HALF_VETO_PX
    assert spread > motion.MOTION_RIGID_TOL_PX
    assert gate.decide(measured)[0] == "veto"


def test_measure_ignores_the_hand_region():
    page = _page(shape=(850, 1200))
    gate = motion.MotionGate()
    gate.measure(_bgr(page), None, None, BASE)
    covered = page.copy()
    covered[:, 600:] = 0.8  # a mao: borrao escuro sobre metade da pagina
    visible = np.ones(page.shape, bool)
    visible[:, 600:] = False
    shift = gate.measure(_bgr(covered), None, visible, BASE)
    assert np.hypot(shift[0], shift[1]) < 0.3
    assert gate.decide(shift)[0] == "accept"


def test_measure_without_enough_visible_page_has_no_verdict():
    page = _page(shape=(850, 1200))
    gate = motion.MotionGate()
    gate.measure(_bgr(page), None, None, BASE)
    visible = np.zeros(page.shape, bool)
    visible[:, :200] = True
    assert gate.measure(_bgr(page), None, visible, BASE) is None


def test_new_anchor_restarts_the_reference():
    page = _page(shape=(850, 1200))
    gate = motion.MotionGate()
    gate.measure(_bgr(page), None, None, BASE)
    moved = BASE + np.float32([5.0, 0.0])
    assert gate.measure(_bgr(motion.translate(page, 3, 0)), None, None, moved) is None


def test_decide_vetoes_while_moving_and_corrects_once_settled():
    gate = motion.MotionGate()
    assert gate.decide((0.3, 0.1, 0.316, 0.0)) == ("accept", None)
    # Folha andando: medidas que nao concordam -> veto.
    moving = ((1.5, 0, 1.5, 0.1), (2.2, 0, 2.2, 0.1), (3.0, 0, 3.0, 0.1))
    assert [gate.decide(m)[0] for m in moving] == ["veto"] * 3
    # Parou em ~3 px, rigida: as 3 ultimas concordam -> corrige pela media.
    assert gate.decide((3.1, 0.0, 3.1, 0.1))[0] == "veto"
    decision, undo = gate.decide((3.0, 0.1, 3.002, 0.1))
    assert decision == "correct" and abs(undo[0] - 3.03) < 0.05
    # Voltou ao lugar: aceita e esquece o historico.
    assert gate.decide((0.2, 0.0, 0.2, 0.0)) == ("accept", None)
    assert gate.decide((2.5, 0.0, 2.5, 0.1))[0] == "veto"


def test_decide_never_corrects_large_or_non_rigid_offsets():
    big = motion.MotionGate()
    assert {big.decide((6.0, 0.0, 6.0, 0.1))[0] for _ in range(6)} == {"veto"}
    bent = motion.MotionGate()
    assert {bent.decide((2.5, 0.0, 2.5, 2.7))[0] for _ in range(6)} == {"veto"}
    # Global parado, uma metade andando (rotacao): veto, nunca corrige.
    turning = motion.MotionGate()
    assert {turning.decide((0.2, 0.0, 3.0, 2.9))[0] for _ in range(6)} == {"veto"}
    # Metade no piso do registro (e5 parado: ate 1,5 px) nao veta.
    assert motion.MotionGate().decide((0.2, 0.2, 1.5, 1.3))[0] == "accept"
    blind = motion.MotionGate()
    assert {blind.decide((2.5, 0.0, 2.5, float("inf")))[0] for _ in range(6)} == {
        "veto"
    }


def test_decide_fails_open_after_too_many_vetoes_in_a_row():
    gate = motion.MotionGate(fail_open_passes=5)
    decisions = [gate.decide((6.0, 0.0, 6.0, 0.1))[0] for _ in range(8)]
    assert decisions == ["veto"] * 5 + ["fail_open"] * 3
    # Uma passada no lugar rearma o veto.
    assert gate.decide((0.1, 0.0, 0.1, 0.0))[0] == "accept"
    assert gate.decide((6.0, 0.0, 6.0, 0.1))[0] == "veto"


def test_config_validates_motion_fields():
    with pytest.raises(ValueError):
        InkConfig(motion_veto_px=0.0)
    with pytest.raises(ValueError):
        InkConfig(motion_settle_passes=1)


# --- trabalhadora: a folha desliza 2,5 px e fica la ---------------------------


class _ProbIsWarp:
    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.005)
    return monkeypatch


def _run(sequence, config):
    worker = worker_mod.PageCleanWorker(_ProbIsWarp(), config=config).start()
    try:
        generation = worker.new_generation()
        for prob in sequence:
            before = worker.stats()["passes_total"]
            # Copia: o page_clean falso devolve o proprio warp como prob,
            # e o motor escreve no prob (o real gera um array novo).
            worker.submit(WarpedPage(prob.copy(), generation, BASE.copy()))
            deadline = time.monotonic() + 5.0
            while worker.stats()["passes_total"] <= before:
                assert time.monotonic() < deadline, "a passada nao executou"
                time.sleep(0.001)
        confirmed = worker.poll_display(generation, 0).confirmed
        return worker.stats(), worker.telemetry_snapshot()["records"], confirmed > 0
    finally:
        worker.stop()


def _slide_and_stay():
    """4 passadas no lugar, depois a folha desliza 2,5 px e PARA; o
    quad (BASE) nao se move nunca. Depois de parada, um traco novo
    e escrito na posicao deslocada."""
    page = _page(seed=5, strokes=160, thickness=5)
    slid = motion.translate(page, 2.5, 0.0)
    new_stroke = np.zeros(SHAPE, np.float32)
    cv2.line(new_stroke, (80, 520), (360, 520), 1.0, thickness=5)
    written = np.maximum(slid, motion.translate(new_stroke, 2.5, 0.0))
    sequence = [page] * 4 + [slid] * 4 + [written] * 6
    return page, new_stroke, sequence


@pytest.mark.slow
def test_slide_and_stay_neither_duplicates_nor_freezes(fast):
    page, new_stroke, sequence = _slide_and_stay()
    on = InkConfig(motion_veto=True, anchor_arbitration=False, anchor_follow=False)
    off = InkConfig(anchor_arbitration=False, anchor_follow=False)
    stats_on, records, conf_on = _run(sequence, on)
    _stats_off, _r, conf_off = _run(sequence, off)

    decisions = [r.get("motion_decision") for r in records]
    # Referencia, aceitas no lugar, vetos ate o deslocamento assentar,
    # e dai em diante passadas corrigidas — nunca veto eterno.
    assert decisions[:4] == ["accept"] * 4  # a 1a vira a referencia
    assert decisions[4:6] == ["veto", "veto"]
    assert set(decisions[6:]) == {"correct"}
    assert stats_on["motion_vetoes_total"] == 2
    assert stats_on["motion_corrections_total"] == len(sequence) - 6

    truth = (page > 0.5) | (new_stroke > 0.5)
    halo = cv2.dilate(truth.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    stray_on = np.count_nonzero(conf_on & ~halo)
    stray_off = np.count_nonzero(conf_off & ~halo)
    # Sem o veto a copia deslocada vira tinta confirmada; com ele, nao.
    assert stray_off > 4 * max(stray_on, 1), (stray_on, stray_off)
    # O traco escrito com a folha ja parada entra, no lugar da ancora:
    # o veto nao silenciou o acumulador.
    line = new_stroke > 0.5
    assert np.count_nonzero(conf_on & line) > 0.8 * np.count_nonzero(line)


@pytest.mark.slow
def test_motion_veto_off_is_byte_identical(fast):
    _page0, _stroke, sequence = _slide_and_stay()
    base = InkConfig(anchor_arbitration=False, anchor_follow=False)
    stats_a, records_a, conf_a = _run(sequence, base)
    stats_b, _records_b, conf_b = _run(sequence, base)
    assert np.array_equal(conf_a, conf_b)
    assert stats_a["motion_vetoes_total"] == 0
    assert all("motion_decision" not in r for r in records_a)


@pytest.mark.slow
def test_rotated_and_stuck_page_fails_open_instead_of_freezing(fast):
    # Folha gira 1,2 grau e fica: nao e translacao (nao corrige), e o
    # quad nunca anda. Depois de MOTION_FAIL_OPEN_PASSES vetos seguidos
    # a passada entra como entraria sem o veto: o acumulador nao fica
    # mudo para sempre.
    page = _page(seed=7, strokes=160, thickness=5)
    h, w = SHAPE
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), 1.2, 1.0)
    turned = cv2.warpAffine(page, rot, (w, h))
    stuck = 4 + motion.MOTION_FAIL_OPEN_PASSES + 4
    sequence = [page] * 4 + [turned] * stuck
    config = InkConfig(motion_veto=True, anchor_arbitration=False, anchor_follow=False)
    stats, records, _confirmed = _run(sequence, config)
    decisions = [r.get("motion_decision") for r in records][4:]
    vetoes = motion.MOTION_FAIL_OPEN_PASSES
    assert decisions[:vetoes] == ["veto"] * vetoes
    assert set(decisions[vetoes:]) == {"fail_open"}
    assert stats["motion_corrections_total"] == 0
    assert stats["motion_fail_open_total"] == len(decisions) - vetoes
