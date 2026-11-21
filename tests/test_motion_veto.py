"""Arbitro de registro e veto por movimento (InkConfig.motion_veto).

Medido no replay da escrita-folha-movida (2026-11-20): passadas
carimbaram tinta deslocada 1,9-4 px ao lado da tinta assentada (39% da
tinta nova do clipe). O registro supoe a folha parada; quando ela anda e
o quad acompanha, ele desfaz o movimento real. O portao mede, por
consenso de blocos com textura 2D, o deslocamento da pagina desta
passada contra a referencia da ancora e decide: aceita, vota sem
registro, corrige pelo modelo afim, veta ou (trava de vivacidade) deixa
passar.
"""

import time

import cv2
import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import motion
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig

SHAPE = (850, 1200)
BASE = np.float32([(0, 0), (1200, 0), (1200, 850), (0, 850)])


def _page(seed=0, strokes=260, shape=SHAPE, thickness=5):
    """Mapa de tinta com tracos em angulos variados: textura 2D por toda parte."""
    rng = np.random.default_rng(seed)
    page = np.zeros(shape, np.float32)
    h, w = shape
    for _ in range(strokes):
        x0, y0 = rng.integers(20, w - 20), rng.integers(20, h - 20)
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(15, 50)
        x1 = int(np.clip(x0 + length * np.cos(ang), 0, w - 1))
        y1 = int(np.clip(y0 + length * np.sin(ang), 0, h - 1))
        cv2.line(page, (int(x0), int(y0)), (x1, y1), 1.0, thickness=thickness)
    return page


def _bgr(prob):
    gray = (255 - prob * 200).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _affine(image, matrix):
    return cv2.warpAffine(
        image,
        np.float32(matrix),
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _shift(image, dx, dy):
    return _affine(image, [[1, 0, dx], [0, 1, dy]])


def _gate_on(reference):
    gate = motion.MotionGate()
    assert gate.measure(reference, None, None, BASE) is None  # vira referencia
    return gate


# --- medida por consenso -----------------------------------------------------


def test_still_page_measures_still():
    page = _bgr(_page())
    still = _gate_on(page).measure(page, None, None, BASE)
    assert still.consistent and still.max_px < 0.1
    assert still.blocks >= 20


def test_translation_is_measured_by_the_consensus():
    page = _bgr(_page())
    m = _gate_on(page).measure(_shift(page, 2.5, -1.0), None, None, BASE)
    assert m.consistent
    assert np.linalg.norm(m.shift - (2.5, -1.0)) < 0.5
    assert np.abs(m.jacobian).max() < 1e-3


def test_rotation_about_the_center_is_seen_and_modelled():
    # Rotacao em torno do centro: o deslocamento medio e ~zero, mas os
    # cantos da grade andam px — o modelo afim tem de ver.
    page = _bgr(_page())
    rot = cv2.getRotationMatrix2D((600, 425), 0.4, 1.0)
    m = _gate_on(page).measure(_affine(page, rot), None, None, BASE)
    assert m.consistent and np.linalg.norm(m.shift) < 0.5
    assert m.max_px > motion.MOTION_VETO_PX
    # E a correcao desfaz: a passada corrigida volta a casar com a referencia.
    fixed = m.warp_back(_affine(page, rot))
    again = _gate_on(page).measure(fixed, None, None, BASE)
    assert again.max_px < 0.5


def test_blank_and_ruling_only_pages_give_no_verdict():
    # Papel liso vota (0, 0) e pauta so horizontal nao mede X: nenhum
    # bloco passa no portao de textura, e sem blocos nao ha juizo — nunca
    # veto falso sobre folha em branco.
    flat = np.full((*SHAPE, 3), 210, np.uint8)
    ruled = flat.copy()
    for y in range(40, SHAPE[0], 30):
        cv2.line(ruled, (0, y), (SHAPE[1], y), (150, 150, 150), 2)
    for page in (flat, ruled):
        assert _gate_on(page).measure(_shift(page, 2.5, 0.0), None, None, BASE) is None


def test_a_hand_over_a_third_of_the_page_does_not_move_the_consensus():
    page = _page()
    covered = page.copy()
    rng = np.random.default_rng(9)
    covered[:, 800:] = rng.uniform(0.3, 0.9, (SHAPE[0], 400))  # mao/sombra sem mascara
    m = _gate_on(_bgr(page)).measure(_bgr(covered), None, None, BASE)
    assert m is not None and m.consistent and m.max_px < motion.MOTION_VETO_PX


def test_the_hand_mask_and_its_margin_keep_blocks_out():
    page = _page()
    visible = np.ones(SHAPE, bool)
    visible[:, :700] = False
    gate = _gate_on(_bgr(page))
    m = gate.measure(_bgr(_shift(page, 2.0, 0.0)), None, visible, BASE)
    assert m.blocks <= 9  # so a faixa visivel, menos a margem da mao


def test_new_anchor_restarts_the_reference():
    page = _bgr(_page())
    gate = _gate_on(page)
    assert gate.measure(_shift(page, 3, 0), None, None, BASE + 5) is None


# --- decisao -------------------------------------------------------------------


def _motion(max_px, consistent=True):
    return motion.Motion(
        jacobian=np.zeros((2, 2)),
        shift=np.array([max_px, 0.0]),
        center=np.zeros(2),
        max_px=max_px,
        blocks=24,
        inliers=24 if consistent else 4,
        corners=np.tile([max_px, 0.0], (4, 1)),
        consistent=consistent,
    )


def test_decide_prefers_registration_then_raw_warp_then_correction():
    gate = motion.MotionGate()
    assert gate.decide(None)[0] == "accept"  # sem juizo
    assert gate.decide(_motion(0.4))[0] == "accept"
    assert gate.decide(_motion(3.5), _motion(0.6))[:3:2] == (
        "unregistered",
        "unregistered",
    )
    decision, fix, hypothesis = gate.decide(_motion(3.5), _motion(2.0))
    assert (decision, hypothesis) == ("correct", "unregistered") and fix.max_px == 2.0
    assert gate.decide(_motion(2.5))[0] == "correct"


def test_decide_vetoes_what_it_cannot_correct():
    gate = motion.MotionGate()
    assert gate.decide(_motion(6.0))[0] == "veto"  # grande demais
    assert gate.decide(_motion(2.0, consistent=False))[0] == "veto"  # sem modelo
    assert gate.decide(_motion(0.5, consistent=False))[0] == "accept"


def test_decide_fails_open_after_too_many_vetoes_in_a_row():
    gate = motion.MotionGate(fail_open_passes=5)
    decisions = [gate.decide(_motion(6.0))[0] for _ in range(8)]
    assert decisions == ["veto"] * 5 + ["fail_open"] * 3
    assert gate.decide(_motion(0.1))[0] == "accept"  # passada alinhada rearma
    assert gate.decide(_motion(6.0))[0] == "veto"


def test_config_validates_motion_fields():
    with pytest.raises(ValueError):
        InkConfig(motion_veto_px=0.0)


# --- trabalhadora ----------------------------------------------------------------


class _InkFromWarp:
    """page_clean falso: o warp e BGR 0-255 (o que o portao mede, como no
    app) e a probabilidade sai da escuridao dele."""

    threshold = 0.7

    def ink_prob(self, warped):
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return np.clip((255.0 - gray) / 200.0, 0.0, 1.0)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.005)
    return monkeypatch


def _run(sequence, config, *, hand=None, coverage=None, spy=None):
    worker = worker_mod.PageCleanWorker(_InkFromWarp(), config=config)
    if spy is not None:
        spy(worker)
    worker.start()
    try:
        generation = worker.new_generation()
        hands = hand if isinstance(hand, list) else [hand] * len(sequence)
        for prob, present in zip(sequence, hands, strict=True):
            if present is not None:
                worker.set_hand_presence(present)
            before = worker.stats()["passes_total"]
            worker.submit(
                WarpedPage(_bgr(prob), generation, BASE.copy(), coverage=coverage)
            )
            deadline = time.monotonic() + 10.0
            while worker.stats()["passes_total"] <= before:
                assert time.monotonic() < deadline, "a passada nao executou"
                time.sleep(0.001)
        confirmed = worker.poll_display(generation, 0).confirmed
        return worker.stats(), worker.telemetry_snapshot()["records"], confirmed > 0
    finally:
        worker.stop()


_PLAIN = dict(anchor_arbitration=False, anchor_follow=False)


def _stray(confirmed, truth):
    halo = cv2.dilate(truth.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return int(np.count_nonzero(confirmed & ~halo))


def _new_stroke():
    stroke = np.zeros(SHAPE, np.float32)
    cv2.line(stroke, (150, 780), (1050, 780), 1.0, thickness=5)
    return stroke


@pytest.mark.slow
def test_slide_and_stay_neither_duplicates_nor_freezes(fast):
    # A folha desliza 2,5 px e PARA la; o quad (BASE) nunca se move — o
    # caso da Q1 (rastreador segurando o quad). Depois de parada, um
    # traco novo e escrito na posicao deslocada. O portao corrige toda
    # passada pelo modelo medido: nada de copia deslocada, nada de
    # acumulador mudo, e o traco novo entra no lugar da ancora.
    page, stroke = _page(seed=5), _new_stroke()
    slid = _shift(page, 2.5, 0.0)
    written = np.maximum(slid, _shift(stroke, 2.5, 0.0))
    sequence = [page] * 4 + [slid] * 4 + [written] * 6
    stats_on, records, conf_on = _run(sequence, InkConfig(motion_veto=True, **_PLAIN))
    _s, _r, conf_off = _run(sequence, InkConfig(**_PLAIN))

    decisions = [r.get("motion_decision") for r in records]
    assert decisions[:4] == ["accept"] * 4  # a 1a vira a referencia
    assert set(decisions[4:]) == {"correct"}
    assert stats_on["motion_vetoes_total"] == 0
    assert stats_on["motion_corrections_total"] == len(sequence) - 4

    truth = (page > 0.5) | (stroke > 0.5)
    assert _stray(conf_off, truth) > 4 * max(_stray(conf_on, truth), 1)
    line = stroke > 0.5
    assert np.count_nonzero(conf_on & line) > 0.8 * np.count_nonzero(line)


@pytest.mark.slow
def test_rotated_and_stuck_page_is_corrected_not_vetoed(fast):
    # 0,3 grau: os cantos da grade andam ~3,2 px, dentro do que o
    # portao corrige (0,4 grau ja passa de MOTION_CORRECT_MAX_PX e veta).
    page = _page(seed=7)
    rot = cv2.getRotationMatrix2D((600, 425), 0.3, 1.0)
    turned = _affine(page, rot)
    sequence = [page] * 4 + [turned] * 6
    stats, records, conf = _run(sequence, InkConfig(motion_veto=True, **_PLAIN))
    assert {r.get("motion_decision") for r in records[4:]} == {"correct"}
    assert _stray(conf, page > 0.5) < 0.02 * np.count_nonzero(page > 0.5)


@pytest.mark.slow
def test_large_stuck_offset_fails_open_instead_of_freezing(fast):
    # Deslocamento parado alem do que o portao corrige e o quad nunca
    # anda: veta MOTION_FAIL_OPEN_PASSES vezes e entao deixa as passadas
    # entrarem como entrariam sem o portao — o acumulador nao fica mudo.
    page = _page(seed=8)
    far = _shift(page, 7.0, 0.0)
    sequence = [page] * 4 + [far] * (motion.MOTION_FAIL_OPEN_PASSES + 4)
    stats, records, _conf = _run(sequence, InkConfig(motion_veto=True, **_PLAIN))
    decisions = [r.get("motion_decision") for r in records][4:]
    vetoes = motion.MOTION_FAIL_OPEN_PASSES
    assert decisions[:vetoes] == ["veto"] * vetoes
    assert set(decisions[vetoes:]) == {"fail_open"}
    assert stats["motion_fail_open_total"] == len(decisions) - vetoes


@pytest.mark.slow
def test_vetoed_passes_do_not_put_the_ink_to_rest(fast):
    # A mao escreve, a folha escorrega para fora do alcance da correcao
    # e a mao sai (descanso total liberado). Passadas vetadas nao contam
    # como quietas: a trabalhadora segue rodando e chega ao fail-open — o
    # traco feito antes de a mao sair nao fica sem confirmar (review do
    # PR #67). Sem a correcao, ela dormia e a passada seguinte nao vinha.
    # O traco escrito com a folha ja deslocada entra (no lugar em que
    # o warp o ve: fail-open e o comportamento de hoje).
    page, stroke = _page(seed=8), _new_stroke()
    far = np.maximum(_shift(page, 7.0, 0.0), stroke)
    n_far = motion.MOTION_FAIL_OPEN_PASSES + 3
    sequence = [page] * 4 + [far] * n_far
    hands = [True] * 5 + [False] * (n_far - 1)
    stats, records, conf = _run(
        sequence, InkConfig(motion_veto=True, **_PLAIN), hand=hands
    )
    assert stats["motion_fail_open_total"] >= 1
    line = stroke > 0.5
    assert np.count_nonzero(conf & line) > 0.8 * np.count_nonzero(line)


@pytest.mark.slow
def test_correction_carries_the_coverage_mask_along(fast):
    # Cobertura de folha so a partir de x=600 no warp; a folha desliza
    # 3 px para a direita e um traco novo nasce colado na borda da
    # cobertura. Corrigido, o traco cai em x 597-599 da ancora: a mascara
    # tem de ir junto, senao essas colunas sao cortadas em toda passada.
    page = _page(seed=6)
    coverage = np.zeros(SHAPE, bool)
    coverage[:, 600:] = True
    stroke = np.zeros(SHAPE, np.float32)
    cv2.line(stroke, (602, 300), (602, 500), 1.0, thickness=5)
    slid = np.maximum(_shift(page, 3.0, 0.0), stroke)
    sequence = [page] * 4 + [slid] * 6
    _s, records, conf = _run(
        sequence, InkConfig(motion_veto=True, **_PLAIN), coverage=coverage
    )
    assert {r.get("motion_decision") for r in records[4:]} == {"correct"}
    assert conf[320:480, 597:600].mean() > 0.8


@pytest.mark.slow
def test_anchor_arbitration_scores_the_camera_map_not_the_corrected_one(fast):
    seen = []

    def spy(worker):
        real = worker._anchor_arbitration_step

        def wrapped(*args):
            seen.append(args[7])  # o mapa "camera" que a arbitragem pontua
            return real(*args)

        worker._anchor_arbitration_step = wrapped

    page = _page(seed=4)
    sequence = [page] * 4 + [_shift(page, 2.5, 0.0)] * 3
    _s, records, _c = _run(
        sequence,
        InkConfig(motion_veto=True, anchor_follow=False),
        spy=spy,
    )
    corrected = [
        i for i, r in enumerate(records) if r.get("motion_decision") == "correct"
    ]
    assert corrected
    # Sem registro (quad parado) o mapa da camera e o proprio prob: a
    # correcao (deslocada 2,5 px) nao pode chegar a arbitragem.
    for i in corrected:
        expected = _InkFromWarp().ink_prob(_bgr(sequence[i]))
        assert np.allclose(seen[i], expected)


@pytest.mark.slow
def test_gate_off_arbitration_sees_the_same_covered_map_as_the_vote(fast):
    # Paridade com cobertura parcial: com o portao desligado a
    # arbitragem pontua exatamente o mapa que vota (ja com o gate de
    # cobertura), como antes do portao. Guardar o mapa da camera antes
    # do gate deixava pixels fora da folha diluirem o escore (review do
    # PR #67).
    pairs = []

    def spy(worker):
        real_arb = worker._anchor_arbitration_step
        real_add = worker._accumulator.add

        def arb(*args):
            pairs.append([args[7]])
            return real_arb(*args)

        def add(prob, *args, **kwargs):
            pairs[-1].append(prob)
            return real_add(prob, *args, **kwargs)

        worker._anchor_arbitration_step = arb
        worker._accumulator.add = add

    coverage = np.ones(SHAPE, bool)
    coverage[:, :300] = False
    page = _page(seed=3)
    _run(
        [page] * 6,
        InkConfig(anchor_follow=False),
        coverage=coverage,
        spy=spy,
    )
    assert len(pairs) >= 4
    for camera, voted in pairs[1:]:
        assert np.array_equal(camera, voted)
        assert not camera[:, :290].any()  # fora da folha nao pontua


@pytest.mark.slow
@pytest.mark.parametrize(
    ("raw_shift", "decision"), [(0.0, "unregistered"), (-2.5, "correct")]
)
def test_camera_coverage_stays_in_camera_geometry_when_vote_changes(
    monkeypatch, raw_shift, decision
):
    # O quad acompanha a folha: o registro da camera desloca 6 px,
    # mas o portao escolhe o warp cru (e pode corrigi-lo). A cobertura
    # da arbitragem precisa continuar na geometria da camera, incluindo
    # a excecao da tinta ja confirmada fora da cobertura.
    now = [1000.0]
    worker = worker_mod.PageCleanWorker(
        _InkFromWarp(),
        config=InkConfig(motion_veto=True, anchor_follow=False),
        clock=lambda: now[0],
    )
    worker.set_hand_presence(True)
    generation = worker.new_generation()
    page = _page(seed=4)
    for _ in range(4):
        worker.submit(WarpedPage(_bgr(page), generation, BASE))
        assert worker.pump()
        now[0] += 1.0
    anchor_ink = worker.poll_display(generation, 0).confirmed > 0
    assert anchor_ink.any()

    camera_maps, votes = [], []
    real_arb = worker._anchor_arbitration_step
    real_add = worker._accumulator.add

    def arb(*args):
        camera_maps.append(args[7].copy())
        return real_arb(*args)

    def add(prob, *args, **kwargs):
        votes.append(prob.copy())
        return real_add(prob, *args, **kwargs)

    monkeypatch.setattr(worker, "_anchor_arbitration_step", arb)
    monkeypatch.setattr(worker._accumulator, "add", add)
    coverage = np.zeros(SHAPE, bool)
    coverage[:, 600:] = True
    raw = _InkFromWarp().ink_prob(_bgr(_shift(page, raw_shift, 0.0)))
    worker.set_hand_presence(True, moved=True)
    # A mascara viaja na geometria da camera (issue #100): o quad
    # deslocou +6 px, entao a cobertura desenhada na grade da pagina
    # chega ao frame por esse mesmo deslocamento — e na ancora
    # (retangulo inteiro -> homografia identidade) ela fica exatamente
    # como a receita antiga produzia com page->anchor.
    matrix = np.float32([[1, 0, 6], [0, 1, 0]])
    size = (SHAPE[1], SHAPE[0])
    camera_coverage = (
        cv2.warpAffine(coverage.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
        > 0
    )
    worker.submit(
        WarpedPage(
            _bgr(_shift(page, raw_shift, 0.0)),
            generation,
            BASE + np.float32([6.0, 0.0]),
            coverage=camera_coverage,
        )
    )
    assert worker.pump()
    assert worker.telemetry_snapshot()["records"][-1]["motion_decision"] == decision
    # Registro de camera conhecido: translacao de +6 px, borda zero.
    camera = cv2.warpAffine(raw, matrix, size, flags=cv2.INTER_LINEAR)
    allowed = camera_coverage | anchor_ink
    expected = np.where(allowed, camera, 0.0)
    assert np.count_nonzero(camera[~allowed]) > 0  # gate exercitado
    assert np.count_nonzero(camera[anchor_ink & ~camera_coverage]) > 0
    assert np.allclose(camera_maps[0], expected)
    assert not np.allclose(camera_maps[0], votes[0])


@pytest.mark.slow
def test_default_path_never_builds_the_gate(fast, monkeypatch):
    # Paridade com o canvas anterior: com motion_veto desligado o portao
    # nem existe — nenhum caminho novo roda (review do PR #67).
    def boom(*_a, **_k):
        raise AssertionError("MotionGate construido com motion_veto desligado")

    monkeypatch.setattr(motion, "MotionGate", boom)
    page = _page(seed=5)
    stats, records, _c = _run([page] * 3 + [_shift(page, 2.5, 0.0)] * 3, InkConfig())
    assert stats["motion_vetoes_total"] == 0
    assert all("motion_decision" not in r for r in records)


@pytest.mark.slow
def test_motion_veto_off_is_byte_identical(fast):
    page = _page(seed=5)
    sequence = [page] * 4 + [_shift(page, 2.5, 0.0)] * 4
    stats_a, records_a, conf_a = _run(sequence, InkConfig(**_PLAIN))
    _stats_b, _records_b, conf_b = _run(sequence, InkConfig(**_PLAIN))
    assert np.array_equal(conf_a, conf_b)
    assert stats_a["motion_vetoes_total"] == 0
    assert all("motion_decision" not in r for r in records_a)
