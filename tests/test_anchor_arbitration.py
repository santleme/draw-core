"""Arbitragem de ancora por conteudo (v3, frente 19 de 2026-11-10).

O defeito (FALHAS M2): no pouso sustentado, migrar o estado pela
inversa do registro (v1) PIORA a duplicacao (0,69-1,14 contra
0,44-0,77 da base), porque a inversa assume a tinta parada na
camera; no modo espurio, trocar so a ancora erra o contrario
(0,74). As hipoteses sao indistinguiveis pelo quad. A v3 arbitra
pelo conteudo (Sp = massa da prob sobre o canvas confirmado na
colocacao identidade; Sc = na colocacao registrada) num streak de
assentamento, e — o mecanismo que faltava na v2 — a passada de
evento aberto NAO vota: a copia deslocada confirmava pelo voto
antes de qualquer veredito.

Contratos:
1. Pouso (Sp >> Sc): exatamente 1 troca de ancora, estado
   confirmado byte-identico ao pre-evento.
2. Espurio (Sc >> Sp): nenhuma acao, ancora intacta, canvas = base.
3. Segmentacao: passadas de meio-movimento (inconsistentes) nao
   fecham o streak; a troca so vem apos a pose assentar.
4. Abstencao: nenhuma confirmacao nasce de passada de evento.
5. Sem massa para julgar: troca por default (custo zero nas duas
   hipoteses).
"""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from draw.ink import registration as registration_mod
from draw.ink import worker as worker_mod
from rr import BASE, SHAPE, _drive, _page, _ProbIsWarp, _run, _settled

STEP = BASE + np.float32([120.0, 0.0])


@pytest.mark.slow
def test_landing_swaps_anchor_without_touching_state(fast):
    content = _page()
    sequence = _settled(content) + [(content, STEP.copy()) for _ in range(3)]
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing"]
    assert events[0]["score_page"] > 0.5
    # O contrato e a separacao (na bancada real, Sc <= 0,012); aqui a
    # pagina miniatura tem 85 px de largura e o degrau de 120 px
    # degenera a homografia, entao Sc mede 0,06 — a margem segue
    # larga.
    assert events[0]["score_page"] - events[0]["score_camera"] > 0.4
    np.testing.assert_allclose(worker._anchor_corners, STEP)
    # O estado nao se moveu: o canvas confirmado e o mesmo que as
    # passadas assentadas construiram (a tinta esta na grade da
    # pagina; quem anda e a ancora).
    _w2, truth, _p2 = _run(_settled(content))
    assert np.array_equal(confirmed, truth)


@pytest.mark.slow
def test_spurious_mode_keeps_anchor_and_canvas(fast):
    content = _page()
    # O quad mente: a prob chega distorcida pela homografia inversa
    # do registro; registrada contra a ancora, ela volta ao lugar.
    matrix = registration_mod._registration_matrix(BASE, STEP, SHAPE)
    distorted = cv2.warpPerspective(
        content, np.linalg.inv(matrix), (SHAPE[1], SHAPE[0])
    ).astype(np.float32)
    sequence = _settled(content) + [(distorted, STEP.copy()) for _ in range(3)]
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot.get("anchor_arbitration_actions", 0) == 0
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["spurious"]
    assert events[0]["score_camera"] > 0.5
    assert events[0]["score_page"] < 0.05
    np.testing.assert_allclose(worker._anchor_corners, BASE)
    _w2, truth, _p2 = _run(_settled(content))
    assert np.array_equal(confirmed, truth)


@pytest.mark.slow
def test_midmovement_passes_do_not_close_the_streak(fast):
    content = _page()
    ramp = [
        (content, (BASE + np.float32([dx, 0.0])).astype(np.float32))
        for dx in (30.0, 60.0, 90.0)
    ]
    settled_step = [(content, STEP.copy()) for _ in range(3)]
    worker, _confirmed, _provisional = _run(_settled(content) + ramp + settled_step)
    snapshot = worker.telemetry_snapshot()
    # As passadas da rampa abstiveram e nenhuma acao nasceu delas;
    # a unica troca vem do streak assentado no fim.
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert len(events) == 1 and events[0]["verdict"] == "landing"
    records = snapshot["records"]
    abstained = [bool(r.get("arb_abstained", False)) for r in records]
    assert abstained[4:7] == [True, True, True]


@pytest.mark.slow
def test_open_event_births_no_confirmation(fast):
    content = _page()
    other = np.zeros(SHAPE, np.float32)
    other[10:14, 10:40] = 1.0  # tinta nova, fora do canvas confirmado
    sequence = _settled(content) + [(other, STEP.copy()) for _ in range(2)]
    worker = worker_mod.PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, sequence)
        _disp = worker.poll_display(generation, 0)
        _version, confirmed, _provisional = (
            _disp.version,
            _disp.confirmed,
            _disp.provisional,
        )
    finally:
        worker.stop()
    _w2, truth, _p2 = _run(_settled(content))
    assert np.array_equal(confirmed, truth), (
        "passada de evento aberto nao pode confirmar tinta nova"
    )


def _permute(img, k):
    # A construcao FISICA da frente 23 (independente do codigo sob
    # teste): a prob que o warp posicional entrega quando a folha
    # assentou rodada de k x 90 graus.
    k %= 4
    if k == 0:
        return img
    if k == 2:
        return np.rot90(img, 2).copy()
    rotated = np.rot90(img, 1 if k == 1 else -1)
    return cv2.resize(
        rotated, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2])
def test_rotated_landing_relabels_anchor_by_k(fast, k):
    # Frente 23/24: sob rotacao, a correspondencia posicional nao
    # distingue "folha rodada" de "quad errado" — o conteudo sim,
    # pontuado como permutacao da imagem entregue. Geometria
    # dedicada onde a pagina ENCHE o quadro entregue (como no
    # pipeline real, em que a prob e o warp da pagina): o veredito
    # tem que ser landing_rotated com o k fisico, a ancora carrega
    # os rotulos re-rotulados, o estado nao se move e as passadas
    # seguintes (ainda entregues rodadas) confirmam sobre o MESMO
    # canvas — e esse ultimo ponto que fixa a direcao do roll.
    full = np.array(
        [[0.0, 0.0], [199.0, 0.0], [199.0, 99.0], [0.0, 99.0]], dtype=np.float32
    )
    moved = full + np.array([30.0, 10.0], dtype=np.float32)
    content = np.zeros(SHAPE, dtype=np.float32)
    content[15:22, 30:110] = 1.0
    content[45:70, 55:63] = 1.0
    content[75:82, 120:180] = 1.0
    rotated = _permute(content, k)
    sequence = (
        [(content, full.copy()) for _ in range(3)]
        + [(rotated, moved.copy()) for _ in range(3)]
        + [(rotated, moved.copy()) for _ in range(3)]
    )
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing_rotated"]
    assert events[0]["rotation_k"] == k
    np.testing.assert_allclose(worker._anchor_corners, np.roll(moved, k, axis=0))
    _tw, truth, _p2 = _run([(content, full.copy()) for _ in range(3)])
    # Retencao exata (nada do canvas se perde) e IoU alto; a
    # igualdade byte a byte dos outros contratos nao esta
    # disponivel aqui: a miniatura reamostra a prob de forma
    # anisotropica e a permutacao acontece antes dessa reamostragem
    # num braco e nao no outro — a borda de ~1 px e artefato da
    # miniatura; quem julga o numero real e a bancada da frente 23.
    assert np.count_nonzero(truth & ~confirmed) == 0
    inter = np.count_nonzero(confirmed & truth)
    union = np.count_nonzero(confirmed | truth)
    assert inter / union >= 0.85


@pytest.mark.slow
def test_straight_landing_after_rotated_landing(fast):
    # Frente 27 (pousos_3_rot): a era 3 — pouso RETO sobre a grade
    # rolada — julga-se pela hipotese-pagina compensada (a prob
    # des-rotada pelo roll parado: 0,193 medidos na captura contra
    # 0,002 do Sc) e a ancora nova e a mediana do streak rolada
    # pelo roll parado, que se preserva. Gemeo dos contratos de
    # pouso rodado, mesma construcao fisica.
    full = np.array(
        [[0.0, 0.0], [199.0, 0.0], [199.0, 99.0], [0.0, 99.0]], dtype=np.float32
    )
    moved = full + np.array([30.0, 10.0], dtype=np.float32)
    moved2 = full + np.array([70.0, 25.0], dtype=np.float32)
    content = np.zeros(SHAPE, dtype=np.float32)
    content[15:22, 30:110] = 1.0
    content[45:70, 55:63] = 1.0
    content[75:82, 120:180] = 1.0
    rotated = _permute(content, 2)
    sequence = (
        [(content, full.copy()) for _ in range(3)]
        + [(rotated, moved.copy()) for _ in range(6)]
        + [(rotated, moved2.copy()) for _ in range(6)]
    )
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 2
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing_rotated", "landing"]
    assert events[1]["score_page_roll_compensated"] is True
    assert worker._anchor_roll == 2
    np.testing.assert_allclose(worker._anchor_corners, np.roll(moved2, 2, axis=0))
    _tw, truth, _p2 = _run([(content, full.copy()) for _ in range(3)])
    assert np.count_nonzero(truth & ~confirmed) == 0
    inter = np.count_nonzero(confirmed & truth)
    union = np.count_nonzero(confirmed | truth)
    assert inter / union >= 0.85


@pytest.mark.slow
def test_no_mass_defaults_to_anchor_swap(fast):
    content = _page()
    empty = np.zeros(SHAPE, np.float32)
    # A geracao assenta numa pagina vazia (ancora nasce, nada
    # confirma) e a folha anda antes de qualquer tinta: sem massa
    # para julgar, a troca e por default.
    sequence = [(empty, BASE.copy()) for _ in range(3)] + [
        (content, STEP.copy()) for _ in range(3)
    ]
    worker, _confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing_default"]
    np.testing.assert_allclose(worker._anchor_corners, STEP)


def test_insufficient_visible_mass_does_not_reanchor_existing_canvas():
    """A partially observed page cannot default-swap an existing canvas."""
    shape = (100, 100)
    anchor = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype=np.float32)
    moved = anchor + np.float32([20, 0])
    snapshot = np.zeros(shape, dtype=bool)
    snapshot[:20, :20] = True  # 400 confirmed pixels, above the canvas floor
    raw_page = np.zeros(shape, dtype=np.float32)
    raw_page[:10, :15] = 1.0  # only 150 visible pixels, below the pass floor
    visible = np.zeros(shape, dtype=bool)
    visible[:10, :15] = True
    camera = raw_page.copy()

    worker = worker_mod.PageCleanWorker(_ProbIsWarp())
    worker._anchor_corners = anchor.copy()
    worker._accumulator._confirmed = snapshot.astype(np.uint8)

    status = None
    event = None
    for _ in range(3):
        status, event = worker._anchor_arbitration_step(
            generation=1,
            corners=moved,
            shape=shape,
            identity_owned=False,
            reg_anchor_median=False,
            reg_deadband=False,
            raw_prob=raw_page,
            registered_prob=camera,
            visible=visible,
            camera_denominator_mass=150.0,
        )

    assert status == "verdict"
    assert event is not None
    assert event["canvas_px"] == 400
    assert event["pass_mass_px"] == 150
    assert event["verdict"] == "indeterminate"
    assert event["action"] is False
    np.testing.assert_array_equal(worker._anchor_corners, anchor)


def _shifted(img, dx, dy):
    out = np.zeros_like(img)
    h, w = img.shape
    y0, y1, x0, x1 = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = img[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return out


@pytest.mark.slow
@pytest.mark.parametrize(
    "residual", [(0, 0), (4, 0), (3, -5)], ids=["T0", "T4x", "T3_5"]
)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotated_landing_with_an_active_residual(fast, k, residual):
    """O resíduo entra depois da permutação: T . Pi_k(P)."""
    dx, dy = residual
    full = np.array(
        [[0.0, 0.0], [199.0, 0.0], [199.0, 99.0], [0.0, 99.0]], dtype=np.float32
    )
    moved = full + np.array([30.0, 10.0], dtype=np.float32)
    content = np.zeros(SHAPE, dtype=np.float32)
    content[15:22, 30:110] = 1.0
    content[45:70, 55:63] = 1.0
    content[75:82, 120:180] = 1.0
    evidence = _shifted(content, -dx, -dy)
    rotated = _permute(evidence, k)
    worker = worker_mod.PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, [(content, full.copy())])
        worker._identity_residual = (dx, dy)
        _drive(worker, generation, [(evidence, full.copy()) for _ in range(3)])
        seen = len(worker.telemetry_snapshot()["anchor_arbitration_events"])
        _drive(worker, generation, [(rotated, moved.copy()) for _ in range(6)])
        events = worker.telemetry_snapshot()["anchor_arbitration_events"][seen:]
        assert [event["verdict"] for event in events] == ["landing_rotated"]
        assert events[0]["rotation_k"] == k
        assert events[0]["scores_cyclic"][k] >= 0.9
        assert events[0]["identity_residual_fresh"] is False
        assert events[0]["pose_displacement_source"] == (
            "pose" if residual == (0, 0) else "stale_residual"
        )
        np.testing.assert_allclose(worker._anchor_corners, np.roll(moved, k, axis=0))
        confirmed = worker.poll_display(generation, 0).confirmed
    finally:
        worker.stop()
    _tw, truth, _p2 = _run([(content, full.copy()) for _ in range(3)])
    assert np.count_nonzero(truth & ~confirmed) == 0
    inter = np.count_nonzero(confirmed & truth)
    union = np.count_nonzero(confirmed | truth)
    assert inter / union >= 0.85


def test_overlap_fraction_uses_the_transformed_mass_after_translation():
    """Rotacao/redimensionamento muda a soma; translacao perde massa."""
    from draw.ink.anchor import _overlap_fraction

    permuted = np.zeros((10, 10), dtype=np.float32)
    permuted[:, 7:] = 1.0  # 30, diferente do raw_total original (20)
    full_snapshot = permuted > 0

    # Sem corte, massa transformada diferente da massa original: o mapa
    # inteiramente coberto ainda pontua 1, em vez de 30/20.
    assert _overlap_fraction(permuted, full_snapshot) == 1.0

    clipped = registration_mod._translate_grid(permuted, 1, 0)
    assert float(clipped.sum()) == 20.0
    assert _overlap_fraction(
        clipped, clipped > 0, denominator_mass=float(permuted.sum())
    ) == pytest.approx(2 / 3)

    half_snapshot = np.zeros_like(full_snapshot)
    half_snapshot[:, 8] = True
    assert _overlap_fraction(
        clipped, half_snapshot, denominator_mass=float(permuted.sum())
    ) == pytest.approx(1 / 3)
    assert _overlap_fraction(np.zeros_like(clipped), half_snapshot) == 0.0


def test_anchor_score_does_not_reward_a_candidate_that_clips_most_mass():
    """The clipped fragment loses to a rival that matches most evidence."""
    from draw.ink.anchor import _overlap_fraction

    source = np.ones((100, 100), dtype=np.float32)
    snapshot = np.zeros_like(source, dtype=bool)
    snapshot[:, -1] = True
    clipped = registration_mod._translate_grid(source, 99, 0)
    clipped_score = _overlap_fraction(
        clipped, snapshot, denominator_mass=float(source.sum())
    )
    legacy_clipped_score = _overlap_fraction(clipped, snapshot)

    rival = np.zeros_like(source)
    rival[:90, -1] = 1.0  # 90 / 100 units match the canvas
    rival[90:, 0] = 1.0  # the remaining 10 are observed but do not match
    rival_score = _overlap_fraction(rival, snapshot)

    assert clipped_score == pytest.approx(0.01)
    assert rival_score == pytest.approx(0.9)
    assert legacy_clipped_score == pytest.approx(1.0)
    assert legacy_clipped_score > rival_score
    assert clipped_score < rival_score


@pytest.mark.parametrize(
    ("matrix", "expected_mass", "expected_canvas_fraction"),
    [
        (np.eye(3, dtype=np.float64), 10_000.0, 1.0),
        (
            np.array([[1, 0, 20], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            10_000.0,
            0.8,
        ),
        (
            np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float64),
            2_500.0,
            1.0,
        ),
    ],
)
def test_padded_camera_warp_keeps_projected_mass_in_its_transformed_domain(
    matrix, expected_mass, expected_canvas_fraction
):
    """Translation crop is penalized; an uncropped scale is not."""
    from draw.ink.anchor import _overlap_fraction

    source = np.ones((100, 100), dtype=np.float32)
    padded = registration_mod._warp_probability_padded(source, matrix)
    assert padded is not None
    complete, _origin, _padded_matrix = padded
    canvas = cv2.warpPerspective(
        source, matrix, (100, 100), flags=cv2.INTER_LINEAR, borderValue=0
    )
    assert float(complete.sum(dtype=np.float64)) == pytest.approx(expected_mass)
    assert _overlap_fraction(
        canvas, canvas > 0, denominator_mass=expected_mass
    ) == pytest.approx(expected_canvas_fraction)


@pytest.mark.parametrize("scale", [3.1, 3.9])
@pytest.mark.parametrize("point", [(0, 0), (0, 99), (99, 0), (99, 99), (50, 0)])
def test_padded_camera_warp_keeps_bilinear_support_of_edge_pixels(scale, point):
    source = np.zeros((100, 100), dtype=np.float32)
    source[point] = 1.0
    matrix = np.diag([scale, scale, 1.0]).astype(np.float64)
    padded = registration_mod._warp_probability_padded(source, matrix)
    assert padded is not None

    # This reference canvas starts well before the mapped source support and
    # extends beyond it, so no destination crop can hide an interpolation halo.
    normalized = matrix / float(np.max(np.abs(matrix)))
    offset = np.array([[1, 0, 32], [0, 1, 32], [0, 0, 1]], dtype=np.float64)
    reference = cv2.warpPerspective(
        source,
        offset @ normalized,
        (500, 500),
        flags=cv2.INTER_LINEAR,
        borderValue=0.0,
    )

    assert float(padded[0].sum(dtype=np.float64)) == pytest.approx(
        float(reference.sum(dtype=np.float64)), abs=1e-4
    )


def test_padded_camera_warp_penalizes_mass_clipped_by_the_canvas():
    from draw.ink.anchor import _overlap_fraction

    source = np.ones((100, 100), dtype=np.float32)
    matrix = np.array([[1, 0, 99], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    padded = registration_mod._warp_probability_padded(source, matrix)
    assert padded is not None
    complete, _origin, _padded_matrix = padded
    canvas = cv2.warpPerspective(
        source, matrix, (100, 100), flags=cv2.INTER_LINEAR, borderValue=0
    )
    snapshot = np.zeros((100, 100), dtype=bool)
    snapshot[:, -1] = True

    assert float(canvas.sum(dtype=np.float64)) == pytest.approx(100.0)
    assert float(complete.sum(dtype=np.float64)) == pytest.approx(10_000.0)
    assert _overlap_fraction(canvas, snapshot, denominator_mass=complete.sum()) == (
        pytest.approx(0.01)
    )


def test_camera_denominator_keeps_mass_for_a_fourfold_scale():
    source = np.ones((100, 100), dtype=np.float32)
    matrix = np.diag([4.0, 4.0, 1.0]).astype(np.float64)
    canvas = cv2.warpPerspective(
        source, matrix, (100, 100), flags=cv2.INTER_LINEAR, borderValue=0.0
    )
    worker_pass = worker_mod._Pass(
        prob=source,
        camera_prob=canvas,
        camera_matrix=matrix,
    )

    assert float(canvas.sum(dtype=np.float64)) == pytest.approx(10_000.0)
    assert worker_mod._camera_probability_denominator(worker_pass) == pytest.approx(
        160_000.0
    )


def test_camera_denominator_applies_exposure_visibility_after_the_warp():
    """The denominator must use the same registered exposure support as camera_prob."""
    shape = (20, 20)
    source = np.zeros(shape, dtype=np.float32)
    source[:, 10:12] = 1.0
    exposure_visible = np.zeros(shape, dtype=bool)
    exposure_visible[:, :11] = True
    matrix = np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    warped = cv2.warpPerspective(
        source, matrix, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue=0
    )
    registered_exposure = (
        cv2.warpPerspective(
            exposure_visible.astype(np.uint8),
            matrix,
            (shape[1], shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        > 0
    )
    camera_prob = np.where(registered_exposure, warped, 0.0).astype(np.float32)
    worker_pass = worker_mod._Pass(
        prob=source,
        camera_prob=camera_prob,
        camera_matrix=matrix,
        exposure=SimpleNamespace(visible=exposure_visible),
    )

    # The independently constructed numerator is exactly 50% mass larger
    # than a source-masked warp at this subpixel boundary.
    assert float(camera_prob.sum(dtype=np.float64)) == pytest.approx(30.0)
    assert worker_mod._camera_probability_denominator(worker_pass) == pytest.approx(
        30.0
    )


def test_camera_exposure_mask_keeps_the_camera_hypothesis_geometry():
    shape = (100, 100)
    anchor = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype=np.float32)
    camera_corners = anchor + np.float32([10, 0])
    # The unregistered-motion branch changes the vote geometry to the
    # anchor, while this camera hypothesis retains its original +10 px map.
    vote_corners = anchor.copy()
    matrix = registration_mod._registration_matrix(anchor, camera_corners, shape)
    source = np.zeros(shape, dtype=np.float32)
    source[5:45, 40:50] = 1.0
    exposure_visible = np.zeros(shape, dtype=bool)
    exposure_visible[:, :50] = True
    camera_prob = cv2.warpPerspective(
        source, matrix, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue=0
    )
    exposure = SimpleNamespace(visible=exposure_visible, paper_exposed=False)
    worker_pass = worker_mod._Pass(
        anchor=anchor,
        prob=source,
        camera_prob=camera_prob.copy(),
        camera_matrix=matrix,
        camera_corners=camera_corners,
        registration_corners=vote_corners,
        exposure=exposure,
    )

    wrong_visible, _unused_exposed = worker_mod._exposure_masks_in_geometry(
        exposure, anchor, vote_corners, shape
    )
    assert float(np.where(wrong_visible, camera_prob, 0.0).sum()) == pytest.approx(0.0)

    worker_mod._mask_camera_probability_by_exposure(worker_pass)

    assert float(worker_pass.camera_prob.sum(dtype=np.float64)) == pytest.approx(400.0)
    assert worker_mod._camera_probability_denominator(worker_pass) == pytest.approx(
        400.0
    )


def test_unregistered_motion_branch_keeps_camera_exposure_geometry(monkeypatch):
    """Run the worker branch that separates camera scoring from vote geometry."""
    shape = (100, 100)
    anchor = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype=np.float32)
    camera_corners = anchor + np.float32([10, 0])
    source = np.zeros(shape, dtype=np.float32)
    source[5:45, 40:50] = 1.0
    exposure_visible = np.zeros(shape, dtype=bool)
    exposure_visible[:, :50] = True
    exposure = SimpleNamespace(visible=exposure_visible, paper_exposed=False)

    class _AlwaysUnregistered:
        def __init__(self):
            self.measure_calls = []

        def reset(self):
            pass

        def measure(self, image, matrix, visible, current_anchor, **kwargs):
            self.measure_calls.append(matrix)
            return SimpleNamespace(max_px=10.0, inliers=24, blocks=24)

        def aligned(self, _measurement):
            return False

        def decide(self, _registered, _unregistered):
            return "unregistered", None, "unregistered"

    worker = worker_mod.PageCleanWorker(_ProbIsWarp())
    fake_motion = _AlwaysUnregistered()
    worker._motion = fake_motion
    # Keep this test on the registration/motion/exposure seam; residual
    # estimation has its own contracts and is not part of this reproduction.
    monkeypatch.setattr(worker, "_pass_identity_residual", lambda _p: None)

    worker_pass = worker_mod._Pass(
        warped=np.zeros((*shape, 3), dtype=np.uint8),
        generation=1,
        corners=camera_corners.copy(),
        visible=None,
        visible_cam=None,
        prob=source,
        first_pass=True,
        anchor=anchor.copy(),
        anchor_ink=None,
        registration_corners=camera_corners.copy(),
        exposure=exposure,
    )

    worker._pass_register_motion_coverage(worker_pass)

    assert worker_pass.motion_decision == "unregistered"
    assert len(fake_motion.measure_calls) == 2
    assert fake_motion.measure_calls[0] is not None
    assert fake_motion.measure_calls[1] is None
    np.testing.assert_array_equal(worker_pass.registration_corners, anchor)
    np.testing.assert_array_equal(worker_pass.camera_corners, camera_corners)

    wrong_visible, _unused_exposed = worker_mod._exposure_masks_in_geometry(
        exposure, anchor, worker_pass.registration_corners, shape
    )
    assert float(np.where(wrong_visible, worker_pass.camera_prob, 0.0).sum()) == 0.0
    assert float(worker_pass.camera_prob.sum(dtype=np.float64)) == pytest.approx(400.0)
    assert worker_pass.camera_denominator_mass == pytest.approx(400.0)


def test_tied_residual_hypotheses_hold_anchor_and_abstain_from_new_ink():
    height, width = 240, 320
    shape = (height, width)
    anchor = np.array([(20, 20), (340, 20), (340, 260), (20, 260)], dtype=np.float32)
    pose = anchor + np.float32([0, 14])
    snapshot = np.zeros(shape, dtype=bool)
    raw = np.zeros(shape, dtype=np.float32)
    for y in range(30, 211, 6):
        snapshot[y : y + 3, 20:60] = True
    # This old mark distinguishes the two mappings, but is absent from the
    # first observation and becomes visible only after the hold begins.
    snapshot[105:108, 20:100] = True
    for y in (36, 42, 48):
        raw[y : y + 3, 20:60] = 1.0

    effective_anchor = registration_mod._shift_anchor_corners(anchor, 0, -6, shape)
    camera_matrix = registration_mod._registration_matrix(effective_anchor, pose, shape)
    camera = cv2.warpPerspective(
        raw,
        camera_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    worker = worker_mod.PageCleanWorker(lambda image: None)
    worker._anchor_corners = anchor.copy()
    worker._accumulator._confirmed = snapshot.astype(np.uint8)
    worker._identity_residual = (0, -6)
    worker._anchor_follow_step = lambda *args, **kwargs: ("closed", None)
    worker_pass = worker_mod._Pass(
        generation=1,
        corners=pose.copy(),
        anchor=anchor.copy(),
        registration_corners=pose.copy(),
        prob=raw,
        camera_prob=camera,
        camera_denominator_mass=float(raw.sum(dtype=np.float64)),
        identity_residual=(0, -6),
        identity_residual_pose_fresh=False,
    )

    for _ in range(3):
        worker_pass.reg_info = {}
        worker._pass_anchor_gates(worker_pass)

    event = worker_pass.reg_info["arbitration_event"]
    assert event["score_page"] == pytest.approx(1.0)
    assert event["score_page_without_residual"] == pytest.approx(1.0)
    assert event["score_camera"] == pytest.approx(1 / 3)
    assert event["verdict"] == "indeterminate"
    assert event["action"] is False
    assert event["identity_residual_pending_reason"] == (
        "residual_hypotheses_indistinguishable"
    )
    assert worker_pass.arbitration_abstain is True
    assert worker._identity_residual == (0, -6)
    np.testing.assert_array_equal(worker._anchor_corners, anchor)
    assert not worker_pass.reg_info.get("identity_residual_retired_on_anchor_landing")

    # A new mark still has the six-pixel raw-warp bias. While evidence cannot
    # distinguish the two old-page mappings, it must not be committed through
    # either mapping or silently lose the residual.
    new_raw = np.zeros(shape, dtype=np.float32)
    new_raw[176:182, 40:280] = 1.0
    new_camera = cv2.warpPerspective(
        new_raw,
        camera_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    next_pass = worker_mod._Pass(
        generation=1,
        corners=pose.copy(),
        anchor=anchor.copy(),
        registration_corners=pose.copy(),
        prob=new_raw,
        registered_prob=new_camera,
        camera_prob=new_camera,
        camera_denominator_mass=float(new_raw.sum(dtype=np.float64)),
        identity_residual=(0, -6),
        identity_residual_pose_fresh=False,
    )
    worker._pass_anchor_gates(next_pass)
    worker._pass_prepare_vote(next_pass)

    assert next_pass.pass_voted is False
    assert worker._identity_residual == (0, -6)
    np.testing.assert_array_equal(worker._anchor_corners, anchor)

    # When the distinguishing old mark becomes visible at its residual-biased
    # raw position, later settled arbitration can land and retain the
    # still-needed correction instead of leaving the worker locked out.
    resolved_raw = raw.copy()
    resolved_raw[111:114, 20:100] = 1.0
    resolved_camera = cv2.warpPerspective(
        resolved_raw,
        camera_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    resolve_pass = worker_mod._Pass(
        generation=1,
        corners=pose.copy(),
        anchor=anchor.copy(),
        registration_corners=pose.copy(),
        prob=resolved_raw,
        registered_prob=resolved_camera,
        camera_prob=resolved_camera,
        camera_denominator_mass=float(resolved_raw.sum(dtype=np.float64)),
        identity_residual=(0, -6),
        identity_residual_pose_fresh=False,
    )
    for _ in range(2):
        resolve_pass.reg_info = {}
        worker._pass_anchor_gates(resolve_pass)

    resolved_event = resolve_pass.reg_info["arbitration_event"]
    assert resolved_event["verdict"] == "landing"
    assert resolved_event["score_page"] > resolved_event["score_page_without_residual"]
    assert resolve_pass.arbitration_abstain is False
    assert worker._identity_residual == (0, -6)
    np.testing.assert_allclose(worker._anchor_corners, pose)

    next_raw = np.zeros(shape, dtype=np.float32)
    next_raw[176:182, 40:280] = 1.0
    residual_anchor = registration_mod._shift_anchor_corners(
        worker._anchor_corners, *worker._identity_residual, shape
    )
    next_matrix = registration_mod._registration_matrix(residual_anchor, pose, shape)
    placed_mark = cv2.warpPerspective(
        next_raw,
        next_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    rows = np.flatnonzero(np.any(placed_mark > 0.5, axis=1))
    assert (int(rows[0]), int(rows[-1]) + 1) == (170, 176)


def test_cyclic_residual_tie_is_indeterminate(monkeypatch):
    height, width = 240, 320
    shape = (height, width)
    anchor = np.array([(20, 20), (340, 20), (340, 260), (20, 260)], dtype=np.float32)
    pose = anchor + np.float32([0, 14])
    snapshot = np.zeros(shape, dtype=bool)
    cyclic_candidate = np.zeros(shape, dtype=np.float32)
    raw = np.zeros(shape, dtype=np.float32)
    for y in range(0, height, 6):
        snapshot[y : y + 3, 20:280] = True
        cyclic_candidate[y : y + 3, 20:280] = 1.0
    raw[0:3, 0:120] = 1.0  # Valid evidence that loses both page hypotheses.

    def permute_candidate(_prob, quarter_turns):
        return cyclic_candidate if quarter_turns == -1 else np.zeros(shape, np.float32)

    monkeypatch.setattr(registration_mod, "_permute_prob", permute_candidate)
    worker = worker_mod.PageCleanWorker(lambda image: None)
    worker._anchor_corners = anchor.copy()
    worker._accumulator._confirmed = snapshot.astype(np.uint8)

    status = None
    event = None
    for _ in range(3):
        status, event = worker._anchor_arbitration_step(
            generation=1,
            corners=pose,
            shape=shape,
            identity_owned=False,
            reg_anchor_median=False,
            reg_deadband=False,
            raw_prob=raw,
            registered_prob=np.zeros(shape, dtype=np.float32),
            grid_shift=(0, -6),
            residual_fresh=False,
            camera_denominator_mass=float(raw.sum(dtype=np.float64)),
        )

    assert status == "verdict"
    assert event is not None
    assert event["scores_cyclic"][1] > 0.9
    assert event["scores_cyclic_without_residual"][0] > 0.9
    assert (
        abs(event["scores_cyclic"][1] - event["scores_cyclic_without_residual"][0])
        <= 0.05
    )
    assert event["verdict"] == "indeterminate"
    assert event["identity_residual_pending_reason"] == (
        "residual_hypotheses_indistinguishable"
    )


def test_camera_denominator_does_not_fall_back_when_geometry_exceeds_memory_cap():
    source = np.ones((100, 100), dtype=np.float32)
    matrix = np.diag([6.0, 6.0, 1.0]).astype(np.float64)
    canvas = cv2.warpPerspective(
        source, matrix, (100, 100), flags=cv2.INTER_LINEAR, borderValue=0.0
    )
    worker_pass = worker_mod._Pass(
        prob=source,
        camera_prob=canvas,
        camera_matrix=matrix,
    )

    assert worker_mod._camera_probability_denominator(worker_pass) is None


def test_padded_camera_warp_matches_normal_warp_and_excludes_invalid_source():
    source = np.ones((100, 100), dtype=np.float32)
    visible = np.ones_like(source, dtype=bool)
    visible[:, :40] = False
    matrix = np.array(
        [[1.0, 0.04, 7.0], [0.02, 1.0, -5.0], [0.0002, 0.0004, 1.0]],
        dtype=np.float64,
    )
    padded = registration_mod._warp_probability_padded(
        source, matrix, source_mask=visible
    )
    assert padded is not None
    complete, origin, _padded_matrix = padded
    normal = cv2.warpPerspective(
        source * visible,
        matrix,
        (100, 100),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    x0, y0 = origin
    overlap = complete[-y0 : -y0 + 100, -x0 : -x0 + 100]

    np.testing.assert_allclose(overlap, normal, atol=2e-5, rtol=0)
    assert float(complete.sum(dtype=np.float64)) < 6_000.0
    assert float(complete.sum(dtype=np.float64)) > 0.0


@pytest.mark.parametrize("homography_scale", [1e-8, -1e-8, 1e8])
def test_padded_camera_warp_is_invariant_to_homography_scale(homography_scale):
    """A homography and any nonzero scalar multiple map the same points."""
    source = np.zeros((100, 100), dtype=np.float32)
    source[20:70, 35:80] = 1.0
    matrix = np.array(
        [[1.0, 0.04, 7.0], [0.02, 1.0, -5.0], [0.0002, 0.0004, 1.0]],
        dtype=np.float64,
    )

    expected = registration_mod._warp_probability_padded(source, matrix)
    actual = registration_mod._warp_probability_padded(
        source, matrix * homography_scale
    )

    assert expected is not None
    assert actual is not None
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-5, rtol=0)
    assert actual[1] == expected[1]


def test_padded_camera_warp_preserves_the_regular_destination_grid():
    """Integer padding must leave ordinary warp samples unchanged."""
    rng = np.random.default_rng(20261124)
    source = rng.random((48, 64), dtype=np.float32)
    for _ in range(24):
        matrix = np.array(
            [
                [
                    rng.uniform(0.88, 1.12),
                    rng.uniform(-0.08, 0.08),
                    rng.uniform(-12, 12),
                ],
                [
                    rng.uniform(-0.08, 0.08),
                    rng.uniform(0.88, 1.12),
                    rng.uniform(-12, 12),
                ],
                [rng.uniform(-0.0003, 0.0003), rng.uniform(-0.0003, 0.0003), 1.0],
            ],
            dtype=np.float64,
        )
        padded = registration_mod._warp_probability_padded(source, matrix)
        assert padded is not None
        complete, (x0, y0), _padded_matrix = padded
        regular = cv2.warpPerspective(
            source,
            matrix,
            (source.shape[1], source.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderValue=0.0,
        )
        crop = complete[-y0 : -y0 + source.shape[0], -x0 : -x0 + source.shape[1]]
        np.testing.assert_allclose(crop, regular, atol=1e-4, rtol=0)


def test_streamed_padded_warp_sum_matches_materialized_masked_reference():
    rng = np.random.default_rng(20261124)
    source = rng.random((48, 64), dtype=np.float32)
    source_mask = rng.random(source.shape) > 0.18
    output_mask = rng.random(source.shape) > 0.22
    destination_mask = np.zeros(source.shape, dtype=bool)
    destination_mask[12:24, 8:28] = True
    matrix = np.array(
        [[1.08, 0.04, 13.0], [-0.02, 1.1, -7.0], [0.0002, -0.0003, 1.0]],
        dtype=np.float64,
    )
    page_matrix = np.array(
        [[0.96, 0.0, 2.0], [0.0, 0.96, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    output_mask_matrix = matrix @ page_matrix

    padded = registration_mod._warp_probability_padded(
        source, matrix, source_mask=source_mask
    )
    assert padded is not None
    complete, origin, _padded_matrix = padded
    padding = np.array(
        [
            [1.0, 0.0, -origin[0]],
            [0.0, 1.0, -origin[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    full_coverage = (
        cv2.warpPerspective(
            output_mask.astype(np.uint8),
            padding @ output_mask_matrix,
            (complete.shape[1], complete.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        > 0
    )
    embedded_destination = registration_mod._embed_destination_mask(
        destination_mask, origin, complete.shape
    )
    assert embedded_destination is not None
    allowed = full_coverage | embedded_destination
    expected = float(np.sum(complete, where=allowed, dtype=np.float64))

    actual = registration_mod._sum_probability_warp_padded(
        source,
        matrix,
        source_mask=source_mask,
        output_mask=output_mask,
        output_mask_matrix=output_mask_matrix,
        destination_mask=destination_mask,
        tile_size=19,
    )

    assert actual == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("homography_scale", [1e-8, -1e-8, 1e8])
def test_streamed_padded_warp_normalizes_coverage_homography_scale(homography_scale):
    source = np.zeros((80, 100), dtype=np.float32)
    source[10:60, 20:80] = 1.0
    output_mask = np.zeros(source.shape, dtype=bool)
    output_mask[:, 15:] = True
    matrix = np.array(
        [[1.0, 0.03, 8.0], [-0.01, 1.0, -4.0], [0.0001, 0.0002, 1.0]],
        dtype=np.float64,
    )
    page_matrix = np.array(
        [[0.96, 0.0, 2.0], [0.0, 0.96, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    output_mask_matrix = matrix @ page_matrix
    expected = registration_mod._sum_probability_warp_padded(
        source,
        matrix,
        output_mask=output_mask,
        output_mask_matrix=output_mask_matrix,
        tile_size=23,
    )
    actual = registration_mod._sum_probability_warp_padded(
        source,
        matrix,
        output_mask=output_mask,
        output_mask_matrix=output_mask_matrix * homography_scale,
        tile_size=23,
    )

    assert actual == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("homography_scale", [1e-300, -1e-300, 1e300, -1e300])
def test_streamed_warp_normalizes_probability_and_composed_coverage_together(
    homography_scale,
):
    source = np.zeros((80, 100), dtype=np.float32)
    source[8:72, 13:88] = np.linspace(0.2, 1.0, 64, dtype=np.float32)[:, None]
    source_mask = np.ones(source.shape, dtype=bool)
    source_mask[25:40, 30:50] = False
    output_mask = np.zeros(source.shape, dtype=bool)
    output_mask[5:70, 20:95] = True
    destination_mask = np.zeros(source.shape, dtype=bool)
    destination_mask[15:25, 15:35] = True

    matrix = np.array(
        [[1.0, 0.03, 8.0], [-0.01, 1.0, -4.0], [0.0001, 0.0002, 1.0]],
        dtype=np.float64,
    )
    page_matrix = np.array(
        [[0.96, 0.0, 2.0], [0.0, 0.96, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    output_mask_matrix = matrix @ page_matrix
    expected = registration_mod._sum_probability_warp_padded(
        source,
        matrix,
        source_mask=source_mask,
        output_mask=output_mask,
        output_mask_matrix=output_mask_matrix,
        destination_mask=destination_mask,
        tile_size=23,
    )
    actual = registration_mod._sum_probability_warp_padded(
        source,
        matrix * homography_scale,
        source_mask=source_mask,
        output_mask=output_mask,
        output_mask_matrix=output_mask_matrix * homography_scale,
        destination_mask=destination_mask,
        tile_size=23,
    )

    assert expected is not None
    assert actual == pytest.approx(expected, abs=1e-4)


def test_streamed_padded_warp_bounds_each_output_allocation(monkeypatch):
    source = np.ones((20, 1000), dtype=np.float32)
    matrix = np.diag([2.1, 1.5, 1.0]).astype(np.float64)
    padded = registration_mod._warp_probability_padded(source, matrix)
    assert padded is not None
    expected = float(padded[0].sum(dtype=np.float64))

    seen_sizes = []
    real_warp = cv2.warpPerspective

    def record_warp(image, transform, size, **kwargs):
        seen_sizes.append(size)
        return real_warp(image, transform, size, **kwargs)

    monkeypatch.setattr(registration_mod.cv2, "warpPerspective", record_warp)
    tile_size = 256
    actual = registration_mod._sum_probability_warp_padded(
        source, matrix, tile_size=tile_size
    )

    assert actual == pytest.approx(expected, abs=1e-3)
    assert len(seen_sizes) > 1
    assert all(
        width <= tile_size and height <= tile_size for width, height in seen_sizes
    )


def test_camera_denominator_applies_the_source_coverage_outside_canvas_too():
    source = np.ones((100, 100), dtype=np.float32)
    visible = np.ones_like(source, dtype=bool)
    coverage = np.zeros_like(source, dtype=bool)
    coverage[:, 50:] = True
    matrix = np.array([[1, 0, 20], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    page_corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    page_matrix = registration_mod._corners_page_matrix(page_corners, 100, 100)
    canvas = cv2.warpPerspective(
        source, matrix, (100, 100), flags=cv2.INTER_LINEAR, borderValue=0
    )
    worker_pass = worker_mod._Pass(
        prob=source,
        visible=visible,
        camera_prob=canvas
        * cv2.warpPerspective(
            coverage.astype(np.uint8),
            matrix @ page_matrix,
            (100, 100),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        ),
        camera_matrix=matrix,
        camera_page_matrix=page_matrix,
        coverage_cam=coverage,
    )

    assert float(worker_pass.camera_prob.sum(dtype=np.float64)) == pytest.approx(
        3_000.0
    )
    assert worker_mod._camera_probability_denominator(worker_pass) == pytest.approx(
        5_000.0
    )


def test_camera_denominator_uses_anchor_ink_only_as_a_coverage_exception():
    source = np.ones((100, 100), dtype=np.float32)
    matrix = np.array([[1, 0, 20], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    canvas = cv2.warpPerspective(
        source, matrix, (100, 100), flags=cv2.INTER_LINEAR, borderValue=0
    )
    anchor_ink = np.zeros(source.shape, dtype=bool)
    anchor_ink[20:30, 20:30] = True
    worker_pass = worker_mod._Pass(
        prob=source,
        camera_prob=canvas,
        camera_matrix=matrix,
        anchor_ink=anchor_ink,
    )

    assert worker_mod._camera_probability_denominator(worker_pass) == pytest.approx(
        10_000.0
    )


def test_anchor_event_reports_camera_score_against_unclipped_mass():
    shape = (100, 100)
    anchor = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    moved = anchor + np.float32([20, 0])
    snapshot = np.zeros(shape, dtype=bool)
    snapshot[:, -1] = True
    raw_page = np.zeros(shape, dtype=np.float32)
    raw_page[:50, -1] = 1.0
    raw_page[50:, 0] = 1.0
    camera = np.zeros(shape, dtype=np.float32)
    camera[:, -1] = 1.0
    worker = worker_mod.PageCleanWorker(_ProbIsWarp())
    worker._anchor_corners = anchor.copy()
    worker._accumulator._confirmed = snapshot.astype(np.uint8)

    for _ in range(3):
        status, event = worker._anchor_arbitration_step(
            generation=1,
            corners=moved,
            shape=shape,
            identity_owned=False,
            reg_anchor_median=False,
            reg_deadband=False,
            raw_prob=raw_page,
            registered_prob=camera,
            camera_denominator_mass=10_000.0,
        )

    assert status == "verdict"
    assert event["score_camera"] == pytest.approx(0.01)
    assert event["camera_denominator_mass"] == pytest.approx(10_000.0)


def test_anchor_page_score_ignores_probability_under_occlusion():
    shape = (100, 100)
    anchor = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype=np.float32)
    moved = anchor + np.float32([20, 0])
    snapshot = np.zeros(shape, dtype=bool)
    snapshot[:30, :20] = True
    raw_page = np.zeros(shape, dtype=np.float32)
    raw_page[:30, :20] = 1.0  # hidden prediction exactly overlaps the canvas
    raw_page[55:75, 55:75] = 1.0  # visible prediction misses the canvas
    visible = np.ones(shape, dtype=bool)
    visible[:30, :20] = False
    camera = snapshot.astype(np.float32)
    worker = worker_mod.PageCleanWorker(_ProbIsWarp())
    worker._anchor_corners = anchor.copy()
    worker._accumulator._confirmed = snapshot.astype(np.uint8)

    status = None
    event = None
    for _ in range(3):
        status, event = worker._anchor_arbitration_step(
            generation=1,
            corners=moved,
            shape=shape,
            identity_owned=False,
            reg_anchor_median=False,
            reg_deadband=False,
            raw_prob=raw_page,
            registered_prob=camera,
            visible=visible,
            camera_denominator_mass=float(np.sum(camera)),
        )

    assert status == "verdict"
    assert event is not None
    assert event["score_page"] == pytest.approx(0.0)
    assert event["page_score_available"] is True
    assert event["verdict"] == "spurious"


def test_page_arbitration_mask_intersects_hand_exposure_and_sheet_coverage():
    shape = (20, 30)
    camera_shape = (40, 60)
    corners = np.array([(0, 0), (60, 0), (60, 40), (0, 40)], dtype=np.float32)
    hand_visible = np.ones(shape, dtype=bool)
    hand_visible[0, 0] = False
    exposure_visible = np.ones(shape, dtype=bool)
    exposure_visible[0, 1] = False
    static_coverage = np.ones(shape, dtype=bool)
    static_coverage[0, 2] = False
    camera_coverage = np.ones(camera_shape, dtype=bool)
    camera_coverage[:, :30] = False
    page_matrix = registration_mod._corners_page_matrix(corners, shape[1], shape[0])
    page = SimpleNamespace(
        prob=np.zeros(shape, dtype=np.float32),
        visible=hand_visible,
        exposure=SimpleNamespace(visible=exposure_visible),
        coverage_page_static=static_coverage,
        coverage_cam=camera_coverage,
        camera_page_matrix=page_matrix,
    )

    observed = worker_mod._page_arbitration_visible(page)

    assert observed.shape == shape
    assert observed[0, 0] is np.False_
    assert observed[0, 1] is np.False_
    assert observed[0, 2] is np.False_
    assert observed[10, 5] is np.False_
    assert observed[10, 20] is np.True_


def test_unmeasurable_camera_support_abstains_instead_of_winning_by_zero():
    shape = (100, 100)
    anchor = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype=np.float32)
    moved = anchor + np.float32([20, 0])
    page = np.ones(shape, dtype=np.float32)
    worker = worker_mod.PageCleanWorker(_ProbIsWarp())
    worker._anchor_corners = anchor.copy()
    worker._accumulator._confirmed = np.ones(shape, dtype=np.uint8)

    for _ in range(3):
        status, event = worker._anchor_arbitration_step(
            generation=1,
            corners=moved,
            shape=shape,
            identity_owned=False,
            reg_anchor_median=False,
            reg_deadband=False,
            raw_prob=page,
            registered_prob=page,
            camera_denominator_mass=None,
            camera_score_available=False,
        )

    assert status == "verdict"
    assert event is not None
    assert event["score_page"] == pytest.approx(1.0)
    assert event["page_score_available"] is True
    assert event["verdict"] == "indeterminate"
    assert event["action"] is False
    assert event["camera_score_available"] is False


def test_empty_registered_map_does_not_erase_valid_page_score():
    """A measurable camera miss must not zero an independently valid page score."""
    shape = (100, 100)
    anchor = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype=np.float32)
    moved = anchor + np.float32([20, 0])
    snapshot = np.zeros(shape, dtype=bool)
    snapshot[20:40, 20:40] = True
    raw_page = snapshot.astype(np.float32)
    registered = np.zeros(shape, dtype=np.float32)

    worker = worker_mod.PageCleanWorker(_ProbIsWarp())
    worker._anchor_corners = anchor.copy()
    worker._accumulator._confirmed = snapshot.astype(np.uint8)

    status = None
    event = None
    for _ in range(3):
        status, event = worker._anchor_arbitration_step(
            generation=1,
            corners=moved,
            shape=shape,
            identity_owned=False,
            reg_anchor_median=False,
            reg_deadband=False,
            raw_prob=raw_page,
            registered_prob=registered,
            camera_denominator_mass=400.0,
        )

    assert status == "verdict"
    assert event is not None
    assert event["score_page"] == pytest.approx(1.0)
    assert event["score_camera"] == pytest.approx(0.0)
    assert event["page_score_available"] is True
    assert event["camera_score_available"] is True
    assert event["verdict"] == "landing"
    assert event["action"] is True


def test_anchor_arbitration_prefers_rival_over_clipped_page_fragment():
    """A 10% surviving page fragment must not beat 90% camera support."""
    shape = (100, 100)
    anchor = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype=np.float32)
    moved = anchor + np.float32([20, 0])
    snapshot = np.zeros(shape, dtype=bool)
    snapshot[:90, -3:] = True  # 270 px, above the minimum canvas mass

    raw_page = np.zeros(shape, dtype=np.float32)
    raw_page[:40, :10] = 0.75  # 300 mass units; 90% leaves the grid
    camera = np.zeros(shape, dtype=np.float32)
    camera[:90, -3:] = 1.0  # 270/300 matched mass
    camera[90:, :3] = 1.0  # 30/300 unmatched mass

    worker = worker_mod.PageCleanWorker(_ProbIsWarp())
    worker._anchor_corners = anchor.copy()
    worker._accumulator._confirmed = snapshot.astype(np.uint8)
    status = None
    event = None
    for _ in range(3):
        status, event = worker._anchor_arbitration_step(
            generation=1,
            corners=moved,
            shape=shape,
            identity_owned=False,
            reg_anchor_median=False,
            reg_deadband=False,
            raw_prob=raw_page,
            registered_prob=camera,
            grid_shift=(99, 0),
            residual_fresh=False,
        )

    assert status == "verdict"
    assert event is not None
    assert event["verdict"] == "spurious", event
    assert event["score_page"] == pytest.approx(0.1)
    assert event["score_camera"] == pytest.approx(0.9)


def test_overlap_fraction_returns_zero_for_nonpositive_mass():
    from draw.ink.anchor import _overlap_fraction

    snapshot = np.ones((8, 8), dtype=bool)
    full_probability = np.ones((8, 8), dtype=np.float32)
    empty_probability = np.zeros((8, 8), dtype=np.float32)

    assert _overlap_fraction(full_probability, snapshot, denominator_mass=0.0) == 0.0
    assert _overlap_fraction(full_probability, snapshot, denominator_mass=-1.0) == 0.0
    assert _overlap_fraction(empty_probability, snapshot) == 0.0


def test_cyclic_linear_resize_keeps_a_thin_stroke_score_bounded():
    """O resize INTER_LINEAR de rot90 pode inflar uma linha de 1 px.

    No canto da grade, a massa transformada fica 11,1% maior que a bruta.
    Dividir pelo total bruto dá 1,111; a definição atual divide pelo mapa
    transformado, que está no mesmo domínio do numerador.
    """
    from draw.ink.anchor import _overlap_fraction

    prob = np.zeros((240, 320), dtype=np.float32)
    prob[:, -1] = 1.0
    transformed = registration_mod._permute_prob(prob, 1)
    snapshot = transformed > 0

    raw_total = float(np.sum(prob))
    # Sem translacao, a regra anterior nao via massa cortada e mantinha
    # `raw_total` como denominador, mesmo com a soma inflada pelo resize.
    old_score = float(np.sum(transformed[snapshot])) / raw_total
    score = _overlap_fraction(transformed, snapshot)
    assert float(np.sum(transformed)) / float(np.sum(prob)) == pytest.approx(10 / 9)
    assert old_score > 1.1
    assert score == 1.0


def test_cyclic_score_uses_resized_mass_before_translation_clips():
    """Rot90 interpolation mass and integer translation loss share a domain.

    The source line expands from 240 to 266.67 mass units after the cyclic
    resize. Translating it one pixel clips 0.83 units. Subtracting that loss
    from the *pre-resize* 240 total would leave a denominator smaller than
    the 265.83 matched units, producing a score above one.
    """
    shape = (240, 320)
    anchor = np.array([(0, 0), (320, 0), (320, 240), (0, 240)], dtype=np.float32)
    moved = anchor + np.float32([20, 0])
    raw = np.zeros(shape, dtype=np.float32)
    raw[:, -1] = 1.0
    permuted = registration_mod._permute_prob(raw, -1)
    translated = registration_mod._translate_grid(permuted, 1, 0)
    snapshot = translated > 0.0

    source_mass = float(np.sum(raw, dtype=np.float64))
    permuted_mass = float(np.sum(permuted, dtype=np.float64))
    translated_mass = float(np.sum(translated, dtype=np.float64))
    matched_mass = float(np.sum(translated[snapshot], dtype=np.float64))
    mixed_domain_score = matched_mass / (
        source_mass - (permuted_mass - translated_mass)
    )
    assert permuted_mass > source_mass
    assert translated_mass < permuted_mass
    assert mixed_domain_score > 1.0

    worker = worker_mod.PageCleanWorker(_ProbIsWarp())
    worker._anchor_corners = anchor.copy()
    worker._accumulator._confirmed = snapshot.astype(np.uint8)
    status = None
    event = None
    for _ in range(3):
        status, event = worker._anchor_arbitration_step(
            generation=1,
            corners=moved,
            shape=shape,
            identity_owned=False,
            reg_anchor_median=False,
            reg_deadband=False,
            raw_prob=raw,
            registered_prob=translated,
            grid_shift=(1, 0),
            residual_fresh=False,
            camera_denominator_mass=translated_mass,
        )

    assert status == "verdict"
    assert event is not None
    expected = matched_mass / permuted_mass
    assert event["scores_cyclic"][1] == pytest.approx(expected)
    assert event["scores_cyclic"][1] < 1.0


@pytest.mark.parametrize(
    ("limit", "expected_displacement", "expected_source"),
    [
        (1.9, 10.0, "fresh_residual_does_not_explain_pose"),
        (2.0, 2.0, "fresh_residual_explains_pose"),
        (2.1, 2.0, "fresh_residual_explains_pose"),
    ],
)
def test_pose_residual_tolerance_is_configured_at_1_9_2_and_2_1_px(
    limit, expected_displacement, expected_source
):
    from draw.ink.config import InkConfig

    worker = worker_mod.PageCleanWorker(
        _ProbIsWarp(), config=InkConfig(residual_explains_px=limit)
    )
    worker._anchor_corners = BASE.copy()
    corners = registration_mod._shift_anchor_corners(BASE, 0, 10, SHAPE)

    displacement, source = worker._pose_displacement(
        corners, SHAPE, (0, 8), residual_fresh=True
    )

    assert displacement == pytest.approx(expected_displacement, abs=0.01)
    assert source == expected_source


def test_stale_identity_residual_cannot_explain_a_pose_displacement():
    from draw.ink.config import InkConfig

    worker = worker_mod.PageCleanWorker(
        _ProbIsWarp(), config=InkConfig(residual_explains_px=2.0)
    )
    worker._anchor_corners = BASE.copy()
    corners = registration_mod._shift_anchor_corners(BASE, 0, 10, SHAPE)

    displacement, source = worker._pose_displacement(
        corners, SHAPE, (0, 8), residual_fresh=False
    )

    assert displacement == pytest.approx(10.0, abs=0.01)
    assert source == "stale_residual"
