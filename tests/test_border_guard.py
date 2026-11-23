"""Regressao da guarda de borda na grade retificada (issue #131)."""

import dataclasses

import cv2
import numpy as np
import pytest

from draw import contracts
from draw.ink import identity as _identity
from draw.ink import registration
from draw.ink.accumulator import LayeredInkAccumulator
from draw.ink.config import InkConfig, resolve_border_guard_px
from draw.ink.worker import PageCleanWorker, _border_guard_mask
from draw.pipeline import Pipeline
from draw.settings import PipelineConfig


class _DarkInkPageClean:
    """Stub deterministico: pixel escuro vira probabilidade alta de tinta."""

    threshold = 0.7

    def ink_prob(self, image):
        gray = image.mean(axis=2)
        return np.where(gray < 130.0, 0.95, 0.0).astype(np.float32)


class _ScriptedDetector:
    def __init__(self, script):
        self._script = script
        self._seq = 0
        self._result = None
        self.error = None

    def submit(self, _frame, captured_at, frame_index=None):
        corners, verdict, extras = self._script(self._seq)
        self._seq += 1
        extras = dict(extras or {})
        self._result = contracts.DetectionResult(
            sequence=self._seq,
            corners=corners,
            verdict=verdict,
            captured_at=captured_at,
            detect_ms=1.0,
            coverage_mask=extras.get("coverage_mask"),
            hand_mask=extras.get("hand_mask"),
        )

    def poll(self, after_seq):
        if self._result is None or self._result.sequence <= after_seq:
            return None
        return self._result

    def take_traces(self, upto_result_seq=None):
        return []


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _drive_with_worker(script, frame, *, pipeline_config, ink_config, steps=18):
    clock = _Clock()
    worker = PageCleanWorker(
        _DarkInkPageClean(),
        config=ink_config,
        clock=clock,
    )
    loop = Pipeline(
        detection_worker=_ScriptedDetector(script),
        ink_worker=worker,
        config=pipeline_config,
        clock=clock,
        log=lambda *_a: None,
    )
    for i in range(1, steps + 1):
        clock.now += 0.2
        loop.step(contracts.FramePacket(image=frame, sequence=i, captured_at=clock.now))
        worker.pump()
        loop.collect_ink()
    return loop, worker


def test_border_guard_config_derivation_and_validation():
    assert resolve_border_guard_px((1200, 850), None) == 7
    assert resolve_border_guard_px((850, 1200), None) == 7
    assert resolve_border_guard_px((1200, 850), 0) == 0
    assert resolve_border_guard_px((1200, 850), 12) == 12
    assert resolve_border_guard_px((40, 60), None, min_auto_side=200) == 0
    assert resolve_border_guard_px((40, 60), 5, min_auto_side=200) == 5

    small_quad = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    mask = _border_guard_mask((1200, 850), None, small_quad)
    assert mask is not None
    assert not mask[:8, :].any()
    assert not mask[:, :8].any()
    assert mask[10:-10, 10:-10].all()
    explicit = _border_guard_mask((1200, 850), 5, small_quad)
    assert explicit is not None
    assert not explicit[:5, :].any()
    assert explicit[5:-5, 5:-5].all()

    # O pixel so entra INTEIRO no inset: a primeira coluna admitida
    # nunca amostra a menos do jitter de camera (3,5 px) da borda da
    # folha. Rasterizar pelo centro do pixel admitia uma coluna a
    # ~3,46 px — dentro da faixa que a guarda existe para excluir.
    span_quad = np.array(
        [[200, 100], [620, 100], [620, 600], [200, 600]], dtype=np.float32
    )
    mask = _border_guard_mask((1200, 850), None, span_quad)
    cols = np.flatnonzero(mask.any(axis=0))
    inv = np.linalg.inv(registration._corners_page_matrix(span_quad, 850, 1200))
    for col in (cols[0], cols[-1]):
        cam = cv2.perspectiveTransform(np.array([[[col, 600]]], dtype=np.float32), inv)[
            0, 0
        ]
        assert cam[0] - 200 > 3.5
        assert 620 - cam[0] > 3.5

    with pytest.raises(ValueError, match="border_guard_px"):
        InkConfig(border_guard_px=-1)


@pytest.mark.parametrize("with_coverage", [True, False])
def test_jitter_against_dark_desk_does_not_confirm_or_provisional_border_ink(
    with_coverage,
):
    """Jitter de 1-3 px contra mesa escura nao carimba margens nem cantos,
    enquanto um traco desenhado na area util confirma normalmente."""
    frame = np.full((720, 1280, 3), 20, np.uint8)  # mesa/teclado escuro
    sheet_y0, sheet_y1, sheet_x0, sheet_x1 = 100, 600, 200, 560
    frame[sheet_y0:sheet_y1, sheet_x0:sheet_x1] = 215  # pagina clara
    # Traco real dentro da area util
    frame[300:380, 360:372] = 25

    base_quad = np.array(
        [
            [sheet_x0, sheet_y0],
            [sheet_x1, sheet_y0],
            [sheet_x1, sheet_y1],
            [sheet_x0, sheet_y1],
        ],
        dtype=np.float32,
    )
    jitters = [
        np.array([[-3, -2], [2, -3], [3, 2], [-2, 3]], dtype=np.float32),
        np.array([[-1, -3], [3, -1], [2, 3], [-3, 1]], dtype=np.float32),
        np.array([[-2, -1], [1, -2], [3, 1], [-1, 2]], dtype=np.float32),
    ]
    cov_mask = np.zeros((720, 1280), np.uint8)
    cov_mask[sheet_y0:sheet_y1, sheet_x0:sheet_x1] = 255

    def script(seq):
        quad = base_quad + jitters[seq % len(jitters)]
        extras = {"coverage_mask": cov_mask} if with_coverage else None
        return quad, "paper", extras

    ink_cfg = dataclasses.replace(InkConfig.production(), paper_guard=False)
    pipe_cfg = PipelineConfig(
        coverage=with_coverage,
        ink_auto_gain=False,
        ink=ink_cfg,
    )
    loop, _worker = _drive_with_worker(
        script,
        frame,
        pipeline_config=pipe_cfg,
        ink_config=ink_cfg,
    )

    confirmed = loop.clean_mask
    provisional = loop.clean_provisional_mask
    assert confirmed is not None
    assert int(np.count_nonzero(confirmed)) > 0, "traco util precisa confirmar"

    guard = resolve_border_guard_px(confirmed.shape, ink_cfg.border_guard_px)
    assert guard > 0
    for mask in (confirmed, provisional):
        if mask is None:
            continue
        assert int(np.count_nonzero(mask[:guard, :])) == 0
        assert int(np.count_nonzero(mask[-guard:, :])) == 0
        assert int(np.count_nonzero(mask[:, :guard])) == 0
        assert int(np.count_nonzero(mask[:, -guard:])) == 0


def test_empty_page_with_jitter_against_dark_desk_stays_empty():
    """Pagina totalmente vazia com jitter de 1-3 px nao gera tinta
    confirmada nem provisoria."""
    frame = np.full((720, 1280, 3), 20, np.uint8)
    frame[100:600, 200:560] = 215

    base_quad = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]],
        dtype=np.float32,
    )
    jitters = [
        np.array([[-3, -3], [3, -2], [2, 3], [-3, 2]], dtype=np.float32),
        np.array([[-2, -2], [2, -3], [3, 2], [-2, 3]], dtype=np.float32),
    ]
    cov_mask = np.zeros((720, 1280), np.uint8)
    cov_mask[100:600, 200:560] = 255

    ink_cfg = dataclasses.replace(InkConfig.production(), paper_guard=False)
    pipe_cfg = PipelineConfig(
        coverage=True,
        ink_auto_gain=False,
        ink=ink_cfg,
    )
    loop, _worker = _drive_with_worker(
        lambda seq: (
            base_quad + jitters[seq % 2],
            "paper",
            {"coverage_mask": cov_mask},
        ),
        frame,
        pipeline_config=pipe_cfg,
        ink_config=ink_cfg,
    )

    assert loop.clean_mask is not None
    assert int(np.count_nonzero(loop.clean_mask)) == 0
    if loop.clean_provisional_mask is not None:
        assert int(np.count_nonzero(loop.clean_provisional_mask)) == 0


def test_border_guard_disabled_discriminates_dark_border_jitter():
    """Sem guarda (border_guard_px=0 e sem cobertura), o jitter contra
    fundo escuro vaza para as margens — prova que o teste discrimina."""
    frame = np.full((720, 1280, 3), 20, np.uint8)
    frame[100:600, 200:560] = 215
    quad = np.array(
        [[197, 97], [563, 97], [563, 603], [197, 603]],
        dtype=np.float32,
    )
    ink_cfg = InkConfig.strict_vote(0.85, border_guard_px=0, paper_guard=False)
    pipe_cfg = PipelineConfig(
        coverage=False,
        ink_auto_gain=False,
        ink=ink_cfg,
    )
    loop, _worker = _drive_with_worker(
        lambda _seq: (quad.copy(), "paper", None),
        frame,
        pipeline_config=pipe_cfg,
        ink_config=ink_cfg,
    )
    assert loop.clean_mask is not None
    assert int(np.count_nonzero(loop.clean_mask[:5, :])) > 0


def test_empty_coverage_mask_still_gates_votes():
    """Cobertura toda vazia continua vedando todos os votos."""
    frame = np.full((720, 1280, 3), 20, np.uint8)
    quad = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]],
        dtype=np.float32,
    )
    empty_cov = np.zeros((720, 1280), np.uint8)
    ink_cfg = dataclasses.replace(InkConfig.production(), paper_guard=False)
    pipe_cfg = PipelineConfig(
        coverage=True,
        coverage_iou_floor=0.0,
        ink_auto_gain=False,
        ink=ink_cfg,
    )
    loop, _worker = _drive_with_worker(
        lambda _seq: (quad.copy(), "paper", {"coverage_mask": empty_cov}),
        frame,
        pipeline_config=pipe_cfg,
        ink_config=ink_cfg,
    )
    assert loop.clean_mask is not None
    assert int(np.count_nonzero(loop.clean_mask)) == 0


@pytest.mark.parametrize(
    "ink_cfg",
    [
        dataclasses.replace(InkConfig.production(), paper_guard=False),
        InkConfig.sketch(),
    ],
)
def test_empty_page_without_coverage_and_small_camera_sheet_stays_empty(ink_cfg):
    """Sem cobertura e com folha pequena na camera (360x500 -> 850x1200),
    a guarda projetada cobre o estiramento de 3 px da camera e mantem
    toda a pagina vazia (inclusive logo dentro da margem de 7 px)."""
    frame = np.full((720, 1280, 3), 20, np.uint8)
    frame[100:600, 200:560] = 215
    quad = np.array(
        [[197, 97], [563, 97], [563, 603], [197, 603]],
        dtype=np.float32,
    )
    pipe_cfg = PipelineConfig(
        coverage=False,
        ink_auto_gain=False,
        ink=ink_cfg,
    )
    loop, _worker = _drive_with_worker(
        lambda _seq: (quad.copy(), "paper", None),
        frame,
        pipeline_config=pipe_cfg,
        ink_config=ink_cfg,
    )
    assert loop.clean_mask is not None
    assert int(np.count_nonzero(loop.clean_mask)) == 0
    if loop.clean_provisional_mask is not None:
        assert int(np.count_nonzero(loop.clean_provisional_mask)) == 0


def test_near_edge_stroke_survives_persistent_inward_quad_shift():
    """Traco proximo a borda util (fora da guarda da ancora) sobrevive
    mesmo quando o quad recua 3 px para dentro e permanece ali."""
    frame = np.full((720, 1280, 3), 20, np.uint8)
    frame[100:600, 200:560] = 215
    # Traco a 5 px da borda esquerda da folha (~12 px na grade 850 px)
    frame[250:350, 205:208] = 25

    anchor_quad = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]],
        dtype=np.float32,
    )
    shifted_quad = np.array(
        [[203, 100], [560, 100], [560, 600], [203, 600]],
        dtype=np.float32,
    )

    def script(seq):
        return (anchor_quad if seq <= 3 else shifted_quad).copy(), "paper", None

    ink_cfg = dataclasses.replace(InkConfig.production(), paper_guard=False)
    pipe_cfg = PipelineConfig(
        coverage=False,
        ink_auto_gain=False,
        ink=ink_cfg,
    )
    loop, _worker = _drive_with_worker(
        script,
        frame,
        pipeline_config=pipe_cfg,
        ink_config=ink_cfg,
        steps=24,
    )
    assert loop.clean_mask is not None
    assert int(np.count_nonzero(loop.clean_mask[:, 9:30])) > 0


def test_trapezoid_guard_suppresses_jitter_and_preserves_less_magnified_edge():
    """Em pagina inclinada (trapezio 220..460 x 400 -> 850x1200), o inset
    projetado pela homografia protege a borda mais comprimida sem cortar
    um traco valido perto da borda inferior menos ampliada."""
    trap = np.array(
        [[530, 100], [750, 100], [870, 500], [410, 500]],
        dtype=np.float32,
    )
    mask = _border_guard_mask((1200, 850), None, trap)
    assert mask is not None
    # A ~9 px da borda inferior (y=1190), 6 px na camera (~9 px no grid):
    # preservado pela projecao em perspectiva (antes 11 px o cortava).
    assert mask[1190, 425]
    # Borda superior mais esticada recebe guarda maior (> 7 px):
    assert not mask[8, 425]


def test_confirmed_stroke_survives_reanchor_that_widens_guard():
    """Quando a mesma folha reancora mais longe na camera e a guarda alarga,
    tracos ja confirmados dentro da nova faixa nao sao apagados (com controle
    mostrando que a remocao os apagaria sem a guarda)."""
    near_quad = np.array(
        [[100, 50], [800, 50], [800, 1050], [100, 1050]], dtype=np.float32
    )
    far_quad = np.array(
        [[100, 50], [320, 50], [320, 350], [100, 350]], dtype=np.float32
    )
    img_with_edge = np.full((1200, 850, 3), 215, np.uint8)
    img_with_edge[200:500, 11:16] = 25
    img_with_edge[300:700, 300:500] = 25  # testemunha no miolo
    img_no_edge = np.full((1200, 850, 3), 215, np.uint8)
    img_no_edge[300:700, 300:500] = 25
    full_visible = np.ones((1200, 850), dtype=bool)

    def run_case(border_guard_px):
        clock = _Clock()
        cfg = InkConfig.strict_vote(
            0.85, border_guard_px=border_guard_px, paper_guard=False
        )
        worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
        for _ in range(5):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(
                    img_with_edge,
                    0,
                    near_quad,
                    visible=full_visible,
                    captured_at=clock.now,
                )
            )
            worker.pump()
        snap = worker.poll_display(0, 0)
        assert snap is not None
        assert int(np.count_nonzero(snap.confirmed[:, 11:16])) > 0

        worker._anchor_corners = far_quad.copy()
        for _ in range(10):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(
                    img_no_edge,
                    0,
                    far_quad,
                    visible=full_visible,
                    captured_at=clock.now,
                )
            )
            worker.pump()
        snap_after = worker.poll_display(0, 0)
        assert snap_after is not None
        return int(np.count_nonzero(snap_after.confirmed[:, 11:16]))

    assert run_case(None) > 0
    assert run_case(0) == 0


def test_erased_interior_stroke_removed_without_visibility_mask():
    """Sem mascara de visibilidade (deteccao classica sem modelo de
    mao), a guarda delimita o dominio de remocao: o traco interior
    apagado sai do canvas enquanto a testemunha permanece. O controle
    (guarda desligada e sem mascara) mantem a abstencao legada."""
    quad = np.array([[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32)
    img = np.full((1200, 850, 3), 215, np.uint8)
    img[300:700, 300:500] = 25  # testemunha no miolo (permanece)
    img[200:500, 100:110] = 25  # traco interior, apagado depois

    def run_case(border_guard_px):
        clock = _Clock()
        cfg = InkConfig.strict_vote(
            0.85, border_guard_px=border_guard_px, paper_guard=False
        )
        worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
        for _ in range(5):
            clock.now += 1.1
            worker.submit(contracts.WarpedPage(img, 0, quad, captured_at=clock.now))
            worker.pump()
        snap = worker.poll_display(0, 0)
        assert snap is not None
        assert int(np.count_nonzero(snap.confirmed[200:500, 100:110])) > 0

        img_erased = img.copy()
        img_erased[200:500, 100:110] = 215
        for _ in range(10):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(img_erased, 0, quad, captured_at=clock.now)
            )
            worker.pump()
        snap_after = worker.poll_display(0, 0)
        assert snap_after is not None
        assert int(np.count_nonzero(snap_after.confirmed[300:700, 300:500])) > 0
        return int(np.count_nonzero(snap_after.confirmed[200:500, 100:110]))

    assert run_case(None) == 0
    assert run_case(0) > 0


def test_unsampled_edge_strip_does_not_count_as_erasure():
    """Quad que recua uma borda para dentro deixa uma faixa sem amostra
    no espaco da ancora: os zeros de preenchimento do registro nao podem
    contar como borracha. Um traco amostrado e apagado de verdade continua
    saindo, e a testemunha interior permanece."""
    anchor_quad = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    shifted_quad = np.array(
        [[206, 100], [560, 100], [560, 600], [206, 600]], dtype=np.float32
    )
    img = np.full((1200, 850, 3), 215, np.uint8)
    img[200:500, 10:14] = 25  # traco na faixa que fica sem amostra
    img[200:500, 100:110] = 25  # traco interior, apagado de verdade
    img[300:700, 300:500] = 25  # testemunha no miolo (permanece)

    clock = _Clock()
    cfg = InkConfig.strict_vote(0.85, border_guard_px=None, paper_guard=False)
    worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
    for _ in range(5):
        clock.now += 1.1
        worker.submit(contracts.WarpedPage(img, 0, anchor_quad, captured_at=clock.now))
        worker.pump()
    snap = worker.poll_display(0, 0)
    assert snap is not None
    assert int(np.count_nonzero(snap.confirmed[200:500, 10:14])) > 0
    assert int(np.count_nonzero(snap.confirmed[200:500, 100:110])) > 0

    img_erased = np.full((1200, 850, 3), 215, np.uint8)
    img_erased[300:700, 300:500] = 25
    for _ in range(10):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(img_erased, 0, shifted_quad, captured_at=clock.now)
        )
        worker.pump()
    snap_after = worker.poll_display(0, 0)
    assert snap_after is not None
    assert int(np.count_nonzero(snap_after.confirmed[200:500, 10:14])) > 0
    assert int(np.count_nonzero(snap_after.confirmed[200:500, 100:110])) == 0
    assert int(np.count_nonzero(snap_after.confirmed[300:700, 300:500])) > 0


def test_deadband_frozen_registration_does_not_erase_unsampled_edge():
    """Variante sob banda morta: o quad recua uma borda menos que
    tau, entao o registro segue no quad CONGELADO — a matriz sai
    identidade e a pegada do registro e a grade inteira. Sem a
    pegada do quad vivo cruzada, os zeros/papel deslocado da faixa
    nao amostrada apagariam o traco de borda como se tivessem sido
    vistos. O controle interior apagado de verdade continua saindo.
    """
    anchor_quad = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    shifted_quad = np.array(
        [[206, 100], [560, 100], [560, 600], [206, 600]], dtype=np.float32
    )
    img = np.full((1200, 850, 3), 215, np.uint8)
    img[200:500, 10:14] = 25  # traco na faixa que fica sem amostra
    img[200:500, 100:110] = 25  # traco interior, apagado de verdade
    img[300:700, 300:500] = 25  # testemunha no miolo (permanece)

    clock = _Clock()
    cfg = InkConfig.strict_vote(
        0.85, border_guard_px=None, paper_guard=False, reg_deadband=True
    )
    worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
    for _ in range(5):
        clock.now += 1.1
        worker.submit(contracts.WarpedPage(img, 0, anchor_quad, captured_at=clock.now))
        worker.pump()
    snap = worker.poll_display(0, 0)
    assert snap is not None
    assert int(np.count_nonzero(snap.confirmed[200:500, 10:14])) > 0
    assert int(np.count_nonzero(snap.confirmed[200:500, 100:110])) > 0

    # O deslocamento de 6 px fica abaixo do tau da banda morta: o
    # registro segue congelado na ancora, entao a unica marca de
    # "nao observei" e a pegada do quad vivo.
    img_erased = np.full((1200, 850, 3), 215, np.uint8)
    img_erased[300:700, 300:500] = 25
    for _ in range(10):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(img_erased, 0, shifted_quad, captured_at=clock.now)
        )
        worker.pump()
    snap_after = worker.poll_display(0, 0)
    assert snap_after is not None
    assert int(np.count_nonzero(snap_after.confirmed[200:500, 10:14])) > 0
    assert int(np.count_nonzero(snap_after.confirmed[200:500, 100:110])) == 0
    assert int(np.count_nonzero(snap_after.confirmed[300:700, 300:500])) > 0


def test_guard_widening_clears_pending_removal_streak():
    """Guarda que alarga por cima de uma borracha em contagem nao pode
    congelar o streak: o reset depende do dominio inspecionado, nao da
    testemunha — contagem velha fora do dominio zera e a trabalhadora
    volta a descansar em vez de rodar inferencia com a folha parada."""
    acc = LayeredInkAccumulator(0.7)
    shape = (1200, 850)
    prob = np.zeros(shape, np.float32)
    prob[200:500, 20:24] = 0.95  # traco A, "apagado" em contagem
    prob[300:700, 300:500] = 0.95  # testemunha B
    for _ in range(4):
        acc.add(prob)
    guard = np.ones(shape, bool)
    partial = prob.copy()
    partial[200:500, 20:24] = 0.0
    for _ in range(3):
        acc.add(partial, valid_mask=guard)
    assert int(acc._removal_streak[200:500, 20:24].max()) > 0
    # A guarda alarga por cima de A e a testemunha some junto: sem
    # ela o incremento para, mas a contagem velha fora do dominio
    # ainda precisa zerar.
    guard2 = guard.copy()
    guard2[:, :30] = False
    acc.add(np.zeros(shape, np.float32), valid_mask=guard2)
    assert int(acc._removal_streak[200:500, 20:24].max()) == 0


def test_confirmed_ink_under_guard_does_not_trigger_page_turn():
    """Tinta confirmada quase toda sob a guarda alargada: a virada se
    abstem em vez de declarar pagina trocada na mesma folha."""
    acc = LayeredInkAccumulator(0.7)
    shape = (1200, 850)
    prob = np.zeros(shape, np.float32)
    prob[150:550, 10:18] = 0.95  # ~3200 px perto da borda esquerda
    for _ in range(4):
        acc.add(prob)
    guard = np.ones(shape, dtype=bool)
    guard[:, :40] = False  # a tinta confirmada ficou sob a guarda
    empty = np.zeros(shape, np.float32)
    turns = [acc.add(empty, valid_mask=guard)["turn"] for _ in range(3)]
    assert not any(turns)


def test_border_noise_does_not_reset_returning_page_identity():
    """Ruido escuro nas faixas de borda (fora da guarda) nao pode
    derrubar o containment da recuperacao de identidade: o canvas
    confirmado — inclusive o que a passada de volta nao reproduz —
    e preservado."""
    quad = np.array([[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32)
    img = np.full((1200, 850, 3), 215, np.uint8)
    img[200:400, 200:320] = 25  # A1: reaparece na volta
    img[200:400, 400:440] = 25  # A2: nao reaparece na volta

    clock = _Clock()
    cfg = InkConfig.strict_vote(0.85, border_guard_px=None, paper_guard=False)
    worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
    for _ in range(5):
        clock.now += 1.1
        worker.submit(contracts.WarpedPage(img, 0, quad, captured_at=clock.now))
        worker.pump()
    gen = worker.new_generation(preserve_identity=True)

    img_return = np.full((1200, 850, 3), 215, np.uint8)
    img_return[200:400, 200:320] = 25  # A1 reaparece
    img_return[:8, :] = 25  # ruido escuro nas 4 faixas de borda
    img_return[-8:, :] = 25
    img_return[:, :8] = 25
    img_return[:, -8:] = 25
    for _ in range(6):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(img_return, gen, quad, captured_at=clock.now)
        )
        worker.pump()
    snap = worker.poll(gen, 0)
    assert snap is not None
    assert int(np.count_nonzero(snap.confirmed[200:400, 400:440])) > 0
    assert int(np.count_nonzero(snap.confirmed[200:400, 200:320])) > 0


def test_saved_ink_under_guard_does_not_time_out_page_recovery():
    """Toda a tinta confirmada sob a guarda nao pode segurar a
    recuperacao em adiamento de oclusao ate o timeout de
    quarentena apagar o canvas — a faixa e estatica, o que ela
    esconde nao volta a aparecer."""
    quad_wide = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    quad_narrow = np.array(
        [[270, 120], [500, 120], [500, 580], [270, 580]], dtype=np.float32
    )
    img = np.full((1200, 850, 3), 215, np.uint8)
    img[100:1100, 9:13] = 25  # traco rente a borda esquerda

    clock = _Clock()
    cfg = InkConfig.strict_vote(0.85, border_guard_px=None, paper_guard=False)
    worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
    for _ in range(5):
        clock.now += 1.1
        worker.submit(contracts.WarpedPage(img, 0, quad_wide, captured_at=clock.now))
        worker.pump()
    # Reancoragem para um quad menor na camera: a guarda alarga
    # (~13 px) e a tinta confirmada rente a borda fica toda sob a
    # faixa — a referencia salva sai daqui.
    worker._anchor_corners = quad_narrow.copy()
    gen = worker.new_generation(preserve_identity=True)

    all_clear = np.ones(img.shape[:2], bool)
    for _ in range(7):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(
                img, gen, quad_narrow, all_clear, captured_at=clock.now
            )
        )
        worker.pump()
    snap = worker.poll(gen, 0)
    assert snap is not None
    assert int(np.count_nonzero(snap.confirmed[100:1100, 9:13])) > 0


def test_occluded_replacement_sheet_does_not_keep_old_ink():
    """Match "vazio" com a mao sobre a folha nova adia: o dominio
    inspecionado pode esconder a tinta do candidato dentro da
    guarda. Revelada, a troca e reconhecida — e a tinta da folha
    velha nao carrega para o canvas novo."""
    quad_wide = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    quad_narrow = np.array(
        [[270, 120], [500, 120], [500, 580], [270, 580]], dtype=np.float32
    )
    img_old = np.full((1200, 850, 3), 215, np.uint8)
    img_old[100:1100, 9:13] = 25  # traco da folha velha (sob a guarda nova)

    clock = _Clock()
    cfg = InkConfig.strict_vote(0.85, border_guard_px=None, paper_guard=False)
    worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
    for _ in range(5):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(img_old, 0, quad_wide, captured_at=clock.now)
        )
        worker.pump()
    worker._anchor_corners = quad_narrow.copy()
    gen = worker.new_generation(preserve_identity=True)

    img_new = np.full((1200, 850, 3), 215, np.uint8)
    img_new[400:600, 500:800] = 25  # sketch da folha nova
    hand = np.ones(img_new.shape[:2], bool)
    hand[:, 350:] = False  # mao esconde a metade direita na camera
    for _ in range(2):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(img_new, gen, quad_narrow, hand, captured_at=clock.now)
        )
        worker.pump()
    all_clear = np.ones(img_new.shape[:2], bool)
    for _ in range(6):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(
                img_new, gen, quad_narrow, all_clear, captured_at=clock.now
            )
        )
        worker.pump()
    snap = worker.poll(gen, 0)
    assert snap is not None
    assert int(np.count_nonzero(snap.confirmed[100:1100, 9:13])) == 0


def test_allclear_unsampled_strip_does_not_erase_saved_ink():
    """A faixa que o registro deixa de amostrar nao e oclusao de
    mao: com mascara all-clear e quad recuado (pegada cortada por
    dentro da guarda), a recuperacao nao pode adiar ate o timeout
    da quarentena e resetar o canvas — a tinta salva sob a guarda
    sobrevive. Com guarda desligada o mesmo vale para o traco
    inteiro nao amostrado numa folha limpa."""
    quad_wide = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    quad_narrow = np.array(
        [[270, 120], [500, 120], [500, 580], [270, 580]], dtype=np.float32
    )
    # Borda direita recuada: a pegada corta uma faixa dentro da
    # guarda direita; o traco esquerdo continua sob a guarda.
    quad_right = np.array(
        [[270, 120], [494, 120], [494, 580], [270, 580]], dtype=np.float32
    )
    # Borda esquerda recuada: o traco inteiro cai fora da pegada.
    quad_left = np.array(
        [[276, 120], [500, 120], [500, 580], [276, 580]], dtype=np.float32
    )
    img_old = np.full((1200, 850, 3), 215, np.uint8)
    img_old[100:1100, 9:13] = 25  # traco salvo sob a guarda nova
    img_blank = np.full((1200, 850, 3), 215, np.uint8)
    all_clear = np.ones(img_old.shape[:2], bool)

    def run_case(border_guard_px, img_return, quad_return):
        clock = _Clock()
        cfg = InkConfig.strict_vote(
            0.85, border_guard_px=border_guard_px, paper_guard=False
        )
        worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
        for _ in range(5):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(img_old, 0, quad_wide, captured_at=clock.now)
            )
            worker.pump()
        worker._anchor_corners = quad_narrow.copy()
        gen = worker.new_generation(preserve_identity=True)

        for _ in range(8):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(
                    img_return, gen, quad_return, all_clear, captured_at=clock.now
                )
            )
            worker.pump()
        snap = worker.poll(gen, 0)
        assert snap is not None
        return int(np.count_nonzero(snap.confirmed[100:1100, 9:13]))

    assert run_case(None, img_old, quad_right) > 0
    assert run_case(0, img_blank, quad_left) > 0


def test_disjoint_replacement_sheet_does_not_keep_old_ink():
    """Folha nova em quad disjunto do salvo na camera: a pegada do
    registro nao toca a regiao usavel da ancora — zero sobreposicao
    nao e "match vazio" (aceitaria mesma pagina e carregaria a
    tinta velha), e ausencia de evidencia. A referencia fica
    pendente, a quarentena estoura o limite de adiamentos e o
    canvas reseta — a folha de substituicao herda nada. Sem
    mascara de visibilidade vale o mesmo: com a guarda ativa a
    pegada vazia dispara o adiamento por "no_overlap"; sem ela a
    troca resolve pelo consenso de virada."""
    quad_wide = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    # Disjunto a direita: nenhum pixel da grade da ancora e amostrado.
    quad_far = np.array(
        [[620, 100], [840, 100], [840, 600], [620, 600]], dtype=np.float32
    )
    img_old = np.full((1200, 850, 3), 215, np.uint8)
    img_old[100:1100, 9:13] = 25  # traco da folha velha
    img_blank = np.full((1200, 850, 3), 215, np.uint8)
    all_clear = np.ones(img_old.shape[:2], bool)

    def run_case(border_guard_px, visible):
        clock = _Clock()
        cfg = InkConfig.strict_vote(
            0.85, border_guard_px=border_guard_px, paper_guard=False
        )
        worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
        for _ in range(5):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(img_old, 0, quad_wide, captured_at=clock.now)
            )
            worker.pump()
        worker._anchor_corners = quad_wide.copy()
        gen = worker.new_generation(preserve_identity=True)

        for _ in range(8):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(
                    img_blank, gen, quad_far, visible, captured_at=clock.now
                )
            )
            worker.pump()
        snap = worker.poll(gen, 0)
        assert snap is not None
        return int(np.count_nonzero(snap.confirmed[100:1100, 9:13]))

    assert run_case(None, all_clear) == 0
    assert run_case(0, all_clear) == 0
    assert run_case(None, None) == 0
    assert run_case(0, None) == 0


def test_deferred_partial_overlap_does_not_mix_ink():
    """Folha nova MARCADA com sobreposicao parcial da pegada (< 25%
    da regiao usavel da ancora): a recuperacao adia por "no_overlap"
    e a passada adiada nao vota — a tinta nova registrada na faixa
    amostrada nao pode confirmar sobre o canvas velho enquanto a
    identidade esta pendente (a quarentena misturando tinta de duas
    folhas). Estourada a quarentena o canvas reseta e a tinta velha
    nao carrega para a folha de substituicao — E a tinta nova
    confirma no canvas novo dentro da janela esperada (um canvas
    que ficasse em branco depois do reset tambem reprovaria)."""
    quad_wide = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    # Sobreposicao parcial a direita: a pegada amostrada cobre ~19%
    # da regiao usavel da ancora (cols ~685-849 da grade) — abaixo
    # do piso de evidencia, a decisao adia por "no_overlap". O
    # traco novo deste quad registra nessa faixa (cols ~684-781) e
    # confirmaria ali em ~3 passadas sob strict_vote se votasse.
    quad_part = np.array(
        [[490, 100], [840, 100], [840, 600], [490, 600]], dtype=np.float32
    )
    img_old = np.full((1200, 850, 3), 215, np.uint8)
    img_old[100:1100, 9:13] = 25  # traco da folha velha
    img_marked = np.full((1200, 850, 3), 215, np.uint8)
    img_marked[100:1100, 0:100] = 25  # traco novo, cai na faixa amostrada
    all_clear = np.ones(img_old.shape[:2], bool)

    def run_case(border_guard_px):
        clock = _Clock()
        cfg = InkConfig.strict_vote(
            0.85, border_guard_px=border_guard_px, paper_guard=False
        )
        worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
        for _ in range(5):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(img_old, 0, quad_wide, captured_at=clock.now)
            )
            worker.pump()
        worker._anchor_corners = quad_wide.copy()
        gen = worker.new_generation(preserve_identity=True)

        snap = None
        first_new_ink_pass = None
        canvas_went_blank = False
        for pass_i in range(1, 11):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(
                    img_marked, gen, quad_part, all_clear, captured_at=clock.now
                )
            )
            worker.pump()
            snap = worker.poll(gen, 0)
            assert snap is not None
            # Toda mascara publicada conta, nao so a final: a tinta
            # da folha nova nunca pode aparecer confirmada na faixa
            # amostrada do canvas velho enquanto a decisao adia.
            assert int(np.count_nonzero(snap.confirmed[:, 685:810])) == 0
            # O timeout da quarentena zera o canvas (a tinta velha
            # nao carrega para a folha de substituicao) — o traco
            # novo cobre as colunas do traco velho na grade do
            # quad_part, entao so o canvas em branco distingue o
            # reset de uma carregada de tinta.
            if not np.count_nonzero(snap.confirmed):
                canvas_went_blank = True
            # O traco novo registra nas proprias colunas (0:100)
            # quando a ancora vira o quad_part apos o reset.
            if first_new_ink_pass is None and np.count_nonzero(
                snap.confirmed[100:1100, :100]
            ):
                first_new_ink_pass = pass_i
        # O canvas passou por branco (reset de fato) e a folha nova
        # nao ficou em branco: a quarentena drena em
        # _PAGE_IDENTITY_MAX_DEFERRALS + 1 passadas (a do estouro
        # abstem) e o traco a 0,95 confirma em 3 votos fortes —
        # limite estrutural: MAX_DEFERRALS + 4.
        assert canvas_went_blank
        assert first_new_ink_pass is not None
        assert first_new_ink_pass <= _identity._PAGE_IDENTITY_MAX_DEFERRALS + 4
        return int(np.count_nonzero(snap.confirmed[100:1100, :100]))

    assert run_case(None) > 50_000
    assert run_case(0) > 50_000


def test_blank_overlap_replacement_does_not_inherit_old_ink():
    """Folha em branco cuja pegada cobre ~metade da regiao usavel da
    ancora (bem acima do piso de 25%) mas NENHUMA tinta salva: a
    referencia nao e vazia de verdade — o "match vazio" nao pode
    aceitar mesma pagina e consumir a copia. A passada adia por
    no_overlap (abstem, sem votar) e a quarentena limitada resolve
    pelo timeout: a substituicao nao herda traco nenhum."""
    quad_wide = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    # ~50% de sobreposicao util: a pegada cobre as cols ~425-849 da
    # grade da ancora — acima do piso, mas o traco salvo em 9:13
    # fica inteiro fora dela.
    quad_half = np.array(
        [[380, 100], [840, 100], [840, 600], [380, 600]], dtype=np.float32
    )
    img_old = np.full((1200, 850, 3), 215, np.uint8)
    img_old[100:1100, 9:13] = 25  # traco salvo na metade nao amostrada
    img_blank = np.full((1200, 850, 3), 215, np.uint8)
    all_clear = np.ones(img_old.shape[:2], bool)

    def run_case(border_guard_px):
        clock = _Clock()
        cfg = InkConfig.strict_vote(
            0.85, border_guard_px=border_guard_px, paper_guard=False
        )
        worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
        for _ in range(5):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(img_old, 0, quad_wide, captured_at=clock.now)
            )
            worker.pump()
        worker._anchor_corners = quad_wide.copy()
        gen = worker.new_generation(preserve_identity=True)

        for _ in range(10):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(
                    img_blank, gen, quad_half, all_clear, captured_at=clock.now
                )
            )
            worker.pump()
        snap = worker.poll(gen, 0)
        assert snap is not None
        # Nunca pode sair "mesma pagina" sem nenhuma tinta salva
        # julgada: a decisao tem de adiar ate o timeout da
        # quarentena, que reseta o canvas para a folha nova.
        decisions = [e["decision"] for e in worker._identity_events]
        assert "same_page" not in decisions
        return int(np.count_nonzero(snap.confirmed))

    assert run_case(None) == 0
    assert run_case(0) == 0


def test_occluded_blank_patch_hold_keeps_reference_recoverable():
    """Referencia com a tinta toda sob a guarda alargada + mao sobre
    um remendo interior em branco: o adiamento por interior oculto
    segura a decisao sem queimar a quarentena — alem do limite
    antigo a copia segue recuperavel. Levantada a mao: a MESMA
    folha resolve same_page com o traco intacto; uma folha marcada
    resolve new_page sem herdar nada."""
    quad_wide = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    quad_narrow = np.array(
        [[270, 120], [500, 120], [500, 580], [270, 580]], dtype=np.float32
    )
    img = np.full((1200, 850, 3), 215, np.uint8)
    img[100:1100, 9:13] = 25  # traco rente a borda, coberto pela guarda
    img_marked = np.full((1200, 850, 3), 215, np.uint8)
    img_marked[200:800, 200:600] = 25  # folha de substituicao marcada
    hand = np.ones(img.shape[:2], bool)
    hand[300:500, 300:400] = False  # mao no interior em branco
    all_clear = np.ones(img.shape[:2], bool)
    held = _identity._PAGE_IDENTITY_MAX_DEFERRALS + 3

    def run_case(reveal_img):
        clock = _Clock()
        cfg = InkConfig.strict_vote(0.85, border_guard_px=None, paper_guard=False)
        worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
        for _ in range(5):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(img, 0, quad_wide, captured_at=clock.now)
            )
            worker.pump()
        # Reancoragem alarga a guarda sobre o traco salvo.
        worker._anchor_corners = quad_narrow.copy()
        gen = worker.new_generation(preserve_identity=True)

        # Mao parada alem do antigo timeout de quarentena.
        for _ in range(held):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(img, gen, quad_narrow, hand, captured_at=clock.now)
            )
            worker.pump()
        snap = worker.poll(gen, 0)
        assert snap is not None
        # Durante o hold o desenho salvo mora so na referencia —
        # o canvas publicavel desta geracao segue vazio (a copia
        # volta no same_page, abaixo). Reter a referencia e
        # separado de apagar o desenho.
        assert int(np.count_nonzero(snap.confirmed[100:1100, 9:13])) == 0
        assert int(np.count_nonzero(worker._identity_reference)) > 0

        for _ in range(8):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(
                    reveal_img, gen, quad_narrow, all_clear, captured_at=clock.now
                )
            )
            worker.pump()
        snap = worker.poll(gen, 0)
        assert snap is not None
        return int(np.count_nonzero(snap.confirmed[100:1100, 9:13])), int(
            np.count_nonzero(snap.confirmed[200:800, 200:600])
        )

    # Mesma folha revelada: mesma pagina, traco preservado.
    old_px, _ = run_case(img)
    assert old_px > 0
    # Folha marcada: pagina nova — a velha nao carrega e a nova
    # confirma no proprio canvas.
    old_px, new_px = run_case(img_marked)
    assert old_px == 0
    assert new_px > 0


@pytest.mark.parametrize("paper_guard", [False, True])
def test_held_recovery_does_not_publish_saved_ink(paper_guard):
    """Caso do review: durante o hold a tinta salva mora so na
    referencia — nenhuma versao publicada (poll/collect_ink) pode
    carrega-la para a folha substituta. Mesma folha revelada:
    same_page devolve o desenho ao canvas e publica sob a geracao
    nova. Cobre border guard e as duas mascaras juntas."""
    quad_wide = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    quad_narrow = np.array(
        [[270, 120], [500, 120], [500, 580], [270, 580]], dtype=np.float32
    )
    img = np.full((1200, 850, 3), 215, np.uint8)
    img[100:1100, 9:13] = 25  # traco rente a borda, coberto pela guarda
    img_marked = np.full((1200, 850, 3), 215, np.uint8)
    img_marked[200:800, 200:600] = 25
    hand = np.ones(img.shape[:2], bool)
    hand[300:500, 300:400] = False
    all_clear = np.ones(img.shape[:2], bool)
    held = _identity._PAGE_IDENTITY_MAX_DEFERRALS + 3

    def run_case(reveal_img):
        clock = _Clock()
        cfg = InkConfig.strict_vote(0.85, border_guard_px=None, paper_guard=paper_guard)
        worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
        for _ in range(5):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(img, 0, quad_wide, captured_at=clock.now)
            )
            worker.pump()
        assert int(np.count_nonzero(worker._accumulator._confirmed)) > 0
        worker._anchor_corners = quad_narrow.copy()
        gen = worker.new_generation(preserve_identity=True)
        # O desenho saiu do acumulador e mora so na referencia.
        assert worker._accumulator._confirmed is None
        assert int(np.count_nonzero(worker._identity_reference)) > 0

        display_version = 0
        for _ in range(held):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(img, gen, quad_narrow, hand, captured_at=clock.now)
            )
            worker.pump()
            # Nenhuma versao publicada durante o hold pode conter
            # o traco velho — e o que o Save consumiria.
            snap = worker.poll(gen, 0)
            if snap is not None:
                assert int(np.count_nonzero(snap.confirmed[100:1100, 9:13])) == 0
            disp = worker.poll_display(gen, display_version)
            if disp is not None:
                display_version = disp.version
                assert int(np.count_nonzero(disp.confirmed[100:1100, 9:13])) == 0
                assert int(np.count_nonzero(disp.provisional[100:1100, 9:13])) == 0
        assert int(np.count_nonzero(worker._identity_reference)) > 0

        marks = []
        for _ in range(8):
            clock.now += 1.1
            worker.submit(
                contracts.WarpedPage(
                    reveal_img, gen, quad_narrow, all_clear, captured_at=clock.now
                )
            )
            worker.pump()
            snap = worker.poll(gen, 0)
            marks.append(
                0
                if snap is None
                else int(np.count_nonzero(snap.confirmed[200:800, 200:600]))
            )
        snap = worker.poll(gen, 0)
        assert snap is not None
        return (
            int(np.count_nonzero(snap.confirmed[100:1100, 9:13])),
            int(np.count_nonzero(snap.confirmed[200:800, 200:600])),
            marks,
        )

    # same_page: o desenho restaurado e publicado sob a geracao nova.
    old_px, _, _ = run_case(img)
    assert old_px > 0
    old_px, new_px, marks = run_case(img_marked)
    assert old_px == 0
    if paper_guard:
        # A folha marcada falha a leitura de papel exposto: a guarda
        # bloqueia a região como oclusor nas primeiras passadas — a
        # tinta nova NÃO confirma de cara. O modelo lendo tinta forte
        # de forma persistente faz a recuperação por preenchimento
        # adotar o bloco (draw.ink.occlusion, PR #144): confirmar ela
        # confirma, mas só depois do veto fotométrico — ao contrário
        # do caminho sem guarda, que vota imediatamente.
        assert max(marks[:4]) == 0
        assert new_px > 0
    else:
        assert new_px > 0


def test_second_tracking_drop_preserves_pending_reference():
    """Segunda queda de tracking com a recuperacao ainda pendente:
    new_generation nao sobrescreve a referencia pelo canvas vazio —
    ela segue recuperavel ate uma passada resolver."""
    quad_wide = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    quad_narrow = np.array(
        [[270, 120], [500, 120], [500, 580], [270, 580]], dtype=np.float32
    )
    img = np.full((1200, 850, 3), 215, np.uint8)
    img[100:1100, 9:13] = 25
    hand = np.ones(img.shape[:2], bool)
    hand[300:500, 300:400] = False
    all_clear = np.ones(img.shape[:2], bool)

    clock = _Clock()
    cfg = InkConfig.strict_vote(0.85, border_guard_px=None, paper_guard=False)
    worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
    for _ in range(5):
        clock.now += 1.1
        worker.submit(contracts.WarpedPage(img, 0, quad_wide, captured_at=clock.now))
        worker.pump()
    worker._anchor_corners = quad_narrow.copy()
    gen = worker.new_generation(preserve_identity=True)
    for _ in range(3):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(img, gen, quad_narrow, hand, captured_at=clock.now)
        )
        worker.pump()
    reference_px = int(np.count_nonzero(worker._identity_reference))
    assert reference_px > 0

    # Segunda queda: a referencia pendente nao some — o canvas vazio
    # desta geracao nao a sobrescreve.
    gen = worker.new_generation(preserve_identity=True)
    assert worker._identity_reference is not None
    assert int(np.count_nonzero(worker._identity_reference)) == reference_px

    for _ in range(3):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(
                img, gen, quad_narrow, all_clear, captured_at=clock.now
            )
        )
        worker.pump()
    snap = worker.poll(gen, 0)
    assert snap is not None
    assert int(np.count_nonzero(snap.confirmed[100:1100, 9:13])) > 0


def test_blank_partial_overlap_without_masks_defers_recovery():
    """Guarda desligada e sem mascara de visibilidade: folha em
    branco cobrindo so uma faixa do quad salvo, com a tinta toda na
    parte nao amostrada. O inconclusivo geometrico (no_overlap)
    adia — e nao o recall vazio — ate o timeout descartar a
    referencia. Antes do fix a pegada nao era medida nesse caminho
    e o ramo era inalcancavel."""
    quad_wide = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    # Cobre so ~60% a direita do quad salvo; o traco fica na faixa
    # nao amostrada.
    quad_shift = np.array(
        [[344, 100], [560, 100], [560, 600], [344, 600]], dtype=np.float32
    )
    img_old = np.full((1200, 850, 3), 215, np.uint8)
    img_old[100:1100, 9:13] = 25
    img_blank = np.full((1200, 850, 3), 215, np.uint8)

    clock = _Clock()
    cfg = InkConfig.strict_vote(0.85, border_guard_px=0, paper_guard=False)
    worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
    for _ in range(5):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(img_old, 0, quad_wide, captured_at=clock.now)
        )
        worker.pump()
    gen = worker.new_generation(preserve_identity=True)
    reference_px = int(np.count_nonzero(worker._identity_reference))
    assert reference_px > 0

    timed_out = False
    for _ in range(_identity._PAGE_IDENTITY_MAX_DEFERRALS + 4):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(img_blank, gen, quad_shift, captured_at=clock.now)
        )
        worker.pump()
        snap = worker.poll(gen, 0)
        # A tinta velha nunca circula pelo canvas publicavel.
        if snap is not None:
            assert int(np.count_nonzero(snap.confirmed)) == 0
        if worker._identity_reference is None:
            timed_out = True
        else:
            # Adiada, a referencia fica intacta — inconclusivo nao
            # apaga evidencia.
            assert int(np.count_nonzero(worker._identity_reference)) == reference_px
    reasons = [e["quarantine_reason"] for e in worker._identity_events]
    assert "no_overlap" in reasons
    assert timed_out
    confirmed = worker._accumulator._confirmed
    assert confirmed is None or int(np.count_nonzero(confirmed)) == 0


def test_cropped_provisional_clears_with_visibility_mask():
    """Provisorio na faixa que o registro deixa de amostrar sai do
    dominio valido mesmo com mascara de visibilidade presente —
    nao pode ficar congelado como "coberto pela mao". Controle:
    provisorio genuinamente ocluido permanece no preview."""
    quad_wide = np.array(
        [[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32
    )
    quad_in = np.array(
        [[206, 100], [560, 100], [560, 600], [206, 600]], dtype=np.float32
    )

    img = np.full((1200, 850, 3), 215, np.uint8)
    img[100:1100, 10:14] = 25  # provisorio na faixa que vai ser cortada
    img[100:1100, 300:340] = 25  # provisorio interior (sera ocluido)

    clock = _Clock()
    cfg = InkConfig.strict_vote(0.85, border_guard_px=None, paper_guard=False)
    worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
    for _ in range(2):
        clock.now += 1.1
        worker.submit(contracts.WarpedPage(img, 0, quad_wide, captured_at=clock.now))
        worker.pump()

    hand = np.ones(img.shape[:2], bool)
    hand[:, 300:400] = False  # mao cobre o traco interior na camera
    for _ in range(10):
        clock.now += 1.1
        worker.submit(
            contracts.WarpedPage(img, 0, quad_in, hand, captured_at=clock.now)
        )
        worker.pump()
    snap = worker.poll_display(0, 0)
    assert snap is not None
    assert int(np.count_nonzero(snap.provisional[100:1100, 10:14])) == 0
    assert int(np.count_nonzero(snap.provisional[100:1100, 300:340])) > 0


def test_stale_provisional_ink_clears_when_reanchor_widens_guard():
    """Marcas provisorias antigas que caem na nova faixa da guarda apos
    reancoragem nao ficam congeladas indefinidamente."""
    clock = _Clock()
    cfg = dataclasses.replace(InkConfig.production(), paper_guard=False)
    worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
    near_quad = np.array(
        [[100, 50], [800, 50], [800, 1050], [100, 1050]], dtype=np.float32
    )
    far_quad = np.array(
        [[100, 50], [320, 50], [320, 350], [100, 350]], dtype=np.float32
    )
    img_prov = np.full((1200, 850, 3), 215, np.uint8)
    img_prov[200:500, 11:16] = 25
    img_clean = np.full((1200, 850, 3), 215, np.uint8)

    clock.now += 0.2
    worker.submit(contracts.WarpedPage(img_prov, 0, near_quad, captured_at=clock.now))
    worker.pump()
    snap = worker.poll_display(0, 0)
    assert snap is not None
    assert int(np.count_nonzero(snap.provisional[:, 11:16])) > 0

    worker._anchor_corners = far_quad.copy()
    clock.now += 0.2
    worker.submit(contracts.WarpedPage(img_clean, 0, far_quad, captured_at=clock.now))
    worker.pump()
    snap_after = worker.poll_display(0, 0)
    assert snap_after is not None
    assert int(np.count_nonzero(snap_after.provisional[:, 11:16])) == 0


def test_corrupt_quad_guard_abstains_instead_of_wiping_provisional():
    """Quad corrompido (canto NaN): a guarda devolve None e a passada
    se abstem — dominio inspecionado vazio nao e "tudo falhou a
    inspecao". Antes: o inset NaN esvaziava a pegada/mascara valida,
    provisional_now &= {} zerava a camada inteira e retracted contava
    40k px de tinta "sumida" numa passada com deteccao quebrada."""
    nan_quad = np.array(
        [[200, 100], [560, 100], [np.nan, 600], [200, 600]], dtype=np.float32
    )
    assert _border_guard_mask((1200, 850), None, nan_quad) is None

    # A mesma invariante na fronteira do acumulador: um valid_mask
    # todo-False e dominio impossivel de julgar — a passada se
    # abstem, nao apaga provisorio nem conta retracao.
    quad = np.array([[200, 100], [560, 100], [560, 600], [200, 600]], dtype=np.float32)
    img = np.full((1200, 850, 3), 215, np.uint8)
    img[100:1100, 300:340] = 25

    clock = _Clock()
    cfg = InkConfig.strict_vote(0.85, border_guard_px=None, paper_guard=False)
    worker = PageCleanWorker(_DarkInkPageClean(), config=cfg, clock=clock)
    for _ in range(2):
        clock.now += 1.1
        worker.submit(contracts.WarpedPage(img, 0, quad, captured_at=clock.now))
        worker.pump()
    snap = worker.poll_display(0, 0)
    assert snap is not None
    before = int(np.count_nonzero(snap.provisional))
    assert before > 0

    clock.now += 1.1
    worker.submit(contracts.WarpedPage(img, 0, nan_quad, captured_at=clock.now))
    worker.pump()
    snap = worker.poll_display(0, 0)
    assert snap is not None
    assert int(np.count_nonzero(snap.provisional)) == before
    assert int(np.count_nonzero(worker._accumulator._retracted_ever)) == 0


def test_sim_writer_rejects_none_for_non_nullable_flags():
    from lab import sim_writer

    with pytest.raises(ValueError, match="booleano|invalido"):
        sim_writer._parse_flags(["coverage=none"])
