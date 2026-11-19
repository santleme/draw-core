"""O laço de orquestração (draw.pipeline) sobre workers falsos.

Cobre os caminhos que o SheetInkLoop legado exercitava com as
flags do preset, agora campos do PipelineConfig: cobertura de
folha no voto, persistência do canvas na queda do rastreador,
salto de passadas sobre cantos retidos, destrave da geometria
congelada sem mão, oclusão semântica da mão e os desfechos
reopen/drop/erro.
"""

import numpy as np
import pytest

from draw import pipeline as pipeline_module
from draw.contracts import (
    DetectionResult,
    FramePacket,
    InkDisplaySnapshot,
)
from draw.settings import PipelineConfig

QUAD = np.array(
    [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]],
    dtype=np.float32,
)


def _coverage_mask(shape=(720, 1280)):
    """Folha falsa: só a metade esquerda do quad é papel; a direita
    do quad (x >= 300) já cai sobre mesa — é o vazamento que a
    cobertura tem de vedar dentro da região retificada."""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[:, :300] = 255
    return mask


class _SyncDetectionWorker:
    """Publica DetectionResult como a worker real, na hora."""

    def __init__(self, script, traces=None):
        self._script = script
        self._traces = list(traces) if traces else []
        self._result = None
        self._seq = 0
        self.error = None

    def submit(self, frame, captured_at, frame_index=None):
        corners, verdict, extras = self._script(self._seq)
        self._seq += 1
        extras = extras or {}
        self._result = DetectionResult(
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
        taken, self._traces = self._traces, []
        return taken


class _FakeInkWorker:
    def __init__(self, display=None, display_generation=None):
        self.display_generation = display_generation
        self.error = None
        self.generation = 0
        self.submitted = []  # (generation, corners, visible, coverage)
        self.new_generation_calls = []
        self.presence_calls = []
        self._display = display

    def new_generation(self, preserve_identity=False):
        self.generation += 1
        self.new_generation_calls.append(preserve_identity)
        return self.generation

    def submit(self, page):
        self.submitted.append(
            (
                page.generation,
                None if page.corners is None else page.corners.copy(),
                None if page.visible is None else page.visible.copy(),
                None if page.coverage is None else page.coverage.copy(),
            )
        )

    def poll_display(self, generation, version):
        if (
            self.display_generation is not None
            and generation != self.display_generation
        ):
            return None
        return self._display

    def set_hand_presence(self, present, moved=False):
        self.presence_calls.append((present, moved))


class _FakeHandPresenceWorker:
    def __init__(self, results=(), error=None):
        self._results = list(results)
        self.error = error
        self.submitted = []

    def submit(self, frame, timestamp_ms):
        self.submitted.append(timestamp_ms)

    def poll(self, after_seq):
        if not self._results:
            return None
        seq, present, moved = self._results[0]
        if seq <= after_seq:
            return None
        return self._results.pop(0)


def _make_loop(
    script, *, config=None, traces=None, ink=None, hand_presence=None, clock=None
):
    det = _SyncDetectionWorker(script, traces=traces)
    ink = _FakeInkWorker() if ink is None else ink
    loop = pipeline_module.Pipeline(
        detection_worker=det,
        ink_worker=ink,
        hand_presence_worker=hand_presence,
        config=config,
        clock=clock or (lambda: 0.0),
        log=lambda *_a: None,
    )
    return loop, det, ink


def _frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _packet(loop, frame):
    return FramePacket(
        image=frame, sequence=loop.frame_number, captured_at=loop.frame_number * 0.1
    )


def _drive(loop, steps, frame=None):
    if frame is None:
        frame = _frame()
    for _ in range(steps):
        loop.step(_packet(loop, frame))


def test_pipeline_config_defaults_are_the_production_preset():
    config = PipelineConfig()
    assert config.coverage is True
    assert config.canvas_persist is True
    assert config.skip_stale is True
    assert config.seg_hand_occlusion is True
    assert config.follow_move is True


def test_no_extras_no_coverage_gate():
    """Sem máscara no resultado, o submit segue sem cobertura."""
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),
        config=PipelineConfig(coverage=True),
    )
    _drive(loop, 40)
    assert loop.tracker.corners is not None
    assert ink.submitted, "a passada de tinta precisa rodar"
    assert all(
        visible is None and coverage is None
        for _g, _c, visible, coverage in ink.submitted
    )


def test_coverage_mask_gates_votes_off_sheet():
    """A máscara da detecção veda o voto fora da folha retificada."""
    mask = _coverage_mask()
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": mask})
    )
    _drive(loop, 40)
    assert loop.tracker.corners is not None
    gated = [cov for _g, _c, _v, cov in ink.submitted if cov is not None]
    assert gated, "a cobertura precisa chegar ao submit"
    # Sem mão sintética o visible segue None — a cobertura viaja
    # em canal próprio e não mistura semântica de oclusão.
    assert all(v is None for _g, _c, v, _cov in ink.submitted)
    # A metade direita do quad cai sobre "mesa": nunca pode votar.
    page_h, page_w = gated[-1].shape
    assert not gated[-1][:, int(page_w * 0.60) :].any()
    # E a região de folha segue aberta para o voto.
    assert gated[-1][:, : int(page_w * 0.30)].all()


def test_coverage_config_off_ignores_the_mask():
    """coverage=False: a máscara publicada não entra no voto."""
    mask = _coverage_mask()
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": mask}),
        config=PipelineConfig(coverage=False),
    )
    _drive(loop, 40)
    assert loop._coverage_mask is None
    assert all(cov is None for _g, _c, _v, cov in ink.submitted)


def test_result_without_mask_clears_coverage():
    """Detecção sem máscara limpa a cobertura anterior."""
    state = {"with_mask": True}
    mask = _coverage_mask()

    def script(_i):
        if state["with_mask"]:
            return QUAD.copy(), "paper", {"coverage_mask": mask}
        return QUAD.copy(), "paper", {}

    loop, _det, ink = _make_loop(script)
    _drive(loop, 30)
    assert loop._coverage_mask is not None
    state["with_mask"] = False
    _drive(loop, 10)
    assert loop._coverage_mask is None
    # Depois da limpeza, o submit volta a ir sem cobertura.
    last = ink.submitted[-1]
    assert last[3] is None


def _mask_block(y0, y1, x0, x1, shape=(720, 1280)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def test_coverage_iou_floor_vetoes_phantom_quad():
    """Quad inflado sobre blob espúrio não sobe para a tinta.

    Mesa vazia real: a seg marca um blob longe do quad que o fit
    inflou (IoU quad×máscara ~0). Medido nas capturas s1–s4:
    fantasmas 0,044–0,108, folha real a partir de 0,298.
    """
    mask = _mask_block(50, 400, 900, 1100)  # blob à direita, quad em x<500
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": mask})
    )
    _drive(loop, 40)
    assert loop.tracker.corners is not None
    assert ink.submitted == []


def test_coverage_iou_floor_vetoes_phantom_through_app_detector():
    """Caminho real do app (review #45): a mascara do fantasma passa pelo
    filtro de 5% do detector combinado, e publicada e o piso veta.

    Medido nas capturas s1 (mesa vazia) com o draw_seg de producao:
    as 52 mascaras fantasmas ocupam 17%-28% do frame (todas >5%) e
    o IoU quad x mascara fica em 0,044-0,108. O fantasma do teste
    reproduz o pior caso medido (IoU ~0,10, sobrepondo o quad): um
    piso frouxo (0,05) deixa passar, o 0,2 veta.
    """
    from draw import app as app_module

    phantom = _mask_block(100, 400, 400, 1100)  # cobre 1/4 do QUAD + mesa
    assert 0.17 < np.count_nonzero(phantom) / phantom.size < 0.28

    class _Neural:
        last_presence_logit = 3.0
        last_coverage_mask = phantom
        last_hand_mask = None

        def __call__(self, frame, *, return_mask=True):
            return QUAD.copy(), "paper", phantom

    corners, verdict, _trace, extras = app_module._detect_sheet_combined_traced(
        _frame(), _Neural()
    )
    assert "coverage_mask" in extras  # sobreviveu ao filtro de 5%
    quad = np.zeros(phantom.shape, bool)
    quad[100:400, 100:500] = True  # QUAD é retângulo alinhado
    mask = phantom > 0
    iou = (quad & mask).sum() / (quad | mask).sum()
    assert 0.09 < iou < 0.11  # pior fantasma medido: 0,108

    loop, _det, ink = _make_loop(lambda _i: (corners, verdict, extras))
    _drive(loop, 40)
    assert loop.tracker.corners is not None
    assert ink.submitted == []

    loose, _det, ink_loose = _make_loop(
        lambda _i: (corners, verdict, extras),
        config=PipelineConfig(coverage_iou_floor=0.05),
    )
    _drive(loose, 40)
    assert ink_loose.submitted, "o teste precisa discriminar o piso"


def test_coverage_iou_floor_passes_aligned_sheet():
    """Máscara alinhada ao quad (folha real) segue submetendo."""
    mask = _mask_block(100, 400, 100, 500)  # coincide com o QUAD
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": mask})
    )
    _drive(loop, 40)
    assert ink.submitted, "a folha real não pode ser vetada"


def test_coverage_iou_floor_zero_disables():
    """coverage_iou_floor=0 desliga o piso (comportamento antigo)."""
    mask = _mask_block(50, 400, 900, 1100)
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": mask}),
        config=PipelineConfig(coverage_iou_floor=0.0),
    )
    _drive(loop, 40)
    assert ink.submitted, "piso zerado mantém a passada de antes"


def test_coverage_iou_floor_inert_without_mask():
    """Resultado sem máscara: piso inerte, submit segue normal."""
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),
        config=PipelineConfig(coverage_iou_floor=0.9),
    )
    _drive(loop, 40)
    assert ink.submitted


def _seen_then_lost():
    """Vê a folha nos 6 primeiros chamados, depois perde tudo."""

    def script(i):
        if i < 6:
            return QUAD.copy(), "paper", None
        return None, None, None

    return script


def test_drop_clears_canvas_without_persist():
    """canvas_persist desligado: drop zera clean_mask na hora."""
    loop, _det, _ink = _make_loop(
        _seen_then_lost(), config=PipelineConfig(canvas_persist=False)
    )
    _drive(loop, 8)
    loop.clean_mask = np.ones((100, 100), np.uint8) * 255
    _drive(loop, 80)
    assert loop.generation_events["drop"] >= 1
    assert loop.clean_mask is None


def test_drop_keeps_canvas_with_persist():
    """canvas_persist: a queda congela o canvas — o usuário segue
    vendo o desenho enquanto a folha não é re-adquirida."""
    loop, _det, _ink = _make_loop(_seen_then_lost())
    _drive(loop, 8)
    canvas = np.ones((100, 100), np.uint8) * 255
    loop.clean_mask = canvas.copy()
    _drive(loop, 80)
    assert loop.generation_events["drop"] >= 1
    assert loop.clean_mask is not None
    np.testing.assert_array_equal(loop.clean_mask, canvas)


def test_drop_arms_identity_recovery_for_the_reopen():
    """A queda guarda a geração: a reabertura pede preservação."""
    state = {"lost": False}

    def script(_i):
        if state["lost"]:
            return None, None, None
        return QUAD.copy(), "paper", None

    loop, _det, ink = _make_loop(script)
    _drive(loop, 8)
    state["lost"] = True
    _drive(loop, 80)
    assert loop.generation_events["drop"] >= 1
    state["lost"] = False
    _drive(loop, 30)
    assert loop.ink_session_open
    # A primeira reabertura depois da queda preserva a identidade;
    # a primeira sessão (antes da queda) não.
    assert ink.new_generation_calls[0] is False
    assert ink.new_generation_calls[-1] is True


def test_stale_grace_passes_skipped_with_skip_stale(monkeypatch):
    """Passadas sobre cantos de graça (> limiar sem confirmação
    nova) não sobem — é mesa/oclusão por construção."""
    monkeypatch.setattr(pipeline_module, "_STALE_PASS_MIN_FRAMES", 5)
    loop, _det, ink = _make_loop(_seen_then_lost())
    stale_seen = False
    for _ in range(60):
        loop.step(_packet(loop, _frame()))
        if loop.tracker._frames_since_seen > 5:
            stale_seen = True
            submits_at_stale = len(ink.submitted)
            for _ in range(6):
                loop.step(_packet(loop, _frame()))
            # graca ativa ou queda consumada: nenhuma submissao a mais
            assert len(ink.submitted) == submits_at_stale
            break
    assert stale_seen


def test_stale_passes_continue_without_skip_stale():
    """skip_stale desligado: o legado segue submetendo na graça."""
    loop, _det, ink = _make_loop(
        _seen_then_lost(), config=PipelineConfig(skip_stale=False)
    )
    submits_grew = False
    for _ in range(60):
        loop.step(_packet(loop, _frame()))
        if loop.tracker._frames_since_seen > 5 and loop.tracker.corners is not None:
            before = len(ink.submitted)
            loop.step(_packet(loop, _frame()))
            if len(ink.submitted) > before:
                submits_grew = True
            break
    assert submits_grew


def test_fresh_passes_unaffected_with_skip_stale():
    """Folha confirmada: _frames_since_seen = 0 — a opção não toca
    em nenhuma passada saudável."""
    loop, _det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    _drive(loop, 40)
    assert len(ink.submitted) > 0


def test_seg_hand_occlusion_blocks_votes_under_hand():
    """A classe mão do seg vira oclusão: a passada não enxerga a
    folha sob ela mesmo sem o masker responder."""
    hand = np.zeros((720, 1280), np.uint8)
    hand[120:280, 200:420] = 255
    loop, _det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper", {"hand_mask": hand}))
    _drive(loop, 40)
    assert loop._hand_mask_det is not None
    visibles = [v for _g, _c, v, _cov in ink.submitted if v is not None]
    assert visibles, "a oclusão semântica precisa chegar ao submit"
    vis = visibles[-1]
    assert vis.dtype == bool
    # A faixa de mão retificada fica invisível para o voto; o resto
    # da página segue visível — nem tudo abstém, nem nada abstém.
    assert not vis.all()
    assert vis.any()


def test_seg_hand_occlusion_off_keeps_visible_none():
    """seg_hand_occlusion desligado: a máscara semântica de mão não
    vira oclusão."""
    hand = np.zeros((720, 1280), np.uint8)
    hand[120:280, 200:420] = 255
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"hand_mask": hand}),
        config=PipelineConfig(seg_hand_occlusion=False),
    )
    _drive(loop, 40)
    assert loop._hand_mask_det is not None
    assert all(v is None for _g, _c, v, _cov in ink.submitted)


def _drive_clock(loop, steps, clock_state, dt=0.05):
    frame = _frame()
    for _ in range(steps):
        loop.step(_packet(loop, frame))
        clock_state["t"] += dt


def _move_scenario(*, follow_move):
    """Folha desliza sem mão: sessão aberta em QUAD, mão sai
    (freeze) e o quad confirmado salta +120 px em x."""
    clock_state = {"t": 0.0}
    moved = QUAD + np.array([120.0, 0.0], dtype=np.float32)
    state = {"move": False}

    def script(_i):
        return (moved if state["move"] else QUAD).copy(), "paper", None

    loop, _det, ink = _make_loop(script, config=PipelineConfig(follow_move=follow_move))
    loop._clock = lambda: clock_state["t"]
    _drive_clock(loop, 40, clock_state)
    assert loop.tracker.corners is not None
    # Mão sai da cena: o histórico congela na pose antiga.
    loop.clean_hand_present = False
    state["move"] = True
    _drive_clock(loop, 60, clock_state)
    return loop, ink, moved


def test_frozen_geometry_tracks_real_move():
    """Com follow_move, o quad congelado destrava num deslocamento
    real e a tinta volta a ser retificada na pose nova."""
    _loop, ink, moved = _move_scenario(follow_move=True)
    last_corners = ink.submitted[-1][1]
    assert last_corners is not None
    assert np.max(np.abs(last_corners - moved)) < 5.0


def test_frozen_geometry_stays_without_follow_move():
    """Sem follow_move, o congelamento de sempre segue: a mediana
    fica na pose antiga mesmo com o rastreador já na posição nova."""
    loop, ink, moved = _move_scenario(follow_move=False)
    assert np.max(np.linalg.norm(loop.tracker.corners - moved, axis=1)) < 40.0, (
        "o rastreador em si já acompanhou a folha"
    )
    last_corners = ink.submitted[-1][1]
    assert last_corners is not None
    assert np.max(np.abs(last_corners - QUAD)) < 5.0


def test_poll_display_feeds_the_display_layers():
    """O InkDisplaySnapshot da trabalhadora vira clean_mask,
    provisória e o canvas acumulado do painel."""
    confirmed = np.zeros((100, 100), np.uint8)
    confirmed[10:20, 10:20] = 255
    provisional = np.zeros((100, 100), np.uint8)
    provisional[30:40, 30:40] = 255
    ink = _FakeInkWorker(
        display=InkDisplaySnapshot(
            version=1, confirmed=confirmed, provisional=provisional
        )
    )
    rendered = np.full((100, 100, 3), 255, np.uint8)
    loop, _det, _ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),
        ink=ink,
    )
    loop._render_display = lambda _c, _p, out=None: rendered
    _drive(loop, 10)
    np.testing.assert_array_equal(loop.clean_mask, confirmed)
    np.testing.assert_array_equal(loop.clean_provisional_mask, provisional)
    np.testing.assert_array_equal(loop.accumulated_canvas, rendered)
    assert loop.clean_display_version == 1


def test_ink_worker_failure_returns_the_signal():
    """Erro da trabalhadora: step devolve INK_WORKER_FAILED."""
    ink = _FakeInkWorker()
    loop, _det, _ink = _make_loop(lambda _i: (QUAD.copy(), "paper", None), ink=ink)
    _drive(loop, 10)
    ink.error = RuntimeError("modelo morreu")
    assert loop.step(_packet(loop, _frame())) == pipeline_module.INK_WORKER_FAILED


def test_collect_ink_without_failure_returns_none():
    loop, _det, _ink = _make_loop(lambda _i: (None, None, None))
    assert loop.collect_ink() is None


def test_on_acquisition_fires_per_consumed_result():
    """Cada rastro de aquisição chega ao callback exatamente uma
    vez, com o evento do rastreador anexado."""
    traces = [{"result_seq": 1, "acq_seq": 7, "source": "fake"}]
    events = []
    det = _SyncDetectionWorker(lambda _i: (QUAD.copy(), "paper", None), traces=traces)
    ink = _FakeInkWorker()
    loop = pipeline_module.Pipeline(
        detection_worker=det,
        ink_worker=ink,
        clock=lambda: 0.0,
        log=lambda *_a: None,
        on_acquisition=events.append,
    )
    _drive(loop, 10)
    assert len(events) == 1
    record = events[0]
    assert record["acq_seq"] == 7 and record["source"] == "fake"
    assert record["consumed_by_tracker"] is True
    assert record["tracker_event"] is not None
    assert record["tracker_before"] is None or isinstance(
        record["tracker_before"], dict
    )


def test_hand_presence_result_feeds_the_ink_gate():
    """A amostra da trabalhadora de presença alimenta a porta da
    tinta e o estado clean_hand_present."""
    hp = _FakeHandPresenceWorker(results=[(1, True, True), (2, False, False)])
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None), hand_presence=hp
    )
    _drive(loop, 10)
    assert loop.clean_hand_present is False
    assert (True, True) in ink.presence_calls
    assert ink.presence_calls[-1] == (False, False)
    # O submit do quadro leva o carimbo de captura em ms
    # (packet.captured_at * 1000): a última detecção consumida foi
    # no quadro 6, captured_at = 0.6.
    assert hp.submitted
    assert hp.submitted[-1] == pytest.approx(600.0)


def test_hand_presence_error_degrades_to_no_signal():
    """Presença morta degrada para 'sem sinal' e a porta recebe
    None — a cadência segue normal."""
    hp = _FakeHandPresenceWorker(error=RuntimeError("landmarker"))
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None), hand_presence=hp
    )
    _drive(loop, 5)
    assert loop.hand_presence_worker is None
    assert loop.clean_hand_present is None
    assert (None,) in ink.presence_calls or (
        any(call[0] is None for call in ink.presence_calls)
    )


def test_reopen_counts_and_resets_display_version():
    """Folha nova: reopen conta, geração sobe, display zera."""
    loop, _det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    _drive(loop, 10)
    assert loop.generation_events["reopen"] >= 1
    assert loop.clean_generation == ink.generation >= 1
    assert loop.clean_display_version == 0


def test_det_hand_mask_reaches_occlusion_without_coverage():
    """coverage=False com seg_hand_occlusion=True: a máscara de mão
    do DetectionResult segue vedando o voto — são campos separados;
    a mão não depende da flag de cobertura."""
    state = {"with_mask": True}
    hand = np.zeros((720, 1280), np.uint8)
    hand[120:280, 200:420] = 255

    def script(_i):
        if state["with_mask"]:
            return QUAD.copy(), "paper", {"hand_mask": hand}
        return QUAD.copy(), "paper", {}

    loop, _det, ink = _make_loop(
        script, config=PipelineConfig(coverage=False, seg_hand_occlusion=True)
    )
    _drive(loop, 30)
    assert loop._hand_mask_det is not None
    visibles = [v for _g, _c, v, _cov in ink.submitted if v is not None]
    assert visibles, "a oclusão semântica precisa chegar ao submit"
    vis = visibles[-1]
    assert not vis.all() and vis.any()
    assert all(cov is None for _g, _c, _v, cov in ink.submitted)
    # Resultado sem máscara limpa a evidência, igual à cobertura.
    state["with_mask"] = False
    _drive(loop, 10)
    assert loop._hand_mask_det is None
    assert ink.submitted[-1][2] is None


def test_drop_without_persist_clears_canvas_and_provisional():
    """canvas_persist desligado: a queda some com o desenho da tela —
    clean_mask, camada provisória e o canvas renderizado zeram
    juntos (antes só a confirmada ia, e o canvas ficava até a
    reabertura)."""
    loop, _det, _ink = _make_loop(
        _seen_then_lost(), config=PipelineConfig(canvas_persist=False)
    )
    _drive(loop, 8)
    loop.clean_mask = np.ones((100, 100), np.uint8) * 255
    loop.clean_provisional_mask = np.ones((100, 100), np.uint8) * 128
    loop.accumulated_canvas = np.ones((100, 100, 3), np.uint8) * 200
    _drive(loop, 80)
    assert loop.generation_events["drop"] >= 1
    assert loop.clean_mask is None
    assert loop.clean_provisional_mask is None
    assert loop.accumulated_canvas is None


def test_reset_source_drops_camera_space_state_and_stale_detections():
    """Troca de camera: estado da camera anterior nao vaza para a nova."""
    loop, det, _ink = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    _drive(loop, 40)
    assert loop.tracker.corners is not None
    assert loop.display_corners is not None and loop.display_gray is not None

    det.submit(_frame(), 1.0)  # deteccao em voo na camera antiga
    loop.reset_source(since=1000.0)
    assert loop.tracker.corners is None
    assert loop.display_corners is None and loop.display_gray is None
    assert len(loop.corner_history) == 0

    small = np.zeros((480, 640, 3), dtype=np.uint8)
    loop.step(FramePacket(image=small, sequence=1, captured_at=1001.0))
    assert loop.tracker.corners is None, "resultado velho nao pode reviver o quad"
    loop.step(FramePacket(image=small, sequence=2, captured_at=1002.0))
    assert loop.tracker.corners is not None, "deteccao da camera nova vale"


def test_reset_source_restarts_the_ink_as_a_new_page():
    """Troca de camera: a tinta nao e preservada, como ao virar a folha."""
    loop, _det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    _drive(loop, 40)
    assert loop.ink_session_open is True
    opened = len(ink.new_generation_calls)

    loop.reset_source(since=1000.0)
    assert loop.ink_session_open is False
    assert loop.clean_recovery_pending is False
    assert loop.clean_mask is None and loop.accumulated_canvas is None

    frame = _frame()
    for i in range(1, 12):
        loop.step(FramePacket(image=frame, sequence=i, captured_at=1000.0 + i * 0.1))
    assert len(ink.new_generation_calls) == opened + 1
    assert ink.new_generation_calls[-1] is False, "pagina nova, sem identidade"


def test_reset_source_blocks_unread_snapshots_of_the_old_generation():
    """Camera nova sem folha: a tinta da camera antiga nao volta nem salva."""
    confirmed = np.zeros((100, 100), np.uint8)
    confirmed[10:20, 10:20] = 255
    snapshot = InkDisplaySnapshot(
        version=7, confirmed=confirmed, provisional=np.zeros((100, 100), np.uint8)
    )
    ink = _FakeInkWorker()
    scene = {"sheet": True}
    loop, _det, _ink = _make_loop(
        lambda _i: (
            (QUAD.copy(), "paper", None) if scene["sheet"] else (None, "none", None)
        ),
        ink=ink,
    )
    loop._render_display = lambda _c, _p, out=None: np.full((100, 100, 3), 9, np.uint8)
    _drive(loop, 20)
    assert loop.ink_session_open is True
    old_generation = ink.generation
    ink._display = snapshot
    ink.display_generation = old_generation  # snapshot ainda nao lido da camera antiga

    scene["sheet"] = False  # a camera nova nao ve folha
    loop.reset_source(since=1000.0)
    frame = _frame()
    for i in range(1, 8):
        loop.step(FramePacket(image=frame, sequence=i, captured_at=1000.0 + i * 0.1))
        loop.collect_ink()
    assert loop.accumulated_canvas is None
    assert loop.clean_mask is None and loop.clean_provisional_mask is None
    assert loop.ink_session_open is False
    assert loop.clean_generation != old_generation
