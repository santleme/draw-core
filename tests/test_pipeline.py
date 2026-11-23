"""O laço de orquestração (draw.pipeline) sobre workers falsos.

Cobre os caminhos que o SheetInkLoop legado exercitava com as
flags do preset, agora campos do PipelineConfig: cobertura de
folha no voto, persistência do canvas na queda do rastreador,
salto de passadas sobre cantos retidos, destrave da geometria
congelada sem mão, oclusão semântica da mão e os desfechos
reopen/drop/erro.
"""

import json

import cv2
import numpy as np
import pytest

from draw import pipeline as pipeline_module
from draw.contracts import (
    DetectionResult,
    FramePacket,
    HandOcclusion,
    HandPresenceResult,
    HandSample,
    InkBaselineState,
    InkDisplaySnapshot,
)
from draw.ink import registration as registration_mod
from draw.ink.config import InkConfig
from draw.ink.worker import PageCleanWorker
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
        self.pages = []
        self._display = display

    def new_generation(self, preserve_identity=False):
        self.generation += 1
        self.new_generation_calls.append(preserve_identity)
        return self.generation

    def submit(self, page):
        self.pages.append(page)
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

    def submit(self, frame, captured_at, *, source_id):
        self.submitted.append((captured_at, source_id))

    def poll(self, after_seq):
        if not self._results:
            return None
        if self._results[0].sequence <= after_seq:
            return None
        return self._results.pop(0)


# Os testes do laco isolam um mecanismo por vez e dependem da cadencia de
# deteccao fixa (um detect por quadro) e do warp cru: ganho e Detect-Wake,
# que sao padrao de producao, ficam desligados aqui salvo quando o teste os
# pede explicitamente.
_BASE = {"ink_auto_gain": False, "detect_wake": False}


def _make_loop(
    script,
    *,
    config=None,
    traces=None,
    ink=None,
    hand_presence=None,
    hand_masker=None,
    clock=None,
):
    det = _SyncDetectionWorker(script, traces=traces)
    ink = _FakeInkWorker() if ink is None else ink
    if config is None:
        config = PipelineConfig(**_BASE)
    loop = pipeline_module.Pipeline(
        detection_worker=det,
        ink_worker=ink,
        hand_presence_worker=hand_presence,
        hand_masker=hand_masker,
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
    assert config.ink_auto_gain is True
    assert config.detect_wake is True
    assert config.ink == InkConfig.production()


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
    # A máscara sobe na grade da CÂMERA (720x1280, issue #100):
    # x>=300 é mesa — nunca pode votar — e o miolo coberto segue
    # aberto (a erosão de borda desconta ~2 px).
    assert gated[-1].shape == (720, 1280)
    assert not gated[-1][:, 320:].any()
    assert gated[-1][10:700, 50:280].all()


def test_paper_mean_bgr_hits_integer_boundary():
    """cv2.mean pode devolver um inteiro exato 1 ulp abaixo (recíproco
    arredondado); o cast antigo truncaria um nível a menos. A soma é
    recuperada exata e dividida direto."""
    from draw.pipeline import _paper_mean_bgr

    image = np.full((1200, 850, 3), 80, np.uint8)
    image[:, :750] = (235, 235, 235)
    coverage = np.zeros((1200, 850), bool)
    coverage[:, :750] = True  # 900.000 px — caso que desvia no cv2
    assert (_paper_mean_bgr(image, coverage) == [235, 235, 235]).all()


def test_paper_mean_bgr_still_truncates_fractional_mean():
    """Média fracionária segue truncando como o astype(uint8) antigo."""
    from draw.pipeline import _paper_mean_bgr

    image = np.zeros((4, 4, 3), np.uint8)
    image[0, 0] = (200, 200, 200)
    image[0, 1] = (201, 201, 201)  # média 200,5 -> 200
    coverage = np.zeros((4, 4), bool)
    coverage[0, :2] = True
    assert (_paper_mean_bgr(image, coverage) == [200, 200, 200]).all()


def test_coverage_fill_uses_the_paper_mean_bgr():
    """O fora-da-folha recebe a cor média do próprio papel por canal.

    Pinca o contrato do preenchimento (a aresta folha->preto vira a
    cor do papel, não um branco fixo): cada pixel fora da cobertura
    sai igual à média BGR dos pixels cobertos do mesmo warp.
    """
    frame = np.full((720, 1280, 3), 220, np.uint8)  # mesa clara
    # Papel em dois tons: com o quad trapezoidal abaixo a media da
    # camera difere da media do retificado — o teste distingue a
    # media no dominio da camera (issue #100) de uma pos-warp.
    frame[100:250, 100:350] = (60, 100, 200)
    frame[250:400, 100:350] = (200, 80, 30)
    coverage = _mask_block(100, 400, 100, 350)
    quad = np.array(
        [[100.0, 100.0], [500.0, 140.0], [480.0, 400.0], [120.0, 380.0]],
        dtype=np.float32,
    )
    loop, _det, ink = _make_loop(
        lambda _i: (quad.copy(), "paper", {"coverage_mask": coverage})
    )
    _drive(loop, 40, frame=frame)
    page = ink.pages[-1]
    # Issue #100: a cobertura sobe em grade de câmera — retifico pelo
    # quad da passada para comparar na grade do warp.
    assert page.coverage is not None and page.coverage.any()
    coverage_page = (
        cv2.warpPerspective(
            page.coverage.astype(np.uint8),
            registration_mod._corners_page_matrix(
                page.corners, page.image.shape[1], page.image.shape[0]
            ),
            (page.image.shape[1], page.image.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        > 0
    )
    paper_cam = (
        cv2.erode((coverage > 127).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ) & pipeline_module._quad_camera_mask(quad, frame.shape)
    expected = pipeline_module._paper_mean_bgr(frame, paper_cam)
    # O anel fronteiriço mistura fill e papel no warp LINEAR da
    # imagem (a máscara comparada é NEAREST) — a comparação exata
    # vale no miolo claramente fora da folha; igualdade por pixel,
    # porque a média esconderia uma faixa estreita de cor errada
    # (review do PR #118).
    inside = page.image[coverage_page]
    outside = page.image[
        cv2.dilate(coverage_page.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    ]
    assert outside.size, "o teste precisa de região fora da folha"
    assert (outside == expected).all()
    # Não-vacuo: papel sombreado + quad trapezoidal fazem a média
    # pós-warp diferir da de câmera — um fill no domínio da página
    # sairia ~6 níveis mais claro no azul e no vermelho.
    assert np.abs(inside.mean(axis=0) - expected.astype(float)).max() > 3


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


def test_coverage_iou_floor_rejections_are_recorded():
    """Issue #105: todo veto do piso de IoU é registrado — total e
    anel dos últimos valores — para a recalibração do limiar, que
    não tem denominador natural de resolução."""
    phantom = _mask_block(50, 400, 900, 1100)  # blob à direita, quad em x<500
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": phantom})
    )
    _drive(loop, 40)
    assert ink.submitted == []  # o veto seguiu ativo em todas as passadas
    stats = loop.stats()
    assert stats["coverage_iou_rejects_total"] >= 1
    recent = stats["coverage_iou_reject_recent"]
    assert recent
    assert stats["coverage_iou_rejects_total"] >= len(recent)
    floor = loop._config.coverage_iou_floor
    assert all(0.0 <= v < floor for v in recent)
    # Nenhum warp passou: o lado aceito da distribuicao fica vazio.
    assert stats["coverage_iou_accepts_total"] == 0
    assert stats["coverage_iou_accept_recent"] == []
    # O registrado é o IoU medido na hora do veto: a cena é estática,
    # recomputar pelo caminho do pipeline devolve o mesmo valor.
    expected = loop._quad_coverage_iou(QUAD)
    assert all(v == pytest.approx(expected, abs=1e-4) for v in recent)
    # O dict sobe ao tel_fields da SessionTelemetry: precisa
    # serializar e voltar intacto.
    assert json.loads(json.dumps(stats)) == stats

    # Fantasma com sobreposição parcial: o anel guarda o valor
    # medido, não um genérico — é o dado da recalibração.
    partial = _mask_block(100, 400, 400, 1100)  # cobre 1/4 do QUAD + mesa
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": partial})
    )
    _drive(loop, 40)
    assert ink.submitted == []
    recent = loop.stats()["coverage_iou_reject_recent"]
    expected = loop._quad_coverage_iou(QUAD)
    assert 0.0 < expected < floor
    assert all(v == pytest.approx(expected, abs=1e-4) for v in recent)

    # Folha alinhada (IoU alto): nenhum veto, anel de rejeitados
    # vazio — e o lado ACEITO registrado: a recalibracao do piso
    # precisa da distribuicao da folha, nao so da do fantasma.
    aligned = _mask_block(100, 400, 100, 500)
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": aligned})
    )
    _drive(loop, 40)
    assert ink.submitted, "a folha real segue submetendo"
    stats = loop.stats()
    assert stats["coverage_iou_rejects_total"] == 0
    assert stats["coverage_iou_reject_recent"] == []
    assert stats["coverage_iou_accepts_total"] >= 1
    accepted = stats["coverage_iou_accept_recent"]
    assert accepted
    assert stats["coverage_iou_accepts_total"] >= len(accepted)
    assert all(v >= floor for v in accepted)
    assert all(
        v == pytest.approx(loop._quad_coverage_iou(QUAD), abs=1e-4) for v in accepted
    )
    assert json.loads(json.dumps(stats)) == stats


def test_paisagem_submete_pagina_deitada_a_tinta():
    """--orientation paisagem: o quad largo chega à tinta numa grade
    deitada (1200x850), sem espremer o conteúdo na grade em pé."""
    from draw import config as config_module

    config_module.apply_format("a4", "paisagem")
    try:
        loop, _det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
        _drive(loop, 40)
        page = ink.pages[-1].image
        assert page.shape[:2] == (850, 1200)
    finally:
        config_module.apply_format("a4")


def test_ink_auto_gain_brightens_dark_page_before_ink():
    """ink_auto_gain: página escura (teto de exposição a 30 fps) chega
    clara à tinta; sem o campo, o warp cru segue intacto."""
    frame = np.full((720, 1280, 3), 80, np.uint8)  # papel subexposto
    frame[200:300, 200:400] = 25  # traço

    loop, _det, ink_raw = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    _drive(loop, 40, frame=frame)
    raw = ink_raw.pages[-1].image
    assert loop.last_ink_gain == 1.0
    assert abs(float(np.percentile(raw, 90)) - 80) < 2

    loop, _det, ink_gain = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),
        config=PipelineConfig(ink_auto_gain=True),
    )
    _drive(loop, 40, frame=frame)
    lit = ink_gain.pages[-1].image
    assert loop.last_ink_gain == pytest.approx(2.5)
    assert abs(float(np.percentile(lit, 90)) - 200) < 3
    # Linear: o traço segue mais escuro que o papel.
    assert float(lit.min()) < 0.5 * float(np.percentile(lit, 90))


def test_ink_auto_gain_smoothing_reaches_the_page_clean_input():
    """Devin Review PR #140: o replay do issue-133 derivava a prob do
    escalar de ganho; aqui a página efetivamente entregue ao page_clean
    prova que a LUT aplicada é a saída do suavizador, não o alvo cru. Uma
    faixa clara local muda o p90 em ±2,5%, sem deslocar a mediana do papel.
    A banda morta segura a LUT e mantém a sombra estável."""
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),
        config=PipelineConfig(ink_auto_gain=True),
    )

    def scene(glint_level):
        frame = np.full((720, 1280, 3), 225, np.uint8)
        frame[100:400, 100:500] = 80
        frame[100:400, 350:500] = glint_level
        frame[150:250, 200:300] = 50  # sombra local não altera o p90 do papel
        return frame

    _drive(loop, 40, frame=scene(80))
    ink.pages.clear()
    shadow_samples = []
    applied_gains = []
    for glint_level in (80, 78, 82, 78, 80, 82, 80):
        loop.step(_packet(loop, scene(glint_level)))
        page = ink.pages[-1].image
        # page[270:295, 300:350] cai sobre a sombra (y 150-250, x 200-300
        # do frame) na grade retificada (1200, 850).
        shadow_samples.append(float(page[270:295, 300:350].mean()))
        # last_ink_gain é o ganho declarado pelo passo que produziu
        # pages[-1]: a cadência pode pular quadros, mas o par
        # (página, ganho) é sempre da mesma passada.
        applied_gains.append(loop.last_ink_gain)

    # A sombra 50 sobe a ~50 x ganho declarado, pixel a pixel: sem o
    # apply_gain a página leria ~50; com o alvo cru leria ~128/122.
    for sample, gain in zip(shadow_samples, applied_gains, strict=True):
        assert sample == pytest.approx(50.0 * gain, abs=1.5)
    assert max(applied_gains) - min(applied_gains) <= 0.01
    assert max(shadow_samples) - min(shadow_samples) <= 1.0


def test_ink_auto_gain_deadband_releases_when_gain_clips_faint_stroke():
    """Devin Review PR #140: uma faixa clara local muda o p90 de 194 para
    199 sem mover a mediana; o alvo novo (1,005) cai dentro da banda morta,
    mas o ganho retido (1,031) clipa a faixa clara junto do traço fraco.
    A banda cede e a página recupera contraste nas passadas seguintes."""
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),
        config=PipelineConfig(ink_auto_gain=True),
    )

    def scene(glint_level):
        frame = np.full((720, 1280, 3), 225, np.uint8)
        frame[100:400, 100:500] = 194
        frame[100:400, 350:500] = glint_level  # p90 muda; a mediana fica em 194
        frame[150:200, 230:280] = 252  # faixa: clipa com 1,031, não com 1,005
        frame[170:175, 240:270] = 248  # traço fraco dentro da faixa
        return frame

    _drive(loop, 40, frame=scene(194))
    assert loop.last_ink_gain == pytest.approx(200 / 194, rel=0.02)

    seen = len(ink.pages)
    patch_means, stroke_means = [], []
    for _ in range(30):
        loop.step(_packet(loop, scene(199)))
        while len(ink.pages) > seen:
            page = ink.pages[seen].image
            seen += 1
            # Na grade retificada (1200, 850): faixa py 200-400/px 276-382,
            # traço py 280-300/px 298-361.
            patch_means.append(float(page[210:390, 285:375].mean()))
            stroke_means.append(float(page[283:297, 305:355].mean()))

    assert len(patch_means) >= 3  # várias submissões no frame claro
    for patch, stroke in zip(patch_means[-3:], stroke_means[-3:], strict=True):
        assert patch < 254.5  # a faixa clara não satura mais
        assert patch - stroke >= 3  # e o traço fraco segue distinto dela


def test_ink_auto_gain_ignores_bright_desk_inside_the_quad():
    """Review #52: o quad abrange papel escuro (86%) e mesa clara (14%);
    o ganho é estimado só sobre a folha (cobertura), então o papel é
    clareado em vez de o p90 da mesa cancelar o ganho."""
    frame = np.full((720, 1280, 3), 225, np.uint8)  # mesa clara
    frame[100:400, 100:442] = 80  # papel: 342 de 400 colunas do QUAD
    coverage = _mask_block(100, 400, 100, 442)
    script = lambda _i: (QUAD.copy(), "paper", {"coverage_mask": coverage})  # noqa: E731

    loop, _det, _ink = _make_loop(script, config=PipelineConfig(ink_auto_gain=False))
    _drive(loop, 40, frame=frame)
    assert loop.last_ink_gain == 1.0

    loop, _det, ink = _make_loop(script, config=PipelineConfig(ink_auto_gain=True))
    _drive(loop, 40, frame=frame)
    assert loop.last_ink_gain == pytest.approx(2.5, rel=0.1)
    page = ink.pages[-1].image
    assert float(np.percentile(page[page.mean(axis=2) < 255], 50)) > 150


def test_ink_auto_gain_ignores_a_bright_hand_over_the_sheet():
    """Review #52: a cobertura do seg inclui a mão; uma mão clara sobre
    metade do quad cancelaria o ganho do papel escuro. O ganho usa só a
    folha visível (cobertura & sem mão)."""
    frame = np.full((720, 1280, 3), 80, np.uint8)  # papel escuro
    frame[100:400, 100:300] = 235  # mão clara sobre metade do quad
    coverage = _mask_block(100, 400, 100, 500)  # inclui a mão
    hand = _mask_block(100, 400, 100, 300)
    script = lambda _i: (  # noqa: E731
        QUAD.copy(),
        "paper",
        {"coverage_mask": coverage, "hand_mask": hand},
    )

    loop, _det, _ink = _make_loop(script, config=PipelineConfig(ink_auto_gain=True))
    _drive(loop, 40, frame=frame)
    assert loop._hand_mask_det is not None
    assert loop.last_ink_gain == pytest.approx(2.5, rel=0.1)  # papel, não a mão


def test_ink_auto_gain_excludes_the_hand_without_sheet_coverage():
    """Review #52: na detecção clássica não há máscara de cobertura; a mão
    (landmarker/seg) ainda precisa sair da estimativa do ganho."""
    frame = np.full((720, 1280, 3), 80, np.uint8)
    frame[100:400, 100:300] = 235  # mão clara sobre metade do quad
    hand = _mask_block(100, 400, 100, 300)
    loop, _det, _ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"hand_mask": hand}),
        config=PipelineConfig(ink_auto_gain=True),
    )
    _drive(loop, 40, frame=frame)
    assert loop._hand_mask_det is not None
    assert loop.last_ink_gain == pytest.approx(2.5, rel=0.1)


def test_ink_auto_gain_excludes_the_hand_from_the_classic_hand_masker():
    """Review #52: sem modelo neural, a detecção clássica só tem o
    HandMasker (landmarker) para saber onde está a mão. Sem cobertura e
    sem máscara semântica, a mão clara ainda sai da estimativa do ganho."""
    frame = np.full((720, 1280, 3), 80, np.uint8)
    frame[100:400, 100:300] = 235  # mão clara sobre metade do quad

    class _Masker:
        def mask(self, image, timestamp_ms=None):
            camera = np.zeros(image.shape[:2], np.uint8)
            camera[100:400, 100:300] = 255
            return camera

    loop, _det, _ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),  # clássica: sem extras
        config=PipelineConfig(ink_auto_gain=True),
        hand_masker=_Masker(),
    )
    _drive(loop, 40, frame=frame)
    assert loop._hand_mask_det is None  # nenhuma máscara semântica
    assert loop.last_ink_gain == pytest.approx(2.5, rel=0.1)

    loop, _det, _ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),
        config=PipelineConfig(ink_auto_gain=True),
    )  # sem o masker, a mão clara cancela o ganho: o teste discrimina
    _drive(loop, 40, frame=frame)
    assert loop.last_ink_gain == 1.0


def test_ink_auto_gain_holds_last_gain_while_hand_covers_the_sheet():
    """Issue #133: a mão cobrindo o quadro inteiro zera a máscara do
    ganho (`visible` todo False) — e paper_gain leria mesa+oclusor como
    "imagem inteira": um quadro claro derrubaria o ganho a 1,0 (flash
    escuro + reconvergência ao sair a mão). Máscara vazia não mede: a
    estimativa congela e o último ganho segue aplicado até o papel
    reaparecer."""

    class _Masker:
        full = False

        def mask(self, image, timestamp_ms=None):
            camera = np.zeros(image.shape[:2], np.uint8)
            if self.full:
                camera[:] = 255
            return camera

    def scene(level, hand=False):
        frame = np.full((720, 1280, 3), 235, np.uint8)  # mesa clara
        frame[100:400, 100:500] = 100 if hand else level  # oclusor ou papel
        return frame

    masker = _Masker()
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),
        config=PipelineConfig(ink_auto_gain=True),
        hand_masker=masker,
    )
    _drive(loop, 40, frame=scene(80))  # papel escuro: ganho estabelecido
    assert loop.last_ink_gain == pytest.approx(2.5, rel=0.05)

    masker.full = True
    seen = len(ink.pages)
    for _ in range(30):
        loop.step(_packet(loop, scene(80, hand=True)))
    occluded = ink.pages[seen:]
    assert occluded, "a passada de tinta segue submetendo sob a mão"
    occluded_visibles = [v for _g, _c, v, _cov in ink.submitted[seen:]]
    assert all(v is not None and not v.any() for v in occluded_visibles)
    # A estimativa não rodou: o ganho não caiu a 1,0 nem andou.
    assert loop.last_ink_gain == pytest.approx(2.5, rel=0.05)
    # E segue aplicado na página: o oclusor 100 sobe ~100 x 2,5 (sem
    # clipar) — com o bug a página subiria crua a ~100.
    for page in occluded:
        assert float(page.image[400:800, 300:500].mean()) == pytest.approx(250.0, abs=3)

    masker.full = False
    for _ in range(20):
        loop.step(_packet(loop, scene(160)))  # papel mais claro
    # Com a folha de volta a estimativa retoma: o ganho segue o alvo
    # novo em vez de ficar preso no valor congelado.
    assert loop.last_ink_gain == pytest.approx(200 / 160, rel=0.05)


def test_ink_divnorm_submits_the_raw_warp_to_the_worker():
    """ink_divnorm=True: o retificado sobe cru à tinta — a
    normalização rola dentro da passada da trabalhadora (issue
    #101), que tem folga de cadência para o blur de página
    inteira. A imagem submetida é a mesma do caminho sem o toggle.
    """
    grad = np.linspace(80, 220, 1280, dtype=np.float32)
    frame = np.repeat(grad[None, :], 720, axis=0)[..., None]
    frame = np.repeat(frame, 3, axis=2).astype(np.uint8)

    loop, _det, ink_raw = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    _drive(loop, 40, frame=frame)
    raw_img = ink_raw.pages[-1].image
    assert float(raw_img.std()) > 10.0  # gradiente cru preservado

    loop, _det, ink_dn = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),
        config=PipelineConfig(ink_divnorm=True),
    )
    _drive(loop, 40, frame=frame)
    np.testing.assert_array_equal(ink_dn.pages[-1].image, raw_img)


def test_ink_divnorm_skips_the_fill_and_the_gain():
    """Com ink_divnorm o laço não preenche o fora-da-folha nem ganha:
    as duas coisas acontecem na trabalhadora sobre o warp cru (o
    preenchimento faz parte da paridade com o treino, review #48; o
    ganho clipava o papel claro antes da normalização, review #52)."""
    frame = np.full((720, 1280, 3), 220, np.uint8)  # mesa clara
    frame[100:400, 100:350] = 100  # papel (mais escuro que a mesa)
    coverage = np.zeros((720, 1280), np.uint8)
    coverage[100:400, 100:350] = 255
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": coverage}),
        config=PipelineConfig(ink_divnorm=True, ink_auto_gain=True),
    )
    _drive(loop, 40, frame=frame)
    assert loop.last_ink_gain == 1.0
    page = ink.pages[-1]
    assert page.coverage is not None and page.coverage.any()
    # Fora da folha fica o warp cru (mesa 220 dentro do quad / preto
    # fora dele) — sem o preenchimento da média do papel no laço.
    # (O quad 100-500 mapeia para a página 0-850; a mesa cobre
    # x>350 -> página x>531.)
    assert float(page.image[100:1100, 600:800].mean()) > 200.0
    # E dentro da cobertura vai o pixel cru da cena (papel 100), não
    # o ganho nem o normalizado. A cobertura sobe na grade da
    # câmera (issue #100) — projeta à grade da página para indexar.
    paper = pipeline_module.paper.warp_sheet(page.coverage.astype(np.uint8), QUAD) > 0
    assert float(page.image[paper].mean()) < 110.0


def test_ink_divnorm_skips_the_gain_so_bright_paper_is_not_clipped():
    """Review #48: com os dois ligados, o ganho clipava em 255 o papel
    claro de uma folha com luz desigual, e o traço fraco sobre ele sumia
    antes da normalização. Como a divisão por I/blur(I) já anula a
    escala, o divnorm dispensa o ganho: o warp sobe cru e a
    normalização que salva o traço rola na trabalhadora
    (tests/test_ink_divnorm.py)."""
    frame = np.full((720, 1280, 3), 70, np.uint8)  # papel escuro (maioria)
    frame[100:400, 440:464] = 200  # faixa bem iluminada (6%: fora do p90)
    frame[200:300, 450:454] = 175  # traço fraco sobre a parte clara

    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),
        config=PipelineConfig(ink_divnorm=True, ink_auto_gain=True),
    )
    _drive(loop, 40, frame=frame)
    assert loop.last_ink_gain == 1.0
    page = ink.pages[-1]
    # Sem o ganho no laço a faixa clara sobe ~200 (não clipa a 255)
    # e o traço de ~175 segue distinguível no warp cru.
    bright = page.image.mean(axis=2)
    assert 185.0 < float(bright.max()) < 235.0


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
        _seen_then_lost(), config=PipelineConfig(skip_stale=False, **_BASE)
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


def _drive_clock(loop, steps, clock_state, dt=0.05, frame=None):
    if frame is None:
        frame = _frame()
    for _ in range(steps):
        loop.step(_packet(loop, frame))
        clock_state["t"] += dt


def _move_scenario(*, frame=None, quad=None, move=(120.0, 0.0), move_steps=60):
    """Folha desliza sem mão: sessão aberta em `quad`, mão sai
    (freeze) e o quad confirmado desliza `move` pixels. `frame`
    define a resolução da captura (padrão 720x1280, a mesa de
    referência)."""
    clock_state = {"t": 0.0}
    quad = QUAD if quad is None else np.asarray(quad, dtype=np.float32)
    moved = quad + np.asarray(move, dtype=np.float32)
    state = {"move": False}
    if frame is None:
        frame = _frame()

    def script(_i):
        return (moved if state["move"] else quad).copy(), "paper", None

    loop, _det, ink = _make_loop(script, config=PipelineConfig())
    loop._clock = lambda: clock_state["t"]
    _drive_clock(loop, 40, clock_state, frame=frame)
    assert loop.tracker.corners is not None
    # Mão sai da cena: o histórico congela na pose antiga.
    loop.clean_hand_present = False
    state["move"] = True
    _drive_clock(loop, move_steps, clock_state, frame=frame)
    return loop, ink, moved


def test_frozen_geometry_tracks_real_move():
    """O quad congelado destrava num deslocamento real da folha e a
    tinta volta a ser retificada na pose nova."""
    _loop, ink, moved = _move_scenario()
    last_corners = ink.submitted[-1][1]
    assert last_corners is not None
    assert np.max(np.abs(last_corners - moved)) < 5.0


def test_frozen_geometry_unlock_scales_with_frame_resolution():
    """Issue #105: o destrave da geometria congelada escala com a
    diagonal do quadro. O MESMO deslize de 60 px destrava o
    histórico a 720p (escala 1.0 → limiar efetivo 25 px) mas NÃO a
    4K (escala 3.0 → limiar 75 px): em 4K o rastreador acompanha a
    folha (60 px < near-limit escalado de 120 px, confirma por
    mistura sem persistência de salto) e mesmo assim o histórico
    congelado não reparte — a tinta segue retificada na pose velha.
    """
    # 720p (mesa de referência): 60 px > 25 px efetivo → destrava.
    _loop, ink_720, moved_720 = _move_scenario(quad=QUAD, move=(60.0, 0.0))
    last = ink_720.submitted[-1][1]
    assert np.max(np.abs(last - moved_720)) < 5.0

    # 4K (escala 3.0): o mesmo deslize de 60 px fica abaixo do
    # limiar efetivo de 75 px → o histórico segue congelado.
    quad_4k = QUAD * 3.0
    frame_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
    loop_4k, ink_4k, moved_4k = _move_scenario(
        frame=frame_4k,
        quad=quad_4k,
        move=(60.0, 0.0),
        move_steps=90,
    )
    # O rastreador em si acompanhou a folha — é o limiar escalado,
    # não o rastreador, que segura o histórico na pose velha.
    assert np.max(np.abs(loop_4k.tracker.corners - moved_4k)) < 15.0
    last = ink_4k.submitted[-1][1]
    assert np.max(np.abs(last - quad_4k)) < 5.0


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
    hp = _FakeHandPresenceWorker(
        results=[
            HandPresenceResult(1, 0.0, 0, HandSample(True, True)),
            HandPresenceResult(2, 0.1, 0, HandSample(False, False)),
        ]
    )
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None), hand_presence=hp
    )
    _drive(loop, 10)
    assert loop.clean_hand_present is False
    assert (True, True) in ink.presence_calls
    assert ink.presence_calls[-1] == (False, False)
    assert hp.submitted[-1] == (pytest.approx(0.6), 0)


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
        script,
        config=PipelineConfig(
            coverage=False, seg_hand_occlusion=True, detect_wake=False
        ),
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


def test_reset_source_resets_the_applied_ink_gain():
    """Devin Review PR #140: o reset de fonte/geração zerava o
    suavizador mas não `last_ink_gain` — com a folha toda coberta a
    estimativa não roda e o frame aplicaria o ganho velho (folha clara
    entraria clipada a 255). Os dois estados zera juntos."""
    loop, _det, _ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None),
        config=PipelineConfig(ink_auto_gain=True),
    )
    frame = np.full((720, 1280, 3), 80, np.uint8)
    _drive(loop, 40, frame=frame)
    assert loop.last_ink_gain == pytest.approx(2.5, rel=0.1)

    loop.reset_source(since=1000.0)
    assert loop.last_ink_gain == 1.0


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


def test_reset_source_clears_the_hand_masker_retention():
    """Marcos retidos da camera anterior nao valem na nova (mesmo tamanho)."""

    class _Masker:
        resets = 0

        def mask(self, frame, timestamp_ms=None):
            return np.zeros(frame.shape[:2], np.uint8)

        def reset_retention(self):
            self.resets += 1

    masker = _Masker()
    loop, _det, _ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None), hand_masker=masker
    )
    loop.reset_source(since=1000.0)
    assert masker.resets == 1


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


def test_ink_page_carries_the_capture_timestamp_of_its_frame():
    """InkDelta.captured_at nasce aqui: o warp entregue à tinta leva o
    carimbo monotônico do quadro de câmera que o originou."""
    loop, _det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    _drive(loop, 40)
    assert ink.pages, "a tinta recebeu paginas"
    stamps = [page.captured_at for page in ink.pages]
    assert all(s is not None for s in stamps)
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)
    # carimbos de _packet: frame_number * 0.1
    assert all(abs(s * 10 - round(s * 10)) < 1e-6 for s in stamps)


def test_reset_source_clears_presence_and_rejects_in_flight_sample():
    hp = _FakeHandPresenceWorker(
        results=[
            HandPresenceResult(1, 0.0, 0, HandSample(True, True)),
            HandPresenceResult(2, 1.0, 0, HandSample(False, False)),
            HandPresenceResult(3, 0.9, 1, HandSample(False, False)),
            HandPresenceResult(4, 1.0, 1, HandSample(False, False)),
        ]
    )
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None), hand_presence=hp
    )
    _drive(loop, 1)
    assert loop.clean_hand_present is True

    loop.reset_source(since=1.0)
    assert loop.clean_hand_present is None
    frame = _frame()
    for sequence in (1, 2):
        loop.step(FramePacket(frame, 1.0 + sequence * 0.1, sequence))
        assert loop.clean_hand_present is None
        assert ink.presence_calls == [(True, True), (None, False)]

    loop.step(FramePacket(frame, 1.3, 3))
    assert loop.clean_hand_present is False
    assert loop._hand_exit_at == 1.0
    assert ink.presence_calls[-1] == (False, False)
    assert hp.submitted[-1][1] == 1


def test_hand_exit_uses_sample_time_to_keep_newer_semantic_mask():
    hp = _FakeHandPresenceWorker(
        results=[HandPresenceResult(1, 0.2, 0, HandSample(False, False))]
    )
    loop, det, _ink = _make_loop(
        lambda _i: (None, None, None),
        config=PipelineConfig(),
        hand_presence=hp,
    )
    frame = _frame()
    loop.step(FramePacket(frame, 0.6, 0))
    assert loop.clean_hand_present is False
    hand = np.zeros(frame.shape[:2], np.uint8)
    hand[150:250, 200:300] = 255
    det._result = DetectionResult(
        sequence=det._seq + 1,
        corners=QUAD,
        verdict="paper",
        captured_at=0.3,
        detect_ms=1.0,
        hand_mask=hand,
    )
    loop.step(FramePacket(frame, 0.7, 1))
    assert loop.last_hand_mask is not None
    assert loop.last_hand_mask.any()


@pytest.mark.parametrize("late_heartbeat", [False, True])
def test_settled_hand_mask_expires_if_heartbeat_stalls(monkeypatch, late_heartbeat):
    frame = _frame()
    hand = np.zeros(frame.shape[:2], np.uint8)
    hand[150:250, 200:300] = 255

    class _CurrentMasker:
        available = False

        def mask(self, frame, timestamp_ms=None):
            if not self.available:
                return None
            current = np.zeros(frame.shape[:2], np.uint8)
            current[300:400, 600:700] = 255
            return current

    masker = _CurrentMasker()
    loop, det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"hand_mask": hand}),
        config=PipelineConfig(detect_wake=True, detect_wake_interval_s=1.0),
        hand_masker=masker,
    )
    while not loop._detect_wake.settled and loop.frame_number < 30:
        loop.step(_packet(loop, frame))
    assert loop._detect_wake.settled
    assert loop.last_hand_mask is not None
    captured_at = det._result.captured_at
    monkeypatch.setattr(det, "submit", lambda *_args: None)

    loop.step(FramePacket(frame, captured_at + 0.9, loop.frame_number))
    assert loop.last_hand_mask is not None
    assert ink.submitted[-1][2] is not None
    loop.step(FramePacket(frame, captured_at + 1.0, loop.frame_number))
    assert loop.last_hand_mask is not None
    submits = len(ink.submitted)
    if late_heartbeat:
        det._result = DetectionResult(
            sequence=det._result.sequence + 1,
            corners=QUAD,
            verdict="paper",
            captured_at=captured_at + 0.01,
            detect_ms=1.0,
            hand_mask=hand,
        )
    loop.step(FramePacket(frame, captured_at + 1.1, loop.frame_number))
    assert loop._detect_wake.settled
    assert loop.last_hand_mask is None
    assert len(ink.submitted) == submits

    masker.available = True
    loop.step(FramePacket(frame, captured_at + 1.3, loop.frame_number))
    assert len(ink.submitted) == submits + 1
    assert not ink.submitted[-1][2][300:400, 600:700].any()


def test_hand_mask_with_sheet_marks_cornerless_results():
    """A máscara de mão acompanha qualquer desfecho neural — veto e
    "none" incluídos (PR #121). Quem a consome como evidência de
    "mão sobre a folha" precisa saber se ela veio de detecção com
    cantos: a flag `last_hand_mask_with_sheet` distingue os dois
    (review do PR #123)."""
    loop, _det, _ink = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    mask = np.zeros((480, 640), np.uint8)
    loop._update_hand_mask(
        DetectionResult(
            sequence=1,
            corners=QUAD.copy(),
            verdict="paper",
            captured_at=1.0,
            detect_ms=1.0,
            hand_mask=mask,
        ),
        1.5,
    )
    assert loop.last_hand_mask is not None
    assert loop.last_hand_mask_with_sheet
    loop._update_hand_mask(
        DetectionResult(
            sequence=2,
            corners=None,
            verdict=None,
            captured_at=2.0,
            detect_ms=1.0,
            hand_mask=mask,
        ),
        2.5,
    )
    # O veto ainda publica a evidência de oclusão — mas sem cantos
    # ela não vale como "mão sobre a folha" para a auto-rotação.
    assert loop.last_hand_mask is not None
    assert not loop.last_hand_mask_with_sheet


def test_hand_mask_with_sheet_drops_during_tracker_grace_after_removal():
    """A graça do tracker segura `sheet_present` por alguns quadros
    depois que a folha some — nessa janela um veto com máscara de mão
    não pode alimentar a auto-rotação. `rotation_hand_mask` é o valor
    que o app consulta antes de `observe_mask`: tem de ir a None no
    resultado sem cantos mesmo com `sheet_present` True (reviews dos
    PRs #123/#125 — caminho do gate real, não só a flag isolada)."""
    mask = np.zeros((480, 640), np.uint8)
    mask[150:250, 200:300] = 255
    seen = [True]

    def detect(_i):
        if seen[0]:
            return QUAD.copy(), "paper", {"hand_mask": mask}
        return None, None, {"hand_mask": mask}

    loop, _det, _ink = _make_loop(detect)
    _drive(loop, 8)
    assert loop.sheet_present
    assert loop.last_hand_mask is not None
    assert loop.rotation_hand_mask is not None

    seen[0] = False
    _drive(loop, 4)
    # A folha segue "presente" na graça do tracker, mas o veto não
    # traz evidência de folha — o gate do app fecha junto.
    assert loop.sheet_present
    assert loop.last_hand_mask is not None
    assert loop.rotation_hand_mask is None


def test_expired_hand_mask_without_mediapipe_cannot_confirm_hand_as_ink(monkeypatch):
    class _DarkIsInk:
        threshold = 0.5

        def ink_prob(self, warped):
            return (warped[:, :, 0] < 128).astype(np.float32)

    frame = np.full_like(_frame(), 255)
    frame[150:250, 200:300] = 0
    frame[160:240, 210:290] = 255  # contorno de traco: preenchido seria sombra
    hand = np.zeros(frame.shape[:2], np.uint8)
    hand[150:250, 200:300] = 255
    cleared = False

    def detect(_i):
        mask = np.zeros_like(hand) if cleared else hand
        return QUAD.copy(), "paper", {"hand_mask": mask}

    now = 0.0

    def clock():
        return now

    config = PipelineConfig(ink=InkConfig(pass_interval_s=0.15), ink_auto_gain=False)
    ink = PageCleanWorker(_DarkIsInk(), config=config.ink, clock=clock)
    loop, det, _ink = _make_loop(detect, config=config, ink=ink, clock=clock)

    def step(at):
        nonlocal now
        now = at
        loop.step(FramePacket(frame, now, loop.frame_number))
        ink.pump()
        loop.collect_ink()

    for index in range(30):
        step(index * 0.1)
        if loop._detect_wake.settled:
            break
    assert loop._detect_wake.settled
    baseline = det._result.captured_at
    # Cadência ~1 Hz com o jitter de uma captura real: a máscara
    # oscila entre "válida e mascarando a mão" e "vencida/abstendo"
    # — nenhum dos dois estados deixa a textura da mão virar tinta
    # confirmada (review do PR #121).
    for at in (0.9, 1.95, 2.9, 3.7, 4.6):
        step(baseline + at)
    assert loop.clean_mask is not None and not loop.clean_mask.any()

    submit = det.submit
    monkeypatch.setattr(det, "submit", lambda *_args: None)
    for at in (5.6, 6.75):
        step(baseline + at)
    assert loop.last_hand_mask is None
    passes = ink.stats()["passes_total"]

    for at in (7.8, 8.95):
        step(baseline + at)
    assert loop.last_hand_mask is None
    assert ink.stats()["passes_total"] == passes
    assert not loop.clean_mask.any()

    # A mão saiu de fato da folha; o classificador volta a ver o papel
    # e um traço novo pode receber votos sem o auxílio do MediaPipe.
    cleared = True
    frame[:] = 255
    frame[190:191, 220:250] = 0
    submit(frame, baseline + 9.8)
    step(baseline + 10.4)
    submit(frame, baseline + 11.3)
    step(baseline + 12.1)
    assert loop.last_hand_mask is not None and not loop.last_hand_mask.any()
    assert loop.clean_mask.any()


def test_off_frame_quad_padding_is_filled_with_the_paper_mean():
    """Folha parcialmente fora do quadro: o preenchimento no frame só
    alcança o que a câmera viu; o padding que o warp cria sobe com a
    média do papel, não preto (review do PR #118)."""
    quad = np.array(
        [[-200.0, 100.0], [400.0, 100.0], [400.0, 400.0], [-200.0, 400.0]],
        dtype=np.float32,
    )
    mask = np.zeros((720, 1280), np.uint8)
    mask[100:400, :420] = 255  # folha segmentada, so o que a camera ve
    frame = np.full((720, 1280, 3), 60, np.uint8)
    frame[:, :420] = (200, 200, 200)
    loop, _det, ink = _make_loop(
        lambda _i: (quad.copy(), "paper", {"coverage_mask": mask})
    )
    _drive(loop, 40, frame=frame)
    image = ink.pages[-1].image
    left = image[:, : image.shape[1] // 3]  # fora do quadro: padding
    assert left.mean() > 150, (
        f"padding {left.mean():.1f} — a média do papel não chegou ao "
        "recorte fora do quadro"
    )


def test_auto_gain_still_fills_off_sheet_inside_the_quad():
    """Com ganho e cobertura juntos a máscara do quad não pode ser
    consumida pelo &= do ganho — o preenchimento fora-folha dentro do
    quad tem de continuar rodando (review do PR #118)."""
    mask = _coverage_mask()  # papel só em x<300
    frame = np.full((720, 1280, 3), 60, np.uint8)
    frame[:, :300] = (200, 200, 200)
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": mask}),
        config=PipelineConfig(ink_auto_gain=True, detect_wake=False),
    )
    _drive(loop, 40, frame=frame)
    image = ink.pages[-1].image
    right = image[:, -image.shape[1] // 5 :]  # mesa dentro do quad
    assert right.mean() > 150, (
        f"fora-folha {right.mean():.1f} — o &= do ganho consumiu a "
        "máscara do quad e o preenchimento não rodou"
    )


@pytest.mark.parametrize("different", [None, "source", "sequence", "timestamp"])
def test_hand_occlusion_reaches_ink_only_on_its_own_capture(different):
    class _HandBackend:
        def observe(self, packet, *, source_id):
            mask = np.zeros(packet.image.shape[:2], dtype=bool)
            mask[150:250, 200:300] = True
            return HandOcclusion(
                frame_sequence=packet.sequence - (different == "sequence"),
                captured_at=packet.captured_at - (different == "timestamp"),
                source_id=source_id + (different == "source"),
                occlusion_mask=mask,
            )

    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", None), hand_masker=_HandBackend()
    )
    _drive(loop, 8)
    visible = ink.submitted[-1][2]
    if different is None:
        assert visible is not None and not visible[200, 250]
        assert visible[350, 450]
    else:
        assert visible is None


class _BaselineInk(_FakeInkWorker):
    def __init__(self):
        super().__init__()
        self.rezero_calls = 0

    def rezero_baseline(self):
        self.rezero_calls += 1
        return True

    def baseline_state(self):
        return InkBaselineState(True, True, 1.0, 0, "ready")


def test_baseline_state_and_rezero_need_page_model_and_active_sheet():
    off, _, off_ink = _make_loop([], ink=_BaselineInk())
    assert off.ink_baseline_state().phase == "disabled"
    assert off.rezero_ink_baseline() is False
    assert off_ink.rezero_calls == 0

    config = PipelineConfig(ink=InkConfig(page_model=True), **_BASE)
    loop, _, ink = _make_loop([], config=config, ink=_BaselineInk())
    # Sem folha ativa: a interface espera a folha e a tecla z nao age.
    assert loop.ink_baseline_state().phase == "waiting_page"
    assert loop.rezero_ink_baseline() is False
    assert ink.rezero_calls == 0

    loop.sheet_present = True
    loop.ink_session_open = True
    assert loop.ink_baseline_state() == InkBaselineState(True, True, 1.0, 0, "ready")
    assert loop.rezero_ink_baseline() is True
    assert ink.rezero_calls == 1
