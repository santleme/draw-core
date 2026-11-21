"""Regressao da guarda de borda na grade retificada (issue #131)."""

import numpy as np
import pytest

from draw import contracts
from draw.ink.config import InkConfig, resolve_border_guard_px, resolve_border_guard_yx
from draw.ink.worker import PageCleanWorker
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
    assert resolve_border_guard_yx((1200, 850), None, small_quad) == (9, 9)
    assert resolve_border_guard_yx((1200, 850), 5, small_quad) == (5, 5)

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

    ink_cfg = InkConfig.production()
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

    ink_cfg = InkConfig.production()
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
    ink_cfg = InkConfig.strict_vote(0.85, border_guard_px=0)
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
    ink_cfg = InkConfig.production()
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


def test_empty_page_without_coverage_and_small_camera_sheet_stays_empty():
    """Sem cobertura e com folha pequena na camera (360x500 -> 850x1200),
    a guarda projetada cobre o estiramento de 3 px da camera e mantem
    toda a pagina vazia (inclusive logo dentro da margem de 7 px)."""
    frame = np.full((720, 1280, 3), 20, np.uint8)
    frame[100:600, 200:560] = 215
    quad = np.array(
        [[197, 97], [563, 97], [563, 603], [197, 603]],
        dtype=np.float32,
    )
    ink_cfg = InkConfig.production()
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

    ink_cfg = InkConfig.production()
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
