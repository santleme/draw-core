"""Recorte do tools/replay_reregister.py so com o que os testes do
motor usam: paginas sinteticas, relogio de cena, trabalhadora
sincrona, a page_clean falsa (_ProbIsWarp) e a bancada reg/anchor
(SHAPE/BASE, drive sincrono e o replay de paridade GOLDEN)."""

import hashlib
import time

import cv2
import numpy as np

from draw.contracts import WarpedPage
from draw.ink.accumulator import LayeredInkAccumulator
from draw.ink.registration import _corners_page_matrix
from draw.ink.worker import PageCleanWorker

CAMERA_W, CAMERA_H = 1280, 720
SYNTHETIC_SHAPE = (600, 424)


def synthetic_pages(shape=SYNTHETIC_SHAPE, stages=10, seed=7):
    """Desenho sintético em estágios, com ombro sub-limiar real.

    Caneta (núcleo ~0,95) e lápis (núcleo ~0,6, só confirma pelo
    voto de hits); blur σ≈1 põe a banda 0,5–0,7 a 1–3 px do núcleo,
    o ombro medido nos mapas reais. Cada estágio tem 3 variantes
    (ganho, ruído, tremor sub-pixel) — a variação passada a passada
    que os warps reais têm.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    ink = np.zeros(shape, np.float32)
    sequence = []
    settled = []
    for _stage in range(stages):
        for _ in range(3):
            n = int(rng.integers(5, 10))
            start = rng.uniform((0.1 * w, 0.1 * h), (0.9 * w, 0.9 * h))
            steps = rng.normal(0.0, 0.06 * min(h, w), size=(n, 2))
            pts = np.clip(
                start + np.cumsum(steps, axis=0),
                (0.05 * w, 0.05 * h),
                (0.95 * w, 0.95 * h),
            )
            strength = 0.85 if rng.random() < 0.8 else 0.55
            layer = np.zeros(shape, np.float32)
            cv2.polylines(
                layer,
                [pts.astype(np.int32).reshape(-1, 1, 2)],
                False,
                1.0,
                int(rng.integers(2, 4)),
                lineType=cv2.LINE_AA,
            )
            ink = np.maximum(ink, layer * strength)
        base = np.clip(cv2.GaussianBlur(ink, (0, 0), 1.0) * 1.15, 0, 0.98)
        variants = []
        for _ in range(3):
            gain = rng.uniform(0.92, 1.04)
            noise = rng.normal(0.0, 0.03, size=shape).astype(np.float32)
            dx, dy = rng.uniform(-0.4, 0.4, size=2)
            moved = cv2.warpAffine(
                base,
                np.float32([[1, 0, dx], [0, 1, dy]]),
                (w, h),
                flags=cv2.INTER_LINEAR,
            )
            variants.append(
                np.clip(moved * gain + noise * (moved > 0.05), 0.0, 1.0).astype(
                    np.float32
                )
            )
        sequence.append(variants)
        settled = variants
    return sequence, settled


# --------------------------------------------------------------------
# Geometria da câmera
# --------------------------------------------------------------------


def base_quad(shape):
    """Quad verdadeiro da página na câmera (TL, TR, BR, BL)."""
    h, w = shape
    height = 466.0
    width = height * w / h
    cx, cy = 560.0, 370.0
    quad = np.array(
        [
            (cx - width / 2 + 12, cy - height / 2),
            (cx + width / 2 - 12, cy - height / 2),
            (cx + width / 2, cy + height / 2),
            (cx - width / 2, cy + height / 2),
        ],
        np.float32,
    )
    return quad


def camera_mask(page_mask, corners):
    """Expressa uma máscara de grade de página na grade da câmera.

    WarpedPage carrega máscaras em espaço de câmera (issue #100):
    para construir a cena do teste, a máscara desenhada na grade
    da página volta ao frame pela homografia inversa do quad.
    """
    matrix = _corners_page_matrix(corners, page_mask.shape[1], page_mask.shape[0])
    return (
        cv2.warpPerspective(
            page_mask.astype(np.uint8) * 255,
            np.linalg.inv(matrix),
            (CAMERA_W, CAMERA_H),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        > 0
    )


def _page_matrix(quad, shape):
    h, w = shape
    return _corners_page_matrix(quad, w, h)


_ONES_FRAME = np.full((CAMERA_H, CAMERA_W), 255, np.uint8)


def render_pass(page, true_quad, warp_corners):
    """O que o modelo devolveria retificando com warp_corners.

    page vive na grade da página; a câmera a vê em true_quad; o app
    retifica com warp_corners. Fora do quadro da câmera não há
    imagem (prob 0).
    """
    shape = page.shape
    matrix = _page_matrix(warp_corners, shape) @ np.linalg.inv(
        _page_matrix(true_quad, shape)
    )
    if np.max(np.abs(matrix - np.eye(3))) < 1e-9:
        prob = page.copy()
    else:
        prob = cv2.warpPerspective(
            page,
            matrix,
            (shape[1], shape[0]),
            flags=cv2.INTER_LINEAR,
            borderValue=0.0,
        )
    visible = cv2.warpPerspective(
        _ONES_FRAME,
        _page_matrix(warp_corners, shape),
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    prob[visible == 0] = 0.0
    return prob


# --------------------------------------------------------------------
# Dependências simuladas do laço real (draw/loop.py)
# --------------------------------------------------------------------


class SceneClock:
    """Relógio de cena: o mesmo instante para o laço e as trabalhadoras."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class CountingAccumulator(LayeredInkAccumulator):
    """O acumulador real; só conta os sinais de virada e guarda o resultado."""

    def __init__(self, threshold=0.7, config=None):
        super().__init__(threshold, config=config)
        self.turn_signals = 0
        self.last_result = None

    def add(self, prob, visible=None, **kwargs):
        result = super().add(prob, visible, **kwargs)
        if result["turn"]:
            self.turn_signals += 1
        self.last_result = result
        return result


def make_ink_worker(page_clean, clock, config=None):
    """`PageCleanWorker` real em modo síncrono, com histórico completo."""
    worker = PageCleanWorker(
        page_clean, clock=clock, history=(None, None), config=config
    )
    worker._accumulator = CountingAccumulator(
        threshold=getattr(page_clean, "threshold", 0.7), config=worker._config
    )
    bind = getattr(page_clean, "bind", None)
    if bind is not None:
        bind(worker)
    return worker


# --------------------------------------------------------------------
# Comparação a menos de re-warp
# --------------------------------------------------------------------


class _ProbIsWarp:
    """page_clean falso: o "warp" submetido já é o mapa."""

    threshold = 0.7

    def ink_prob(self, warped):
        return warped


# --------------------------------------------------------------------
# Bancada reg/anchor: página miniatura 120x85, drive síncrono e o
# replay de paridade compartilhado pelos testes de registro e âncora.
# --------------------------------------------------------------------

SHAPE = (120, 85)
BASE = base_quad(SHAPE)


def _page():
    """A página assentada do desenho sintético fixo da bancada."""
    _drawing, settled = synthetic_pages(SHAPE, stages=3, seed=3)
    return settled[0]


def _pass(page, warp_corners):
    """A pagina parada em BASE, retificada com o quad do detector."""
    corners = np.asarray(warp_corners, np.float32)
    return render_pass(page, BASE, corners), corners


def _settled(content, passes=4):
    return [(content, BASE.copy()) for _ in range(passes)]


def _drive(worker, generation, sequence):
    for prob, corners in sequence:
        before = worker.stats()["passes_total"]
        worker.submit(WarpedPage(prob, generation, corners))
        deadline = time.monotonic() + 5.0
        while worker.stats()["passes_total"] <= before:
            assert time.monotonic() < deadline, "a passada nao executou"
            time.sleep(0.001)


def _run(sequence, config=None):
    worker = PageCleanWorker(_ProbIsWarp(), config=config).start()
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


# sha256 de confirmado + provisorio publicados, gerados na main
# 8dcce18 (antes das flags), no replay sintetico de
# tools/replay_reregister.py com tremor de 4 px e recuperacao.
GOLDEN = {
    "jitter": "f5c9b83c2462b5e2d8608a824d24e8b6b5c28b391129e9388684eab5e4f1afbf",
    "recover": "42a9851ef5e027e0b868ae8f75f86d548da49b1dde3a2426dbb772405c7a151b",
}


def _parity_canvas(scenario, config=None):
    drawing, settled = synthetic_pages(SHAPE, stages=3, seed=3)
    rng = np.random.default_rng(0)
    sequence = []
    for stage in drawing + [settled]:
        for k in range(4):
            corners = (BASE + rng.normal(0, 4.0, size=(4, 2))).astype(np.float32)
            sequence.append(
                (render_pass(stage[k % len(stage)], BASE, corners), corners)
            )
    return_at = None
    if scenario == "recover":
        return_at = len(sequence)
        moved = BASE + np.float32([6.0, -4.0])
        sequence += [(render_pass(settled[0], BASE, moved), moved)] * 3
    worker = PageCleanWorker(_ProbIsWarp(), config=config).start()
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
