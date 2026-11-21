"""Recorte do tools/replay_reregister.py so com o que os testes do
motor usam: paginas sinteticas, relogio de cena, trabalhadora
sincrona e a page_clean falsa (_ProbIsWarp / ScenePageClean)."""

import copy
import threading
import zlib
from pathlib import Path

import cv2
import numpy as np

from draw.ink.accumulator import LayeredInkAccumulator
from draw.ink.registration import _corners_page_matrix
from draw.ink.worker import PageCleanWorker

HOME = Path.home()
THRESHOLD = 0.7
PASSES_PER_WARP = 3
FPS = 30
DT = 1.0 / FPS
CAMERA_W, CAMERA_H = 1280, 720
DETECTOR_NOISE_PX = 1.0
DETECT_MS = 36.7
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


def transform_quad(quad, shift=(0.0, 0.0), angle=0.0, scale=1.0):
    center = quad.mean(axis=0)
    rad = np.deg2rad(angle)
    rot = np.array(
        [[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]],
        np.float32,
    )
    out = (quad - center) @ rot.T * scale + center
    return (out + np.asarray(shift, np.float32)).astype(np.float32)


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


class SceneWarp:
    """O "warp" que a câmera de cena entrega ao laço.

    Carrega a página, o quad verdadeiro e o quad com que o laço
    retificou; o mapa só é renderizado quando a passada da
    trabalhadora roda (latest-wins: warps substituídos nunca custam).
    """

    __slots__ = ("page", "true_quad", "corners")

    def __init__(self, page, true_quad, corners):
        self.page = page
        self.true_quad = true_quad
        self.corners = corners


class ScenePageClean:
    """page_clean falso: o mapa de probabilidade é o da cena re-warpado."""

    threshold = THRESHOLD

    def __init__(self):
        self.worker = None

    def bind(self, worker):
        self.worker = worker

    def ink_prob(self, warped):
        return self.observe(warped.page, warped.true_quad, warped.corners)

    def observe(self, page, true_quad, corners):
        return render_pass(page, true_quad, corners)


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


class ReplayInkWorker(PageCleanWorker):
    """A trabalhadora real, dirigida por pump() (sem thread).

    Só acrescenta cópia profunda (os cenários partem de uma cópia do
    app assentado); nenhum comportamento é sobrescrito — o teste
    tests/test_replay_reregister.py cobra isso.
    """

    def __deepcopy__(self, memo):
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        for key, value in self.__dict__.items():
            if key in ("_condition", "_thread"):
                continue
            setattr(clone, key, copy.deepcopy(value, memo))
        clone._condition = threading.Condition()
        clone._thread = threading.Thread(
            target=clone._work, name="draw-page-clean", daemon=True
        )
        return clone


def make_ink_worker(page_clean, clock, config=None):
    """`PageCleanWorker` real em modo síncrono, com histórico completo."""
    worker = ReplayInkWorker(
        page_clean, clock=clock, history=(None, None), config=config
    )
    worker._accumulator = CountingAccumulator(
        threshold=getattr(page_clean, "threshold", 0.7), config=worker._config
    )
    bind = getattr(page_clean, "bind", None)
    if bind is not None:
        bind(worker)
    return worker


def _noisy(quad, rng, sigma=DETECTOR_NOISE_PX):
    return (quad + rng.normal(0.0, sigma, size=quad.shape)).astype(np.float32)


def _rng(*names):
    return np.random.default_rng(zlib.crc32("/".join(names).encode()))


# --------------------------------------------------------------------
# Comparação a menos de re-warp
# --------------------------------------------------------------------


class _ProbIsWarp:
    """page_clean falso: o "warp" submetido já é o mapa."""

    threshold = 0.7

    def ink_prob(self, warped):
        return warped
