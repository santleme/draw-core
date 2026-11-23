"""Detecção de folha em thread própria: o laço nunca espera.

O contrato, com um detector falso que bloqueia de propósito:

1. submit() volta na hora mesmo com a detecção travada.
2. Quadros submetidos durante uma detecção em voo são
   substituídos (latest-wins): só o mais novo roda depois, e a
   contagem de descartes registra os que ficaram pelo caminho.
3. O resultado carrega o carimbo de captura do quadro que foi
   detectado, não o instante em que a detecção terminou.

Cobre também o formato do DetectionResult publicado (campos
opcionais de cobertura) e o parâmetro coverage da
trabalhadora, que decide se as máscaras da detecção são publicadas.
"""

import threading
import time

import numpy as np
import pytest

from draw.tracker import SheetDetectionWorker

CORNERS = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
QUAD = np.array(
    [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]],
    dtype=np.float32,
)


class _BlockingDetector:
    def __init__(self):
        self.release = threading.Event()
        self.seen = []

    def __call__(self, frame):
        self.seen.append(frame)
        self.release.wait(timeout=5.0)
        return CORNERS, "paper"


def _poll_until(worker, after_seq):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = worker.poll(after_seq)
        if result is not None:
            return result
        time.sleep(0.002)
    raise AssertionError("trabalhadora não publicou resultado")


def _wait_result(worker, after_seq=0, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = worker.poll(after_seq)
        if result is not None:
            return result
        time.sleep(0.005)
    raise AssertionError("a detecção nunca publicou resultado")


@pytest.mark.slow
def test_submit_never_blocks_and_latest_wins():
    detector = _BlockingDetector()
    worker = SheetDetectionWorker(detector).start()
    try:
        first = np.full((4, 4, 3), 1, dtype=np.uint8)
        second = np.full((4, 4, 3), 2, dtype=np.uint8)
        third = np.full((4, 4, 3), 3, dtype=np.uint8)
        started = time.monotonic()
        worker.submit(first, 100.0)
        deadline = time.monotonic() + 5.0
        while not detector.seen and time.monotonic() < deadline:
            time.sleep(0.002)
        assert detector.seen, "a thread nunca pegou o primeiro quadro"
        worker.submit(second, 100.1)
        worker.submit(third, 100.2)
        assert time.monotonic() - started < 0.5
        detector.release.set()
        result = _wait_result(worker)
        assert result.verdict == "paper"
        assert np.array_equal(result.corners, CORNERS)
        # O poll devolve o resultado mais novo: o do terceiro
        # quadro, com o carimbo dele, nao o do primeiro.
        captured_at = result.captured_at
        deadline = time.monotonic() + 5.0
        while captured_at != 100.2 and time.monotonic() < deadline:
            newer = worker.poll(result.sequence)
            if newer is not None:
                result = newer
                captured_at = result.captured_at
            time.sleep(0.005)
        assert captured_at == 100.2
        stats = worker.stats()
        assert stats["detect_submitted"] == 3
        assert stats["detect_completed"] == 2
        assert stats["detect_dropped"] == 1
        assert detector.seen[0] is first
        assert detector.seen[1] is third
    finally:
        detector.release.set()
        worker.stop()


@pytest.mark.slow
def test_detector_error_is_kept_for_the_loop():
    def boom(_frame):
        raise RuntimeError("detector quebrou")

    worker = SheetDetectionWorker(boom).start()
    try:
        worker.submit(np.zeros((4, 4, 3), dtype=np.uint8), 1.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and worker.error is None:
            time.sleep(0.005)
        assert isinstance(worker.error, RuntimeError)
    finally:
        worker.stop()


# ---------- rastro da detecção -> trabalhadora ----------


def test_worker_result_fields_with_and_without_trace():
    def detect_plain(_frame):
        return QUAD.copy(), "paper"

    worker = SheetDetectionWorker(detect_plain).start()
    try:
        worker.submit(np.zeros((8, 8, 3), np.uint8), 1.0, 0)
        result = _poll_until(worker, 0)
        assert result.coverage_mask is None
        assert result.hand_mask is None
        assert worker.take_traces() == []
    finally:
        worker.stop()

    def detect_traced(_frame):
        return QUAD.copy(), "paper", {"source": "neural"}

    worker = SheetDetectionWorker(detect_traced).start()
    try:
        worker.submit(np.zeros((8, 8, 3), np.uint8), 1.0, 0)
        result = _poll_until(worker, 0)
        traces = worker.take_traces()
        assert len(traces) == 1
        assert traces[0]["source"] == "neural"
        assert traces[0]["result_seq"] == result.sequence
        assert worker.take_traces() == []
    finally:
        worker.stop()


# ---------- extras de cobertura: parâmetro coverage ----------


def test_worker_forwards_extras_with_coverage_on():
    mask = np.zeros((8, 8), dtype=np.uint8)

    def detect_with_extras(_frame):
        return QUAD.copy(), "paper", None, {"coverage_mask": mask}

    worker = SheetDetectionWorker(detect_with_extras, coverage=True).start()
    try:
        worker.submit(np.zeros((8, 8, 3), np.uint8), 1.0, 0)
        result = _poll_until(worker, 0)
        np.testing.assert_array_equal(result.coverage_mask, mask)
    finally:
        worker.stop()


def test_worker_drops_extras_with_coverage_off():
    def detect_with_extras(_frame):
        return QUAD.copy(), "paper", None, {"coverage_mask": "x"}

    worker = SheetDetectionWorker(detect_with_extras, coverage=False).start()
    try:
        worker.submit(np.zeros((8, 8, 3), np.uint8), 1.0, 0)
        result = _poll_until(worker, 0)
        assert result.coverage_mask is None
        assert result.hand_mask is None
    finally:
        worker.stop()


# ---------- vereditos do detector ----------


def test_blank_maybe_verdict_is_published_and_worker_survives():
    """O detector pode devolver "blank_maybe" (candidato branco liso
    demais para afirmar papel): o resultado sai com esse veredito e
    a trabalhadora segue viva para o quadro seguinte."""

    def detect_blank(_frame):
        return QUAD.copy(), "blank_maybe"

    worker = SheetDetectionWorker(detect_blank).start()
    try:
        worker.submit(np.zeros((8, 8, 3), np.uint8), 1.0, 0)
        result = _poll_until(worker, 0)
        assert result.verdict == "blank_maybe"
        np.testing.assert_array_equal(result.corners, QUAD)
        worker.submit(np.zeros((8, 8, 3), np.uint8), 1.1, 1)
        newer = _poll_until(worker, result.sequence)
        assert newer.verdict == "blank_maybe"
        assert worker.error is None
    finally:
        worker.stop()


def test_invalid_verdict_reports_worker_error():
    """Um veredito fora dos aceitos falha na construção do contrato:
    a falha chega ao laço como erro (worker.error), não mata a
    thread em silêncio deixando poll/error presos em None."""

    def detect_bogus(_frame):
        return QUAD.copy(), "table"

    worker = SheetDetectionWorker(detect_bogus).start()
    try:
        worker.submit(np.zeros((8, 8, 3), np.uint8), 1.0, 0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and worker.error is None:
            time.sleep(0.005)
        assert isinstance(worker.error, ValueError)
        assert worker.poll(0) is None
    finally:
        worker.stop()
