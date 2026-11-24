"""Testes da captura latest-frame e da escritora de PNG de draw/camera.py.

Comportamento das classes com capturas falsas: quem consome so ve
o quadro mais novo e nunca reprocessa; quadros substituidos contam
como descartados; falha seguida de leitura declara a fonte morta
no limite de _CAPTURE_MAX_FAILURES; falha transitoria nao mata a
fonte; a escritora grava os PNG enfileirados e drena no close().

Uso: python tests/test_capture_thread.py
"""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from draw.camera import _CAPTURE_MAX_FAILURES, LatestFrameCapture, PngWriter

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"ok - {name}")
    else:
        print(f"FALHOU - {name} {detail}")
        FAILURES.append(name)
        # Sob pytest a verificacao tem de falhar de verdade; no modo
        # script (__main__) o relatorio no fim decide a saida.
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise AssertionError(f"{name} {detail}")


class _PacedCapture:
    """read() devolve um quadro novo a cada periodo, como uma camera."""

    def __init__(self, period_s):
        self._period = period_s
        self._next_at = time.monotonic()
        self.produced = 0
        self.released = False

    def read(self):
        now = time.monotonic()
        if now < self._next_at:
            time.sleep(self._next_at - now)
        self._next_at = max(self._next_at + self._period, time.monotonic())
        self.produced += 1
        frame = np.full((48, 64, 3), self.produced % 256, dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


class _FailingCapture:
    def __init__(self):
        self.calls = 0
        self.released = False

    def read(self):
        self.calls += 1
        return False, None

    def release(self):
        self.released = True


class _FlakyCapture(_PacedCapture):
    """Falha as primeiras leituras e depois entrega quadros."""

    def __init__(self, failures, period_s):
        super().__init__(period_s)
        self._failures_left = failures

    def read(self):
        if self._failures_left > 0:
            self._failures_left -= 1
            return False, None
        return super().read()


class _RaisingCapture:
    """read() levanta cv2.error — a bomba morre com a excecao."""

    def __init__(self):
        self.released = threading.Event()

    def read(self):
        raise cv2.error("driver quebrou no meio do read")

    def release(self):
        self.released.set()


class _BurstCapture:
    """Entrega uma rajada de quadros e depois trava no read()."""

    def __init__(self, count):
        self._left = count
        self._released = threading.Event()
        self.released = False

    def read(self):
        if self._left > 0:
            self._left -= 1
            return True, np.full((48, 64, 3), 7, dtype=np.uint8)
        self._released.wait(timeout=30.0)
        return False, None

    def release(self):
        self.released = True
        self._released.set()


@pytest.mark.slow
def test_latest_only_and_no_reprocess():
    capture = _PacedCapture(0.01)
    source = LatestFrameCapture(capture).start()
    time.sleep(0.3)
    item = source.wait_next(0, timeout=2.0)
    check("consumidor recebe quadro", item is not None)
    check(
        "o retorno e um FramePacket do contrato",
        item.image.shape == (48, 64, 3)
        and item.image.dtype == np.uint8
        and item.captured_at <= time.monotonic(),
    )
    check(
        "imagem do pacote e somente-leitura",
        not item.image.flags.writeable,
    )
    seq = item.sequence
    check(
        "quadro consumido e o mais novo, nao o primeiro da fila",
        seq >= capture.produced - 1,
        f"seq={seq} produced={capture.produced}",
    )
    check(
        "substituidos antes do consumo contam como descartados",
        source.dropped > 0,
        f"dropped={source.dropped}",
    )
    newer = source.wait_next(seq, timeout=2.0)
    check(
        "sequencia seguinte chega",
        newer is not None and newer.sequence > seq,
    )
    source.stop()
    check("stop libera a captura", capture.released)


@pytest.mark.slow
def test_no_reprocess_without_new_frame():
    capture = _BurstCapture(3)
    source = LatestFrameCapture(capture).start()
    time.sleep(0.2)
    item = source.wait_next(0, timeout=2.0)
    check("rajada consumida ate o ultimo quadro", item is not None)
    seq = item.sequence
    check("ultimo da rajada tem a sequencia da rajada", seq == 3, f"seq={seq}")
    again = source.wait_next(seq, timeout=0.3)
    check("sem quadro novo, a mesma sequencia nao volta", again is None)
    source.stop()


@pytest.mark.slow
def test_failure_limit_marks_dead():
    capture = _FailingCapture()
    source = LatestFrameCapture(capture).start()
    deadline = time.monotonic() + 5.0
    while not source.dead and time.monotonic() < deadline:
        time.sleep(0.01)
    check("fonte morta apos o limite de falhas", source.dead)
    check(
        "contagem de falhas para no limite",
        source.failures == _CAPTURE_MAX_FAILURES,
        f"failures={source.failures}",
    )
    check(
        "quem espera acorda com None",
        source.wait_next(0, timeout=1.0) is None,
    )
    source.stop()


def test_pump_exception_marks_dead_releases_and_exposes_error():
    """read() que levanta cv2.error: a bomba declara a fonte morta,
    libera a captura no finally proprio e publica a excecao em
    .error — sem a propriedade, o motivo da morte era invisivel."""
    capture = _RaisingCapture()
    source = LatestFrameCapture(capture).start()
    assert capture.released.wait(5.0)
    check("fonte morta com read que levanta", source.dead)
    check(
        "excecao exposta no .error",
        isinstance(source.error, cv2.error),
        f"error={source.error!r}",
    )
    source.stop()


@pytest.mark.slow
def test_transient_failures_recover():
    capture = _FlakyCapture(failures=3, period_s=0.01)
    source = LatestFrameCapture(capture).start()
    item = source.wait_next(0, timeout=2.0)
    check("falha transitoria nao impede o quadro", item is not None)
    check("fonte nao morreu", not source.dead)
    check(
        "leitura boa zera a contagem de falhas",
        source.failures == 0,
        f"failures={source.failures}",
    )
    source.stop()


@pytest.mark.slow
def test_png_writer_drains_on_close():
    writer = PngWriter()
    done = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        expected = {}
        for index, value in enumerate((30, 120, 220)):
            path = base / f"img_{index}.png"
            expected[path] = value
            writer.save(
                path,
                np.full((40, 50, 3), value, dtype=np.uint8),
                lambda p: done.append(p),
            )
        writer.close()
        check(
            "close drena a fila e a thread termina",
            not writer._thread.is_alive(),
        )
        check(
            "callbacks rodam em ordem, um por arquivo",
            done == list(expected),
            f"done={done}",
        )
        for path, value in expected.items():
            img = cv2.imread(str(path)) if path.exists() else None
            check(
                f"{path.name} gravado com o conteudo enfileirado",
                img is not None and int(img[0, 0, 0]) == value,
            )


def test_on_done_error_does_not_kill_writer(tmp_path):
    writer = PngWriter()
    done = []

    def boom(_path):
        raise RuntimeError("confirmacao quebrou")

    first = tmp_path / "one.png"
    second = tmp_path / "two.png"
    writer.save(first, np.zeros((4, 4, 3), np.uint8), on_done=boom)
    writer.save(second, np.zeros((4, 4, 3), np.uint8), on_done=done.append)
    writer.close()
    assert first.exists()
    assert second.exists()
    assert done == [second]


class _HelperLikeCapture:
    """Fonte que expõe identidade do emissor, como HelperCapture."""

    def __init__(self, records):
        # records: [(frame_id, capture_ts_ms), ...]
        self._records = list(records)
        self.last_frame_id = None
        self.last_capture_ts_ms = None
        self.source_dropped = 0
        self._last_seen_id = None

    def read(self):
        if not self._records:
            time.sleep(0.05)
            return False, None
        self.last_frame_id, self.last_capture_ts_ms = self._records.pop(0)
        if (
            self._last_seen_id is not None
            and self.last_frame_id > self._last_seen_id + 1
        ):
            self.source_dropped += self.last_frame_id - self._last_seen_id - 1
        self._last_seen_id = self.last_frame_id
        frame = np.full((8, 8, 3), 7, dtype=np.uint8)
        return True, frame

    def release(self):
        pass


@pytest.mark.slow
def test_source_metadata_reaches_packet():
    capture = _HelperLikeCapture([(10, 42_000), (11, 42_033)])
    source = LatestFrameCapture(capture).start()
    time.sleep(0.2)
    item = source.wait_next(0, timeout=2.0)
    check("pacote chegou", item is not None)
    check(
        "frame_id do emissor vai no pacote",
        item.source_frame_id == 11,
        f"source_frame_id={item.source_frame_id}",
    )
    check(
        "ts do emissor vai no pacote",
        item.source_ts_ms == 42_033,
        f"source_ts_ms={item.source_ts_ms}",
    )
    check(
        "sem saltos, nada descartado na fonte",
        source.stats()["capture_source_dropped"] == 0,
    )
    source.stop()


@pytest.mark.slow
def test_source_dropped_counter_reaches_stats():
    """stats() repassa o contador de descarte na fonte do adaptador."""
    capture = _HelperLikeCapture([(0, 0), (1, 33), (4, 100), (5, 133)])
    source = LatestFrameCapture(capture).start()
    time.sleep(0.2)
    item = source.wait_next(0, timeout=2.0)
    check("pacote chegou", item is not None)
    stats = source.stats()
    check(
        "salto 1->4 conta 2 quadros descartados na fonte",
        stats["capture_source_dropped"] == 2,
        f"stats={stats}",
    )
    source.stop()


@pytest.mark.slow
def test_plain_capture_has_no_source_metadata():
    capture = _PacedCapture(0.01)
    source = LatestFrameCapture(capture).start()
    time.sleep(0.1)
    item = source.wait_next(0, timeout=2.0)
    check("pacote chegou", item is not None)
    check(
        "fonte sem identidade deixa campos None",
        item.source_frame_id is None and item.source_ts_ms is None,
    )
    check(
        "sem fonte identificada nao ha contador de origem",
        "capture_source_dropped" not in source.stats(),
    )
    source.stop()


if __name__ == "__main__":
    test_latest_only_and_no_reprocess()
    test_no_reprocess_without_new_frame()
    test_failure_limit_marks_dead()
    test_transient_failures_recover()
    test_png_writer_drains_on_close()
    print()
    if FAILURES:
        print(f"FALHARAM {len(FAILURES)} verificacoes: {FAILURES}")
        sys.exit(1)
    print("TODOS OS TESTES DE CAPTURA OK.")
