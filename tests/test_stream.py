"""Stream de tinta (draw.stream, bridge/STREAM.md)."""

import io
import os
import socket
import struct
import sys
import threading
import time

import numpy as np
import pytest

from draw import contracts, stream
from draw.protocol import (
    ProtocolError,
    client_handshake,
)

TOKEN = "t" * 32


needs_private_files = pytest.mark.skipif(
    not stream._PRIVATE_FILES,
    reason="gravacao privada exige os.fchmod (nao no Windows nativo)",
)


def _pub(pub, generation, version, mask):
    pub.publish(contracts.InkPublication(generation, version, mask))


def _mask(h=40, w=30, fill=0):
    return np.full((h, w), fill, np.uint8)


def _canvas_from(data):
    canvas = stream.InkCanvas()
    boxes = [canvas.apply(r) for r in stream.iter_records(io.BytesIO(data).read)]
    return canvas, boxes


def test_header_is_32_bytes():
    assert stream.HEADER_SIZE == 32


def test_full_then_delta_roundtrip():
    a = _mask()
    b = a.copy()
    b[5:9, 3:12] = 255
    data = stream.pack_full(1, 1, 10, a, (210.0, 297.0))
    data += stream.pack_delta(1, 2, 20, b, stream.diff_box(a, b))
    canvas, boxes = _canvas_from(data)
    assert boxes == [(0, 0, 30, 40), (3, 5, 9, 4)]
    assert np.array_equal(canvas.mask, b)
    assert canvas.paper_mm == pytest.approx((210.0, 297.0))
    assert (canvas.generation, canvas.ink_version) == (1, 2)


def test_diff_box_none_when_equal():
    assert stream.diff_box(_mask(), _mask()) is None


def test_delta_without_full_is_protocol_error():
    a = _mask()
    b = a.copy()
    b[0, 0] = 255
    data = stream.pack_delta(1, 2, 0, b, (0, 0, 1, 1))
    with pytest.raises(ProtocolError):
        _canvas_from(data)


def test_delta_other_generation_is_protocol_error():
    a = _mask()
    b = a.copy()
    b[0, 0] = 255
    data = stream.pack_full(1, 1, 0, a, (1.0, 1.0))
    data += stream.pack_delta(2, 2, 0, b, (0, 0, 1, 1))
    with pytest.raises(ProtocolError):
        _canvas_from(data)


def test_bad_magic_and_oversized_payload_rejected_before_alloc():
    good = stream.pack_full(1, 1, 0, _mask(), (1.0, 1.0))
    with pytest.raises(ProtocolError):
        list(stream.iter_records(io.BytesIO(b"XXXX" + good[4:]).read))
    huge = stream.HEADER.pack(
        stream.MAGIC,
        1,
        stream.KIND_FULL,
        0,
        1,
        1,
        0,
        10,
        10,
        stream.MAX_PAYLOAD_LEN + 1,
    )
    with pytest.raises(ProtocolError):
        list(stream.iter_records(io.BytesIO(huge).read))


def test_truncated_tail_is_error():
    good = stream.pack_full(1, 1, 0, _mask(), (1.0, 1.0))
    with pytest.raises(ProtocolError):
        list(stream.iter_records(io.BytesIO(good[:-3]).read))


def test_zlib_bomb_rejected():
    import zlib

    bomb = zlib.compress(bytes(10_000_000))
    payload = struct.pack("<ff", 1.0, 1.0) + bomb
    data = (
        stream.HEADER.pack(
            stream.MAGIC, 1, stream.KIND_FULL, 0, 1, 1, 0, 4, 4, len(payload)
        )
        + payload
    )
    with pytest.raises(ProtocolError):
        _canvas_from(data)


def _connect(port, token=TOKEN):
    sock = socket.create_connection(("127.0.0.1", port), timeout=3)
    client_handshake(sock, token)
    return sock


def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _read_until(sock, canvas, version):
    reader = stream.iter_records(sock.recv)
    for rec in reader:
        canvas.apply(rec)
        if canvas.ink_version >= version:
            return


@needs_private_files
def test_publisher_socket_and_record_reconstruct_same_canvas(tmp_path):
    rec_path = tmp_path / "s.drawink"
    pub = stream.InkStreamPublisher(
        port=0,
        token=TOKEN,
        paper_mm=(210.0, 297.0),
        record_path=rec_path,
        log=lambda *_: None,
    )
    try:
        sock = _connect(pub.port)
        assert _wait(lambda: len(pub._sinks) == 2)
        a = _mask()
        _pub(pub, 1, 1, a)
        b = a.copy()
        b[2:6, 2:6] = 255
        time.sleep(0.1)
        _pub(pub, 1, 2, b)
        canvas = stream.InkCanvas()
        _read_until(sock, canvas, 2)
        assert np.array_equal(canvas.mask, b)
        sock.close()
    finally:
        pub.close()
    file_canvas = stream.InkCanvas()
    for rec in stream.records_from_file(rec_path):
        file_canvas.apply(rec)
    assert np.array_equal(file_canvas.mask, b)


def test_wrong_token_gets_nothing():
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    try:
        sock = socket.create_connection(("127.0.0.1", pub.port), timeout=3)
        sock.sendall(struct.pack("<4sHH", b"DRAH", 1, 32) + b"x" * 32)
        assert sock.recv(4) == b""
        sock.close()
        assert pub._sinks == []
    finally:
        pub.close()


def test_late_consumer_gets_full_of_latest():
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    try:
        a = _mask()
        a[1:3, 1:3] = 255
        _pub(pub, 3, 7, a)
        time.sleep(0.1)
        sock = _connect(pub.port)
        canvas = stream.InkCanvas()
        _read_until(sock, canvas, 7)
        assert np.array_equal(canvas.mask, a)
        assert canvas.generation == 1  # geracao do publicador, nao a da trabalhadora
        sock.close()
    finally:
        pub.close()


def test_publish_never_blocks_with_stalled_consumer():
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    try:
        sock = _connect(pub.port)  # nunca le
        big = np.random.default_rng(0).integers(0, 256, (1200, 850), dtype=np.uint8)
        start = time.monotonic()
        for i in range(50):
            _pub(pub, 1, i + 1, big ^ np.uint8(i))
        assert time.monotonic() - start < 0.5
        sock.close()
    finally:
        pub.close()


def test_no_consumer_publish_is_cheap_and_close_is_clean():
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    for i in range(20):
        _pub(pub, 1, i + 1, _mask())
    pub.close()
    assert not any(t.is_alive() for t in pub._threads)


def test_missing_token_refused():
    with pytest.raises(Exception, match="token"):
        stream.InkStreamPublisher(port=0, token=None)


def test_pipeline_forwards_on_ink():
    from draw import pipeline as pipeline_module

    calls = []

    class Worker:
        error = None

        def poll_display(self, generation, version):
            from draw.contracts import InkDisplaySnapshot

            return InkDisplaySnapshot(
                version=4, confirmed=_mask(fill=255), provisional=_mask()
            )

    pipe = pipeline_module.Pipeline.__new__(pipeline_module.Pipeline)
    pipe.ink_worker = Worker()
    pipe.clean_generation = 2
    pipe.clean_display_version = 0
    pipe._on_ink = lambda *a: calls.append(a)
    pipe._render_display = lambda *a, **k: None
    pipe.canvas_buffer = None
    pipe._log = lambda *_: None
    pipe.collect_ink()
    assert len(calls) == 1
    pub = calls[0][0]
    assert isinstance(pub, contracts.InkPublication)
    assert (pub.generation, pub.version) == (2, 4)
    assert pub.confirmed.min() == 255
    assert threading.active_count() >= 1


def test_zlib_garbage_and_trailing_bytes_are_protocol_errors():
    import zlib

    prefix = struct.pack("<ff", 1.0, 1.0)
    for body in (b"not zlib at all", zlib.compress(bytes(16)) + b"extra"):
        payload = prefix + body
        data = stream.HEADER.pack(
            stream.MAGIC, 1, stream.KIND_FULL, 0, 1, 1, 0, 4, 4, len(payload)
        )
        with pytest.raises(ProtocolError):
            _canvas_from(data + payload)


@needs_private_files
def test_record_only_needs_no_token_and_keeps_final_ink(tmp_path):
    path = tmp_path / "r.drawink"
    pub = stream.InkStreamPublisher(port=None, record_path=path, log=lambda *_: None)
    assert pub.port is None
    final = _mask()
    final[3:5, 3:5] = 255
    _pub(pub, 1, 1, _mask())
    time.sleep(0.1)
    _pub(pub, 1, 2, final)
    pub.close()  # a ultima publicacao pendente ainda precisa ser gravada
    canvas = stream.InkCanvas()
    for rec in stream.records_from_file(path):
        canvas.apply(rec)
    assert canvas.ink_version == 2
    assert np.array_equal(canvas.mask, final)


def test_nothing_to_publish_is_refused():
    with pytest.raises(ValueError):
        stream.InkStreamPublisher(port=None)


@needs_private_files
def test_recording_is_private(tmp_path):
    import os
    import stat

    path = tmp_path / "p.drawink"
    old = os.umask(0)
    try:
        pub = stream.InkStreamPublisher(
            port=None, record_path=path, log=lambda *_: None
        )
    finally:
        os.umask(old)
    pub.close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@needs_private_files
def test_failing_recording_does_not_stop_live_stream(tmp_path):
    pub = stream.InkStreamPublisher(
        port=0, token=TOKEN, record_path=tmp_path / "f.drawink", log=lambda *_: None
    )
    try:
        record_sink = next(s for s in pub._sinks if not s.is_socket)

        def boom(_data):
            raise OSError("disco cheio")

        def boom_close():
            raise OSError("flush falhou")

        record_sink.write = boom
        record_sink.close = boom_close
        sock = _connect(pub.port)
        assert _wait(lambda: len(pub._sinks) == 2)
        _pub(pub, 1, 1, _mask())
        time.sleep(0.2)
        b = _mask()
        b[1:3, 1:3] = 255
        _pub(pub, 1, 2, b)
        canvas = stream.InkCanvas()
        _read_until(sock, canvas, 2)
        assert np.array_equal(canvas.mask, b)
        sock.close()
    finally:
        pub.close()


def test_closed_idle_consumer_is_pruned():
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    try:
        sock = _connect(pub.port)
        assert _wait(lambda: len(pub._sinks) == 1)
        sock.close()
        assert _wait(lambda: len(pub._sinks) == 0, timeout=3)
    finally:
        pub.close()


def test_consumer_cap_evicts_oldest(monkeypatch):
    monkeypatch.setattr(stream, "MAX_CONSUMERS", 2)
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    try:
        # A adocao roda numa thread por conexao: registrar em sequencia
        # garante que a ordem dos sinks bate com a ordem dos sockets,
        # senao a vitima da expulsao e aleatoria.
        socks = [_connect(pub.port)]
        assert _wait(lambda: len(pub._sinks) == 1)
        socks.append(_connect(pub.port))
        socks.append(_connect(pub.port))
        assert _wait(lambda: len(pub._sinks) == 2)
        # len==2 ja vale antes da 3a conexao entrar; espera o EOF com folga.
        socks[0].settimeout(10)
        assert socks[0].recv(1) == b""
        for s in socks:
            s.close()
    finally:
        pub.close()


def test_pending_handshake_cap_refuses_without_thread(monkeypatch):
    monkeypatch.setattr(stream, "MAX_PENDING_HANDSHAKES", 1)
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    try:
        silent = socket.create_connection(("127.0.0.1", pub.port), timeout=3)
        time.sleep(0.4)  # ocupa o unico slot (handshake sem HELLO)
        second = socket.create_connection(("127.0.0.1", pub.port), timeout=3)
        second.settimeout(2)
        assert second.recv(1) == b""  # recusado de cara
        silent.close()
        second.close()
    finally:
        pub.close()


def test_clear_bumps_generation_and_sends_empty_canvas():
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    try:
        sock = _connect(pub.port)
        assert _wait(lambda: len(pub._sinks) == 1)
        full = _mask(fill=255)
        _pub(pub, 5, 3, full)
        canvas = stream.InkCanvas()
        _read_until(sock, canvas, 3)
        first_generation = canvas.generation
        time.sleep(0.1)
        pub.publish(contracts.InkPublication(-1, 0, None))
        _read_until(sock, canvas, 0)
        assert canvas.generation == first_generation + 1
        assert canvas.mask.max() == 0
        sock.close()
    finally:
        pub.close()


def test_clear_before_any_publication_is_a_noop(monkeypatch):
    monkeypatch.setattr(stream.InkStreamPublisher, "_send_loop", lambda self: None)
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    try:
        pub.publish(contracts.InkPublication(-1, 0, None))
        assert pub._pending is None
    finally:
        pub.close()


def test_worker_generation_maps_to_monotonic_stream_generation(monkeypatch):
    # Sem thread de envio: _pending fica estavel e o teste nao corre contra ela.
    monkeypatch.setattr(stream.InkStreamPublisher, "_send_loop", lambda self: None)
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    try:
        _pub(pub, 7, 1, _mask())
        g1 = pub._pending[0]
        _pub(pub, 7, 2, _mask())
        assert pub._pending[0] == g1
        _pub(pub, 9, 1, _mask())
        assert pub._pending[0] == g1 + 1
    finally:
        pub.close()


def test_reset_source_publishes_clear():
    from draw import pipeline as pipeline_module

    calls = []
    pipe = pipeline_module.Pipeline.__new__(pipeline_module.Pipeline)
    pipe._on_ink = calls.append
    pipe.tracker = type("T", (), {"reset": lambda self: None})()
    pipe.corner_history = type("C", (), {"clear": lambda self: None})()
    pipe.reset_source(1.0)
    assert len(calls) == 1 and calls[0].confirmed is None


def test_reconnect_at_capacity_does_not_evict_healthy_consumer(monkeypatch):
    monkeypatch.setattr(stream, "MAX_CONSUMERS", 2)
    # Sem a thread de envio nao ha poda periodica: so a poda do _adopt age,
    # e o teste fica deterministico.
    monkeypatch.setattr(stream.InkStreamPublisher, "_send_loop", lambda self: None)
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    try:
        healthy = _connect(pub.port)
        assert _wait(lambda: len(pub._sinks) == 1)
        healthy_sink = pub._sinks[0]
        gone = _connect(pub.port)
        assert _wait(lambda: len(pub._sinks) == 2)
        gone_sink = pub._sinks[1]
        gone.close()
        time.sleep(0.05)
        again = _connect(pub.port)
        assert _wait(lambda: gone_sink not in pub._sinks and len(pub._sinks) == 2)
        assert healthy_sink in pub._sinks and len(pub._sinks) == 2
        healthy.settimeout(0.3)
        with pytest.raises(TimeoutError):
            healthy.recv(1)  # segue aberto, sem EOF
        healthy.close()
        again.close()
    finally:
        pub.close()


def test_recording_refused_without_private_files(monkeypatch, tmp_path):
    monkeypatch.setattr(stream, "_PRIVATE_FILES", False)
    path = tmp_path / "w.drawink"
    with pytest.raises(RuntimeError, match="privada"):
        stream.InkStreamPublisher(port=None, record_path=path, log=lambda *_: None)
    assert not path.exists()


@needs_private_files
def test_close_reports_recording_flush_failure(tmp_path):
    messages = []
    pub = stream.InkStreamPublisher(
        port=None, record_path=tmp_path / "x.drawink", log=messages.append
    )
    sink = pub._sinks[0]

    def boom():
        raise OSError("flush falhou")

    sink.close = boom
    pub.close()
    assert any("falha ao fechar" in m for m in messages)


# ------------------------------------------------ v2: captura, ancora e FULL

_QUAD = np.array([[1, 2], [31, 3], [30, 41], [2, 40]], np.float32)


def test_v2_full_and_delta_carry_capture_stamp_and_anchor():
    a, b = _mask(), _mask()
    b[2:5, 3:7] = 255
    data = stream.pack_full(1, 1, 10, a, (210.0, 297.0), 123_456, _QUAD)
    data += stream.pack_delta(1, 2, 20, b, stream.diff_box(a, b), 654_321)
    recs = list(stream.iter_records(io.BytesIO(data).read))
    assert [r.proto for r in recs] == [2, 2]
    canvas = stream.InkCanvas()
    canvas.apply(recs[0])
    assert canvas.captured_at_us == 123_456
    assert np.allclose(canvas.anchor_corners, _QUAD)
    canvas.apply(recs[1])
    assert canvas.captured_at_us == 654_321
    assert np.array_equal(canvas.mask, b)


def test_v2_without_capture_stamp_or_anchor_reads_as_none():
    data = stream.pack_full(1, 1, 10, _mask(), (210.0, 297.0))
    canvas, _ = _canvas_from(data)
    assert canvas.captured_at_us is None and canvas.anchor_corners is None


def test_v1_recordings_are_still_readable():
    mask = _mask(fill=255)
    payload = struct.pack("<ff", 210.0, 297.0) + zlib_compress(mask)
    header = stream.HEADER.pack(
        stream.MAGIC, 1, stream.KIND_FULL, 0, 3, 5, 99, 30, 40, len(payload)
    )
    canvas, _ = _canvas_from(header + payload)
    assert canvas.paper_mm == (210.0, 297.0)
    assert canvas.captured_at_us is None and canvas.anchor_corners is None
    assert (canvas.mask == 255).all() and canvas.generation == 3


def zlib_compress(mask):
    import zlib

    return zlib.compress(np.ascontiguousarray(mask).tobytes(), 1)


def _kinds(path):
    return [(r.kind, r.proto) for r in stream.records_from_file(path)]


@needs_private_files
def test_publication_metadata_reaches_the_wire(tmp_path):
    path = tmp_path / "rec.dral"
    mask = _mask()
    mask[1:3, 1:3] = 255
    pub = stream.InkStreamPublisher(port=None, record_path=str(path))
    pub.publish(
        contracts.InkPublication(1, 1, mask, captured_at=12.5, anchor_corners=_QUAD)
    )
    pub.close()
    canvas = stream.InkCanvas()
    for rec in stream.records_from_file(path):
        canvas.apply(rec)
    assert canvas.captured_at_us == 12_500_000
    assert np.allclose(canvas.anchor_corners, _QUAD)


@needs_private_files
def test_full_flag_forces_a_full_even_with_identical_mask(tmp_path):
    path = tmp_path / "rec.dral"
    mask = _mask()
    mask[1:3, 1:3] = 255
    pub = stream.InkStreamPublisher(port=None, record_path=str(path))
    calls = _count_dispatches(pub)
    pub.publish(contracts.InkPublication(1, 1, mask))
    _wait_dispatches(calls, 1)
    pub.publish(contracts.InkPublication(1, 2, mask))  # igual: nao gera registro
    _wait_dispatches(calls, 2)
    pub.publish(contracts.InkPublication(1, 3, mask, full=True))  # re-ancorou
    _wait_dispatches(calls, 3)
    pub.close()
    assert [k for k, _ in _kinds(path)] == [stream.KIND_FULL, stream.KIND_FULL]


def _count_dispatches(pub):
    """Conta despachos concluidos (o arquivo gravado e bufferizado ate o close)."""
    calls = []
    real = pub._dispatch

    def counted(*args):
        real(*args)
        calls.append(1)

    pub._dispatch = counted
    return calls


def _wait_dispatches(calls, n, timeout=3.0):
    deadline = time.monotonic() + timeout
    while len(calls) < n and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(calls) >= n, "o publicador nao despachou"


@needs_private_files
def test_delta_over_half_the_page_becomes_full(tmp_path):
    path = tmp_path / "rec.dral"
    small = _mask()
    small[0:2, 0:2] = 255
    big = small.copy()
    big[:, 0:20] = 255  # 2/3 da largura da pagina
    tiny = big.copy()
    tiny[30:33, 25:28] = 255
    pub = stream.InkStreamPublisher(port=None, record_path=str(path))
    calls = _count_dispatches(pub)
    for version, m in enumerate((small, big, tiny), start=1):
        pub.publish(contracts.InkPublication(1, version, m))
        _wait_dispatches(calls, version)
    pub.close()
    assert [k for k, _ in _kinds(path)] == [
        stream.KIND_FULL,
        stream.KIND_FULL,  # bbox grande -> FULL
        stream.KIND_DELTA,  # mudanca pequena segue DELTA
    ]


@needs_private_files
def test_latest_wins_keeps_the_full_request_of_a_replaced_publication():
    seen = []
    gate = threading.Event()
    import os
    import tempfile

    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        pub = stream.InkStreamPublisher(port=None, record_path=path)
        real = pub._dispatch

        def slow(*args):
            seen.append(args)
            if len(seen) == 1:
                gate.wait(2.0)
            return real(*args)

        pub._dispatch = slow
        mask = _mask()
        pub.publish(contracts.InkPublication(1, 1, mask))  # bloqueia no 1o dispatch
        deadline = time.monotonic() + 2.0
        while not seen and time.monotonic() < deadline:
            time.sleep(0.005)
        b = mask.copy()
        b[0, 0] = 255
        c = mask.copy()
        c[1, 1] = 255
        pub.publish(contracts.InkPublication(1, 2, b, full=True))
        pub.publish(contracts.InkPublication(1, 3, c))  # substitui a anterior
        gate.set()
        pub.close()
        assert seen[-1][-1] is True  # force_full sobreviveu a substituicao
        assert seen[-1][1] == 3
    finally:
        os.unlink(path)


def test_pipeline_ink_publication_takes_meta_only_when_asked():
    from draw import pipeline as pipeline_module

    class Delta:
        version = 7
        captured_at = 3.25
        anchor_corners = _QUAD
        full = True

    class Worker:
        calls = []

        def poll_delta(self, generation, after, wait=0.0):
            self.calls.append((generation, after, wait))
            return Delta()

    class Got:
        version = 4
        confirmed = _mask(fill=255)

    pipe = pipeline_module.Pipeline.__new__(pipeline_module.Pipeline)
    pipe.ink_worker = Worker()
    pipe.clean_generation = 2

    pub = pipe._ink_publication(Got())  # sem ink_meta: nao paga o poll_delta
    assert pub.captured_at is None and not pub.full and Worker.calls == []

    pipe._ink_meta = True
    pub = pipe._ink_publication(Got())
    assert (pub.captured_at, pub.full) == (3.25, True)
    assert np.allclose(pub.anchor_corners, _QUAD)
    assert Worker.calls == [(2, -1, 0.005)]
    assert pipe._delta_version == 7  # a versao confirmada propria do delta
    pipe._ink_publication(Got())
    assert Worker.calls[-1] == (2, 7, 0.005)

    pipe.ink_worker = object()  # trabalhadora sem poll_delta: sem metadados
    assert pipe._ink_publication(Got()).captured_at is None


def test_ink_publication_validates_metadata():
    with pytest.raises(ValueError):
        contracts.InkPublication(1, 1, _mask(), captured_at=float("nan"))
    with pytest.raises(ValueError):
        contracts.InkPublication(1, 1, _mask(), anchor_corners=np.zeros((3, 2)))
    pub = contracts.InkPublication(1, 1, _mask(), captured_at=1, anchor_corners=_QUAD)
    assert pub.captured_at == 1 and pub.anchor_corners.dtype == np.float32


def test_late_consumer_keeps_anchor_after_a_provisional_only_update():
    # Review #64: a atualizacao so do provisorio chega sem metadados e nao pode
    # apagar a ancora/captura que o FULL de um consumidor tardio precisa.
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    calls = _count_dispatches(pub)
    try:
        a = _mask()
        a[1:3, 1:3] = 255
        pub.publish(
            contracts.InkPublication(
                3, 7, a, captured_at=4.5, anchor_corners=_QUAD, full=True
            )
        )
        _wait_dispatches(calls, 1)
        pub.publish(contracts.InkPublication(3, 8, a))  # sem metadados
        _wait_dispatches(calls, 2)
        sock = _connect(pub.port)
        canvas = stream.InkCanvas()
        _read_until(sock, canvas, 7)
        sock.close()
        assert (
            canvas.ink_version == 8
        )  # o FULL tardio sai do ultimo estado (sem metadados)
        assert canvas.captured_at_us == 4_500_000
        assert np.allclose(canvas.anchor_corners, _QUAD)
        # outra geracao nao herda a ancora da anterior
        pub.publish(contracts.InkPublication(4, 1, a))
        _wait_dispatches(calls, 3)
        sock = _connect(pub.port)
        canvas = stream.InkCanvas()
        _read_until(sock, canvas, 1)
        sock.close()
        assert canvas.anchor_corners is None and canvas.captured_at_us is None
    finally:
        pub.close()


def test_full_discontinuity_does_not_inherit_the_old_anchor():
    # Review #64: a recuperacao de identidade zera a ancora da trabalhadora sem
    # mudar a geracao; o FULL do reset nao pode carregar a ancora velha.
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    calls = _count_dispatches(pub)
    try:
        a = _mask()
        a[1:3, 1:3] = 255
        pub.publish(
            contracts.InkPublication(
                3, 7, a, captured_at=4.5, anchor_corners=_QUAD, full=True
            )
        )
        _wait_dispatches(calls, 1)
        b = a.copy()
        b[10:12, 10:12] = 255
        pub.publish(contracts.InkPublication(3, 8, b, full=True))  # sem ancora agora
        _wait_dispatches(calls, 2)
        sock = _connect(pub.port)
        canvas = stream.InkCanvas()
        _read_until(sock, canvas, 8)
        sock.close()
        assert canvas.ink_version == 8
        assert canvas.anchor_corners is None and canvas.captured_at_us is None
    finally:
        pub.close()


def test_close_wakes_accept_loop_immediately(monkeypatch):
    entered = threading.Event()
    real_selector = stream.selectors.DefaultSelector

    class _WatchingSelector(real_selector):
        def select(self, timeout=None):
            entered.set()
            return super().select(timeout=timeout)

    monkeypatch.setattr(stream.selectors, "DefaultSelector", _WatchingSelector)
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    try:
        assert entered.wait(timeout=1.0)
        time.sleep(0.01)
        started = time.monotonic()
        pub.close()
        elapsed = time.monotonic() - started
        assert elapsed < 0.15
        assert not any(t.is_alive() for t in pub._threads)
        pub.close()  # segunda chamada e no-op seguro
    finally:
        pub.close()


@pytest.mark.skipif(
    sys.platform != "linux", reason="limite de 1024 FD do select.select e do Linux"
)
def test_accept_loop_handles_high_fd_on_linux():
    import resource

    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 1100:
        pytest.skip("RLIMIT_NOFILE < 1100")
    dummy_fds = []
    pub = sock = None
    try:
        while True:
            fd = os.dup(0)
            dummy_fds.append(fd)
            if fd >= 1024:
                break
        pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
        assert pub._wake_r.fileno() >= 1024
        a = _mask()
        a[1:3, 1:3] = 255
        _pub(pub, 1, 1, a)
        time.sleep(0.05)
        sock = _connect(pub.port)
        assert sock.fileno() >= 1024
        canvas = stream.InkCanvas()
        _read_until(sock, canvas, 1)
        assert np.array_equal(canvas.mask, a)
    finally:
        if sock is not None:
            sock.close()
        if pub is not None:
            pub.close()
        for fd in dummy_fds:
            try:
                os.close(fd)
            except OSError:
                pass


@needs_private_files
def test_clear_after_orientation_change_keeps_grid_and_mm_together(tmp_path):
    path = tmp_path / "rec.dral"
    pub = stream.InkStreamPublisher(port=None, record_path=str(path))
    pub.set_paper_mm((210.0, 297.0))
    portrait = _mask(40, 30)
    portrait[2:5, 2:5] = 255
    pub.publish(contracts.InkPublication(1, 1, portrait))
    # Orientacao muda; o clear seguinte ainda usa a grade retrato antiga.
    pub.set_paper_mm((297.0, 210.0))
    pub.publish(contracts.InkPublication(-1, 0, None))
    pub.close()
    canvas = stream.InkCanvas()
    for rec in stream.records_from_file(path):
        canvas.apply(rec)
    assert canvas.mask.shape == (40, 30)
    assert canvas.paper_mm == pytest.approx((210.0, 297.0))
