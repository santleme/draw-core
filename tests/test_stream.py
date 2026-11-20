"""Stream de tinta (draw.stream, bridge/STREAM.md)."""

import io
import socket
import struct
import threading
import time

import numpy as np
import pytest

from draw import stream
from draw.protocol import (
    ProtocolError,
    client_handshake,
)

TOKEN = "t" * 32


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


def test_publisher_socket_and_record_equal(tmp_path):
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
        pub.publish(1, 1, a)
        b = a.copy()
        b[2:6, 2:6] = 255
        time.sleep(0.1)
        pub.publish(1, 2, b)
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
        pub.publish(3, 7, a)
        time.sleep(0.1)
        sock = _connect(pub.port)
        canvas = stream.InkCanvas()
        _read_until(sock, canvas, 7)
        assert np.array_equal(canvas.mask, a)
        assert canvas.generation == 3
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
            pub.publish(1, i + 1, big ^ np.uint8(i))
        assert time.monotonic() - start < 0.5
        sock.close()
    finally:
        pub.close()


def test_no_consumer_publish_is_cheap_and_close_is_clean():
    pub = stream.InkStreamPublisher(port=0, token=TOKEN, log=lambda *_: None)
    for i in range(20):
        pub.publish(1, i + 1, _mask())
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
    assert calls[0][:2] == (2, 4)
    assert calls[0][2].min() == 255
    assert threading.active_count() >= 1
