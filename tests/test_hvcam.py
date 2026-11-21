"""Fonte hvcam:// — adaptador VideoCapture sobre o transporte hvcam.

O Camera ctypes em si depende de libhvcam.so (artefato do build do
santleme/hvcam) e nao e exercitado aqui: HvcamCapture recebe uma
camera_factory, e os testes injetam um fake com as mesmas chamadas
(grab/to_bgr/close). Cobertura: is_hvcam_source, parse_hvcam_endpoint,
o contrato read()/release() do adaptador e a fiacao de open_capture
e _check_hvcam_source.
"""

import errno

import cv2
import numpy as np
import pytest

from draw import app as app_module
from draw import camera as camera_module
from draw import hvcam as hvcam_module
from draw.hvcam import HvcamCapture, is_hvcam_source, parse_hvcam_endpoint


def _frame(width=64, height=48):
    return np.full((height, width, 3), 127, dtype=np.uint8)


def _payload(w=64, h=48, fmt=1):
    return np.zeros(h * w * 3, dtype=np.uint8)


_DEFAULT = object()


class FakeCamera:
    """Mesma superficie usada do Camera: grab/to_bgr/close."""

    def __init__(self, grabs, bgr_result=_DEFAULT, fail=None):
        self.kwargs = None
        self.grabs = list(grabs)
        self.bgr_result = _frame() if bgr_result is _DEFAULT else bgr_result
        self.fail = fail
        self.closed = False

    def grab(self, timeout_ms=1000):
        item = self.grabs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def to_bgr(self, info, payload):
        if isinstance(self.bgr_result, Exception):
            raise self.bgr_result
        return self.bgr_result

    def close(self):
        self.closed = True


def _factory(camera, recorder=None):
    def factory(**kwargs):
        if recorder is not None:
            recorder.append(kwargs)
        if camera.fail is not None:
            raise camera.fail
        return camera

    return factory


def _info(seq=1, ts_us=100_000):
    return {
        "seq": seq,
        "ts_us": ts_us,
        "flags": 1,
        "width": 64,
        "height": 48,
        "format": 1,
        "payload": 64 * 48 * 3,
    }


@pytest.mark.parametrize(
    "source,expected",
    [
        ("hvcam://", True),
        ("hvcam://:9100", True),
        ("hvcam+tcp://10.0.0.5", True),
        ("hvcam+reverse://", True),
        ("HVCAM://", True),
        ("hvcam://10.0.0.5", True),  # prefixo casa; o host e recusado no parse
        ("tcp://10.0.0.5", False),
        ("video.mp4", False),
        (0, False),
        (None, False),
    ],
)
def test_is_hvcam_source(source, expected):
    assert is_hvcam_source(source) is expected


@pytest.mark.parametrize(
    "url,expected",
    [
        ("hvcam://", (None, 8765)),
        ("hvcam://:", (None, 8765)),
        ("hvcam://:9100", (None, 9100)),
        ("hvcam://:70000", (None, 70000)),
        ("hvcam+tcp://10.0.0.5", ("10.0.0.5", 8765)),
        ("hvcam+tcp://10.0.0.5:9999", ("10.0.0.5", 9999)),
        ("hvcam+reverse://", (None, 8765)),
        ("hvcam+reverse://:", (None, 8765)),
        ("hvcam+reverse://:9100", (None, 9100)),
        ("hvcam+reverse://:70000", (None, 70000)),
    ],
)
def test_parse_hvcam_endpoint(url, expected):
    assert parse_hvcam_endpoint(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "hvcam://10.0.0.5",  # vsock nao tem host
        "hvcam+tcp://",  # tcp sem host
        "hvcam+tcp://10.0.0.5:70000",  # tcp e 16-bit
        "hvcam+reverse://10.0.0.5",  # reverse e listener sem hostname
        "hvcam://:0",
        "hvcam+reverse://:0",
        "hvcam+reverse://:4294967295",  # VMADDR_PORT_ANY reservado
        "hvcam+reverse://:4294967296",  # acima de 32 bits
    ],
)
def test_parse_hvcam_endpoint_invalido(url):
    with pytest.raises(ValueError):
        parse_hvcam_endpoint(url)


def test_hvcam_capture_reads_frame():
    cam = FakeCamera([(_info(seq=7, ts_us=250_000), _payload())])
    capture = HvcamCapture("hvcam://", camera_factory=_factory(cam))
    assert capture.isOpened()
    ok, frame = capture.read()
    assert ok and frame.shape == (48, 64, 3)
    assert capture.last_frame_id == 7
    assert capture.last_capture_ts_ms == 250
    capture.release()
    assert cam.closed


def test_hvcam_capture_vsock_port_passed():
    cam = FakeCamera([])
    seen = []
    HvcamCapture("hvcam://:9100", camera_factory=_factory(cam, seen))
    assert seen == [{"port": 9100}]


def test_hvcam_capture_reverse_uses_listener():
    cam = FakeCamera([])
    seen = []
    HvcamCapture("hvcam+reverse://:70000", camera_factory=_factory(cam, seen))
    assert seen == [{"port": 70000, "listen": True}]


def test_hvcam_capture_reverse_forwards_connect_timeout():
    cam = FakeCamera([])
    seen = []
    # timeout_ms de leitura (ex.: 5000 no --check) NAO encurta o prazo de
    # accept do listener reverso (35 s); so connect_timeout_ms explicito.
    HvcamCapture(
        "hvcam+reverse://:9100",
        timeout_ms=5000,
        camera_factory=_factory(cam, seen),
    )
    assert seen == [{"port": 9100, "listen": True}]
    seen.clear()
    HvcamCapture(
        "hvcam+reverse://:9100",
        connect_timeout_ms=12000,
        camera_factory=_factory(cam, seen),
    )
    assert seen == [{"port": 9100, "listen": True, "timeout_ms": 12000}]


def test_hvcam_capture_tcp_endpoint_passed():
    cam = FakeCamera([])
    seen = []
    HvcamCapture("hvcam+tcp://10.0.0.5:9999", camera_factory=_factory(cam, seen))
    assert seen == [{"host": "10.0.0.5", "port": 9999}]


def test_hvcam_capture_tcp_ipv6_endpoint_passed():
    cam = FakeCamera([])
    seen = []
    HvcamCapture("hvcam+tcp://[::1]:9999", camera_factory=_factory(cam, seen))
    assert seen == [{"host": "::1", "port": 9999}]


def test_hvcam_capture_counts_seq_gaps_as_source_dropped():
    cam = FakeCamera(
        [
            (_info(seq=10), _payload()),
            (_info(seq=13), _payload()),
            (_info(seq=14), _payload()),
        ]
    )
    capture = HvcamCapture("hvcam://", camera_factory=_factory(cam))
    assert capture.read()[0]
    assert capture.read()[0]
    assert capture.read()[0]
    assert capture.source_dropped == 2  # seqs 11 e 12 nunca chegaram
    capture.release()


@pytest.mark.parametrize("error", [TimeoutError("t"), OSError("stream dropped")])
def test_hvcam_capture_grab_errors_are_failed_reads(error):
    cam = FakeCamera([error, (_info(), _payload())])
    capture = HvcamCapture("hvcam://", camera_factory=_factory(cam))
    ok, frame = capture.read()
    assert not ok and frame is None
    ok, frame = capture.read()  # fonte segue viva depois do erro
    assert ok and frame is not None
    capture.release()


def test_hvcam_capture_placeholder_is_failed_read():
    cam = FakeCamera([(_info(), _payload())], bgr_result=None)
    capture = HvcamCapture("hvcam://", camera_factory=_factory(cam))
    ok, frame = capture.read()
    assert not ok and frame is None
    capture.release()


def test_hvcam_capture_bad_payload_is_failed_read():
    cam = FakeCamera(
        [(_info(), _payload()), (_info(), _payload())],
        bgr_result=ValueError("payload != esperado"),
    )
    capture = HvcamCapture("hvcam://", camera_factory=_factory(cam))
    ok, frame = capture.read()
    assert not ok and frame is None
    capture.release()


def test_hvcam_capture_unopened_when_connect_fails():
    cam = FakeCamera([], fail=OSError("connect failed (-110)"))
    capture = HvcamCapture("hvcam://", camera_factory=_factory(cam))
    assert not capture.isOpened()
    assert "connect failed" in str(capture.open_error)
    ok, frame = capture.read()
    assert not ok and frame is None


def test_hvcam_capture_unopened_on_bad_url():
    capture = HvcamCapture("hvcam://10.0.0.5", camera_factory=_factory(FakeCamera([])))
    assert not capture.isOpened()
    assert "vsock" in str(capture.open_error)


def test_hvcam_capture_release_without_open():
    capture = HvcamCapture("hvcam://10.0.0.5", camera_factory=_factory(FakeCamera([])))
    capture.release()  # nao levanta
    capture.release()


@pytest.mark.parametrize("source", ["hvcam://", "hvcam+reverse://"])
def test_open_capture_accepts_hvcam(monkeypatch, source):
    cam = FakeCamera([(_info(), _payload())])
    monkeypatch.setattr(
        camera_module,
        "HvcamCapture",
        lambda source: HvcamCapture(source, camera_factory=_factory(cam)),
    )
    capture = camera_module.open_capture(source)
    try:
        ok, frame = capture.read()
        assert ok and frame.shape == (48, 64, 3)
    finally:
        capture.release()


def test_open_capture_hvcam_raises_with_open_error(monkeypatch):
    cam = FakeCamera([], fail=OSError("libhvcam.so nao encontrada"))
    monkeypatch.setattr(
        camera_module,
        "HvcamCapture",
        lambda source: HvcamCapture(source, camera_factory=_factory(cam)),
    )
    # O erro real (lib ausente) aparece na mensagem — nunca um
    # "hvcam-host fora do ar" generico que mandaria a pessoa no caminho
    # errado.
    with pytest.raises(RuntimeError, match="libhvcam.so nao encontrada"):
        camera_module.open_capture("hvcam://")


def test_check_hvcam_source_reads_frame(monkeypatch, capsys):
    cam = FakeCamera([(_info(), _payload())])
    monkeypatch.setattr(
        app_module,
        "HvcamCapture",
        lambda source, timeout_ms=None: HvcamCapture(
            source, camera_factory=_factory(cam)
        ),
    )
    assert app_module._check_hvcam_source("hvcam://") == 1
    assert "quadro lido da fonte hvcam" in capsys.readouterr().out


def test_check_hvcam_source_fails_unopened(monkeypatch, capsys):
    cam = FakeCamera([], fail=OSError("connect failed (-110)"))
    monkeypatch.setattr(
        app_module,
        "HvcamCapture",
        lambda source, timeout_ms=None: HvcamCapture(
            source, camera_factory=_factory(cam)
        ),
    )
    assert app_module._check_hvcam_source("hvcam://") == 0
    out = capsys.readouterr().out
    assert "FAIL" in out and "connect failed" in out


class _FakeLib:
    """Conector nativo falso: registra host/porta exatos repassados."""

    def __init__(self, fd=3):
        self.calls = []
        self.fd = fd
        self.closed = []

    def hvcam_connect(self, port):
        self.calls.append(("vsock", port))
        return self.fd

    def hvcam_connect_tcp(self, host, port):
        self.calls.append((host, port))
        return self.fd

    def hvcam_grab_next(self, fd, hdr_ptr, buf, cap, last_ptr, timeout_ms):
        self.calls.append(("grab", fd, timeout_ms))
        hdr = hdr_ptr._obj
        hdr.seq_num = 1
        hdr.timestamp_us = 100_000
        hdr.flags = 1
        hdr.width = 4
        hdr.height = 2
        hdr.format = 1
        hdr.payload_bytes = 24
        return 24

    def hvcam_close(self, fd):
        self.closed.append(fd)


def test_camera_tcp_passes_host_and_port(monkeypatch):
    lib = _FakeLib()
    monkeypatch.setattr(hvcam_module, "_LIB", lib)
    cam = hvcam_module.Camera(host="::1", port=9999)
    # IPv6 chega intacto ao conector nativo — sem repartir ":"
    assert lib.calls == [(b"::1", 9999)]
    cam.close()
    assert lib.closed == [3]


def test_camera_vsock_connect_without_host(monkeypatch):
    lib = _FakeLib()
    monkeypatch.setattr(hvcam_module, "_LIB", lib)
    cam = hvcam_module.Camera(port=9100)
    assert lib.calls == [("vsock", 9100)]
    cam.close()


def test_camera_vsock_listen_handoff_to_grab_and_close(monkeypatch):
    """O descritor destacado pelo listener Python em modo reverso alimenta
    hvcam_grab_next e e fechado por hvcam_close sem exigir hvcam_listen na
    libhvcam.so (review #75)."""
    lib = _FakeLib()
    monkeypatch.setattr(hvcam_module, "_LIB", lib)
    seen = []

    def fake_listener(port, timeout_ms):
        seen.append((port, timeout_ms))
        return 42

    cam = hvcam_module.Camera(port=70000, listen=True, listener=fake_listener)
    assert seen == [(70000, hvcam_module._LISTEN_TIMEOUT_MS)]
    info, payload = cam.grab(timeout_ms=750)
    assert info["seq"] == 1 and payload.size == 24
    assert lib.calls == [("grab", 42, 750)]
    cam.close()
    assert lib.closed == [42]


def test_camera_connect_failure_raises_oserror(monkeypatch):
    lib = _FakeLib(fd=-1)
    monkeypatch.setattr(hvcam_module, "_LIB", lib)
    with pytest.raises(OSError, match="connect failed"):
        hvcam_module.Camera(port=8765)


def test_camera_timeout_suggests_reverse_mode(monkeypatch):
    lib = _FakeLib(fd=-errno.ETIMEDOUT)
    monkeypatch.setattr(hvcam_module, "_LIB", lib)
    with pytest.raises(OSError, match=r"hvcam\+reverse://:9100") as error:
        hvcam_module.Camera(port=9100)
    assert "--register-vsock --port 9100" in str(error.value)


def test_listen_vsock_bounded_times_out_and_filters_non_host_cid(monkeypatch):
    """O listener AF_VSOCK fecha no timeout (-ETIMEDOUT), rejeita conexoes
    que nao venham de VMADDR_CID_HOST (2) e entrega o fd destacado ao
    HvcamCapture.read()/release() (review #75)."""

    class _FakeConn:
        def __init__(self, fd):
            self._fd = fd
            self.closed = False
            self.blocking = None

        def setblocking(self, flag):
            self.blocking = flag

        def detach(self):
            return self._fd

        def close(self):
            self.closed = True

    class _FakeListener:
        def __init__(self, accepts):
            self._accepts = list(accepts)
            self.closed = False
            self.bound = None

        def setblocking(self, _flag):
            pass

        def bind(self, addr):
            self.bound = addr

        def listen(self, _backlog):
            pass

        def accept(self):
            return self._accepts.pop(0)

        def close(self):
            self.closed = True

    class _FakeSelector:
        def __init__(self, events):
            self._events = list(events)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            pass

        def register(self, *_a):
            pass

        def select(self, _timeout):
            return self._events.pop(0) if self._events else []

    # 1) Timeout sem conexao: fecha o listener e devolve -ETIMEDOUT.
    now = [100.0]
    listener_timeout = _FakeListener([])
    rc = hvcam_module._listen_vsock_bounded(
        8765,
        timeout_ms=500,
        socket_factory=lambda *_a: listener_timeout,
        selector_factory=lambda: _FakeSelector([[]]),
        clock=lambda: now[0],
    )
    assert rc == -errno.ETIMEDOUT
    assert listener_timeout.closed

    # 2) Conexao de CID != VMADDR_CID_HOST (ex.: loopback CID 1) e fechada;
    #    a seguinte, de VMADDR_CID_HOST (2), e aceita, destacada, lida via
    #    HvcamCapture.read() e fechada em release().
    rogue = _FakeConn(fd=11)
    host_conn = _FakeConn(fd=42)
    listener_ok = _FakeListener([(rogue, (1, 50000)), (host_conn, (2, 50001))])
    lib = _FakeLib()
    real_listen = hvcam_module._listen_vsock_bounded
    monkeypatch.setattr(hvcam_module, "_LIB", lib)
    monkeypatch.setattr(
        hvcam_module,
        "_listen_vsock_bounded",
        lambda port, timeout_ms=hvcam_module._LISTEN_TIMEOUT_MS: real_listen(
            port,
            timeout_ms=timeout_ms,
            socket_factory=lambda *_a: listener_ok,
            selector_factory=lambda: _FakeSelector([[True], [True]]),
            clock=lambda: now[0],
        ),
    )
    capture = HvcamCapture("hvcam+reverse://:70000")
    assert capture.isOpened()
    assert rogue.closed is True
    assert host_conn.closed is False
    assert host_conn.blocking is True
    assert listener_ok.closed is True
    ok, frame = capture.read()
    assert ok and frame.shape == (2, 4, 3)
    assert lib.calls == [("grab", 42, hvcam_module._TIMEOUT_MS)]
    capture.release()
    assert lib.closed == [42]


def test_camera_to_bgr_mjpeg_honors_bottom_up():
    top = np.zeros((24, 32, 3), dtype=np.uint8)
    bottom = np.full((24, 32, 3), 255, dtype=np.uint8)
    img = np.concatenate([top, bottom])
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    cam = hvcam_module.Camera.__new__(hvcam_module.Camera)
    info = {
        "seq": 1,
        "ts_us": 0,
        "flags": 3,  # bit0 presente + bit1 bottom-up
        "width": 32,
        "height": 48,
        "format": 4,
        "payload": encoded.size,
    }
    out = cam.to_bgr(info, np.frombuffer(encoded.tobytes(), dtype=np.uint8))
    expected = cv2.flip(cv2.imdecode(encoded, cv2.IMREAD_COLOR), 0)
    np.testing.assert_array_equal(out, expected)


def test_camera_to_bgr_rejects_wrong_payload_size():
    cam = hvcam_module.Camera.__new__(hvcam_module.Camera)
    info = {
        "seq": 1,
        "ts_us": 0,
        "flags": 1,
        "width": 64,
        "height": 48,
        "format": 1,
        "payload": 64 * 48 * 3 + 8,
    }
    with pytest.raises(ValueError, match="payload"):
        cam.to_bgr(info, np.zeros(64 * 48 * 3 + 8, dtype=np.uint8))


def test_pure_python_hvcam_lib_pulls_frame_over_socketpair(monkeypatch):
    """Sem libhvcam.so compilada, _PurePythonHvcamLib implementa o protocolo
    de fio (REQ 'Q' -> 36-byte header + payload) em Python puro."""
    import socket
    import struct

    monkeypatch.setattr(hvcam_module, "_LIB", None)
    monkeypatch.setattr(
        hvcam_module,
        "_load",
        lambda: (_ for _ in ()).throw(OSError("sem libhvcam.so")),
    )
    client_sock, server_sock = socket.socketpair()
    try:
        payload = np.arange(4 * 2 * 3, dtype=np.uint8).tobytes()
        header = struct.pack(
            "<IHHQQHHII",
            0x4856434D,  # HVCAM_MAGIC
            1,  # HVCAM_VERSION
            1,  # flags: valid
            42,  # seq_num
            999_000,  # timestamp_us
            4,  # width
            2,  # height
            1,  # format: BGR24
            len(payload),
        )
        fd = client_sock.detach()
        cam = hvcam_module.Camera(
            port=8765,
            listen=True,
            listener=lambda _port, _timeout: fd,
        )
        # O servidor responde ao byte 'Q' enviado por grab().
        import threading

        def serve():
            req = server_sock.recv(1)
            assert req == b"Q"
            server_sock.sendall(header + payload)

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        info, raw = cam.grab(timeout_ms=1000)
        t.join(timeout=1.0)
        assert info["seq"] == 42
        assert info["width"] == 4 and info["height"] == 2
        bgr = cam.to_bgr(info, raw)
        assert bgr.shape == (2, 4, 3)
        np.testing.assert_array_equal(bgr.reshape(-1), np.frombuffer(payload, np.uint8))
        cam.close()
    finally:
        server_sock.close()


def test_open_capture_on_wsl_without_dev_video0_falls_back_to_hvcam(monkeypatch):
    class _ClosedLocal:
        def isOpened(self):
            return False

        def release(self):
            pass

    cam = FakeCamera([(_info(), _payload())])
    tried = []

    def fake_hvcam(source):
        tried.append(source)
        if source == "hvcam+reverse://":
            return HvcamCapture(source, camera_factory=_factory(cam))
        return HvcamCapture(
            source,
            camera_factory=_factory(FakeCamera([], fail=OSError("no vsock service"))),
        )

    monkeypatch.setattr(camera_module, "is_wsl", lambda: True)
    monkeypatch.setattr(camera_module.os.path, "exists", lambda _p: False)
    monkeypatch.setattr(camera_module, "_video_nodes", lambda: [])
    monkeypatch.setattr(camera_module, "HvcamCapture", fake_hvcam)
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda _s: _ClosedLocal())

    capture = camera_module.open_capture(0)
    try:
        assert tried == ["hvcam://", "hvcam+reverse://"]
        ok, frame = capture.read()
        assert ok and frame.shape == (48, 64, 3)
    finally:
        capture.release()


def test_pure_python_hvcam_lib_tcp_loopback_end_to_end(monkeypatch):
    """Testa o caminho publico hvcam+tcp:// usando socket.socket (send/recv/close),
    compativel tambem com Winsock no Windows nativo."""
    import socket
    import struct
    import threading

    monkeypatch.setattr(hvcam_module, "_LIB", None)
    monkeypatch.setattr(
        hvcam_module,
        "_load",
        lambda: (_ for _ in ()).throw(OSError("sem libhvcam.so")),
    )
    payload = np.arange(4 * 2 * 3, dtype=np.uint8).tobytes()
    header = struct.pack(
        "<IHHQQHHII",
        0x4856434D,
        1,
        1,
        7,
        555_000,
        4,
        2,
        1,
        len(payload),
    )
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        try:
            req = conn.recv(1)
            assert req == b"Q"
            conn.sendall(header + payload)
        finally:
            conn.close()
            server.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    cap = HvcamCapture(f"hvcam+tcp://127.0.0.1:{port}", timeout_ms=2000)
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok and frame.shape == (2, 4, 3)
        assert cap.last_frame_id == 7
    finally:
        cap.release()
        t.join(timeout=2.0)


def test_run_check_on_wsl_without_dev_video_uses_hvcam_fallback(monkeypatch, capsys):
    cam = FakeCamera([(_info(), _payload())])
    seen = []

    def fake_hvcam(source, **kw):
        seen.append((source, kw))
        if source == "hvcam+reverse://":
            return HvcamCapture(source, camera_factory=_factory(cam))
        return HvcamCapture(
            source,
            camera_factory=_factory(FakeCamera([], fail=OSError("no service"))),
        )

    monkeypatch.setattr(app_module.sys, "platform", "linux")
    monkeypatch.setattr(camera_module.sys, "platform", "linux")
    monkeypatch.setattr(app_module, "_require_gui_backend", lambda: None)
    monkeypatch.setattr(app_module, "_check_models", lambda: True)
    monkeypatch.setattr(app_module, "_video_nodes", lambda: [])
    monkeypatch.setattr(camera_module, "_video_nodes", lambda: [])
    monkeypatch.setattr(camera_module.os.path, "exists", lambda _p: False)
    monkeypatch.setattr(app_module, "is_wsl", lambda: True)
    monkeypatch.setattr(camera_module, "is_wsl", lambda: True)
    monkeypatch.setattr(camera_module, "HvcamCapture", fake_hvcam)

    assert app_module.run_check(0) == 0
    assert cam.closed is True
    assert seen == [
        ("hvcam://", {"timeout_ms": 5000}),
        ("hvcam+reverse://", {"timeout_ms": 5000}),
    ]
    out = capsys.readouterr().out
    assert "quadro lido da fonte hvcam (hvcam+reverse://)" in out
