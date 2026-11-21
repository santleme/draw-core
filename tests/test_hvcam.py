"""Fonte hvcam:// — adaptador VideoCapture sobre o transporte hvcam.

O Camera ctypes em si depende de libhvcam.so (artefato do build do
santleme/hvcam) e nao e exercitado aqui: HvcamCapture recebe uma
camera_factory, e os testes injetam um fake com as mesmas chamadas
(grab/to_bgr/close). Cobertura: is_hvcam_source, parse_hvcam_endpoint,
o contrato read()/release() do adaptador e a fiacao de open_capture
e _check_hvcam_source.
"""

import errno
import os

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


def test_ensure_hvcam_host_downloads_verifies_and_caches(tmp_path):
    import hashlib
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(hvcam_module.HVCAM_HOST_ZIP_MEMBER, b"MZ-fake-hvcam-exe")
    zip_bytes = buf.getvalue()
    sha = hashlib.sha256(zip_bytes).hexdigest()

    calls = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def fake_urlopen(req, timeout=None):
        calls.append((req.full_url, timeout))
        return _Resp(zip_bytes)

    exe = hvcam_module.ensure_hvcam_host(
        cache_dir=tmp_path,
        urlopen=fake_urlopen,
        expected_sha256=sha,
        local_candidates=(),
    )
    assert exe == tmp_path / "hvcam-host.exe"
    assert exe.read_bytes() == b"MZ-fake-hvcam-exe"
    assert len(calls) == 1

    # Segunda chamada reutiliza o cache sem baixar de novo.
    exe2 = hvcam_module.ensure_hvcam_host(
        cache_dir=tmp_path,
        urlopen=fake_urlopen,
        expected_sha256=sha,
        local_candidates=(),
    )
    assert exe2 == exe
    assert len(calls) == 1


def test_ensure_hvcam_host_rejects_bad_sha256(tmp_path):
    import io

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    with pytest.raises(RuntimeError, match="sha256 divergente"):
        hvcam_module.ensure_hvcam_host(
            cache_dir=tmp_path,
            urlopen=lambda _req, timeout=None: _Resp(b"corrupted"),
            expected_sha256="0" * 64,
            local_candidates=(),
        )
    assert not (tmp_path / "hvcam-host.exe").exists()


def test_list_hvcam_devices_parses_cp1252_stderr(monkeypatch, tmp_path):
    raw = b"[0] 1200W Camera\r\n[1] POCO M3 (C\xe2mera Virtual)\r\n"
    monkeypatch.setattr(
        hvcam_module.subprocess,
        "check_output",
        lambda *_a, **_k: raw,
    )
    devices = hvcam_module.list_hvcam_devices(tmp_path / "hvcam-host.exe")
    assert devices == [(0, "1200W Camera"), (1, "POCO M3 (Câmera Virtual)")]


def test_listen_vsock_bounded_calls_on_listening_and_aborts():
    events = []
    aborted = [False]

    class _FakeListener:
        def setblocking(self, _flag):
            pass

        def bind(self, _addr):
            events.append("bind")

        def listen(self, _backlog):
            events.append("listen")

        def close(self):
            events.append("close")

    class _FakeSelector:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def register(self, *_):
            pass

        def select(self, _timeout):
            aborted[0] = True
            return []

    def on_listening():
        events.append("spawn")

    rc = hvcam_module._listen_vsock_bounded(
        8765,
        timeout_ms=10_000,
        socket_factory=lambda *_a: _FakeListener(),
        selector_factory=_FakeSelector,
        on_listening=on_listening,
        abort_check=lambda: aborted[0],
    )
    assert rc == -errno.ECONNABORTED
    assert events == ["bind", "listen", "spawn", "close"]


def test_open_managed_reverse_falls_back_to_working_device_and_cleans_up(tmp_path):
    fake_exe = tmp_path / "hvcam-host.exe"
    fake_exe.write_bytes(b"MZ")
    procs = []

    class _FakeProc:
        def __init__(self, cmd):
            self.cmd = cmd
            self.terminated = False
            self.stdout = None
            self.stderr = None
            self.stdin = None
            self._rc = None

        def poll(self):
            return self._rc

        def terminate(self):
            self.terminated = True
            self._rc = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.terminated = True
            self._rc = -9

    def fake_popen(cmd, **_kw):
        proc = _FakeProc(cmd)
        procs.append(proc)
        return proc

    class _FakeManagedCap:
        def __init__(self, url, **kw):
            self.url = url
            self._host_proc = None
            kw["on_listening"]()
            self._device = procs[-1].cmd[-1]
            # Dispositivo 0 abre mas nao entrega quadros; dispositivo 1 abre e entrega.
            self._opened = self._device in ("0", "1")

        def isOpened(self):
            return self._opened

        def read(self):
            if self._device == "1":
                return True, _frame()
            return False, None

        def release(self):
            self._opened = False
            if self._host_proc is not None:
                hvcam_module._terminate_host_process(self._host_proc)
                self._host_proc = None

    cap, dev_idx, dev_name = hvcam_module.open_managed_reverse(
        device=0,
        try_all=True,
        host_exe=fake_exe,
        devices=[(0, "Frameless Cam"), (1, "Working Cam")],
        popen=fake_popen,
        capture_cls=_FakeManagedCap,
    )
    assert cap is not None
    assert (dev_idx, dev_name) == (1, "Working Cam")
    # Dispositivo 0: 720p e depois nativo (ambos sem quadros), entao dispositivo 1.
    assert len(procs) == 3
    assert procs[0].terminated is True
    assert procs[1].terminated is True
    assert procs[2].terminated is False

    cap.release()
    assert procs[2].terminated is True


def test_windows_cache_dir_handles_accented_utf8_profile(monkeypatch):
    def fake_check_output(cmd, **_kw):
        if cmd[0] == "powershell.exe":
            return "C:\\Users\\João\\AppData\\Local\r\n".encode("utf-8")
        if cmd[0] == "wslpath":
            assert cmd[1] == "C:\\Users\\João\\AppData\\Local"
            return "/mnt/c/Users/João/AppData/Local\n".encode("utf-8")
        raise AssertionError(cmd)

    monkeypatch.setattr(hvcam_module.subprocess, "check_output", fake_check_output)
    result = hvcam_module._windows_cache_dir()
    assert result is not None
    assert str(result).endswith("/Users/João/AppData/Local/draw")


def test_try_other_local_cameras_skips_frameless_candidate(monkeypatch):
    class _FakeLocalCap:
        def __init__(self, idx, has_frame):
            self.idx = idx
            self._has_frame = has_frame
            self._opened = True
            self.released = False

        def isOpened(self):
            return self._opened

        def set(self, *_):
            return True

        def read(self):
            if self._has_frame:
                return True, np.full((32, 32, 3), 100, dtype=np.uint8)
            return False, None

        def release(self):
            self._opened = False
            self.released = True

    caps = {1: _FakeLocalCap(1, False), 2: _FakeLocalCap(2, True)}
    monkeypatch.setattr(camera_module.sys, "platform", "linux")
    monkeypatch.setattr(
        camera_module, "_video_nodes", lambda: ["/dev/video1", "/dev/video2"]
    )
    chosen = camera_module._try_other_local_cameras(0, lambda idx: caps[idx])
    assert chosen is caps[2]
    assert caps[1].released is True


def test_try_other_local_cameras_accepts_camera_that_recovers_after_missed_read(
    monkeypatch,
):
    class _RecoveringCap:
        def __init__(self):
            self._calls = 0
            self.released = False

        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def read(self):
            self._calls += 1
            if self._calls == 2:
                return False, None  # falha pontual no warm-up
            return True, np.full((32, 32, 3), 100, dtype=np.uint8)

        def release(self):
            self.released = True

    recovering = _RecoveringCap()
    monkeypatch.setattr(camera_module.sys, "platform", "linux")
    monkeypatch.setattr(camera_module, "_video_nodes", lambda: ["/dev/video1"])
    chosen = camera_module._try_other_local_cameras(0, lambda _idx: recovering)
    assert chosen is recovering
    assert recovering.released is False


def test_confirm_stream_alive_recovers_after_short_gap_and_fails_on_stall():
    now = [0.0]

    def fake_clock():
        return now[0]

    def fake_sleep(dt):
        now[0] += dt

    class _GapCap:
        def __init__(self, resume_at):
            self._resume_at = resume_at

        def read(self):
            if self._resume_at is not None and now[0] >= self._resume_at:
                return True, np.full((16, 16, 3), 100, dtype=np.uint8)
            return False, None

    # Falhas imediatas por 30 ms e retomada em seguida -> aceita.
    assert (
        camera_module._confirm_stream_alive(
            _GapCap(resume_at=0.03),
            window_s=0.35,
            retry_s=0.005,
            now=fake_clock,
            sleep=fake_sleep,
        )
        is True
    )
    assert now[0] < 0.1

    # Camera permanentemente travada -> encerra no teto da janela.
    now[0] = 0.0
    assert (
        camera_module._confirm_stream_alive(
            _GapCap(resume_at=None),
            window_s=0.05,
            retry_s=0.01,
            now=fake_clock,
            sleep=fake_sleep,
        )
        is False
    )
    assert now[0] >= 0.05


def test_confirm_stream_alive_bounds_blocking_backend():
    now = [0.0]
    reads = []

    def fake_clock():
        return now[0]

    def fake_sleep(dt):
        now[0] += dt

    class _BlockingCap:
        def read(self):
            reads.append(now[0])
            now[0] += 0.03
            return False, None

    # Janela 50 ms, retry 10 ms: primeira leitura em 0.0 termina em 0.03.
    # Sleep de 10 ms leva a 0.04 (< 0.05). Segunda leitura inicia em 0.04
    # e termina em 0.07 (ultrapassa a janela). O loop nao deve disparar
    # uma terceira leitura no/apos o prazo de 50 ms.
    alive = camera_module._confirm_stream_alive(
        _BlockingCap(),
        window_s=0.05,
        retry_s=0.01,
        now=fake_clock,
        sleep=fake_sleep,
    )
    assert alive is False
    assert len(reads) == 2
    assert all(t < 0.05 for t in reads)


def test_confirm_stream_alive_does_not_read_at_deadline():
    now = 0.0
    reads = []

    def sleep(delay):
        nonlocal now
        now += delay

    class Capture:
        def read(self):
            reads.append(now)
            return False, None

    assert not camera_module._confirm_stream_alive(
        Capture(), window_s=0.125, retry_s=0.25, now=lambda: now, sleep=sleep
    )
    assert now == 0.125
    assert reads == [0.0]


def test_open_managed_reverse_allows_host_to_write_large_stderr(tmp_path):
    import subprocess
    import sys

    procs = []

    def popen(_cmd, **kwargs):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stderr.buffer.write(b'x' * 2**20)"],
            **kwargs,
        )
        procs.append(proc)
        return proc

    class ConnectedCapture:
        def __init__(self, _url, **kwargs):
            kwargs["on_listening"]()

        def isOpened(self):
            return True

        def release(self):
            hvcam_module._terminate_host_process(self._host_proc)

    cap, _, _ = hvcam_module.open_managed_reverse(
        host_exe=tmp_path / "hvcam-host.exe",
        devices=[],
        popen=popen,
        capture_cls=ConnectedCapture,
    )
    try:
        assert procs[0].wait(timeout=3) == 0
    finally:
        cap.release()


def test_open_managed_reverse_surfaces_host_stderr_on_abort(tmp_path):
    fake_exe = tmp_path / "hvcam-host.exe"
    fake_exe.write_bytes(b"MZ")

    class _FailingProc:
        def __init__(self, cmd):
            self.cmd = cmd
            self._rc = 1
            self.stdout = None
            self.stdin = None
            self.stderr = None

        def poll(self):
            return self._rc

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 1

        def kill(self):
            pass

    def fake_popen(cmd, *, stderr, **_kw):
        stderr.write(b"x" * 32_768 + b"\nFailed to open camera: 0x80070490\n")
        return _FailingProc(cmd)

    class _AbortedCap:
        def __init__(self, url, **kw):
            self.url = url
            self._host_proc = None
            kw["on_listening"]()
            assert kw["abort_check"]() is True
            self._opened = False
            self.open_error = OSError("aborted")

        def isOpened(self):
            return False

        def release(self):
            pass

    with pytest.raises(RuntimeError) as exc_info:
        hvcam_module.open_managed_reverse(
            device=0,
            try_all=False,
            host_exe=fake_exe,
            popen=fake_popen,
            capture_cls=_AbortedCap,
        )
    assert str(exc_info.value).endswith("Failed to open camera: 0x80070490")
    assert len(str(exc_info.value)) <= 16_384


def test_try_other_local_cameras_skips_camera_that_stalls_permanently(monkeypatch):
    class _StalledAfterOneCap:
        def __init__(self):
            self._calls = 0
            self.released = False

        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def read(self):
            self._calls += 1
            if self._calls == 1:
                return True, np.full((32, 32, 3), 100, dtype=np.uint8)
            return False, None  # morre permanentemente

        def release(self):
            self.released = True

    class _GoodCap:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def read(self):
            return True, np.full((32, 32, 3), 100, dtype=np.uint8)

        def release(self):
            self.released = True

    stalled = _StalledAfterOneCap()
    good = _GoodCap()
    caps = {1: stalled, 2: good}
    monkeypatch.setattr(camera_module.sys, "platform", "linux")
    monkeypatch.setattr(
        camera_module, "_video_nodes", lambda: ["/dev/video1", "/dev/video2"]
    )
    chosen = camera_module._try_other_local_cameras(0, lambda idx: caps[idx])
    assert chosen is good
    assert stalled.released is True


def test_run_check_succeeds_without_fail_message_when_video1_works(monkeypatch, capsys):
    monkeypatch.setattr(app_module.sys, "platform", "linux")
    monkeypatch.setattr(app_module, "_require_gui_backend", lambda: None)
    monkeypatch.setattr(app_module, "_check_models", lambda: True)
    monkeypatch.setattr(
        app_module, "_video_nodes", lambda: ["/dev/video0", "/dev/video1"]
    )
    monkeypatch.setattr(app_module.os, "access", lambda p, _m: p == "/dev/video1")
    monkeypatch.setattr(
        app_module,
        "_check_camera_open",
        lambda idx, _label: idx == 1,
    )
    assert app_module.run_check(0) == 0
    assert "FAIL:" not in capsys.readouterr().out


def test_open_wsl_hvcam_fallback_does_not_use_manual_host_for_nonzero_source(
    monkeypatch,
):
    seen = []

    def fake_hvcam(source, **kw):
        seen.append(source)
        return HvcamCapture(
            source,
            camera_factory=_factory(FakeCamera([], fail=OSError("no host"))),
        )

    monkeypatch.setattr(camera_module.sys, "platform", "linux")
    monkeypatch.setattr(camera_module, "is_wsl", lambda: True)
    monkeypatch.setattr(camera_module.os.path, "exists", lambda _p: False)
    monkeypatch.setattr(camera_module, "_video_nodes", lambda: [])
    monkeypatch.setattr(camera_module, "HvcamCapture", fake_hvcam)

    candidate, cap = camera_module.open_wsl_hvcam_fallback(1)
    assert (candidate, cap) == (None, None)
    assert seen == []


def _strip_jpeg_dht(raw_jpeg):
    out = bytearray()
    i = 0
    while i < len(raw_jpeg):
        if raw_jpeg[i] == 0xFF and i + 1 < len(raw_jpeg):
            marker = raw_jpeg[i + 1]
            if marker == 0xDA:
                out.extend(raw_jpeg[i:])
                break
            if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
                out.extend(raw_jpeg[i : i + 2])
                i += 2
                continue
            seg_len = int.from_bytes(raw_jpeg[i + 2 : i + 4], "big")
            if marker != 0xC4:
                out.extend(raw_jpeg[i : i + 2 + seg_len])
            i += 2 + seg_len
        else:
            out.append(raw_jpeg[i])
            i += 1
    return bytes(out)


def test_sanitize_mjpeg_trims_uvc_padding_and_injects_missing_dht():
    img = np.full((64, 96, 3), (40, 120, 210), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    raw = encoded.tobytes()

    # 1. Padding UVC antes do SOI e apos o EOI (causa comum de "extraneous bytes").
    padded = b"\x00\x00" + raw + b"\x00" * 64
    clean_padded = hvcam_module._sanitize_mjpeg(np.frombuffer(padded, dtype=np.uint8))
    assert clean_padded[:2].tobytes() == b"\xff\xd8"
    assert clean_padded[-2:].tobytes() == b"\xff\xd9"
    dec1 = cv2.imdecode(clean_padded, cv2.IMREAD_COLOR)
    assert dec1 is not None and dec1.shape == img.shape

    # 2. MJPEG sem segmento DHT (padrao UVC/RFC 2435): DHT padrao e injetado.
    no_dht = _strip_jpeg_dht(raw) + b"\x00" * 16
    cam = object.__new__(hvcam_module.Camera)
    dec2 = cam.to_bgr(
        {"width": 96, "height": 64, "format": 4, "flags": 1},
        np.frombuffer(no_dht, dtype=np.uint8),
    )
    assert dec2 is not None and dec2.shape == img.shape
    assert np.allclose(dec2.mean(axis=(0, 1)), (40, 120, 210), atol=5.0)


def test_sanitize_mjpeg_rejects_truncated_payload_before_decode():
    img = np.full((32, 48, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    truncated = encoded.tobytes()[:-10]  # Remove o marcador EOI (0xFFD9).
    with pytest.raises(ValueError, match="EOI"):
        hvcam_module._sanitize_mjpeg(np.frombuffer(truncated, dtype=np.uint8))


def test_open_managed_reverse_requests_720p_and_falls_back_to_nv12_on_mjpeg_errors(
    tmp_path,
):
    fake_exe = tmp_path / "hvcam-host.exe"
    fake_exe.write_bytes(b"MZ")
    cmds = []

    class _FakeProc:
        def __init__(self, cmd):
            self.cmd = cmd
            self.stdout = None
            self.stderr = None
            self.stdin = None
            self._rc = None

        def poll(self):
            return self._rc

        def terminate(self):
            self._rc = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self._rc = -9

    def fake_popen(cmd, **_kw):
        cmds.append(list(cmd))
        return _FakeProc(cmd)

    class _MjpegBrokenCap:
        def __init__(self, url, **kw):
            self.url = url
            self._host_proc = None
            self.mjpeg_decode_errors = 0
            kw["on_listening"]()
            self._cmd = cmds[-1]
            self._opened = True

        def isOpened(self):
            return self._opened

        def read(self):
            if "--format" in self._cmd and "nv12" in self._cmd:
                return True, _frame()
            self.mjpeg_decode_errors += 1
            return False, None

        def release(self):
            self._opened = False

    cap, dev_idx, dev_name = hvcam_module.open_managed_reverse(
        device=0,
        try_all=True,
        host_exe=fake_exe,
        devices=[(0, "Corrupt MJPEG Cam")],
        popen=fake_popen,
        capture_cls=_MjpegBrokenCap,
    )
    assert cap is not None
    assert (dev_idx, dev_name) == (0, "Corrupt MJPEG Cam")
    assert len(cmds) == 2
    # Primeira tentativa pede 1280x720@30fps.
    assert "--width" in cmds[0] and "1280" in cmds[0]
    assert "--height" in cmds[0] and "720" in cmds[0]
    assert "--fps" in cmds[0] and "30" in cmds[0]
    assert cmds[0][-2:] == ["--device", "0"]
    # Segunda tentativa cai para NV12 (sem JPEG) no mesmo dispositivo.
    assert "--format" in cmds[1] and "nv12" in cmds[1]
    assert cmds[1][-2:] == ["--device", "0"]


def test_premature_mjpeg_scan_with_eoi_rejected_by_real_decoder_and_triggers_nv12(
    tmp_path,
):
    """Scan MJPEG cortado no meio mantendo FF D9 no fim (imagem metade-cinza
    com aviso do libjpeg) deve ser rejeitado pelo caminho real de `to_bgr` e
    acionar o fallback NV12 em `open_managed_reverse`."""
    img = np.random.default_rng(42).integers(0, 255, (128, 192, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    raw = encoded.tobytes()
    sos = raw.find(b"\xff\xda")
    assert sos > 0
    half_scan_with_eoi = raw[: sos + (len(raw) - sos) // 2] + b"\xff\xd9"
    arr = np.frombuffer(half_scan_with_eoi, dtype=np.uint8)

    # 1. Caminho real de Camera.to_bgr rejeita o quadro metade-cinza.
    real_cam = object.__new__(hvcam_module.Camera)
    with pytest.raises(ValueError, match="corrompido"):
        real_cam.to_bgr({"width": 192, "height": 128, "format": 4, "flags": 1}, arr)

    # 2. Acoplado no HvcamCapture + open_managed_reverse, cai para NV12.
    fake_exe = tmp_path / "hvcam-host.exe"
    fake_exe.write_bytes(b"MZ")
    cmds = []

    class _FakeProc:
        def __init__(self, cmd):
            self.cmd = cmd
            self.stdout = self.stderr = self.stdin = None
            self._rc = None

        def poll(self):
            return self._rc

        def terminate(self):
            self._rc = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self._rc = -9

    def fake_popen(cmd, **_kw):
        cmds.append(list(cmd))
        return _FakeProc(cmd)

    class _StreamCam:
        def __init__(self, cmd):
            self._cmd = cmd
            self._seq = 0

        def grab(self, _timeout_ms):
            self._seq += 1
            if "--format" in self._cmd and "nv12" in self._cmd:
                nv12 = np.full((64 * 3 // 2) * 64, 128, dtype=np.uint8)
                return {
                    "seq": self._seq,
                    "ts_us": self._seq * 33_000,
                    "flags": 1,
                    "width": 64,
                    "height": 64,
                    "format": 3,
                }, nv12
            return {
                "seq": self._seq,
                "ts_us": self._seq * 33_000,
                "flags": 1,
                "width": 192,
                "height": 128,
                "format": 4,
            }, arr

        def to_bgr(self, info, payload):
            return real_cam.to_bgr(info, payload)

        def close(self):
            pass

    def _cap_factory(url, **kw):
        def _cam_factory(**_cam_kw):
            kw["on_listening"]()
            return _StreamCam(cmds[-1])

        return HvcamCapture(url, camera_factory=_cam_factory)

    cap, dev_idx, _name = hvcam_module.open_managed_reverse(
        device=0,
        try_all=True,
        host_exe=fake_exe,
        devices=[(0, "Half-Gray MJPEG Cam")],
        popen=fake_popen,
        capture_cls=_cap_factory,
    )
    assert cap is not None
    assert dev_idx == 0
    assert len(cmds) == 2
    assert "--format" in cmds[1] and "nv12" in cmds[1]


def test_open_managed_reverse_negotiates_native_30fps_when_720p_is_15fps(tmp_path):
    """Quando 720p entrega apenas 15 FPS e o modo nativo (ex.: 1080p) entrega
    30 FPS, `open_managed_reverse` detecta a taxa baixa no probe e prefere o
    modo de 30 FPS."""
    fake_exe = tmp_path / "hvcam-host.exe"
    fake_exe.write_bytes(b"MZ")
    cmds = []

    class _FakeProc:
        def __init__(self, cmd):
            self.cmd = cmd
            self.stdout = self.stderr = self.stdin = None
            self._rc = None

        def poll(self):
            return self._rc

        def terminate(self):
            self._rc = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self._rc = -9

    def fake_popen(cmd, **_kw):
        cmds.append(list(cmd))
        return _FakeProc(cmd)

    class _TwoModeCap:
        def __init__(self, url, **kw):
            self.url = url
            self._host_proc = None
            self.mjpeg_decode_errors = 0
            self.last_capture_ts_ms = None
            kw["on_listening"]()
            self._cmd = cmds[-1]
            self._opened = True
            self._n = 0

        def isOpened(self):
            return self._opened

        def read(self):
            self._n += 1
            # 720p -> 15 FPS (67 ms/frame); nativo -> 30 FPS (33 ms/frame).
            step_ms = 67 if "--width" in self._cmd else 33
            self.last_capture_ts_ms = self._n * step_ms
            return True, _frame()

        def release(self):
            self._opened = False

    cap, dev_idx, _name = hvcam_module.open_managed_reverse(
        device=0,
        try_all=True,
        host_exe=fake_exe,
        devices=[(0, "Two-Mode Cam")],
        popen=fake_popen,
        capture_cls=_TwoModeCap,
    )
    assert cap is not None
    assert dev_idx == 0
    assert len(cmds) == 2
    assert "--width" in cmds[0]
    assert "--width" not in cmds[1]
    assert "--width" not in cap._cmd


def test_probe_rejects_mjpeg_stream_after_any_decode_error():
    class _IntermittentMjpegCap:
        def __init__(self):
            self.calls = 0
            self.mjpeg_decode_errors = 0
            self.last_capture_ts_ms = None

        def read(self):
            self.calls += 1
            if self.calls == 1:
                self.mjpeg_decode_errors += 1
                return False, None
            self.last_capture_ts_ms = self.calls * 33
            return True, _frame()

    ok, _fps = hvcam_module._probe_hvcam_stream(_IntermittentMjpegCap(), attempts=3)
    assert ok is False


def test_sanitize_mjpeg_zero_copy_on_clean_ndarray():
    img = np.full((32, 48, 3), 150, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    arr = np.frombuffer(encoded.tobytes(), dtype=np.uint8)
    assert hvcam_module._sanitize_mjpeg(arr) is arr


class _ScriptedProc:
    def __init__(self, cmd):
        self.cmd = cmd
        self.stdout = self.stderr = self.stdin = None
        self._rc = None

    def poll(self):
        return self._rc

    def terminate(self):
        self._rc = 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._rc = -9


def _scripted_open(tmp_path, behave):
    """Roda open_managed_reverse com uma camera cujo comportamento depende do
    comando e do indice da tentativa: behave(cmd, n) -> (corrupto, passo_ms)."""
    fake_exe = tmp_path / "hvcam-host.exe"
    fake_exe.write_bytes(b"MZ")
    cmds = []

    def fake_popen(cmd, **_kw):
        cmds.append(list(cmd))
        return _ScriptedProc(cmd)

    class _Cap:
        def __init__(self, url, **kw):
            self.mjpeg_decode_errors = 0
            self.last_capture_ts_ms = None
            kw["on_listening"]()
            self._cmd = cmds[-1]
            self._corrupt, self._step = behave(self._cmd, len(cmds))
            self._opened = True
            self._n = 0

        def isOpened(self):
            return self._opened

        def read(self):
            if self._corrupt:
                self.mjpeg_decode_errors += 1
                return False, None
            self._n += 1
            self.last_capture_ts_ms = self._n * self._step
            return True, _frame()

        def release(self):
            self._opened = False

    cap, _idx, _name = hvcam_module.open_managed_reverse(
        device=0,
        try_all=True,
        host_exe=fake_exe,
        devices=[(0, "Scripted Cam")],
        popen=fake_popen,
        capture_cls=_Cap,
    )
    return cap, cmds


def test_reopened_720p_with_corrupt_mjpeg_falls_back_to_raw(tmp_path):
    def behave(cmd, n):
        raw = "--format" in cmd
        if n == 1:
            return False, 67  # 720p MJPEG limpo, 15 FPS
        if "--width" not in cmd:
            return False, 67  # nativo: sem ganho
        return (not raw), 67  # 720p reaberto: MJPEG corrompido, raw ok

    cap, cmds = _scripted_open(tmp_path, behave)
    assert cap is not None
    assert "--format" in cap._cmd and "--width" in cap._cmd


def test_native_raw_slower_than_clean_720p_is_not_adopted(tmp_path):
    def behave(cmd, n):
        if "--width" in cmd and "--format" not in cmd:
            return False, 67  # 720p MJPEG limpo, 15 FPS
        if "--width" not in cmd and "--format" not in cmd:
            return True, 67  # nativo MJPEG corrompido
        if "--width" not in cmd:
            return False, 200  # nativo raw: 5 FPS
        return False, 67

    cap, _cmds = _scripted_open(tmp_path, behave)
    assert cap is not None
    assert "--width" in cap._cmd and "--format" not in cap._cmd


def _open_fd_count():
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="requer /proc")
def test_strict_decoder_does_not_leak_fd_when_tempfile_fails(monkeypatch):
    img = np.full((16, 16, 3), 90, dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok

    def boom(*_a, **_k):
        raise OSError("sem espaco")

    monkeypatch.setattr(hvcam_module.tempfile, "TemporaryFile", boom)
    before = _open_fd_count()
    for _ in range(20):
        hvcam_module._imdecode_mjpeg_strict(enc)
    assert _open_fd_count() == before


def _jpeg_with_tail_before_eoi(tail):
    img = np.random.default_rng(7).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    raw = enc.tobytes()
    assert raw.endswith(b"\xff\xd9")
    return np.frombuffer(raw[:-2] + tail + b"\xff\xd9", dtype=np.uint8)


def test_strict_decoder_accepts_real_padding_before_eoi():
    buf = _jpeg_with_tail_before_eoi(b"\x00" * 40)
    assert hvcam_module._imdecode_mjpeg_strict(buf).shape == (96, 96, 3)


def test_strict_decoder_rejects_extraneous_scan_bytes_before_eoi():
    # Bytes de dados (nao padding) antes do EOI: libjpeg avisa "extraneous
    # bytes", mas a estrutura independente mostra que nao e padding.
    buf = _jpeg_with_tail_before_eoi(bytes(range(1, 41)))
    with pytest.raises(ValueError):
        hvcam_module._imdecode_mjpeg_strict(buf)


def test_classify_warning_is_per_line():
    pad = np.frombuffer(b"\x01\x02" + b"\x00" * 8 + b"\xff\xd9", dtype=np.uint8)
    ext = "Corrupt JPEG data: 8 extraneous bytes before marker 0xd9"
    assert hvcam_module._classify_mjpeg_warning(ext, pad) == "padding"
    bad = "Corrupt JPEG data: bad Huffman code"
    assert hvcam_module._classify_mjpeg_warning(bad, pad) == "corrupt"
    wrong_n = "Corrupt JPEG data: 9 extraneous bytes before marker 0xd9"
    assert hvcam_module._classify_mjpeg_warning(wrong_n, pad) == "corrupt"


def test_native_raw_tries_yuy2_when_nv12_is_too_slow(tmp_path):
    def behave(cmd, n):
        fmt = cmd[cmd.index("--format") + 1] if "--format" in cmd else None
        if "--width" in cmd and fmt is None:
            return False, 67  # 720p MJPEG limpo, 15 FPS
        if "--width" not in cmd and fmt is None:
            return True, 67  # nativo MJPEG corrompido
        if "--width" not in cmd:
            return False, 200 if fmt == "nv12" else 33
        return False, 67

    cap, _cmds = _scripted_open(tmp_path, behave)
    assert cap is not None
    assert "yuy2" in cap._cmd and "--width" not in cap._cmd


def test_slow_native_raw_is_last_resort_when_720p_is_gone(tmp_path):
    def behave(cmd, n):
        fmt = cmd[cmd.index("--format") + 1] if "--format" in cmd else None
        if n == 1:
            return False, 67  # 720p MJPEG limpo, 15 FPS
        if "--width" in cmd:
            return True, 67  # 720p reaberto corrompido; sem raw em 720p
        if fmt is None:
            return True, 67  # nativo MJPEG corrompido
        return False, 200  # nativo raw lento, mas funciona

    cap, _cmds = _scripted_open(tmp_path, behave)
    assert cap is not None
    assert "--format" in cap._cmd and "--width" not in cap._cmd


def test_requested_mode_that_cannot_start_falls_back_to_native(tmp_path):
    """720p nao produz video algum (nem raw); o modo nativo (sem argumentos)
    funcionava antes e precisa ser tentado antes de abandonar a camera."""

    def behave(cmd, n):
        if "--width" in cmd:
            return True, 67  # 720p (e 720p raw) corrompidos
        return False, 33  # nativo OK

    cap, cmds = _scripted_open(tmp_path, behave)
    assert cap is not None
    assert "--width" not in cap._cmd and "--format" not in cap._cmd


def test_unmeasured_native_stream_does_not_replace_working_720p(tmp_path):
    """Nativo entrega 1 quadro e para: FPS desconhecido nao e upgrade."""
    fake_exe = tmp_path / "hvcam-host.exe"
    fake_exe.write_bytes(b"MZ")
    cmds = []

    def fake_popen(cmd, **_kw):
        cmds.append(list(cmd))
        return _ScriptedProc(cmd)

    class _Cap:
        def __init__(self, url, **kw):
            self.mjpeg_decode_errors = 0
            self.last_capture_ts_ms = None
            kw["on_listening"]()
            self._cmd = cmds[-1]
            self._native = "--width" not in self._cmd
            self._opened = True
            self._n = 0

        def isOpened(self):
            return self._opened

        def read(self):
            self._n += 1
            if self._native and self._n > 1:
                return False, None  # nativo trava depois do 1o quadro
            self.last_capture_ts_ms = self._n * 67
            return True, _frame()

        def release(self):
            self._opened = False

    cap, _idx, _name = hvcam_module.open_managed_reverse(
        device=0,
        try_all=True,
        host_exe=fake_exe,
        devices=[(0, "Stalling Native Cam")],
        popen=fake_popen,
        capture_cls=_Cap,
    )
    assert cap is not None
    assert "--width" in cap._cmd


def test_no_native_retry_when_requested_mode_never_connects(tmp_path):
    """Sem conexao nem erro do host (so timeout), nao gasta outro timeout com
    uma tentativa nativa por camera."""
    fake_exe = tmp_path / "hvcam-host.exe"
    fake_exe.write_bytes(b"MZ")
    cmds = []

    def fake_popen(cmd, **_kw):
        cmds.append(list(cmd))
        return _ScriptedProc(cmd)

    class _NeverConnects:
        def __init__(self, url, **kw):
            kw["on_listening"]()

        def isOpened(self):
            return False

        def release(self):
            pass

    cap, _idx, _name = hvcam_module.open_managed_reverse(
        device=0,
        try_all=True,
        host_exe=fake_exe,
        devices=[(0, "Silent Cam")],
        popen=fake_popen,
        capture_cls=_NeverConnects,
    )
    assert cap is None
    assert len(cmds) == 1
