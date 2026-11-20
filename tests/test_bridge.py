"""Fumaça do bridge: windows_helper.py compila e é stdlib-only.

O helper é um esboço Windows (ctypes/SendInput) — importar teria
efeitos colaterais de plataforma, então o teste compila e varre os
imports via ast em vez de executar.
"""

import ast
import importlib.util
import py_compile
import queue
import socket
import struct
import sys
import threading
import time
import types
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parent.parent / "bridge" / "windows_helper.py"


def test_windows_helper_compiles():
    py_compile.compile(str(HELPER), doraise=True)


def test_windows_helper_has_no_repo_imports():
    tree = ast.parse(HELPER.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not ({"draw", "tools", "lab"} & imported)


# --- testes da logica de rede (sem Windows nem camera) ---

_spec = importlib.util.spec_from_file_location("windows_helper", HELPER)
wh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wh)

TOKEN = b"token-de-teste-para-o-bridge"  # >= TOKEN_MIN_LEN bytes


def _hello(token):
    return wh.HELLO.pack(wh.HELLO_MAGIC, wh.HELLO_VERSION, len(token)) + token


def _serve_one_client(host="127.0.0.1"):
    """Abre um listening socket efemero para alimentar o accept loop."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, 0))
    server.listen(1)
    return server


def _client(server, token=TOKEN):
    conn = socket.create_connection(server.getsockname()[:2], timeout=5)
    conn.settimeout(5)
    conn.sendall(_hello(token))
    return conn


def _start_loop(server, frames):
    """Sobe o accept loop com parada cooperativa deterministica."""
    stop = threading.Event()
    logs = []
    thread = threading.Thread(
        target=wh._accept_loop,
        args=(server, TOKEN, frames, 0),
        kwargs={"log": logs.append, "stop": stop},
        daemon=True,
    )
    thread.start()
    return thread, stop, logs


def _stop_loop(thread, stop, server):
    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    server.close()


def test_accept_loop_refuses_bad_handshake_and_keeps_listening():
    """Um par sem token e recusado e o laco continua para o proximo."""
    server = _serve_one_client()
    thread, stop, _logs = _start_loop(server, lambda: iter([b"jpeg"]))
    try:
        with _client(server, token=b"token-errado-de-teste!") as bad:
            assert bad.recv(4) == b""  # recusado: conexao fechada sem ACCEPT
        with _client(server) as good:
            assert good.recv(4) == wh.ACCEPT
            assert good.recv(wh.HEADER.size + 4)  # segundo cliente atendido
    finally:
        _stop_loop(thread, stop, server)


def test_accept_loop_survives_client_disconnect_mid_stream():
    """WSL caindo no meio do fluxo nao mata o ajudante residente."""
    server = _serve_one_client()
    thread, stop, _logs = _start_loop(server, lambda: iter([b"x" * 8] * 40))
    try:
        conn = _client(server)
        conn.recv(4)  # ACCEPT
        conn.recv(wh.HEADER.size + 8)  # um frame
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        conn.close()  # RST: o proximo sendall do servidor levanta OSError
        with _client(server) as again:
            assert again.recv(4) == wh.ACCEPT
            assert again.recv(wh.HEADER.size + 8)
    finally:
        _stop_loop(thread, stop, server)


def test_accept_loop_survives_capture_failure_and_serves_next_client():
    """Camera falhando para um cliente nao mata o ajudante residente.

    O generador da primeira conexao levanta CaptureError (camera em
    uso/desconectada); a conexao morre, o laco volta ao accept e o
    proximo cliente recebe frames normalmente.
    """
    server = _serve_one_client()

    def _fail():
        raise wh.CaptureError("camera 0 indisponivel ou em uso")
        yield  # pragma: no cover — inalcancavel; mantem gerador

    gens = [_fail(), iter([b"x" * 8])]
    thread, stop, logs = _start_loop(server, lambda: gens.pop(0))
    try:
        conn = _client(server)
        assert conn.recv(4) == wh.ACCEPT
        while conn.recv(4096):  # a falha de captura fecha a conexao
            pass
        with _client(server) as again:
            assert again.recv(4) == wh.ACCEPT
            assert again.recv(wh.HEADER.size + 8)
        assert any("Captura falhou" in line for line in logs)
    finally:
        _stop_loop(thread, stop, server)


def test_accept_loop_closes_capture_generator_on_disconnect():
    """Cliente caindo fecha o gerador de captura (stop no produtor).

    Antes, um OSError no sendall abandonava o gerador de captura: o
    finally dele (stop da thread produtora + release do VideoCapture)
    so rodava quando o GC coletasse — a camera podia ficar refem ate
    la. O accept loop agora chama close() no gerador ao fim de cada
    cliente, em qualquer caminho de saida.
    """
    server = _serve_one_client()
    closed = threading.Event()

    def _gen():
        try:
            while True:
                yield b"x" * 8
        finally:
            closed.set()

    thread, stop, _logs = _start_loop(server, lambda: _gen())
    try:
        conn = _client(server)
        conn.recv(4)  # ACCEPT
        conn.recv(wh.HEADER.size + 8)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        conn.close()  # RST: o proximo sendall do servidor levanta OSError
        assert closed.wait(2.0)  # gerador fechado pelo accept loop
    finally:
        _stop_loop(thread, stop, server)


def test_accept_loop_survives_eager_factory_failure():
    """Factory que levanta na chamada nao mata o helper nem vaza socket.

    Nem toda factory e um gerador preguicoso: uma que abre a camera
    ANTES de retornar o iterador pode levantar CaptureError na propria
    chamada de make_frames(). Isso precisa cair na mesma recuperacao
    das falhas de iteracao — conexao fechada, laco volta ao accept.
    """
    server = _serve_one_client()

    def _boom():
        raise wh.CaptureError("camera ocupada ja na abertura")

    factories = [_boom, lambda: iter([b"x" * 8])]
    thread, stop, logs = _start_loop(server, lambda: factories.pop(0)())
    try:
        conn = _client(server)
        assert conn.recv(4) == wh.ACCEPT
        while conn.recv(4096):  # a falha na factory fecha a conexao
            pass
        with _client(server) as again:
            assert again.recv(4) == wh.ACCEPT
            assert again.recv(wh.HEADER.size + 8)
        assert any("Captura falhou" in line for line in logs)
    finally:
        _stop_loop(thread, stop, server)


def test_camera_producer_reports_unexpected_error(monkeypatch):
    """Excecao no produtor (imencode etc.) vira CaptureError, nao stall.

    Sem o catch-all, a excecao morria com a thread daemon e o
    consumidor so via a fila vazia — reportava 'camera parou de
    entregar frames' em vez do erro real.
    """

    class _Cap:
        def isOpened(self):
            return True

        def read(self):
            return True, object()

        def release(self):
            pass

    def _imencode(*_args):
        raise RuntimeError("encoder quebrou")

    fake_cv2 = types.SimpleNamespace(
        CAP_DSHOW=0,
        VideoCapture=lambda *_a: _Cap(),
        imencode=_imencode,
        IMWRITE_JPEG_QUALITY=1,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    out = queue.Queue()
    wh._camera_jpeg_producer(0, None, None, 85, out, threading.Event())
    item = out.get_nowait()
    assert isinstance(item, wh.CaptureError)
    assert "encoder quebrou" in str(item)


def test_accept_loop_stop_event_shuts_down_between_accepts():
    """O stop cooperativo encerra o laco sem cliente nenhum."""
    server = _serve_one_client()
    thread, stop, _logs = _start_loop(server, lambda: iter([b"jpeg"]))
    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    server.close()


def test_capture_timestamp_is_relative_to_process_epoch():
    """capture_ts_us = (perf_counter_ns - start_ns)//1000 (PROTOCOL)."""
    server_conn, client_conn = socket.socketpair()
    ticks = iter([1_000_000_000, 1_500_000_000])

    def clock():
        return next(ticks)

    with server_conn:
        sent = wh._handle_client(server_conn, [b"ab", b"cd"], start_ns=0, clock=clock)
    assert sent == 2
    data = client_conn.recv(4096)
    client_conn.close()
    magic, version, _flags, frame_id, capture_ts_us, length = wh.HEADER.unpack(
        data[: wh.HEADER.size]
    )
    assert (magic, version, frame_id, length) == (wh.MAGIC, wh.VERSION, 0, 2)
    assert capture_ts_us == 1_000_000_000 // 1000  # epoch = start_ns = 0
    _, _, _, frame_id2, ts2, _ = wh.HEADER.unpack(
        data[wh.HEADER.size + 2 : wh.HEADER.size * 2 + 2]
    )
    assert frame_id2 == 1
    assert ts2 == 1_500_000_000 // 1000


def test_authenticate_enforces_total_handshake_deadline():
    """Par lento (1 byte antes de cada timeout por-recv) nao pode
    segurar o accept serial alem do prazo total."""
    server_conn, client_conn = socket.socketpair()
    message = wh.HELLO.pack(wh.HELLO_MAGIC, wh.HELLO_VERSION, len(TOKEN)) + TOKEN

    def drip():
        try:
            for byte in message:
                time.sleep(0.12)
                client_conn.sendall(bytes([byte]))
        except OSError:
            pass
        finally:
            client_conn.close()

    threading.Thread(target=drip, daemon=True).start()
    start = time.monotonic()
    result = wh.authenticate_client(server_conn, TOKEN, timeout=0.3)
    elapsed = time.monotonic() - start
    server_conn.close()
    assert result is False
    assert elapsed < 2.0


def test_authenticate_accepts_valid_hello():
    server_conn, client_conn = socket.socketpair()
    client_conn.sendall(_hello(TOKEN))
    assert wh.authenticate_client(server_conn, TOKEN, timeout=5.0) is True
    assert client_conn.recv(4) == wh.ACCEPT
    server_conn.close()
    client_conn.close()


@pytest.mark.parametrize("host", ["0", "0.0", "0.0.0.0", "::", ""])
def test_resolve_bind_host_rejects_wildcard_forms(host):
    """Qualquer forma que resolve para endereco nao especificado
    (wildcard) exige --allow-all-interfaces, como o 0.0.0.0 literal."""
    with pytest.raises(SystemExit):
        wh.resolve_bind_host(host)


@pytest.mark.parametrize("host", ["0", "0.0.0.0", "::"])
def test_resolve_bind_host_wildcard_with_opt_in(host):
    assert wh.resolve_bind_host(host, allow_all_interfaces=True) == host


def test_resolve_bind_host_keeps_specific_addresses():
    assert wh.resolve_bind_host("127.0.0.1") == "127.0.0.1"
    assert wh.resolve_bind_host("localhost") == "localhost"


# --- capture_frames: cv2 falso (sem camera nem Windows) ---


class _Encoded:
    def tobytes(self):
        return b"jpeg"


class _FakeCap:
    def __init__(self, frames, opens=True):
        self._frames = list(frames)
        self._opens = opens
        self.sets = []
        self.released = False

    def isOpened(self):
        return self._opens

    def set(self, prop, val):
        self.sets.append((prop, val))
        return True

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True


def _fake_cv2(monkeypatch, frames, opens=True):
    import sys
    import types

    cap = _FakeCap(frames, opens)
    fake = types.SimpleNamespace(
        CAP_DSHOW=700,
        CAP_PROP_FOURCC=6,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        IMWRITE_JPEG_QUALITY=1,
        VideoCapture=lambda *args: cap,
        VideoWriter_fourcc=lambda *args: 0,
        imencode=lambda ext, img, params: (True, _Encoded()),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return cap


def test_capture_frames_yields_jpeg_per_frame_and_releases(monkeypatch):
    import itertools

    cap = _fake_cv2(monkeypatch, frames=["f0", "f1", "f2"])
    gen = wh.capture_frames()
    jpegs = list(itertools.islice(gen, 3))
    assert jpegs == [b"jpeg"] * 3
    gen.close()
    assert cap.released


def test_capture_frames_defaults_to_no_format_negotiation(monkeypatch):
    """Sem width/height nao se chama cap.set: negociar formato (ex.:
    FOURCC=MJPG) ja travou o driver DirectShow de uma webcam real e
    deixou o dispositivo refem. Padrao seguro e o formato nativo."""
    cap = _fake_cv2(monkeypatch, frames=["f0"])
    gen = wh.capture_frames()
    next(gen)
    gen.close()
    assert cap.sets == []


def test_capture_frames_sets_resolution_only_when_asked(monkeypatch):
    cap = _fake_cv2(monkeypatch, frames=["f0"])
    gen = wh.capture_frames(index=1, width=640, height=480, quality=70)
    next(gen)
    gen.close()
    assert cap.sets == [(3, 640), (4, 480)]  # FRAME_WIDTH, FRAME_HEIGHT


def test_capture_frames_raises_when_camera_unavailable(monkeypatch):
    _fake_cv2(monkeypatch, frames=[], opens=False)
    gen = wh.capture_frames()
    with pytest.raises(RuntimeError, match="indisponivel"):
        next(gen)


def test_capture_frames_raises_on_read_failure(monkeypatch):
    cap = _fake_cv2(monkeypatch, frames=["f0"])
    gen = wh.capture_frames()
    assert next(gen) == b"jpeg"
    with pytest.raises(RuntimeError, match="leitura"):
        next(gen)
    assert cap.released


def test_enqueue_put_never_drops_and_honors_stop():
    """Protocolo push: nada se descarta; stop destrava o produtor."""
    import queue as _queue

    q = _queue.Queue(maxsize=1)
    wh._enqueue_put(q, b"frame", threading.Event())
    assert q.get_nowait() == b"frame"

    full = _queue.Queue(maxsize=1)
    full.put(b"stuck")
    stop = threading.Event()
    stop.set()
    wh._enqueue_put(full, b"item", stop)  # retorna sem bloquear
    assert full.get_nowait() == b"stuck"  # nada evictado


def test_capture_frames_stalled_camera_becomes_capture_error():
    """read() encalhado no DirectShow: o prazo vira CaptureError.

    Reproduz a falha vista em hardware (o stream morreu no meio da
    sessao e o cliente seguinte nem completava o handshake porque o
    accept loop ficou preso no _handle_client): o produtor nunca
    entrega, o consumidor nao pode congelar — estoura CaptureError
    que so derruba aquele cliente.
    """

    def _stalled(index, width, height, quality, out, stop):
        stop.wait(5.0)  # simula cap.read() bloqueado para sempre

    gen = wh.capture_frames(stall_timeout=0.05, _producer=_stalled)
    with pytest.raises(wh.CaptureError, match="parou de entregar"):
        next(gen)
    gen.close()


def test_capture_frames_propagates_producer_error():
    """Erro do produtor (camera indisponivel) chega ao consumidor."""

    def _dead(index, width, height, quality, out, stop):
        wh._enqueue_put(out, wh.CaptureError("camera 0 indisponivel ou em uso"), stop)

    gen = wh.capture_frames(_producer=_dead)
    with pytest.raises(wh.CaptureError, match="indisponivel"):
        next(gen)
    gen.close()


def test_accept_loop_stop_wakes_immediately(monkeypatch):
    entered = threading.Event()
    gate = threading.Event()
    real_selector = wh.selectors.DefaultSelector

    class _WatchingSelector(real_selector):
        def select(self, timeout=None):
            entered.set()
            return super().select(timeout=timeout)

    def _frames():
        gate.wait(timeout=2.0)
        yield b"jpeg"

    monkeypatch.setattr(wh.selectors, "DefaultSelector", _WatchingSelector)
    server = _serve_one_client()
    thread, stop, _logs = _start_loop(server, _frames)
    try:
        with _client(server) as conn:
            assert conn.recv(4) == wh.ACCEPT
            # Accept loop esta dentro de _handle_client esperando gate; zera
            # entered agora para comprovar que voltou ao select() apos sair.
            entered.clear()
            gate.set()
            assert conn.recv(wh.HEADER.size + 4)
        assert entered.wait(timeout=1.0)
        time.sleep(0.01)
        started = time.monotonic()
        stop.set()
        thread.join(timeout=1.0)
        elapsed = time.monotonic() - started
        assert not thread.is_alive()
        assert elapsed < 0.15
    finally:
        gate.set()
        stop.set()
        server.close()
