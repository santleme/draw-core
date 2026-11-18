"""Adaptador de fonte do ajudante: .drawcap/tcp:// no open_capture.

O núcleo do protocolo (framing, handshake, jpeg) já é coberto por
test_bridge_security.py; aqui ficam HelperCapture, is_helper_source,
parse_tcp_endpoint e a fiação de open_capture/run().
"""

import socket
import threading
import time

import cv2
import numpy as np
import pytest

from draw.camera import open_capture
from draw.protocol import (
    DEFAULT_PORT,
    AuthenticationError,
    HelperCapture,
    is_helper_source,
    pack_record,
    parse_tcp_endpoint,
    server_handshake,
)

TOKEN = "t" * 32


def _jpeg(width=64, height=48):
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _drawcap(path, frames):
    with open(path, "wb") as handle:
        for frame_id, ts_us, payload in frames:
            handle.write(pack_record(frame_id, ts_us, payload))
    return path


@pytest.mark.parametrize(
    "source,expected",
    [
        ("sessao.drawcap", True),
        ("PASTA/SESSAO.DRAWCAP", True),
        ("tcp://192.168.0.10", True),
        ("tcp://127.0.0.1:8765", True),
        ("drawcap://127.0.0.1", True),
        ("http://192.168.0.10:8080/video", False),
        ("video.mp4", False),
        (0, False),
        (None, False),
    ],
)
def test_is_helper_source(source, expected):
    assert is_helper_source(source) is expected


@pytest.mark.parametrize(
    "url,expected",
    [
        ("tcp://192.168.0.10", ("192.168.0.10", DEFAULT_PORT)),
        ("tcp://192.168.0.10:9000", ("192.168.0.10", 9000)),
        ("drawcap://127.0.0.1:1234", ("127.0.0.1", 1234)),
    ],
)
def test_parse_tcp_endpoint(url, expected):
    assert parse_tcp_endpoint(url) == expected


def test_parse_tcp_endpoint_sem_host():
    with pytest.raises(ValueError, match="sem host"):
        parse_tcp_endpoint("tcp://")


def test_helper_capture_reads_drawcap(tmp_path):
    frames = [(0, 100_000, _jpeg()), (1, 133_333, _jpeg())]
    path = _drawcap(tmp_path / "s.drawcap", frames)
    capture = HelperCapture(str(path))
    assert capture.isOpened()

    ok, frame = capture.read()
    assert ok and frame.shape == (48, 64, 3)
    assert capture.last_frame_id == 0
    assert capture.last_capture_ts_ms == 100

    ok, _frame = capture.read()
    assert ok and capture.last_frame_id == 1

    ok, frame = capture.read()
    assert not ok and frame is None  # fim do arquivo: fonte parada
    capture.release()


def test_helper_capture_missing_drawcap(tmp_path):
    capture = HelperCapture(str(tmp_path / "ausente.drawcap"))
    assert not capture.isOpened()
    ok, frame = capture.read()
    assert not ok and frame is None


def test_open_capture_accepts_drawcap(tmp_path):
    path = _drawcap(tmp_path / "s.drawcap", [(0, 0, _jpeg())])
    capture = open_capture(str(path))
    try:
        assert isinstance(capture, HelperCapture)
        ok, frame = capture.read()
        assert ok and frame.shape == (48, 64, 3)
    finally:
        capture.release()


def test_open_capture_missing_drawcap(tmp_path):
    with pytest.raises(RuntimeError, match="fonte do ajudante"):
        open_capture(str(tmp_path / "ausente.drawcap"))


def _serve(token, frames, results):
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def run():
        conn, _addr = server.accept()
        with conn:
            ok = server_handshake(conn, token, timeout=2)
            results.append(ok)
            if ok:
                for frame_id, ts_us, payload in frames:
                    conn.sendall(pack_record(frame_id, ts_us, payload))
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return server.getsockname()[1], thread


def test_helper_capture_tcp_stream():
    results = []
    payload = _jpeg()
    port, thread = _serve(TOKEN, [(0, 42_000, payload)], results)
    capture = HelperCapture(f"tcp://127.0.0.1:{port}", timeout=5, token=TOKEN)
    thread.join(5)
    assert results == [True]
    assert capture.isOpened()
    ok, frame = capture.read()
    assert ok and frame.shape == (48, 64, 3)
    assert capture.last_frame_id == 0
    assert capture.last_capture_ts_ms == 42
    capture.release()


def test_helper_capture_tcp_sem_token_recusado():
    # Sem token válido a conexão nem é tentada: o erro é de
    # autenticação, não de "não consegui abrir".
    with pytest.raises(AuthenticationError):
        HelperCapture("tcp://127.0.0.1:1", timeout=1)


def test_helper_capture_tcp_token_file(tmp_path):
    results = []
    (tmp_path / "token.txt").write_text(TOKEN + "\n", encoding="utf-8")
    port, thread = _serve(TOKEN, [(0, 0, _jpeg())], results)
    capture = HelperCapture(
        f"tcp://127.0.0.1:{port}",
        timeout=5,
        token_file=str(tmp_path / "token.txt"),
    )
    thread.join(5)
    assert capture.isOpened()
    capture.release()


def test_helper_capture_tcp_token_errado():
    results = []
    port, thread = _serve(TOKEN, [(0, 0, _jpeg())], results)
    with pytest.raises(AuthenticationError):
        HelperCapture(f"tcp://127.0.0.1:{port}", timeout=5, token="w" * 32)
    thread.join(5)
    assert results == [False]


def test_helper_capture_tcp_servidor_fora():
    # Ninguém escutando: falha de abertura, não exceção.
    capture = HelperCapture("tcp://127.0.0.1:1", timeout=1, token=TOKEN)
    assert not capture.isOpened()


def test_drawcap_read_respeita_release(tmp_path):
    path = _drawcap(tmp_path / "s.drawcap", [(0, 0, _jpeg())])
    capture = HelperCapture(str(path))
    capture.release()
    ok, frame = capture.read()
    assert not ok and frame is None


class _FakeClock:
    """Relógio determinístico: avança no sleep() e pelo 'decode' do teste."""

    def __init__(self, start=100.0):
        self.now = start
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


def _patch_clock(monkeypatch, clock):
    monkeypatch.setattr("draw.protocol.time.monotonic", clock.monotonic)
    monkeypatch.setattr("draw.protocol.time.sleep", clock.sleep)


def test_drawcap_replay_respeita_cadencia_gravada(tmp_path, monkeypatch):
    """Replay não vaza quadros: read() espera o intervalo entre
    capture_ts_us em vez de consumir na velocidade do disco."""
    clock = _FakeClock()
    _patch_clock(monkeypatch, clock)
    frames = [
        (0, 1_000_000, _jpeg()),
        (1, 1_040_000, _jpeg()),
        (2, 1_080_000, _jpeg()),
    ]
    capture = HelperCapture(str(_drawcap(tmp_path / "s.drawcap", frames)))
    for _ in range(3):
        ok, _frame = capture.read()
        assert ok
    capture.release()
    assert clock.sleeps == [pytest.approx(0.04), pytest.approx(0.04)]


def test_drawcap_replay_nao_acumula_decode(tmp_path, monkeypatch):
    """O prazo é absoluto: 10 ms de decode por quadro não somam 10 ms
    a cada intervalo de 40 ms — a cadência entregue fica em 40 ms."""
    clock = _FakeClock()
    _patch_clock(monkeypatch, clock)
    decode_s = 0.010
    frames = [(i, 1_000_000 + i * 40_000, _jpeg()) for i in range(3)]
    capture = HelperCapture(str(_drawcap(tmp_path / "s.drawcap", frames)))
    delivered = []
    for _ in range(3):
        ok, _frame = capture.read()
        assert ok
        delivered.append(clock.now)
        clock.now += decode_s  # o decode do quadro "gasta" 10 ms
    capture.release()
    # Entregas ancoradas na cadência de 40 ms da gravação.
    assert delivered == [
        pytest.approx(100.0),
        pytest.approx(100.04),
        pytest.approx(100.08),
    ]
    assert clock.sleeps == [pytest.approx(0.03), pytest.approx(0.03)]


def test_drawcap_primeiro_frame_sem_espera(tmp_path, monkeypatch):
    clock = _FakeClock()
    _patch_clock(monkeypatch, clock)
    capture = HelperCapture(
        str(_drawcap(tmp_path / "s.drawcap", [(7, 9_000_000, _jpeg())]))
    )
    ok, _frame = capture.read()
    assert ok and clock.sleeps == []
    capture.release()


def test_drawcap_pace_desligado_nao_espera(tmp_path, monkeypatch):
    clock = _FakeClock()
    _patch_clock(monkeypatch, clock)
    frames = [(0, 0, _jpeg()), (1, 5_000_000, _jpeg())]
    capture = HelperCapture(str(_drawcap(tmp_path / "s.drawcap", frames)), pace=False)
    capture.read()
    capture.read()
    capture.release()
    assert clock.sleeps == []


def test_drawcap_atrasado_nao_espera(tmp_path, monkeypatch):
    """Prazo já vencido (ts no passado ou consumo lento) não dorme."""
    clock = _FakeClock()
    _patch_clock(monkeypatch, clock)
    frames = [(0, 5_000_000, _jpeg()), (1, 1_000_000, _jpeg())]
    capture = HelperCapture(str(_drawcap(tmp_path / "s.drawcap", frames)))
    capture.read()
    capture.read()
    capture.release()
    assert clock.sleeps == []


def test_token_file_ilegivel_vira_erro_de_autenticacao(tmp_path):
    from draw.protocol import resolve_token

    with pytest.raises(AuthenticationError, match="arquivo de token"):
        resolve_token(token_file=str(tmp_path / "ausente.txt"))


def test_check_helper_source_token_file_ruim(tmp_path, capsys):
    from draw.app import _check_helper_source

    result = _check_helper_source(
        "tcp://127.0.0.1:1", token_file=str(tmp_path / "ausente.txt")
    )
    assert result == 0
    assert "FAIL" in capsys.readouterr().out


def test_tools_frames_from_socket_respeita_env(monkeypatch):
    """O fallback de DRAW_HELPER_TOKEN continua valendo no tools."""
    from tools import helper_receiver as rx

    monkeypatch.delenv(rx.TOKEN_ENV, raising=False)
    with pytest.raises(rx.AuthenticationError):
        list(rx.frames_from_socket("127.0.0.1", 1, timeout=1))
    # Com o env, a resolução passa e a falha é de conexão (OSError),
    # não de autenticação.
    monkeypatch.setenv(rx.TOKEN_ENV, TOKEN)
    with pytest.raises(OSError):
        list(rx.frames_from_socket("127.0.0.1", 1, timeout=1))


def test_tools_record_rejeita_max_frames_antes_do_token(tmp_path, monkeypatch):
    from tools import helper_receiver as rx

    monkeypatch.delenv(rx.TOKEN_ENV, raising=False)
    with pytest.raises(ValueError, match="max_frames"):
        rx.record("127.0.0.1", 1, tmp_path / "x.drawcap", max_frames=0)


def test_source_dropped_conta_salto_sem_confundir_jpeg_ruim(tmp_path):
    """JPEG que o decode rejeita chegou ao receptor: não é descarte
    do emissor. Só o salto real de frame_id conta."""
    bad = b"nao-e-jpeg"
    frames = [
        (0, 0, _jpeg()),
        (1, 33_000, bad),
        (2, 66_000, _jpeg()),
        (5, 130_000, _jpeg()),
    ]
    capture = HelperCapture(str(_drawcap(tmp_path / "s.drawcap", frames)), pace=False)
    ok, _frame = capture.read()
    assert ok and capture.last_frame_id == 0
    ok, frame = capture.read()
    assert not ok and frame is None  # id 1: payload inválido, leitura falha
    ok, _frame = capture.read()
    assert ok and capture.last_frame_id == 2
    ok, _frame = capture.read()
    assert ok and capture.last_frame_id == 5
    capture.release()
    # ids 0,1,2,5 chegaram; o único descarte do emissor é 3 e 4.
    assert capture.source_dropped == 2


def test_source_dropped_conta_salto_no_socket():
    results = []
    payload = _jpeg()
    records = [(0, 0, payload), (3, 33_000, payload)]
    port, thread = _serve(TOKEN, records, results)
    capture = HelperCapture(f"tcp://127.0.0.1:{port}", timeout=5, token=TOKEN)
    thread.join(5)
    capture.read()
    capture.read()
    capture.release()
    assert capture.source_dropped == 2


def test_iter_records_rejeita_ts_negativo():
    from draw.protocol import HEADER, MAGIC, VERSION, ProtocolError, _iter_records

    payload = _jpeg()
    data = HEADER.pack(MAGIC, VERSION, 0, 0, -1000, len(payload)) + payload
    with pytest.raises(ProtocolError, match="capture_ts_us"):
        list(_iter_records(_reader_of(data)))


def _reader_of(data):
    view = memoryview(data)
    pos = 0

    def read(n):
        nonlocal pos
        chunk = bytes(view[pos : pos + n])
        pos += len(chunk)
        return chunk

    return read


def test_iter_records_tolera_stall_transitorio():
    """socket.timeout entre bytes: o registro chega mesmo assim."""
    from draw.protocol import HEADER, MAGIC, VERSION, _iter_records

    payload = _jpeg()
    data = HEADER.pack(MAGIC, VERSION, 0, 7, 1000, len(payload)) + payload
    base = _reader_of(data)
    calls = {"n": 0}

    def read(n):
        calls["n"] += 1
        if calls["n"] == 2:
            raise socket.timeout("stall")
        return base(n)

    records = list(_iter_records(read, stall_retries=1))
    assert [r[0] for r in records] == [7]


def test_iter_records_esgota_janelas_de_stall():
    from draw.protocol import ProtocolError, _iter_records

    def read(n):
        raise socket.timeout("stall")

    with pytest.raises(ProtocolError, match="emissor parado"):
        list(_iter_records(read, stall_retries=2))


def test_helper_capture_tcp_stall_nao_mata_iterador():
    """Stall breve entre dois frames: a segunda leitura ainda entrega."""
    payload = _jpeg()
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def run():
        conn, _addr = server.accept()
        with conn:
            if server_handshake(conn, TOKEN, timeout=2):
                conn.sendall(pack_record(0, 42_000, payload))
                time.sleep(0.35)  # stall ~2 janelas de timeout do cliente
                conn.sendall(pack_record(1, 84_000, payload))
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    capture = HelperCapture(
        f"tcp://127.0.0.1:{server.getsockname()[1]}", timeout=0.15, token=TOKEN
    )
    try:
        ok1, _ = capture.read()
        ok2, frame = capture.read()
    finally:
        capture.release()
        thread.join(5)
    assert ok1 and ok2 and frame.shape == (48, 64, 3)
    assert capture.last_frame_id == 1


def test_iter_records_sem_stall_budget_propaga_timeout():
    """stall_retries=0: o tipo original sobe — TimeoutError, nao ProtocolError."""
    from draw.protocol import _iter_records

    def read(n):
        raise socket.timeout("stall")

    with pytest.raises(TimeoutError):
        list(_iter_records(read))


def test_helper_capture_stall_windows_zero_falha_no_deadline():
    """Diagnostico: sem stall retries, quadro alem do timeout nao passa."""
    payload = _jpeg()
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def run():
        conn, _addr = server.accept()
        with conn:
            if server_handshake(conn, TOKEN, timeout=2):
                time.sleep(0.35)
                conn.sendall(pack_record(0, 42_000, payload))
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    capture = HelperCapture(
        f"tcp://127.0.0.1:{server.getsockname()[1]}",
        timeout=0.15,
        token=TOKEN,
        stall_windows=0,
    )
    started = time.monotonic()
    ok, _ = capture.read()
    elapsed = time.monotonic() - started
    capture.release()
    thread.join(5)
    assert not ok and elapsed < 0.5


def test_check_helper_source_respeita_deadline(monkeypatch, tmp_path, capsys):
    """`draw --check` recusa o helper cujo primeiro quadro atrasa.

    Com token de verdade: sem ele o check falhava na autenticacao
    antes de conectar e o deadline nunca era exercitado. E o timeout
    so e encurtado por subclasse — functools.partial(timeout=...)
    seria sobreposto pelo timeout=5.0 explicito do check.
    """
    from draw import app as app_module

    class _ShortTimeout(HelperCapture):
        def __init__(self, source, **kwargs):
            kwargs["timeout"] = 0.15
            super().__init__(source, **kwargs)

    payload = _jpeg()
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def run():
        conn, _addr = server.accept()
        with conn:
            if server_handshake(conn, TOKEN, timeout=2):
                time.sleep(0.35)  # quadro chega alem do deadline do check
                conn.sendall(pack_record(0, 42_000, payload))
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    token_file = tmp_path / "helper.token"
    token_file.write_text(TOKEN)
    monkeypatch.setattr(app_module, "HelperCapture", _ShortTimeout)
    started = time.monotonic()
    rc = app_module._check_helper_source(
        f"tcp://127.0.0.1:{server.getsockname()[1]}", token_file=str(token_file)
    )
    elapsed = time.monotonic() - started
    thread.join(5)
    assert rc == 0 and elapsed < 0.5


def test_iter_records_preserva_bytes_no_stall_mid_payload():
    """Timeout no meio do payload: os bytes ja lidos nao se perdem."""
    from draw.protocol import HEADER, HEADER_SIZE, MAGIC, VERSION, _iter_records

    payload = _jpeg()
    data = HEADER.pack(MAGIC, VERSION, 0, 9, 2000, len(payload)) + payload
    base = _reader_of(data)
    calls = {"n": 0}

    def read(n):
        calls["n"] += 1
        if calls["n"] == 1:
            return base(HEADER_SIZE)  # cabecalho inteiro
        if calls["n"] == 2:
            return base(len(payload) // 3)  # pedaco do payload
        if calls["n"] == 3:
            raise socket.timeout("stall")  # stall dentro do registro
        return base(n)  # o restante chega no retry

    records = list(_iter_records(read, stall_retries=1))
    assert len(records) == 1
    frame_id, _ts, got, _raw = records[0]
    assert frame_id == 9 and got == payload


def test_iter_records_respeita_prazo_do_registro():
    """Frame pingando byte a byte: o prazo total corta a montagem."""
    from draw.protocol import ProtocolError, _iter_records

    clock = [0.0]

    def read(n):
        clock[0] += 1.0
        return b"x"  # um byte por segundo, para sempre

    with pytest.raises(ProtocolError, match="prazo"):
        list(_iter_records(read, record_timeout_s=5.0, clock=lambda: clock[0]))


def test_iter_records_sem_prazo_nao_limita():
    """record_timeout_s=None (padrao streaming): so stall/tamanho limitam."""
    from draw.protocol import HEADER, MAGIC, VERSION, _iter_records

    payload = _jpeg()
    data = HEADER.pack(MAGIC, VERSION, 0, 1, 100, len(payload)) + payload
    slow = _reader_of(data)

    def read(n):
        return slow(min(n, 3))  # dribla 3 bytes por chamada

    records = list(_iter_records(read))
    assert records[0][2] == payload


def test_iter_records_clampa_recv_ao_orcamento_do_prazo():
    """recv_timeout recebe o orcamento restante — decresce a cada recv.

    A leitura simulada honra o timeout configurado (espera seen[-1]
    ou menos), entao os valores sao exatos: um orcamento constante
    ou a janela cheia falhariam a asserção.
    """
    from draw.protocol import ProtocolError, _iter_records

    clock = [0.0]
    seen = []

    def read(n):
        clock[0] += min(seen[-1], 0.1)  # responde dentro do orcamento
        return b"x"

    with pytest.raises(ProtocolError, match="prazo"):
        list(
            _iter_records(
                read,
                stall_retries=10,
                record_timeout_s=0.4,
                clock=lambda: clock[0],
                recv_timeout=seen.append,
            )
        )
    assert seen == pytest.approx([0.4, 0.3, 0.2, 0.1])
