"""Núcleo do protocolo do ajudante Windows (bridge/PROTOCOL.md).

Um fluxo de registros (cabeçalho binário de 28 bytes + payload JPEG)
chega por socket TCP ou por arquivo `.drawcap` gravado, sem o
consumidor distinguir a fonte. Cada frame sai como a tupla
(frame_id, capture_ts_us, jpeg_bytes); capture_ts_us é o relógio
monotônico do ajudante, ancorado na captura, e só serve para
diferenças dentro da mesma sessão.

Toda conexão TCP começa pelo handshake de token (PROTOCOL.md,
"Authentication handshake"). O token entra por parâmetro ou por
arquivo: a única leitura de ambiente permitida no runtime é
DRAW_MODEL_STRICT (regra do repositório), então o app recebe
--helper-token-file e o tools/helper_receiver.py aceita
DRAW_HELPER_TOKEN.

HelperCapture adapta `.drawcap`, `tcp://host[:porta]` e
`drawcap://host[:porta]` a uma fonte compatível com VideoCapture
para camera.open_capture: read() -> (ok, frame) preservando o
carimbo monotônico real de captura em last_capture_ts_ms.
"""

import hmac
import socket
import struct
import time
from pathlib import Path
from urllib.parse import urlparse

from .infra.safe_io import decode_image, image_format

MAGIC = b"DRAW"
VERSION = 1
# magic, versao, reservado, frame_id, capture_ts_us, payload_len.
HEADER = struct.Struct("<4sHHQqI")
HEADER_SIZE = HEADER.size  # 28 bytes
DEFAULT_PORT = 8765

# Teto de payload_len: conferido ANTES de ler/alocar o payload. Um
# JPEG 4K de qualidade alta fica bem abaixo disso.
MAX_PAYLOAD_LEN = 16 << 20
MAX_FRAME_SIDE = 8192
MAX_FRAME_PIXELS = 40_000_000

# Handshake de autenticação (PROTOCOL.md): o cliente envia
# HELLO(magic, versão, len) + token; o ajudante responde ACCEPT ou fecha.
HELLO_MAGIC = b"DRAH"
HELLO_VERSION = 1
HELLO = struct.Struct("<4sHH")
ACCEPT = b"DROK"
TOKEN_ENV = "DRAW_HELPER_TOKEN"
TOKEN_MIN_LEN = 16
TOKEN_MAX_LEN = 256
HANDSHAKE_TIMEOUT_S = 5.0


class ProtocolError(RuntimeError):
    """O fluxo violou bridge/PROTOCOL.md (magic, versão ou truncamento)."""


class AuthenticationError(ProtocolError):
    """Token ausente, inválido ou recusado pelo ajudante."""


def resolve_token(token=None, token_file=None):
    """Token como bytes: argumento ou arquivo (--helper-token-file).

    Sem fallback de ambiente — no runtime a única variável lida é
    DRAW_MODEL_STRICT. O fallback para DRAW_HELPER_TOKEN mora no
    wrapper de ferramentas (tools/helper_receiver.py).
    """
    if token is None and token_file:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise AuthenticationError(
                f"nao consegui ler o arquivo de token {token_file!r}: {error}"
            ) from error
    if token is None:
        token = ""
    if isinstance(token, str):
        token = token.encode("utf-8")
    if not TOKEN_MIN_LEN <= len(token) <= TOKEN_MAX_LEN:
        raise AuthenticationError(
            f"token do ajudante ausente ou fora de {TOKEN_MIN_LEN}..{TOKEN_MAX_LEN} "
            "bytes; use --helper-token-file"
        )
    return token


def client_handshake(sock, token):
    """Lado WSL: envia o token e exige ACCEPT antes de qualquer frame."""
    token = resolve_token(token)
    sock.sendall(HELLO.pack(HELLO_MAGIC, HELLO_VERSION, len(token)) + token)
    try:
        reply = _read_exact(sock.recv, len(ACCEPT))
    except (ProtocolError, OSError) as error:
        raise AuthenticationError(f"ajudante fechou o handshake: {error}") from error
    if reply != ACCEPT:
        raise AuthenticationError("ajudante recusou o token (handshake sem ACCEPT)")


def server_handshake(conn, token, timeout=HANDSHAKE_TIMEOUT_S):
    """Lado ajudante: True e ACCEPT enviado se o token confere.

    Espelha bridge/windows_helper.authenticate_client (usado no
    selfcheck e nos testes). Qualquer falha devolve False sem
    responder; quem chamou fecha a conexão.
    """
    expected = resolve_token(token)
    previous = conn.gettimeout()
    conn.settimeout(timeout)
    try:
        hello = _read_exact(conn.recv, HELLO.size)
        if hello is None:
            return False
        magic, version, length = HELLO.unpack(hello)
        if magic != HELLO_MAGIC or version != HELLO_VERSION:
            return False
        if not TOKEN_MIN_LEN <= length <= TOKEN_MAX_LEN:
            return False
        offered = _read_exact(conn.recv, length)
        if offered is None or not hmac.compare_digest(offered, expected):
            return False
        conn.sendall(ACCEPT)
        return True
    except (ProtocolError, OSError):
        return False
    finally:
        conn.settimeout(previous)


def _connect(host, port, token, timeout):
    token = resolve_token(token)
    sock = socket.create_connection(
        (host, port), timeout=timeout if timeout is not None else HANDSHAKE_TIMEOUT_S
    )
    try:
        client_handshake(sock, token)
        sock.settimeout(timeout)
    except BaseException:
        sock.close()
        raise
    return sock


def pack_record(frame_id, capture_ts_us, payload):
    """Monta um registro completo (cabeçalho + payload) para escrever."""
    return (
        HEADER.pack(MAGIC, VERSION, 0, frame_id, capture_ts_us, len(payload)) + payload
    )


def _read_exact(read, size):
    """Lê exatamente size bytes via read(n) -> bytes.

    Retorna None se a fonte acabar limpa antes do primeiro byte.
    Levanta ProtocolError se acabar no meio.
    """
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = read(remaining)
        if not chunk:
            if not chunks:
                return None
            raise ProtocolError(f"fluxo truncado: faltaram {remaining} de {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _iter_records(read, max_payload=MAX_PAYLOAD_LEN):
    """Itera registros sobre uma função read(n) -> bytes.

    Produz (frame_id, capture_ts_us, payload, bruto), onde bruto é o
    registro byte a byte como chegou (o que o gravador escreve).
    payload_len fora de 1..max_payload é erro de protocolo antes de
    qualquer leitura do payload.
    """
    while True:
        header = _read_exact(read, HEADER_SIZE)
        if header is None:
            return
        magic, version, _reserved, frame_id, capture_ts_us, payload_len = HEADER.unpack(
            header
        )
        if magic != MAGIC:
            raise ProtocolError(
                f"magic invalido: {magic!r} (esperado {MAGIC!r}); "
                "fluxo dessincronizado ou arquivo de outro formato"
            )
        if version != VERSION:
            raise ProtocolError(f"versao {version} nao suportada (esperada {VERSION})")
        if capture_ts_us < 0:
            raise ProtocolError(
                f"capture_ts_us negativo ({capture_ts_us}) no frame_id "
                f"{frame_id}; fluxo corrompido"
            )
        if payload_len == 0 or payload_len > max_payload:
            raise ProtocolError(
                f"payload_len {payload_len} fora de 1..{max_payload} "
                f"(frame_id {frame_id}); fluxo corrompido ou hostil"
            )
        payload = _read_exact(read, payload_len)
        if payload is None:
            raise ProtocolError(
                "fluxo truncado: cabecalho sem payload "
                f"(frame_id {frame_id}, {payload_len} bytes esperados)"
            )
        yield frame_id, capture_ts_us, payload, header + payload


def frames_from_file(path, max_payload=MAX_PAYLOAD_LEN):
    """Itera (frame_id, capture_ts_us, jpeg_bytes) de um arquivo gravado."""
    with open(path, "rb") as handle:
        for frame_id, capture_ts_us, payload, _raw in _iter_records(
            handle.read, max_payload
        ):
            yield frame_id, capture_ts_us, payload


def frames_from_socket(
    host, port=DEFAULT_PORT, timeout=None, token=None, max_payload=MAX_PAYLOAD_LEN
):
    """Itera (frame_id, capture_ts_us, jpeg_bytes) do ajudante via TCP.

    Autentica com o token (argumento; no tools, DRAW_HELPER_TOKEN)
    antes de ler.
    """
    with _connect(host, port, token, timeout) as sock:
        for frame_id, capture_ts_us, payload, _raw in _iter_records(
            sock.recv, max_payload
        ):
            yield frame_id, capture_ts_us, payload


def record(
    host,
    port,
    out_path,
    max_frames=None,
    timeout=None,
    token=None,
    max_payload=MAX_PAYLOAD_LEN,
):
    """Grava o fluxo do socket byte a byte num arquivo. Retorna n.

    max_frames<=0 é rejeitado: zero frames não exige conexão, e
    aceitar o valor faria o laço gravar um frame antes de parar.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames precisa ser um inteiro positivo")
    count = 0
    with _connect(host, port, token, timeout) as sock:
        with open(out_path, "wb") as handle:
            for _fid, _ts, _payload, raw in _iter_records(sock.recv, max_payload):
                handle.write(raw)
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
    return count


def is_complete_jpeg(payload):
    """True se o payload começa em FF D8 e termina em FF D9."""
    return (
        len(payload) >= 4
        and image_format(payload) == "jpeg"
        and bytes(payload[-2:]) == b"\xff\xd9"
    )


def decode_jpeg(jpeg_bytes, max_side=MAX_FRAME_SIDE, max_pixels=MAX_FRAME_PIXELS):
    """Decodifica um payload em imagem BGR (numpy) ou None se inválido.

    Payload truncado, sem cabeçalho SOF, com dimensões acima do teto
    ou que o decoder recuse devolve None; nunca levanta.
    """
    import cv2

    if not is_complete_jpeg(jpeg_bytes):
        return None
    return decode_image(
        bytes(jpeg_bytes),
        cv2.IMREAD_COLOR,
        max_bytes=MAX_PAYLOAD_LEN,
        max_side=max_side,
        max_pixels=max_pixels,
    )


def is_helper_source(source):
    """True para arquivos `.drawcap` ou URLs `tcp://`/`drawcap://`."""
    if not isinstance(source, str):
        return False
    text = source.strip()
    lower = text.lower()
    return (
        lower.endswith(".drawcap")
        or lower.startswith("tcp://")
        or lower.startswith("drawcap://")
    )


def parse_tcp_endpoint(url):
    """Extrai (host, porta) de `tcp://host[:porta]` ou `drawcap://host[:porta]`."""
    parsed = urlparse(url.strip())
    if not parsed.hostname:
        raise ValueError(f"endereco TCP sem host: {url!r}")
    port = parsed.port if parsed.port is not None else DEFAULT_PORT
    return parsed.hostname, int(port)


class HelperCapture:
    """Adaptador compatível com VideoCapture sobre `.drawcap` ou `tcp://`.

    Além de `read() -> (ok, frame)`, expõe `last_capture_ts_ms` e
    `last_frame_id` do registro mais recente, para alimentar o
    acumulador com o relógio real de captura do emissor.
    """

    def __init__(self, source, timeout=5.0, token=None, token_file=None, pace=True):
        self.source = str(source).strip()
        self._handle = None
        self._sock = None
        self._iter = None
        self._opened = False
        # Replay de arquivo respeita a cadência gravada (capture_ts_us):
        # sem pacing o read() consome a velocidade do disco e o
        # LatestFrameCapture descarta os quadros intermediários.
        self._pace = pace
        self._pace_stream = False
        # Âncora do primeiro registro: ts gravado + relógio local. Os
        # prazos seguintes são absolutos — o tempo de decodificação
        # não se soma ao intervalo gravado.
        self._pace_origin_us = None
        self._pace_clock_s = None
        self.last_frame_id = None
        self.last_capture_ts_ms = None
        # Descarte na fonte: salto de frame_id sobre TODOS os registros
        # recebidos (PROTOCOL.md: ids nunca repetem). Fica aqui e não
        # no LatestFrameCapture porque um JPEG que o decode rejeita
        # não é descarte do emissor.
        self.source_dropped = 0
        self._last_seen_id = None
        self._open(timeout, token, token_file)

    def _open(self, timeout, token, token_file):
        lower = self.source.lower()
        if lower.startswith("tcp://") or lower.startswith("drawcap://"):
            try:
                host, port = parse_tcp_endpoint(self.source)
            except ValueError:
                return
            # Sem token válido a conexão nem vale a pena tentar: o
            # erro de autenticação sobe para o operador corrigir.
            resolved = resolve_token(token, token_file)
            try:
                self._sock = _connect(host, port, resolved, timeout)
            except OSError:
                self.release()
                return
            self._iter = _iter_records(self._sock.recv)
            self._opened = True
            return
        path = Path(self.source)
        if not path.is_file():
            return
        self._handle = open(path, "rb")
        self._iter = _iter_records(self._handle.read)
        self._opened = True
        self._pace_stream = True

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened or self._iter is None:
            return False, None
        try:
            frame_id, capture_ts_us, payload, _raw = next(self._iter)
        except (StopIteration, ProtocolError, OSError):
            return False, None
        if self._last_seen_id is not None and frame_id > self._last_seen_id + 1:
            self.source_dropped += frame_id - self._last_seen_id - 1
        self._last_seen_id = frame_id
        if self._pace and self._pace_stream:
            if self._pace_origin_us is None:
                self._pace_origin_us = capture_ts_us
                self._pace_clock_s = time.monotonic()
            else:
                deadline = (
                    self._pace_clock_s + (capture_ts_us - self._pace_origin_us) / 1e6
                )
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        frame = decode_jpeg(payload)
        if frame is None:
            return False, None
        self.last_frame_id = int(frame_id)
        self.last_capture_ts_ms = int(capture_ts_us // 1000)
        return True, frame

    def release(self):
        self._opened = False
        self._iter = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None
