"""Núcleo do protocolo do ajudante Windows (bridge/PROTOCOL.md).

Um fluxo de registros (cabeçalho binário de 28 bytes + payload JPEG)
chega por socket TCP ou por arquivo `.drawcap` gravado, sem o
consumidor distinguir a fonte. Cada frame sai como a tupla
(frame_id, capture_ts_us, jpeg_bytes); capture_ts_us é o relógio
monotônico do ajudante, ancorado na captura, e só serve para
diferenças dentro da mesma sessão.

O handshake de token (PROTOCOL.md, "Authentication handshake"), os
erros e o leitor `_read_exact` moram agora em `draw_dral.wire`
(pacote draw-dral), compartilhados com o fio DRAL — reexportados
aqui para compatibilidade. A reexportação preserva imports e
leituras; escrever num nome reexportado (ex.: monkeypatch) não
alcança `draw_dral.wire` — quem precisa mira lá. O token entra por
parâmetro ou por
arquivo: a única leitura de ambiente permitida no runtime é
DRAW_MODEL_STRICT (regra do repositório), então o app recebe
--helper-token-file e o tools/helper_receiver.py aceita
DRAW_HELPER_TOKEN.

HelperCapture adapta `.drawcap`, `tcp://host[:porta]` e
`drawcap://host[:porta]` a uma fonte compatível com VideoCapture
para camera.open_capture: read() -> (ok, frame) preservando o
carimbo monotônico real de captura em last_capture_ts_ms.
"""

import socket
import struct
import time
from pathlib import Path
from urllib.parse import urlparse

import draw_dral.wire as _wire
from draw_dral.wire import (
    HANDSHAKE_TIMEOUT_S,
    ProtocolError,
    _read_exact,
    client_handshake,
    resolve_token,
)

from .infra.safe_io import decode_image, image_format

# Reexportados para compatibilidade (ver docstring): nomes do handshake
# que este modulo nao consome internamente, mas consumidores como
# tools/helper_receiver.py e draw.stream importam daqui.
ACCEPT = _wire.ACCEPT
HELLO = _wire.HELLO
HELLO_MAGIC = _wire.HELLO_MAGIC
HELLO_VERSION = _wire.HELLO_VERSION
TOKEN_ENV = _wire.TOKEN_ENV
TOKEN_MAX_LEN = _wire.TOKEN_MAX_LEN
TOKEN_MIN_LEN = _wire.TOKEN_MIN_LEN
AuthenticationError = _wire.AuthenticationError
server_handshake = _wire.server_handshake

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

# Janelas de socket vazias (timeout de recv sem nenhum byte) que uma
# leitura do HelperCapture tolera antes de desistir do registro. Um
# stall breve não mata o iterador; um emissor morto ainda é
# declarado morto depois de ~4 janelas de silêncio.
_STALL_WINDOWS = 3

# Pacing do replay .drawcap. Um capture_ts_us corrompido podia prender a
# thread num sleep() de horas: acima do teto o replay entrega na hora. A
# espera vai em fatias para release() interrompe-la em ate uma fatia —
# e continua passando por time.sleep/monotonic do modulo, que os testes
# monkeypatcham com um relogio falso.
_MAX_PACE_DELAY_S = 5.0
_PACE_SLICE_S = 0.05


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


def _iter_records(
    read,
    max_payload=MAX_PAYLOAD_LEN,
    stall_retries=0,
    record_timeout_s=None,
    clock=time.monotonic,
    recv_timeout=None,
):
    """Itera registros sobre uma função read(n) -> bytes.

    Produz (frame_id, capture_ts_us, payload, bruto), onde bruto é o
    registro byte a byte como chegou (o que o gravador escreve).
    payload_len fora de 1..max_payload é erro de protocolo antes de
    qualquer leitura do payload. record_timeout_s, quando dado, e o
    prazo total para montar UM registro (cabecalho + payload) — o
    timeout do socket so limita cada recv isolado.
    """
    while True:
        deadline = None
        if record_timeout_s is not None:
            deadline = clock() + record_timeout_s
        header = _read_exact(
            read,
            HEADER_SIZE,
            stall_retries=stall_retries,
            deadline=deadline,
            clock=clock,
            recv_timeout=recv_timeout,
        )
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
        payload = _read_exact(
            read,
            payload_len,
            stall_retries=stall_retries,
            deadline=deadline,
            clock=clock,
            recv_timeout=recv_timeout,
        )
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
    acumulador com o relógio real de captura do emissor, e
    `last_error`, que separa o fim por falha (registro truncado ou
    payload sem decode) do EOF limpo — os dois chegam como o mesmo
    (False, None).
    """

    def __init__(
        self,
        source,
        timeout=5.0,
        token=None,
        token_file=None,
        pace=True,
        stall_windows=_STALL_WINDOWS,
        record_timeout_s=None,
    ):
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
        # Janelas de socket vazias toleradas por leitura. O padrao
        # serve o streaming; o diagnostico (`draw --check`) passa 0
        # para honrar o deadline do proprio timeout.
        self._stall_windows = stall_windows
        # Prazo total de UM registro (cabecalho+payload). O timeout
        # do socket e por recv: sem isto, um frame pingando byte a
        # byte passava de qualquer deadline — o diagnostico usa o
        # mesmo valor do timeout como prazo de quadro.
        self._record_timeout_s = record_timeout_s
        self.last_frame_id = None
        self.last_capture_ts_ms = None
        # Fim por erro (registro truncado/corrompido, payload sem
        # decode, falha de socket): read() devolve (False, None) do
        # mesmo jeito que no EOF limpo — este atributo e o que separa
        # os dois desfechos para quem audita a fonte.
        self.last_error = None
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
            # recv clampeado ao orcamento do prazo do registro: sem
            # isto, o recv podia bloquear a janela inteira do socket e
            # um frame cruzava o deadline sendo aceito. O min() com o
            # timeout configurado impede afrouxar o detector de stall.
            base_timeout = self._sock.gettimeout()
            self._iter = _iter_records(
                self._sock.recv,
                stall_retries=self._stall_windows,
                record_timeout_s=self._record_timeout_s,
                recv_timeout=lambda s: self._sock.settimeout(
                    min(base_timeout, s) if base_timeout is not None else s
                ),
            )
            self._opened = True
            return
        path = Path(self.source)
        if not path.is_file():
            return
        try:
            self._handle = open(path, "rb")
        except OSError as error:
            # Existe mas nao abre (permissao, bloqueio): mesmo desfecho de
            # "nao abriu" — isOpened() False — em vez de um traceback cru
            # para o open_capture; o motivo fica em last_error.
            self.last_error = error
            return
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
        except StopIteration:
            return False, None
        except (ProtocolError, OSError) as error:
            self.last_error = error
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
                # Teto + fatias: um salto de timestamp nao pode dormir
                # para sempre e release() interrompe em ate uma fatia.
                wait_until = min(deadline, time.monotonic() + _MAX_PACE_DELAY_S)
                while self._opened:
                    remaining = wait_until - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, _PACE_SLICE_S))
                if not self._opened:
                    return False, None
        frame = decode_jpeg(payload)
        if frame is None:
            self.last_error = ProtocolError(
                f"payload sem decode jpeg (frame_id {frame_id})"
            )
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


def __getattr__(name):
    return getattr(_wire, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_wire)))
