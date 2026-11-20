"""Saída ao vivo da tinta confirmada (bridge/STREAM.md).

O Draw publica o canvas confirmado a cada versão nova para um consumidor
externo (o drawlink) por um fluxo binário de registros: cabeçalho de
32 bytes + payload. Socket e arquivo gravado têm a mesma gramática e o
consumidor não distingue a fonte (cada destino tem o próprio diff, então os
bytes de um consumidor tardio e os do arquivo podem diferir).

O publicador NUNCA bloqueia o laço do Draw: `publish` só guarda a última
publicação (latest-wins) e acorda uma thread que codifica e envia. Sem
consumidor, consumidor lento ou conexão caída, o laço segue igual. O
consumidor inicia a conexão (o Draw escuta em 127.0.0.1) e passa pelo mesmo
handshake de token do PROTOCOL.md.
"""

import os
import select
import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass

import numpy as np

from .protocol import ProtocolError, _read_exact, resolve_token, server_handshake

MAGIC = b"DRAL"
VERSION = 1
# magic, versão, tipo, reservado, geração, versão da tinta, ts_us,
# largura, altura, payload_len.
HEADER = struct.Struct("<4sHBBIIqHHI")
HEADER_SIZE = HEADER.size  # 32 bytes
KIND_FULL = 1  # payload: <ff (largura_mm, altura_mm) + zlib(máscara h*w)
KIND_DELTA = 2  # payload: <HHHH (x, y, w, h) + zlib(sub-retângulo h*w)
_FULL_PREFIX = struct.Struct("<ff")
_DELTA_PREFIX = struct.Struct("<HHHH")
MAX_PAYLOAD_LEN = 16 << 20
MAX_SIDE = 8192
MAX_PIXELS = 40_000_000
DEFAULT_PORT = 8766
# Prazo de envio por registro: consumidor que não drena é derrubado em vez
# de segurar a thread (o laço do Draw nunca espera por ele).
SEND_TIMEOUT_S = 2.0
# Teto de handshakes simultaneos (clientes sem token ainda) e de consumidores
# autenticados: o excedente e fechado, o laco do Draw nunca paga por isso.
MAX_PENDING_HANDSHAKES = 8
MAX_CONSUMERS = 8
_PRIVATE_FILES = hasattr(os, "fchmod")


@dataclass(frozen=True, slots=True)
class InkRecord:
    kind: int
    generation: int
    ink_version: int
    ts_us: int
    width: int
    height: int
    payload: bytes


def _pack(kind, generation, ink_version, ts_us, width, height, payload):
    return (
        HEADER.pack(
            MAGIC,
            VERSION,
            kind,
            0,
            generation,
            ink_version,
            ts_us,
            width,
            height,
            len(payload),
        )
        + payload
    )


def pack_full(generation, ink_version, ts_us, mask, paper_mm):
    height, width = mask.shape
    payload = _FULL_PREFIX.pack(*paper_mm) + zlib.compress(
        np.ascontiguousarray(mask).tobytes(), 1
    )
    return _pack(KIND_FULL, generation, ink_version, ts_us, width, height, payload)


def pack_delta(generation, ink_version, ts_us, mask, box):
    x, y, w, h = box
    sub = np.ascontiguousarray(mask[y : y + h, x : x + w])
    payload = _DELTA_PREFIX.pack(x, y, w, h) + zlib.compress(sub.tobytes(), 1)
    height, width = mask.shape
    return _pack(KIND_DELTA, generation, ink_version, ts_us, width, height, payload)


def diff_box(previous, current):
    """Retângulo (x, y, w, h) das diferenças, ou None se iguais."""
    changed = previous != current
    rows = np.flatnonzero(changed.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(changed.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return x0, y0, x1 - x0, y1 - y0


def iter_records(read, max_payload=MAX_PAYLOAD_LEN):
    """Itera InkRecord de read(n) -> bytes; fim limpo encerra o laço.

    Valida magic, versão, tipo e limites ANTES de alocar o payload; qualquer
    violação levanta ProtocolError, sem tentar ressincronizar.
    """
    while True:
        header = _read_exact(read, HEADER_SIZE)
        if header is None:
            return
        magic, version, kind, _res, gen, ink_ver, ts_us, w, h, length = HEADER.unpack(
            header
        )
        if magic != MAGIC or version != VERSION:
            raise ProtocolError(
                f"registro DRAL invalido: magic={magic!r} versao={version}"
            )
        if kind not in (KIND_FULL, KIND_DELTA):
            raise ProtocolError(f"tipo de registro desconhecido: {kind}")
        if not 1 <= length <= max_payload:
            raise ProtocolError(f"payload_len fora de 1..{max_payload}: {length}")
        if not (0 < w <= MAX_SIDE and 0 < h <= MAX_SIDE and w * h <= MAX_PIXELS):
            raise ProtocolError(f"dimensoes invalidas: {w}x{h}")
        payload = _read_exact(read, length)
        if payload is None:
            raise ProtocolError("fluxo truncado: registro sem payload")
        yield InkRecord(kind, gen, ink_ver, ts_us, w, h, payload)


def records_from_file(path, max_payload=MAX_PAYLOAD_LEN):
    with open(path, "rb") as handle:
        yield from iter_records(handle.read, max_payload)


def _inflate(data, size):
    # Limite a size+1 bytes: payload malicioso nao vira bomba de descompressao.
    decomp = zlib.decompressobj()
    try:
        out = decomp.decompress(data, size + 1)
    except zlib.error as error:
        raise ProtocolError("payload zlib invalido") from error
    if len(out) != size or decomp.unconsumed_tail or not decomp.eof:
        raise ProtocolError("payload zlib com tamanho diferente do declarado")
    if decomp.unused_data:
        raise ProtocolError("bytes extras apos o payload zlib")
    return out


class InkCanvas:
    """Lado consumidor: aplica registros e mantém a máscara confirmada."""

    def __init__(self):
        self.mask = None
        self.generation = None
        self.ink_version = None
        self.paper_mm = None

    def apply(self, rec):
        """Aplica um registro; devolve o retângulo (x, y, w, h) alterado."""
        if rec.kind == KIND_FULL:
            if len(rec.payload) < _FULL_PREFIX.size:
                raise ProtocolError("registro FULL truncado")
            self.paper_mm = _FULL_PREFIX.unpack_from(rec.payload)
            raw = _inflate(rec.payload[_FULL_PREFIX.size :], rec.width * rec.height)
            self.mask = (
                np.frombuffer(raw, np.uint8).reshape(rec.height, rec.width).copy()
            )
            box = (0, 0, rec.width, rec.height)
        else:
            if (
                self.mask is None
                or self.generation != rec.generation
                or self.mask.shape != (rec.height, rec.width)
            ):
                raise ProtocolError("DELTA sem FULL anterior da mesma geracao")
            if len(rec.payload) < _DELTA_PREFIX.size:
                raise ProtocolError("registro DELTA truncado")
            x, y, w, h = _DELTA_PREFIX.unpack_from(rec.payload)
            if w == 0 or h == 0 or x + w > rec.width or y + h > rec.height:
                raise ProtocolError("retangulo do DELTA fora da pagina")
            raw = _inflate(rec.payload[_DELTA_PREFIX.size :], w * h)
            self.mask[y : y + h, x : x + w] = np.frombuffer(raw, np.uint8).reshape(h, w)
            box = (x, y, w, h)
        self.generation = rec.generation
        self.ink_version = rec.ink_version
        return box


class _Sink:
    """Um destino (socket ou arquivo) com o próprio estado de diff."""

    def __init__(self, write, close, alive=None, is_socket=False):
        self.write = write
        self.close = close
        self.alive = alive or (lambda: True)
        self.is_socket = is_socket
        self.mask = None
        self.generation = None


def _socket_alive(conn):
    """False se o consumidor fechou (leitura devolve b"" sem bloquear)."""
    try:
        readable, _, _ = select.select([conn], [], [], 0)
        if not readable:
            return True
        return conn.recv(1, socket.MSG_PEEK) != b""
    except (OSError, ValueError):
        return False


def _safe_close(sink, log=None):
    try:
        sink.close()
    except (OSError, ValueError) as error:
        if log is not None:
            log(f"Stream de tinta: falha ao fechar destino ({error}).")


class InkStreamPublisher:
    """Publica a tinta confirmada sem nunca bloquear quem publica.

    `port`: porta em 127.0.0.1 (0 = efêmera; None = sem socket, só gravação;
    com socket o token é obrigatório). `paper_mm`: (largura_mm, altura_mm) da
    folha — vai no registro FULL, para o consumidor travar a escala.
    `record_path`: grava o fluxo num arquivo privado (0600) para replay.

    `generation` no fio é do publicador, monotônica: sobe a cada geração da
    trabalhadora nova e a cada limpeza (troca de câmera).
    """

    def __init__(
        self,
        *,
        port=DEFAULT_PORT,
        host="127.0.0.1",
        token=None,
        token_file=None,
        paper_mm=(210.0, 297.0),
        record_path=None,
        clock=time.monotonic,
        log=print,
    ):
        if port is None and record_path is None:
            raise ValueError("sem porta e sem arquivo: nada a publicar")
        self._token = None if port is None else resolve_token(token, token_file)
        self._paper_mm = tuple(float(v) for v in paper_mm)
        self._clock = clock
        self._log = log
        self._cond = threading.Condition()
        self._pending = None
        self._closed = False
        self._sinks = []
        self._sinks_lock = threading.Lock()
        self._handshakes = threading.BoundedSemaphore(MAX_PENDING_HANDSHAKES)
        self._src_generation = None
        self._stream_generation = 0
        self._last_shape = None
        self._server = None
        self.port = None
        threads = []
        try:
            if port is not None:
                self._server = socket.create_server((host, port))
                self.port = self._server.getsockname()[1]
                # accept() com prazo: fechar o socket de outra thread nao o
                # destrava.
                self._server.settimeout(0.2)
                threads.append(
                    threading.Thread(
                        target=self._accept_loop, name="ink-stream-accept", daemon=True
                    )
                )
            if record_path is not None:
                self._sinks.append(self._open_record(record_path))
        except BaseException:
            if self._server is not None:
                self._server.close()
            raise
        threads.append(
            threading.Thread(
                target=self._send_loop, name="ink-stream-send", daemon=True
            )
        )
        self._threads = threads
        for thread in self._threads:
            thread.start()

    @staticmethod
    def _open_record(path):
        # 0600: a tinta gravada nao pode depender do token do socket. Sem
        # fchmod (Windows nativo) nao ha como garantir isso (chmod nao aplica
        # ACL): recusa em vez de gravar a tinta legivel para outros usuarios.
        if not _PRIVATE_FILES:
            raise RuntimeError(
                "gravacao privada do stream nao e garantida sem os.fchmod "
                "(Windows nativo); grave de dentro do WSL"
            )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
            handle = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        return _Sink(handle.write, handle.close)

    def publish(self, publication):
        """Guarda a última publicação e retorna na hora (latest-wins).

        publication: contracts.InkPublication; confirmed None = canvas limpo.
        """
        ts_us = int(self._clock() * 1_000_000)
        mask = publication.confirmed
        with self._cond:
            if mask is None:
                if self._last_shape is None:
                    return  # nunca publicou: nada a limpar
                self._src_generation = None
                self._stream_generation += 1
                mask = np.zeros(self._last_shape, np.uint8)
                mask.flags.writeable = False
                version = 0
            else:
                if publication.generation != self._src_generation:
                    self._src_generation = publication.generation
                    self._stream_generation += 1
                self._last_shape = mask.shape
                version = publication.version
            self._pending = (self._stream_generation, int(version), ts_us, mask)
            self._cond.notify()

    def close(self):
        """Encerra; a última publicação pendente ainda é despachada."""
        with self._cond:
            self._closed = True
            self._cond.notify()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        for thread in self._threads:
            thread.join(timeout=SEND_TIMEOUT_S + 1)
        with self._sinks_lock:
            sinks, self._sinks = self._sinks, []
        for sink in sinks:
            # No encerramento, perda de gravacao nao pode ser silenciosa.
            _safe_close(sink, self._log)

    def _accept_loop(self):
        while not self._closed:
            try:
                conn, _addr = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            if not self._handshakes.acquire(blocking=False):
                conn.close()  # handshakes demais em aberto: recusa sem thread
                continue
            threading.Thread(
                target=self._adopt, args=(conn,), name="ink-stream-hello", daemon=True
            ).start()

    def _adopt(self, conn):
        # Handshake em thread própria: cliente lento não trava o accept.
        try:
            try:
                if not server_handshake(conn, self._token):
                    conn.close()
                    return
                conn.settimeout(SEND_TIMEOUT_S)
            except OSError:
                conn.close()
                return
        finally:
            self._handshakes.release()
        sink = _Sink(
            conn.sendall, conn.close, lambda: _socket_alive(conn), is_socket=True
        )
        # Poda os mortos ANTES de contar: quem acabou de reconectar nao pode
        # custar a vaga de um consumidor saudavel.
        self._prune_dead()
        evicted = []
        with self._sinks_lock:
            self._sinks.append(sink)
            # Teto de consumidores: o mais antigo (nunca o arquivo) sai.
            consumers = [s for s in self._sinks if s.is_socket]
            while len(consumers) > MAX_CONSUMERS:
                oldest = consumers.pop(0)
                self._sinks.remove(oldest)
                evicted.append(oldest)
        for old in evicted:
            _safe_close(old)
        # Consumidor novo recebe FULL da última publicação mesmo parada.
        with self._cond:
            self._cond.notify()

    def _prune_dead(self):
        with self._sinks_lock:
            dead = [s for s in self._sinks if not s.alive()]
            if dead:
                self._sinks = [s for s in self._sinks if s not in dead]
        for sink in dead:
            _safe_close(sink)

    def _send_loop(self):
        last = None
        while True:
            with self._cond:
                if self._pending is None and not self._closed:
                    self._cond.wait(timeout=0.5)
                item, self._pending = self._pending, None
                closed = self._closed
            if item is None:
                if closed:
                    return
                # Acordou sem publicacao: limpa mortos e atende quem chegou.
                self._prune_dead()
                if not self._sinks_wanting_full(last):
                    continue
                item = last
            last = item
            try:
                self._dispatch(*item)
            except Exception as error:  # noqa: BLE001 - o envio nunca morre
                self._log(f"Stream de tinta: falha ao despachar ({error!r}).")
            if closed:
                return

    def _sinks_wanting_full(self, last):
        if last is None:
            return False
        with self._sinks_lock:
            return any(sink.mask is None for sink in self._sinks)

    def _dispatch(self, generation, ink_version, ts_us, mask):
        with self._sinks_lock:
            sinks = list(self._sinks)
        dead = []
        for sink in sinks:
            try:
                data = self._encode_for(sink, generation, ink_version, ts_us, mask)
                if data:
                    sink.write(data)
            except (OSError, ValueError):
                dead.append(sink)
        if dead:
            with self._sinks_lock:
                self._sinks = [s for s in self._sinks if s not in dead]
            for sink in dead:
                _safe_close(sink)
            self._log("Stream de tinta: destino removido (conexao ou disco).")

    def _encode_for(self, sink, generation, ink_version, ts_us, mask):
        full = (
            sink.mask is None
            or sink.generation != generation
            or sink.mask.shape != mask.shape
        )
        if full:
            data = pack_full(generation, ink_version, ts_us, mask, self._paper_mm)
        else:
            box = diff_box(sink.mask, mask)
            if box is None:
                return None
            data = pack_delta(generation, ink_version, ts_us, mask, box)
        sink.mask = mask
        sink.generation = generation
        return data
