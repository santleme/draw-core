"""Fonte hvcam:// — camera do Windows via Hyper-V socket (santleme/hvcam).

O hvcam é o transporte nativo para a webcam que mora no lado Windows:
hvcam-host.exe captura por Media Foundation e serve o quadro mais novo
por AF_HYPERV (vsock) na porta 8765 — sem IP, sem firewall, sem usbipd.
`hvcam://` conecta ao host (requer o serviço vsock registrado uma vez no
Windows); `hvcam+reverse://` escuta no WSL para `hvcam-host.exe --reverse`
e não requer registro; fora do WSL, `hvcam+tcp://host` usa TCP.

O cliente ctypes abaixo é vendored de santleme/hvcam
(bindings/python/hvcam.py). A libhvcam.so é um artefato do build do
hvcam — o loader procura ao lado deste arquivo, em <repo>/build ao
lado do pacote e no LD_LIBRARY_PATH, e a carga é lazy: importar este
módulo nunca exige a lib; só abrir uma fonte hvcam:// sim.

HvcamCapture adapta a fonte ao contrato do VideoCapture que
camera.open_capture consome: read() -> (ok, frame BGR), isOpened(),
release(), e a identidade do emissor (last_frame_id,
last_capture_ts_ms, source_dropped por salto de seq) que o
LatestFrameCapture usa na telemetria.
"""

import atexit
import ctypes
import errno
import hashlib
import io
import os
import re
import selectors
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

# Porta padrao do servico vsock — a mesma do protocolo hvcam e do
# ajudante draw (coincidencia util: ambos falam 8765).
DEFAULT_PORT = 8765

# Release oficial do hvcam-host.exe (santleme/hvcam) para provisionamento
# automatico no WSL: o binario roda no lado Windows e transmite a webcam
# via AF_HYPERV sem admin nem instalacao manual.
HVCAM_HOST_RELEASE_URL = (
    "https://github.com/santleme/hvcam/releases/download/"
    "v0.1.1-rc.1/hvcam-v0.1.1-rc.1-windows-x86_64.zip"
)
HVCAM_HOST_ZIP_SHA256 = (
    "4bb9aeaf72ca983950b7ed65df3188ead146b7fdcd5a2b53e4280d82e8247c6e"
)
HVCAM_HOST_ZIP_MEMBER = "hvcam-v0.1.1-rc.1-windows-x86_64/hvcam-host.exe"
_HVCAM_HOST_MAX_ZIP_BYTES = 16 << 20
_HVCAM_HOST_DOWNLOAD_TIMEOUT_S = 60
_MANAGED_CONNECT_TIMEOUT_MS = 10_000
_DEVICE_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*$")

_MANAGED_HOST_PROCS = set()


def _terminate_host_process(proc):
    """Encerra um hvcam-host.exe iniciado pelo Draw, sem vazar processo."""
    if proc is None:
        return
    _MANAGED_HOST_PROCS.discard(proc)
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
    except OSError:
        pass
    finally:
        for stream in (
            getattr(proc, "stdout", None),
            getattr(proc, "stderr", None),
            getattr(proc, "stdin", None),
        ):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


def _cleanup_managed_hosts():
    for proc in list(_MANAGED_HOST_PROCS):
        _terminate_host_process(proc)


atexit.register(_cleanup_managed_hosts)

_TIMEOUT_MS = 1000
# No Hyper-V (AF_HYPERV), uma tentativa de connect() do host antes de o
# listener existir no WSL pode levar 30 s para o host rescindir a oferta
# mais 1 s de espera antes do retry do hvcam-host.exe --reverse: 35 s
# mantem o accept limitado e cobre um ciclo completo de timeout + retry.
_LISTEN_TIMEOUT_MS = 35_000
_MAX_PAYLOAD = 64 * 1024 * 1024
# Portas AF_VSOCK sao uint32; 0xFFFFFFFF e VMADDR_PORT_ANY (reservado).
_VSOCK_PORT_MAX = 0xFFFFFFFF - 1
_TCP_PORT_MAX = 65535
_HVCAM_MAGIC = 0x4856434D  # "HVCM"
_HVCAM_VERSION = 1
_HVCAM_REQ = b"Q"
_HEADER_STRUCT = struct.Struct("<IHHQQHHII")
_MIN_REQ_SECONDS = 0.020
_NAP_SECONDS = 0.015


class HvFrameHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint16),
        ("flags", ctypes.c_uint16),
        ("seq_num", ctypes.c_uint64),
        ("timestamp_us", ctypes.c_uint64),
        ("width", ctypes.c_uint16),
        ("height", ctypes.c_uint16),
        ("format", ctypes.c_uint32),
        ("payload_bytes", ctypes.c_uint32),
    ]


def _candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(here, "libhvcam.so"),
        os.path.join(here, "..", "..", "build", "libhvcam.so"),
        "libhvcam.so",
        os.path.join(here, "..", "..", "..", "hvcam", "build", "libhvcam.so"),
    ]
    try:
        import pwd

        home_dir = pwd.getpwuid(os.getuid()).pw_dir
    except (ImportError, AttributeError, KeyError, OSError):
        home_dir = None
    if home_dir:
        paths.append(os.path.join(home_dir, "hvcam", "build", "libhvcam.so"))
    return tuple(paths)


def _load():
    for cand in _candidates():
        try:
            lib = ctypes.CDLL(cand)
            break
        except OSError:
            continue
    else:
        raise OSError(
            "libhvcam.so nao encontrada — ela vem do build do "
            "santleme/hvcam (cmake --build build); exporte "
            "LD_LIBRARY_PATH apontando para o diretorio dela"
        )
    lib.hvcam_connect.argtypes = [ctypes.c_uint32]
    lib.hvcam_connect.restype = ctypes.c_int
    lib.hvcam_connect_tcp.argtypes = [ctypes.c_char_p, ctypes.c_uint16]
    lib.hvcam_connect_tcp.restype = ctypes.c_int
    lib.hvcam_grab_next.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(HvFrameHeader),
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
    ]
    lib.hvcam_grab_next.restype = ctypes.c_int
    lib.hvcam_close.argtypes = [ctypes.c_int]
    return lib


_LIB = None


def _lib():
    """Carga lazy: o import deste modulo nao exige libhvcam.so."""
    global _LIB
    if _LIB is None:
        _LIB = _load()
    return _LIB


class _PurePythonHvcamLib:
    """Implementacao Python pura da API de libhvcam.so (protocol.h).

    Permite rodar hvcam://, hvcam+reverse:// e hvcam+tcp:// numa maquina
    recem-clonada sem precisar compilar o repositorio C santleme/hvcam.
    Mantem a mesma assinatura e semantica de hvcam_client.c:
    - hvcam_connect(port) -> fd ou -errno
    - hvcam_connect_tcp(host_bytes, port) -> fd ou -1
    - hvcam_grab_next(fd, hdr_ptr, buf, cap, last_ptr, timeout_ms) -> bytes, -2 (timeout) ou -1 (erro)
    - hvcam_close(fd)

    Usa `socket.socket` (send/recv/close) para funcionar tambem no Windows
    nativo (onde um handle Winsock nao e um fd da CRT para os.read/os.write).
    """

    def __init__(self):
        self._sockets = {}

    def _register(self, sock):
        fd = sock.fileno()
        self._sockets[fd] = sock
        return fd

    def _socket_for(self, fd):
        sock = self._sockets.get(fd)
        if sock is not None:
            return sock
        # Conexao aceita externamente (ex.: _listen_vsock_bounded que retorna
        # um fd destacado via conn.detach()): reencapsula sem duplicar o fd.
        sock = socket.socket(fileno=fd)
        self._sockets[fd] = sock
        return sock

    def hvcam_connect(self, port):
        af_vsock = getattr(socket, "AF_VSOCK", 40)
        cid_host = getattr(socket, "VMADDR_CID_HOST", 2)
        try:
            sock = socket.socket(af_vsock, socket.SOCK_STREAM)
        except OSError as error:
            return -(error.errno or errno.EIO)
        try:
            sock.connect((cid_host, int(port)))
            sock.setblocking(True)
            return self._register(sock)
        except OSError as error:
            sock.close()
            return -(error.errno or errno.EIO)

    def hvcam_connect_tcp(self, host_bytes, port):
        host = (
            host_bytes.decode("utf-8", errors="replace")
            if isinstance(host_bytes, (bytes, bytearray))
            else str(host_bytes or "127.0.0.1")
        )
        if not host:
            host = "127.0.0.1"
        try:
            infos = socket.getaddrinfo(
                host, int(port), socket.AF_UNSPEC, socket.SOCK_STREAM
            )
        except OSError:
            return -1
        for family, socktype, proto, _canon, sockaddr in infos:
            sock = None
            try:
                sock = socket.socket(family, socktype, proto)
                sock.connect(sockaddr)
                sock.setblocking(True)
                return self._register(sock)
            except OSError:
                if sock is not None:
                    sock.close()
                continue
        return -1

    @staticmethod
    def _wait_readable(sock, deadline):
        with selectors.DefaultSelector() as sel:
            sel.register(sock, selectors.EVENT_READ)
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    return -2
                try:
                    ready = sel.select(left)
                except InterruptedError:
                    continue
                except OSError:
                    return -1
                if not ready:
                    return -2
                return 0

    @classmethod
    def _read_full(cls, sock, total, deadline):
        chunks = bytearray()
        remaining = total
        while remaining > 0:
            w = cls._wait_readable(sock, deadline)
            if w == -2:
                return None, (-1 if len(chunks) > 0 else -2)
            if w != 0:
                return None, w
            try:
                data = sock.recv(remaining)
            except InterruptedError:
                continue
            except BlockingIOError:
                continue
            except OSError:
                return None, -1
            if not data:
                return None, -1
            chunks.extend(data)
            remaining -= len(data)
        return bytes(chunks), 0

    @staticmethod
    def _write_full(sock, data):
        view = memoryview(data)
        sent = 0
        while sent < len(view):
            try:
                n = sock.send(view[sent:])
            except InterruptedError:
                continue
            except OSError:
                return -1
            if n <= 0:
                return -1
            sent += n
        return 0

    @classmethod
    def _pull_frame(cls, sock, hdr, buf, cap, deadline):
        if cls._write_full(sock, _HVCAM_REQ) < 0:
            return -1
        raw_hdr, rc = cls._read_full(sock, _HEADER_STRUCT.size, deadline)
        if rc != 0:
            return rc
        (
            magic,
            version,
            flags,
            seq_num,
            timestamp_us,
            width,
            height,
            fmt,
            payload_bytes,
        ) = _HEADER_STRUCT.unpack(raw_hdr)
        hdr.magic = magic
        hdr.version = version
        hdr.flags = flags
        hdr.seq_num = seq_num
        hdr.timestamp_us = timestamp_us
        hdr.width = width
        hdr.height = height
        hdr.format = fmt
        hdr.payload_bytes = payload_bytes
        if (
            magic != _HVCAM_MAGIC
            or version != _HVCAM_VERSION
            or payload_bytes > _MAX_PAYLOAD
        ):
            return -1
        raw_payload, rc = cls._read_full(sock, payload_bytes, deadline)
        if rc == -2:
            return -1
        if rc != 0:
            return rc
        stored = min(payload_bytes, int(cap))
        if stored > 0 and buf is not None:
            ctypes.memmove(buf, raw_payload, stored)
        return stored

    def hvcam_grab_next(self, fd, hdr_ptr, buf, cap, last_ptr, timeout_ms):
        try:
            sock = self._socket_for(fd)
        except OSError:
            return -1
        hdr = hdr_ptr._obj if hasattr(hdr_ptr, "_obj") else hdr_ptr.contents
        last = last_ptr._obj if hasattr(last_ptr, "_obj") else last_ptr.contents
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        drained = False
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return -2
            if drained and left < _MIN_REQ_SECONDS:
                return -2
            n = self._pull_frame(sock, hdr, buf, cap, deadline)
            if n == -2:
                return -1
            if n < 0:
                return n
            drained = True
            if (hdr.flags & 1) and hdr.seq_num != last.value:
                last.value = hdr.seq_num
                return n
            nap_end = min(time.monotonic() + _NAP_SECONDS, deadline)
            w = self._wait_readable(sock, nap_end)
            if w == -1:
                return -1

    def hvcam_close(self, fd):
        if fd < 0:
            return
        sock = self._sockets.pop(fd, None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
            return
        try:
            os.close(fd)
        except OSError:
            pass


_PYTHON_LIB = _PurePythonHvcamLib()


def _lib_or_fallback():
    """Carrega libhvcam.so se disponivel; senao usa o cliente Python puro."""
    try:
        return _lib()
    except OSError:
        return _PYTHON_LIB


def _listen_vsock_bounded(
    port,
    timeout_ms=_LISTEN_TIMEOUT_MS,
    *,
    socket_factory=socket.socket,
    selector_factory=selectors.DefaultSelector,
    clock=time.monotonic,
    on_listening=None,
    abort_check=None,
):
    """Escuta em AF_VSOCK com prazo de accept e exige peer CID do host (2).

    `hvcam_listen` da libhvcam C chama `accept()` bloqueante sem prazo e
    sem checar o CID de origem; aqui o listener fecha no timeout com
    `-ETIMEDOUT` (para nao travar o startup nem o `--check`) e descarta
    conexoes que nao venham de `VMADDR_CID_HOST`.

    `on_listening`, se fornecido, roda logo apos `listen(1)` (ideal para
    subir `hvcam-host.exe --reverse` sem corrida de 30 s no AF_HYPERV).
    `abort_check`, se fornecido, e consultado periodicamente para abortar
    cedo caso o processo host tenha saido com erro.
    """
    if timeout_ms is not None and timeout_ms <= 0:
        return -errno.ETIMEDOUT
    af_vsock = getattr(socket, "AF_VSOCK", 40)
    cid_any = getattr(socket, "VMADDR_CID_ANY", 0xFFFFFFFF)
    cid_host = getattr(socket, "VMADDR_CID_HOST", 2)
    try:
        listener = socket_factory(af_vsock, socket.SOCK_STREAM)
    except OSError as error:
        return -(error.errno or errno.EIO)
    try:
        listener.setblocking(False)
        listener.bind((cid_any, int(port)))
        listener.listen(1)
        if on_listening is not None:
            on_listening()
        deadline = None if timeout_ms is None else clock() + (timeout_ms / 1000.0)
        with selector_factory() as selector:
            selector.register(listener, selectors.EVENT_READ)
            while True:
                if abort_check is not None and abort_check():
                    return -errno.ECONNABORTED
                remaining = None if deadline is None else deadline - clock()
                if remaining is not None and remaining <= 0:
                    return -errno.ETIMEDOUT
                wait_s = remaining
                if abort_check is not None:
                    wait_s = 0.2 if wait_s is None else min(wait_s, 0.2)
                if not selector.select(wait_s):
                    if abort_check is not None:
                        continue
                    return -errno.ETIMEDOUT
                try:
                    conn, peer = listener.accept()
                except BlockingIOError:
                    continue
                peer_cid = (
                    peer[0] if isinstance(peer, tuple) and len(peer) >= 1 else None
                )
                if peer_cid != cid_host:
                    conn.close()
                    continue
                conn.setblocking(True)
                return conn.detach()
    except OSError as error:
        return -(error.errno or errno.EIO)
    finally:
        listener.close()


class Camera:
    """Uma conexao hvcam; puxa o quadro mais novo sob demanda.

    Diverge do binding upstream num ponto: host e porta chegam
    separados (host=None = vsock), em vez de uma string "host:port"
    repartida no ':' — que quebrava IPv6 (`[::1]` -> "::1:8765").
    """

    def __init__(
        self,
        port=DEFAULT_PORT,
        host=None,
        listen=False,
        timeout_ms=_LISTEN_TIMEOUT_MS,
        listener=None,
        on_listening=None,
        abort_check=None,
    ):
        if host is not None and listen:
            raise ValueError("hvcam: listen mode cannot use a TCP host")
        if listen and not 1 <= int(port) <= _VSOCK_PORT_MAX:
            raise ValueError(f"porta hvcam invalida: {port!r}")
        lib = _lib_or_fallback()
        self._lib = lib
        if host is not None:
            self.fd = lib.hvcam_connect_tcp(host.encode(), port)
        elif listen:
            if listener is None:
                if on_listening is None and abort_check is None:
                    self.fd = _listen_vsock_bounded(port, timeout_ms)
                else:
                    self.fd = _listen_vsock_bounded(
                        port,
                        timeout_ms,
                        on_listening=on_listening,
                        abort_check=abort_check,
                    )
            else:
                self.fd = listener(port, timeout_ms)
        else:
            self.fd = lib.hvcam_connect(port)
        if self.fd < 0:
            operation = "listen" if listen else "connect"
            detail = f"hvcam: {operation} failed ({self.fd})"
            if self.fd == -errno.ETIMEDOUT:
                if listen:
                    detail += (
                        "; timeout: hvcam-host.exe --reverse nao conectou a "
                        f"tempo na porta {port}. No Windows, inicie "
                        f"hvcam-host.exe --reverse --port {port}"
                    )
                else:
                    reverse_source = (
                        "hvcam+reverse://"
                        if port == DEFAULT_PORT
                        else f"hvcam+reverse://:{port}"
                    )
                    detail += (
                        "; timeout: hvcam-host.exe nao respondeu. No Windows, "
                        f"inicie hvcam-host.exe --reverse --port {port} e use "
                        f"--source {reverse_source} no Draw, ou registre o "
                        "servico uma vez como administrador com "
                        f"hvcam-host.exe --register-vsock --port {port}"
                    )
            raise OSError(detail)
        self._buf = (ctypes.c_ubyte * _MAX_PAYLOAD)()
        self._hdr = HvFrameHeader()
        self._last = ctypes.c_uint64(0xFFFFFFFFFFFFFFFF)

    def grab(self, timeout_ms=_TIMEOUT_MS):
        """Devolve (info, payload) do quadro mais novo.

        Bloqueia ate chegar um quadro mais novo que o anterior ou o
        timeout vencer (TimeoutError). O ndarray e uma view sobre o
        buffer persistente — to_bgr() ja materializa a imagem.
        """
        lib = getattr(self, "_lib", None) or _lib_or_fallback()
        n = lib.hvcam_grab_next(
            self.fd,
            ctypes.byref(self._hdr),
            self._buf,
            _MAX_PAYLOAD,
            ctypes.byref(self._last),
            timeout_ms,
        )
        if n == -2:
            raise TimeoutError("hvcam: nenhum quadro novo")
        if n < 0:
            raise OSError("hvcam: stream dropped")
        h = self._hdr
        arr = np.frombuffer(self._buf, dtype=np.uint8, count=n)
        info = {
            "seq": h.seq_num,
            "ts_us": h.timestamp_us,
            "flags": h.flags,
            "width": h.width,
            "height": h.height,
            "format": h.format,
            "payload": n,
        }
        return info, arr

    @staticmethod
    def _expected_bytes(w, h, fmt):
        if fmt == 1:
            return w * h * 3
        if fmt == 2:
            return w * h * 2
        if fmt == 3:
            return (h * 3 // 2) * w
        if fmt == 4:
            return None
        if fmt == 5:
            return w * h * 4
        raise ValueError(f"hvcam: formato de fio desconhecido {fmt}")

    def to_bgr(self, info, frame):
        """Converte o payload para BGR; None num placeholder reply.

        Formatos de fio (hvcam protocol.h): 1=BGR24 2=YUY2 3=NV12
        4=MJPEG 5=BGRX32. flags bit0 clear = placeholder; bit1 =
        linhas bottom-up (flip no fim). Payload de tamanho nao exato
        e rejeitado — padding entrelinhas viraria pixel.
        """
        w, hgt, fmt = info["width"], info["height"], info["format"]
        flags = info.get("flags", 1)
        if not flags & 1:
            return None
        expected = self._expected_bytes(w, hgt, fmt)
        if expected is not None and frame.nbytes != expected:
            raise ValueError(
                f"hvcam: payload {frame.nbytes}B != {expected}B para "
                f"{w}x{hgt} fmt={fmt} (stride solto nao suportado)"
            )
        if fmt == 1:
            img = frame.reshape(hgt, w, 3).copy()
        elif fmt == 2:
            img = cv2.cvtColor(frame.reshape(hgt, w, 2), cv2.COLOR_YUV2BGR_YUY2)
        elif fmt == 3:
            img = cv2.cvtColor(
                frame.reshape(hgt * 3 // 2, w),
                cv2.COLOR_YUV2BGR_NV12,
            )
        elif fmt == 4:
            img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("hvcam: payload MJPEG falhou no decode")
        else:
            img = cv2.cvtColor(frame.reshape(hgt, w, 4), cv2.COLOR_BGRA2BGR)
        return cv2.flip(img, 0) if flags & 2 else img

    def close(self):
        if self.fd >= 0:
            lib = getattr(self, "_lib", None) or _lib_or_fallback()
            lib.hvcam_close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def is_hvcam_source(source):
    """True para fontes vsock, reverse-vsock ou TCP do hvcam."""
    if not isinstance(source, str):
        return False
    lower = source.strip().lower()
    return lower.startswith(("hvcam://", "hvcam+tcp://", "hvcam+reverse://"))


def parse_hvcam_endpoint(source):
    """Extrai (host, porta) da URL hvcam. host None = vsock.

    `hvcam://` e `hvcam://:porta` falam vsock com o host Windows da
    propria maquina; `hvcam+reverse://[:porta]` escuta por uma conexao
    iniciada pelo hvcam-host. Vsock nao tem hostname e aceita portas de
    32 bits (1..2^32-2; 0xFFFFFFFF e VMADDR_PORT_ANY).
    `hvcam+tcp://host[:porta]` usa o fallback TCP (1..65535).
    """
    text = source.strip()
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme == "hvcam+tcp":
        try:
            port = parsed.port if parsed.port is not None else DEFAULT_PORT
        except ValueError as error:
            raise ValueError(f"porta hvcam invalida: {source!r}") from error
        if not 1 <= port <= _TCP_PORT_MAX:
            raise ValueError(f"porta hvcam invalida: {source!r}")
        if not parsed.hostname:
            raise ValueError(f"hvcam+tcp sem host: {source!r}")
        return parsed.hostname, port

    host_part, sep, port_text = parsed.netloc.partition(":")
    if host_part:
        hostname = parsed.hostname or host_part
        if scheme == "hvcam+reverse":
            raise ValueError(
                f"hvcam+reverse nao aceita host ({hostname!r}); "
                "use hvcam:// para conexao direta"
            )
        shown_port = port_text if (sep and port_text) else DEFAULT_PORT
        raise ValueError(
            f"vsock nao tem host ({hostname!r}); para TCP use "
            f"hvcam+tcp://{hostname}:{shown_port}"
        )
    if not sep or not port_text:
        port = DEFAULT_PORT
    else:
        if not port_text.isascii() or not port_text.isdigit():
            raise ValueError(f"porta hvcam invalida: {source!r}")
        port = int(port_text, 10)
    if not 1 <= port <= _VSOCK_PORT_MAX:
        raise ValueError(f"porta hvcam invalida: {source!r}")
    return None, port


class HvcamCapture:
    """Adaptador compativel com VideoCapture sobre fontes hvcam.

    Alem de read() -> (ok, frame), expoe last_frame_id,
    last_capture_ts_ms e source_dropped (saltos de seq do emissor)
    para o LatestFrameCapture — mesmo contrato do HelperCapture.
    A camera_factory injeta um Camera fake nos testes.
    """

    def __init__(
        self,
        source,
        timeout_ms=_TIMEOUT_MS,
        camera_factory=None,
        connect_timeout_ms=None,
        on_listening=None,
        abort_check=None,
        host_process=None,
    ):
        self.source = str(source).strip()
        self._timeout_ms = timeout_ms
        self._connect_timeout_ms = connect_timeout_ms
        self._on_listening = on_listening
        self._abort_check = abort_check
        self._host_proc = host_process
        self._cam = None
        self._opened = False
        self.last_frame_id = None
        self.last_capture_ts_ms = None
        self.source_dropped = 0
        self._last_seq = None
        # Por que a fonte nao abriu (URL invalida, libhvcam.so ausente
        # ou connect recusado): os chamadores exibem isso em vez de um
        # "host fora do ar" generico.
        self.open_error = None
        self._open(camera_factory or Camera)

    def _open(self, camera_factory):
        try:
            host, port = parse_hvcam_endpoint(self.source)
        except ValueError as error:
            self.open_error = error
            return
        try:
            scheme = urlparse(self.source).scheme.lower()
            if host is not None:
                self._cam = camera_factory(host=host, port=port)
            elif scheme == "hvcam+reverse":
                listen_kwargs = {"port": port, "listen": True}
                if self._connect_timeout_ms is not None:
                    listen_kwargs["timeout_ms"] = self._connect_timeout_ms
                if self._on_listening is not None:
                    listen_kwargs["on_listening"] = self._on_listening
                if self._abort_check is not None:
                    listen_kwargs["abort_check"] = self._abort_check
                self._cam = camera_factory(**listen_kwargs)
            else:
                self._cam = camera_factory(port=port)
        except OSError as error:
            self._cam = None
            self.open_error = error
            _terminate_host_process(self._host_proc)
            self._host_proc = None
            return
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened or self._cam is None:
            return False, None
        try:
            info, payload = self._cam.grab(self._timeout_ms)
        except (TimeoutError, OSError):
            return False, None
        seq = info.get("seq")
        if seq is not None:
            if self._last_seq is not None and seq > self._last_seq + 1:
                self.source_dropped += seq - self._last_seq - 1
            self._last_seq = seq
        try:
            frame = self._cam.to_bgr(info, payload)
        except (ValueError, cv2.error):
            return False, None
        if frame is None:
            return False, None
        self.last_frame_id = int(seq) if seq is not None else None
        self.last_capture_ts_ms = int(info["ts_us"] // 1000)
        return True, frame

    def release(self):
        self._opened = False
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:  # noqa: BLE001 - release nunca levanta
                pass
            self._cam = None
        if self._host_proc is not None:
            _terminate_host_process(self._host_proc)
            self._host_proc = None


def _windows_cache_dir():
    """Pasta de cache no filesystem do Windows (LocalApplicationData\\draw).

    Windows Media Foundation bloqueia o acesso a camera com 0x80070490
    quando hvcam-host.exe e executado a partir de um caminho UNC do WSL
    (\\\\wsl.localhost\\...). Copiar/extrair o binario para o disco do
    Windows evita esse bloqueio sem exigir configuracao manual.

    Usa a API de Known Folders do .NET com saida UTF-8 explicita para nao
    ler variaveis de ambiente nem quebrar em perfis com acento (CP850/1252).
    """
    cwd = "/mnt/c" if os.path.isdir("/mnt/c") else None
    ps_cmd = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        "[Environment]::GetFolderPath('LocalApplicationData')"
    )
    try:
        raw_bytes = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        raw = raw_bytes.decode("utf-8").strip()
        if not raw:
            return None
        wsl_bytes = subprocess.check_output(
            ["wslpath", raw],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        wsl_path = wsl_bytes.decode("utf-8").strip()
        if not wsl_path:
            return None
        return Path(wsl_path) / "draw"
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None


def _local_hvcam_host_candidates():
    """Locais onde um checkout/instalacao local do hvcam-host.exe pode morar."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parents[2] / "hvcam" / "dist" / "hvcam-host.exe",
        here.parents[2] / "hvcam" / "build" / "Release" / "hvcam-host.exe",
        Path.home() / "hvcam" / "dist" / "hvcam-host.exe",
        Path.home() / "hvcam" / "build" / "Release" / "hvcam-host.exe",
    ]
    found_in_path = shutil.which("hvcam-host.exe")
    if found_in_path:
        candidates.append(Path(found_in_path))
    return tuple(candidates)


def _atomic_write_executable(target, payload_bytes):
    """Grava `payload_bytes` em `target` de forma atomica com permissao +x."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".part", dir=str(target.parent)
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(payload_bytes)
        try:
            os.chmod(tmp_path, 0o755)
        except OSError:
            pass
        os.replace(tmp_path, target)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return target


def ensure_hvcam_host(
    *,
    cache_dir=None,
    urlopen=None,
    release_url=None,
    expected_sha256=None,
    local_candidates=None,
):
    """Garante hvcam-host.exe pronto no disco do Windows (copia local ou download).

    Verifica o SHA-256 do pacote zip oficial antes de extrair (fail-closed).
    Devolve o Path do executavel ou levanta RuntimeError em caso de falha.
    """
    win_dir = Path(cache_dir) if cache_dir is not None else _windows_cache_dir()
    if win_dir is None:
        raise RuntimeError(
            "nao consegui localizar %LOCALAPPDATA% no Windows via interop do WSL"
        )
    target = win_dir / "hvcam-host.exe"
    if target.is_file() and target.stat().st_size > 0:
        try:
            os.chmod(target, 0o755)
        except OSError:
            pass
        return target

    candidates = (
        _local_hvcam_host_candidates() if local_candidates is None else local_candidates
    )
    for cand in candidates:
        cand_path = Path(cand)
        if cand_path.is_file() and cand_path.stat().st_size > 0:
            try:
                _atomic_write_executable(target, cand_path.read_bytes())
                return target
            except OSError:
                continue

    url = (release_url or HVCAM_HOST_RELEASE_URL).strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError(f"URL de release do hvcam precisa ser https://: {url!r}")
    want_sha = (expected_sha256 or HVCAM_HOST_ZIP_SHA256).strip().lower()
    opener = urllib.request.urlopen if urlopen is None else urlopen
    print("Baixando hvcam-host.exe da Release oficial (santleme/hvcam)...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "draw-core/0.1"})
        with opener(req, timeout=_HVCAM_HOST_DOWNLOAD_TIMEOUT_S) as resp:
            chunks = bytearray()
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                chunks.extend(chunk)
                if len(chunks) > _HVCAM_HOST_MAX_ZIP_BYTES:
                    raise RuntimeError(
                        f"download do hvcam excedeu {_HVCAM_HOST_MAX_ZIP_BYTES} bytes"
                    )
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(
            f"falha ao baixar hvcam-host.exe de {url}: {error}"
        ) from error

    actual_sha = hashlib.sha256(chunks).hexdigest()
    if actual_sha != want_sha:
        raise RuntimeError(
            f"sha256 divergente no download do hvcam: {actual_sha} != {want_sha}"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(chunks)) as zf:
            member = HVCAM_HOST_ZIP_MEMBER
            if member not in zf.namelist():
                matches = [
                    name
                    for name in zf.namelist()
                    if name.lower().endswith("/hvcam-host.exe")
                    or name.lower() == "hvcam-host.exe"
                ]
                if not matches:
                    raise RuntimeError("hvcam-host.exe nao encontrado no pacote zip")
                member = matches[0]
            exe_bytes = zf.read(member)
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"pacote zip do hvcam corrompido: {error}") from error

    _atomic_write_executable(target, exe_bytes)
    print(f"hvcam-host.exe salvo e verificado em {target}.")
    return target


def list_hvcam_devices(host_exe):
    """Lista webcams disponiveis no Windows via `hvcam-host.exe --list`."""
    exe = Path(host_exe)
    cwd = "/mnt/c" if os.path.isdir("/mnt/c") else str(exe.parent)
    try:
        output = subprocess.check_output(
            [str(exe), "--list"],
            cwd=cwd,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    try:
        text = output.decode("utf-8")
    except UnicodeDecodeError:
        text = output.decode("cp1252", errors="replace")
    devices = []
    for line in text.splitlines():
        match = _DEVICE_LINE_RE.match(line)
        if match:
            devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def open_managed_reverse(
    port=DEFAULT_PORT,
    device=0,
    *,
    timeout_ms=_TIMEOUT_MS,
    connect_timeout_ms=_MANAGED_CONNECT_TIMEOUT_MS,
    host_exe=None,
    devices=None,
    try_all=False,
    popen=subprocess.Popen,
    capture_cls=None,
):
    """Sobe `hvcam-host.exe --reverse` automaticamente e conecta via vsock.

    Quando `try_all=True` (padrao para `--source 0`), tenta primeiro o
    indice pedido e, se a camera nao abrir, percorre as demais webcams
    encontradas no Windows ate achar uma que funcione. O processo host
    fica preso ao `HvcamCapture` retornado e e encerrado no `release()`.
    Devolve `(HvcamCapture, device_index, device_name)` ou `(None, None, None)`.
    """
    exe = Path(host_exe) if host_exe is not None else ensure_hvcam_host()
    discovered = list_hvcam_devices(exe) if devices is None else list(devices)
    names = dict(discovered)

    order = [(int(device), names.get(int(device), f"Camera {device}"))]
    if try_all:
        for idx, name in discovered:
            if idx != int(device):
                order.append((idx, name))

    url = (
        "hvcam+reverse://"
        if int(port) == DEFAULT_PORT
        else f"hvcam+reverse://:{int(port)}"
    )
    run_cwd = "/mnt/c" if os.path.isdir("/mnt/c") else str(exe.parent)
    cls = HvcamCapture if capture_cls is None else capture_cls

    last_host_error = None
    for dev_idx, dev_name in order:
        proc_box = [None]
        # A file lets the host keep logging after connect without a pipe reader.
        # The child keeps its own descriptor until it exits.
        with tempfile.TemporaryFile() as stderr_log:

            def _spawn(idx=dev_idx, box=proc_box):
                proc = popen(
                    [
                        str(exe),
                        "--reverse",
                        "--port",
                        str(int(port)),
                        "--device",
                        str(idx),
                    ],
                    cwd=run_cwd,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_log,
                )
                box[0] = proc
                _MANAGED_HOST_PROCS.add(proc)

            def _aborted(box=proc_box):
                proc = box[0]
                return proc is not None and proc.poll() is not None

            try:
                cap = cls(
                    url,
                    timeout_ms=timeout_ms,
                    connect_timeout_ms=connect_timeout_ms,
                    on_listening=_spawn,
                    abort_check=_aborted,
                )
                if cap.isOpened():
                    if not try_all or _probe_hvcam_frame(cap):
                        cap._host_proc = proc_box[0]
                        proc_box[0] = None
                        return cap, dev_idx, dev_name
                elif _aborted():
                    size = stderr_log.seek(0, os.SEEK_END)
                    stderr_log.seek(max(0, size - 16_384))
                    err_msg = stderr_log.read(16_384).decode("utf-8", errors="replace")
                    if err_msg.strip():
                        last_host_error = err_msg.strip()
                cap.release()
            finally:
                _terminate_host_process(proc_box[0])
    if last_host_error:
        raise RuntimeError(last_host_error)
    return None, None, None


def _probe_hvcam_frame(capture, attempts=3):
    """Confirma que a fonte hvcam entrega ao menos um quadro valido."""
    for _ in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None:
            return True
    return False
