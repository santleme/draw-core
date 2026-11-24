"""Cliente da libhvcam nativa e fallback Python puro (pacote draw.hvcam).

Vendored de santleme/hvcam (bindings/python/hvcam.py). A libhvcam.so e um
artefato do build do hvcam — o loader procura ao lado deste arquivo, em
<repo>/build ao lado do pacote e no LD_LIBRARY_PATH, e a carga e lazy:
importar este modulo nunca exige a lib; so abrir uma fonte hvcam:// sim.
"""

import ctypes
import errno
import os
import selectors
import socket
import time

from ._wire import (
    _HEADER_STRUCT,
    _HVCAM_MAGIC,
    _HVCAM_REQ,
    _HVCAM_VERSION,
    _MAX_PAYLOAD,
    _MIN_REQ_SECONDS,
    _NAP_SECONDS,
)


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
    # A base e src/draw/ (o pacote desceu um nivel: hvcam/ virou diretorio).
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    Os kwargs `timeout_ms` e `socket_factory` sao extensoes so-Python:
    limitam o connect() (que senao espera o SYN timeout do kernel) e dao
    uma seam de teste. A libhvcam.so nativa tem ABI C fixa — quem chama
    nunca repassa kwargs para ela.
    - hvcam_connect(port, *, timeout_ms=None, socket_factory=socket.socket)
    - hvcam_connect_tcp(host_bytes, port, *, timeout_ms=None, socket_factory=socket.socket)
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

    @staticmethod
    def _connect_errno(error):
        if error.errno is not None:
            return error.errno
        # socket.timeout/TimeoutError nao carrega errno; o chamador usa
        # -ETIMEDOUT para exibir a dica de "host nao respondeu".
        return errno.ETIMEDOUT if isinstance(error, TimeoutError) else errno.EIO

    def hvcam_connect(self, port, *, timeout_ms=None, socket_factory=socket.socket):
        af_vsock = getattr(socket, "AF_VSOCK", 40)
        cid_host = getattr(socket, "VMADDR_CID_HOST", 2)
        try:
            sock = socket_factory(af_vsock, socket.SOCK_STREAM)
        except OSError as error:
            return -(error.errno or errno.EIO)
        try:
            if timeout_ms is not None:
                sock.settimeout(timeout_ms / 1000.0)
            sock.connect((cid_host, int(port)))
            sock.setblocking(True)
            return self._register(sock)
        except OSError as error:
            sock.close()
            return -self._connect_errno(error)

    def hvcam_connect_tcp(
        self, host_bytes, port, *, timeout_ms=None, socket_factory=socket.socket
    ):
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
        timeout_s = None if timeout_ms is None else timeout_ms / 1000.0
        for family, socktype, proto, _canon, sockaddr in infos:
            sock = None
            try:
                sock = socket_factory(family, socktype, proto)
                if timeout_s is not None:
                    sock.settimeout(timeout_s)
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
    def _read_into(cls, sock, view, deadline):
        total = len(view)
        received = 0
        while received < total:
            w = cls._wait_readable(sock, deadline)
            if w == -2:
                return -1 if received > 0 else -2
            if w != 0:
                return w
            try:
                n = sock.recv_into(view[received:])
            except InterruptedError:
                continue
            except BlockingIOError:
                continue
            except OSError:
                return -1
            if n <= 0:
                return -1
            received += n
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
        if (
            buf is not None
            and payload_bytes > 0
            and payload_bytes <= int(cap)
            and hasattr(sock, "recv_into")
        ):
            try:
                view = memoryview(buf).cast("B")[:payload_bytes]
            except TypeError:
                view = None
            if view is not None:
                rc = cls._read_into(sock, view, deadline)
                if rc == -2:
                    return -1
                if rc != 0:
                    return rc
                return payload_bytes
        raw_payload, rc = cls._read_full(sock, payload_bytes, deadline)
        if rc == -2:
            return -1
        if rc != 0:
            return rc
        # O payload inteiro ja saiu do fio; so a parte que cabe no buffer
        # do chamador e copiada. Sem o min(), o memmove transborda buf.
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
