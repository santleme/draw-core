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

import ctypes
import errno
import os
import selectors
import socket
import time
from urllib.parse import urlparse

import cv2
import numpy as np

# Porta padrao do servico vsock — a mesma do protocolo hvcam e do
# ajudante draw (coincidencia util: ambos falam 8765).
DEFAULT_PORT = 8765

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


def _listen_vsock_bounded(
    port,
    timeout_ms=_LISTEN_TIMEOUT_MS,
    *,
    socket_factory=socket.socket,
    selector_factory=selectors.DefaultSelector,
    clock=time.monotonic,
):
    """Escuta em AF_VSOCK com prazo de accept e exige peer CID do host (2).

    `hvcam_listen` da libhvcam C chama `accept()` bloqueante sem prazo e
    sem checar o CID de origem; aqui o listener fecha no timeout com
    `-ETIMEDOUT` (para nao travar o startup nem o `--check`) e descarta
    conexoes que nao venham de `VMADDR_CID_HOST`.
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
        deadline = None if timeout_ms is None else clock() + (timeout_ms / 1000.0)
        with selector_factory() as selector:
            selector.register(listener, selectors.EVENT_READ)
            while True:
                remaining = None if deadline is None else deadline - clock()
                if remaining is not None and remaining <= 0:
                    return -errno.ETIMEDOUT
                if not selector.select(remaining):
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
    ):
        if host is not None and listen:
            raise ValueError("hvcam: listen mode cannot use a TCP host")
        if listen and not 1 <= int(port) <= _VSOCK_PORT_MAX:
            raise ValueError(f"porta hvcam invalida: {port!r}")
        lib = _lib()
        self._lib = lib
        if host is not None:
            self.fd = lib.hvcam_connect_tcp(host.encode(), port)
        elif listen:
            listen_fn = _listen_vsock_bounded if listener is None else listener
            self.fd = listen_fn(port, timeout_ms)
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
        lib = getattr(self, "_lib", None) or _lib()
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
            lib = getattr(self, "_lib", None) or _lib()
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
    ):
        self.source = str(source).strip()
        self._timeout_ms = timeout_ms
        self._connect_timeout_ms = connect_timeout_ms
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
                self._cam = camera_factory(**listen_kwargs)
            else:
                self._cam = camera_factory(port=port)
        except OSError as error:
            self._cam = None
            self.open_error = error
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
