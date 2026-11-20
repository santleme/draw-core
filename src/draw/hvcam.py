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
from urllib.parse import urlparse

import cv2
import numpy as np

# Porta padrao do servico vsock — a mesma do protocolo hvcam e do
# ajudante draw (coincidencia util: ambos falam 8765).
DEFAULT_PORT = 8765

_TIMEOUT_MS = 1000
_MAX_PAYLOAD = 64 * 1024 * 1024


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


def _load():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, "libhvcam.so"),
        os.path.join(here, "..", "..", "build", "libhvcam.so"),
        os.path.expanduser("~/hvcam/build/libhvcam.so"),
        "libhvcam.so",
    ):
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
    # Older builds can still use direct vsock/TCP. Reverse mode needs the
    # listener symbol added by hvcam's no-admin connection path.
    try:
        lib.hvcam_listen.argtypes = [ctypes.c_uint32]
        lib.hvcam_listen.restype = ctypes.c_int
    except AttributeError:
        pass
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


class Camera:
    """Uma conexao hvcam; puxa o quadro mais novo sob demanda.

    Diverge do binding upstream num ponto: host e porta chegam
    separados (host=None = vsock), em vez de uma string "host:port"
    repartida no ':' — que quebrava IPv6 (`[::1]` -> "::1:8765").
    """

    def __init__(self, port=DEFAULT_PORT, host=None, listen=False):
        lib = _lib()
        if host is not None and listen:
            raise ValueError("hvcam: listen mode cannot use a TCP host")
        if host is not None:
            self.fd = lib.hvcam_connect_tcp(host.encode(), port)
        elif listen:
            listen_fn = getattr(lib, "hvcam_listen", None)
            if listen_fn is None:
                raise OSError(
                    "libhvcam.so sem hvcam_listen; reconstrua a biblioteca "
                    "com a versao atual de santleme/hvcam"
                )
            self.fd = listen_fn(port)
        else:
            self.fd = lib.hvcam_connect(port)
        if self.fd < 0:
            operation = "listen" if listen else "connect"
            detail = f"hvcam: {operation} failed ({self.fd})"
            if not listen and self.fd == -errno.ETIMEDOUT:
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
        n = _lib().hvcam_grab_next(
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
            _lib().hvcam_close(self.fd)
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
    iniciada pelo hvcam-host. Vsock nao tem hostname. `hvcam+tcp://host[:porta]`
    usa o fallback TCP.
    """
    text = source.strip()
    parsed = urlparse(text)
    port = parsed.port if parsed.port is not None else DEFAULT_PORT
    if port < 1:
        raise ValueError(f"porta hvcam invalida: {source!r}")
    if parsed.scheme.lower() == "hvcam+tcp":
        if not parsed.hostname:
            raise ValueError(f"hvcam+tcp sem host: {source!r}")
        return parsed.hostname, port
    if parsed.scheme.lower() == "hvcam+reverse" and parsed.hostname:
        raise ValueError(
            f"hvcam+reverse nao aceita host ({parsed.hostname!r}); "
            "use hvcam:// para conexao direta"
        )
    if parsed.hostname:
        raise ValueError(
            f"vsock nao tem host ({parsed.hostname!r}); para TCP use "
            f"hvcam+tcp://{parsed.hostname}:{port}"
        )
    return None, port


class HvcamCapture:
    """Adaptador compativel com VideoCapture sobre fontes hvcam.

    Alem de read() -> (ok, frame), expoe last_frame_id,
    last_capture_ts_ms e source_dropped (saltos de seq do emissor)
    para o LatestFrameCapture — mesmo contrato do HelperCapture.
    A camera_factory injeta um Camera fake nos testes.
    """

    def __init__(self, source, timeout_ms=_TIMEOUT_MS, camera_factory=None):
        self.source = str(source).strip()
        self._timeout_ms = timeout_ms
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
                self._cam = camera_factory(port=port, listen=True)
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
