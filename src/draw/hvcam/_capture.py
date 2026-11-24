"""Conexao hvcam e adaptador VideoCapture (pacote draw.hvcam).

HvcamCapture adapta a fonte ao contrato do VideoCapture que
camera.open_capture consome: read() -> (ok, frame BGR), isOpened(),
release(), e a identidade do emissor (last_frame_id,
last_capture_ts_ms, source_dropped por salto de seq) que o
LatestFrameCapture usa na telemetria.
"""

import ctypes
import errno
import logging
from urllib.parse import urlparse

import cv2
import numpy as np

from .._hvcam_mjpeg import decode_mjpeg as _imdecode_mjpeg_strict
from ._lib import HvFrameHeader, _lib_or_fallback, _PurePythonHvcamLib
from ._mjpeg import _sanitize_mjpeg
from ._process import _terminate_host_process
from ._transport import _listen_vsock_bounded
from ._wire import (
    _LISTEN_TIMEOUT_MS,
    _MAX_PAYLOAD,
    _TCP_PORT_MAX,
    _TIMEOUT_MS,
    _VSOCK_PORT_MAX,
    DEFAULT_PORT,
)

_LOG = logging.getLogger(__name__)


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
        connect_timeout_ms=None,
    ):
        if host is not None and listen:
            raise ValueError("hvcam: listen mode cannot use a TCP host")
        if listen and not 1 <= int(port) <= _VSOCK_PORT_MAX:
            raise ValueError(f"porta hvcam invalida: {port!r}")
        lib = _lib_or_fallback()
        self._lib = lib
        # O timeout de connect so existe no fallback Python puro: a
        # libhvcam.so nativa tem ABI C fixa e nao aceita kwargs.
        lib_kwargs = {}
        if connect_timeout_ms is not None and isinstance(lib, _PurePythonHvcamLib):
            lib_kwargs["timeout_ms"] = connect_timeout_ms
        if host is not None:
            self.fd = lib.hvcam_connect_tcp(host.encode(), port, **lib_kwargs)
        elif listen:
            if listener is None:
                listen_kwargs = {}
                if on_listening is not None:
                    listen_kwargs["on_listening"] = on_listening
                if abort_check is not None:
                    listen_kwargs["abort_check"] = abort_check
                self.fd = _listen_vsock_bounded(port, timeout_ms, **listen_kwargs)
            else:
                self.fd = listener(port, timeout_ms)
        else:
            self.fd = lib.hvcam_connect(port, **lib_kwargs)
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
        n = self._lib.hvcam_grab_next(
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
            clean = _sanitize_mjpeg(frame)
            img = _imdecode_mjpeg_strict(clean)
        else:
            img = cv2.cvtColor(frame.reshape(hgt, w, 4), cv2.COLOR_BGRA2BGR)
        return cv2.flip(img, 0) if flags & 2 else img

    def close(self):
        if self.fd >= 0:
            self._lib.hvcam_close(self.fd)
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
        try:
            shown_port = parsed.port if parsed.port is not None else DEFAULT_PORT
        except ValueError:
            shown_port = port_text
        # host:port e o formato de hvcam+tcp://; IPv6 precisa de colchetes.
        tcp_host = f"[{hostname}]" if ":" in hostname else hostname
        raise ValueError(
            f"vsock nao tem host ({hostname!r}); para TCP use "
            f"hvcam+tcp://{tcp_host}:{shown_port}"
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
    ):
        self.source = str(source).strip()
        self._timeout_ms = timeout_ms
        self._connect_timeout_ms = connect_timeout_ms
        self._on_listening = on_listening
        self._abort_check = abort_check
        # O modo reverse prende o hvcam-host.exe aqui depois do open
        # (_reverse.py atribui cap._host_proc); release() o encerra.
        self._host_proc = None
        self._managed_host = on_listening is not None
        self._cam = None
        self._opened = False
        self.last_frame_id = None
        self.last_capture_ts_ms = None
        self.source_dropped = 0
        self.mjpeg_decode_errors = 0
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
            kwargs = {"port": port}
            if host is not None:
                kwargs["host"] = host
                if self._connect_timeout_ms is not None:
                    kwargs["connect_timeout_ms"] = self._connect_timeout_ms
            elif scheme == "hvcam+reverse":
                kwargs["listen"] = True
                if self._connect_timeout_ms is not None:
                    kwargs["timeout_ms"] = self._connect_timeout_ms
                if self._on_listening is not None:
                    kwargs["on_listening"] = self._on_listening
                if self._abort_check is not None:
                    kwargs["abort_check"] = self._abort_check
            elif self._connect_timeout_ms is not None:
                kwargs["connect_timeout_ms"] = self._connect_timeout_ms
            self._cam = camera_factory(**kwargs)
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
            if info.get("format") == 4:
                self.mjpeg_decode_errors += 1
                if self.mjpeg_decode_errors == 1 and not self._managed_host:
                    # HVCAM v1 so permite pedir o quadro mais recente.
                    # O formato de captura pertence ao host externo.
                    _LOG.warning(
                        "hvcam: MJPEG corrompido recebido do host externo. "
                        "Se a falha persistir, reinicie o hvcam-host.exe "
                        "com --format nv12 ou --format yuy2. Em WSL sem "
                        "/dev/video*, encerre o host externo e execute ./run.sh "
                        "sem --source para tentar o host gerenciado."
                    )
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
