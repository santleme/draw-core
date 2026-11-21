"""Decoder MJPEG isolado. O stderr do libjpeg pertence ao processo filho."""

import atexit
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
from contextlib import suppress

import cv2
import numpy as np

_SIZE = struct.Struct("<I")
# Resposta: largura e altura, ou largura zero e tamanho do texto do erro.
_RESULT = struct.Struct("<II")
_EXTRANEOUS_RE = re.compile(r"(\d+) extraneous bytes before marker 0xd9")


def _classify_mjpeg_warning(line, buf):
    lower = line.lower()
    if "premature end" in lower:
        return "corrupt"
    if "corrupt jpeg" not in lower:
        return "other"
    match = _EXTRANEOUS_RE.search(lower)
    if match is None:
        return "corrupt"
    n = int(match.group(1))
    # O aviso so e benigno se os bytes antes do EOI forem padding UVC.
    if n <= 0 or buf.size < n + 2 or buf[-2] != 0xFF or buf[-1] != 0xD9:
        return "corrupt"
    tail = buf[-2 - n : -2]
    return "padding" if bool(np.all((tail == 0x00) | (tail == 0xFF))) else "corrupt"


def _decode_in_worker(buf, warnings):
    """Libjpeg pode devolver imagem parcial junto com aviso no fd 2."""
    warnings.seek(0)
    warnings.truncate()
    saved_fd = os.dup(2)
    try:
        os.dup2(warnings.fileno(), 2)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    finally:
        try:
            os.dup2(saved_fd, 2)
        finally:
            os.close(saved_fd)

    warnings.seek(0)
    corruption = None
    for raw_line in warnings:
        line = raw_line.decode("utf-8", errors="replace")
        verdict = _classify_mjpeg_warning(line, buf)
        if verdict == "other":
            sys.stderr.write(line)
        if verdict == "corrupt" and corruption is None:
            corruption = line.strip()
    if corruption is not None:
        raise ValueError(f"hvcam: payload MJPEG corrompido ({corruption})")
    if img is None or img.size == 0:
        raise ValueError("hvcam: payload MJPEG falhou no decode")
    return img


def _serve():
    source = sys.stdin.buffer
    # O pipe binário fica reservado ao protocolo mesmo se o codec escrever
    # mensagens no stdout do filho.
    with os.fdopen(os.dup(1), "wb") as sink, tempfile.TemporaryFile() as warnings:
        os.dup2(2, 1)
        while len(header := source.read(_SIZE.size)) == _SIZE.size:
            (size,) = _SIZE.unpack(header)
            payload = source.read(size)
            if len(payload) != size:
                return
            try:
                image = _decode_in_worker(
                    np.frombuffer(payload, dtype=np.uint8), warnings
                )
            except (ValueError, cv2.error) as error:
                message = str(error).encode("utf-8", errors="replace")
                sink.write(_RESULT.pack(0, len(message)))
                sink.write(message)
            else:
                height, width = image.shape[:2]
                sink.write(_RESULT.pack(width, height))
                sink.write(memoryview(image).cast("B"))
            sink.flush()


class _MjpegDecoder:
    """Reusa um processo e os pipes entre quadros, inclusive ao trocar de câmera."""

    def __init__(self):
        self._lock = threading.Lock()
        self._process = None

    def decode(self, payload):
        with self._lock:
            try:
                if self._process is None:
                    self._process = subprocess.Popen(
                        [sys.executable, os.path.abspath(__file__)],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                    )
                proc = self._process
                proc.stdin.write(_SIZE.pack(payload.nbytes))
                proc.stdin.write(memoryview(payload).cast("B"))
                proc.stdin.flush()
                return self._receive(proc.stdout)
            except (OSError, EOFError) as error:
                self.close()
                raise ValueError(
                    "hvcam: processo de decode MJPEG indisponivel"
                ) from error

    @staticmethod
    def _receive(pipe):
        header = pipe.read(_RESULT.size)
        if len(header) != _RESULT.size:
            raise EOFError("resposta incompleta do decoder MJPEG")
        width, height = _RESULT.unpack(header)
        if width == 0:
            message = pipe.read(height)
            if len(message) != height:
                raise EOFError("mensagem incompleta do decoder MJPEG")
            raise ValueError(message.decode("utf-8", errors="replace"))
        image = np.empty((height, width, 3), dtype=np.uint8)
        if pipe.readinto(memoryview(image).cast("B")) != image.nbytes:
            raise EOFError("quadro incompleto do decoder MJPEG")
        return image

    def close(self):
        proc, self._process = self._process, None
        if proc is not None:
            proc.kill()
            proc.wait()
            with suppress(OSError):
                proc.stdin.close()
            proc.stdout.close()


_decoder = _MjpegDecoder()
decode_mjpeg = _decoder.decode
atexit.register(_decoder.close)


if __name__ == "__main__":
    _serve()
