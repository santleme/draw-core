"""Decoder MJPEG isolado. O stderr do libjpeg pertence ao processo filho."""

import atexit
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, suppress

import cv2
import numpy as np

_SIZE = struct.Struct("<I")
# Resposta: largura e altura, ou largura zero e tamanho do texto do erro.
_RESULT = struct.Struct("<II")
_EXTRANEOUS_RE = re.compile(r"(\d+) extraneous bytes before marker 0xd9")
_DECODE_TIMEOUT_S = 2.0


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
    return img


def _serve():
    source = sys.stdin.buffer
    # O pipe binário fica reservado ao protocolo mesmo se o codec escrever
    # mensagens no stdout do filho.
    with os.fdopen(os.dup(1), "wb") as sink, ExitStack() as resources:
        os.dup2(2, 1)
        try:
            warnings = resources.enter_context(tempfile.TemporaryFile())
        except OSError:
            warnings = None
            sys.stderr.write(
                "hvcam: temporario indisponivel para avisos do decoder MJPEG.\n"
            )
        while len(header := source.read(_SIZE.size)) == _SIZE.size:
            (size,) = _SIZE.unpack(header)
            payload = source.read(size)
            if len(payload) != size:
                return
            try:
                buf = np.frombuffer(payload, dtype=np.uint8)
                image = (
                    cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if warnings is None
                    else _decode_in_worker(buf, warnings)
                )
                if image is None or image.size == 0:
                    raise ValueError("hvcam: payload MJPEG falhou no decode")
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

    def __init__(self, *, timeout_s=_DECODE_TIMEOUT_S):
        self._lock = threading.Lock()
        self._watch = threading.Condition()
        self._timeout_s = timeout_s
        self._deadline = None
        self._timed_out = False
        self._process = None
        self._watchdog = None

    def _kill_if_stalled(self, proc):
        with self._watch:
            while self._process is proc:
                if self._deadline is None:
                    self._watch.wait()
                    continue
                remaining = self._deadline - time.monotonic()
                if remaining > 0:
                    self._watch.wait(remaining)
                    continue
                self._timed_out = True
                # Encerrar o filho destrava tanto a escrita como a leitura
                # do pipe, inclusive no Windows (onde select nao os atende).
                proc.kill()
                return

    def _start(self):
        with self._watch:
            proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            self._process = proc
            self._watchdog = threading.Thread(
                target=self._kill_if_stalled,
                args=(proc,),
                name="draw-mjpeg-watchdog",
                daemon=True,
            )
            self._watchdog.start()
        return proc

    def decode(self, payload):
        with self._lock:
            self._timed_out = False
            try:
                proc = self._process or self._start()
                with self._watch:
                    self._deadline = time.monotonic() + self._timeout_s
                    self._watch.notify()
                try:
                    proc.stdin.write(_SIZE.pack(payload.nbytes))
                    proc.stdin.write(memoryview(payload).cast("B"))
                    proc.stdin.flush()
                    image = self._receive(proc.stdout)
                finally:
                    with self._watch:
                        self._deadline = None
                        self._watch.notify()
                    if self._timed_out:
                        raise TimeoutError
                return image
            except (OSError, EOFError) as error:
                self.close()
                if self._timed_out:
                    raise ValueError(
                        "hvcam: tempo limite do decoder MJPEG excedido"
                    ) from error
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
        with self._watch:
            proc, self._process = self._process, None
            watchdog, self._watchdog = self._watchdog, None
            self._watch.notify()
        if proc is not None:
            proc.kill()
            proc.wait()
            with suppress(OSError):
                proc.stdin.close()
            proc.stdout.close()
        if watchdog is not None:
            watchdog.join()


_decoder = _MjpegDecoder()
decode_mjpeg = _decoder.decode
atexit.register(_decoder.close)


if __name__ == "__main__":
    _serve()
